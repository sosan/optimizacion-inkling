#!/usr/bin/env python3
"""Cross-platform benchmark for the Inkling 2 Mini engine.

Replaces bench_perf.sh and bench_perf.ps1. Works on Linux and Windows.

Usage:
    python tests/bench_perf.py cli                     # engine CLI: prefill + decode tok/s
    python tests/bench_perf.py http                    # through serve/server.py (/v1/chat/completions)
    python tests/bench_perf.py both --gen 256 --ctx 163840
    python tests/bench_perf.py cli --vram-sample

Logs and CSV go to bench-results/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def find_root() -> Path:
    """Walk up from this file to the repo root."""
    p = Path(__file__).resolve()
    for d in [p.parent, *p.parent.parents]:
        if (d / "engine-inkling-custom" / "CMakeLists.txt").exists():
            return d
    raise SystemExit("ERROR: repo root not found (looking for engine-inkling-custom/CMakeLists.txt)")


ROOT = find_root()

DEFAULT_PROMPT = (
    "Write a short story about a lighthouse and the sea. Describe the light, the weather "
    "and the keeper's thoughts. Keep it coherent and descriptive."
)


def http_ok(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def vram_mib() -> tuple[int, int] | None:
    """(used, total) MiB via nvidia-smi, or None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).splitlines()[0]
        parts = [x.strip() for x in out.split(",")[:2]]
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def make_ids(args, sys_path: Path) -> None:
    if args.ids.exists():
        return
    if not args.prompt_file.exists():
        args.prompt_file.write_text(DEFAULT_PROMPT + "\n", encoding="utf-8")
    print(f"Generating token ids from {args.prompt_file} ...")
    code = (
        "import sys, pathlib\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from strata_tokenizer import Tokenizer\n"
        "prompt = pathlib.Path(sys.argv[2]).read_text(encoding='utf-8')\n"
        "ids = Tokenizer.from_dir(sys.argv[3]).encode(prompt)\n"
        "pathlib.Path(sys.argv[4]).write_text(','.join(str(i) for i in ids), encoding='utf-8')\n"
        "print(f'  {len(ids)} tokens')\n"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(sys_path), str(args.prompt_file),
         str(args.tokens_dir), str(args.ids)],
        check=True, cwd=ROOT,
    )


def config_env(path: Path) -> dict[str, str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}


def start_vram_sampler(csv_path: Path):
    if not args.vram_sample or not shutil.which("nvidia-smi"):
        return None
    fh = csv_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", "1000"],
        stdout=fh, stderr=subprocess.DEVNULL,
    ), fh


def stop_vram_sampler(proc_fh) -> None:
    if not proc_fh:
        return
    proc, fh = proc_fh
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    fh.close()


def print_timing(log: Path, gen: int) -> None:
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    p_line = next((l for l in reversed(lines) if l.startswith("prompt: ")), None)
    g_line = next((l for l in reversed(lines) if l.startswith("generation: ")), None)
    e_line = next((l for l in reversed(lines) if l.startswith("@END ")), None)
    if p_line and g_line:
        p, g = p_line.split(), g_line.split()
        ptok, psec, gtok, gsec = float(p[1]), float(p[4]), float(g[1]), float(g[4])
        pr = ptok / psec if psec > 0 else 0.0
        gr = gtok / gsec if gsec > 0 else 0.0
        print(f"  prompt={ptok:.0f} tokens  generated={gtok:.0f} tokens")
        print(f"  prefill: {psec:.2f} s  -> {pr:.1f} tok/s (engine: {p[6].lstrip('(')})")
        print(f"  decode : {gsec:.2f} s  -> {gr:.1f} tok/s (engine: {g[6].lstrip('(')})")
    elif e_line:
        e = e_line.split()
        ptok, psec, dsec = float(e[2]), float(e[4]), float(e[5])
        pr = ptok / psec if psec > 0 else 0.0
        gr = gen / dsec if dsec > 0 else 0.0
        print(f"  reason={e[1]}  prompt={ptok:.0f} tokens  generated={gen}")
        print(f"  prefill: {psec:.2f} s  -> {pr:.1f} tok/s")
        print(f"  decode : {dsec:.2f} s  -> {gr:.1f} tok/s")
    else:
        print("ERROR: no timing line in log. Last 30 lines:", file=sys.stderr)
        print("\n".join(lines[-30:]), file=sys.stderr)
        raise SystemExit(1)


def run_cli(args, stamp: str) -> None:
    print(f"\n=== CLI benchmark: gen={args.gen} ctx={args.ctx} ===")
    if http_ok(args.port):
        raise SystemExit(
            f"ERROR: a server is already listening on port {args.port} and holds the VRAM. "
            "Stop it and re-run."
        )
    vram = vram_mib()
    if vram:
        free = vram[1] - vram[0]
        print(f"VRAM now: {vram[0]}/{vram[1]} MiB used, {free} MiB free")
        if free < 6000:
            raise SystemExit(
                f"ERROR: only {free} MiB free — something else holds the GPU. "
                "Check with nvidia-smi, or pass a smaller --ctx."
            )

    make_ids(args, ROOT / "tools")
    log = args.out / f"cli_{stamp}.log"
    sampler = start_vram_sampler(args.out / f"cli_{stamp}_vram.csv")
    env = dict(os.environ)
    env.update(config_env(args.config))
    cmd = [str(args.bin), "--model", str(args.model), "--ctx", str(args.ctx),
           "--chunk", "2048", "--tokens", f"@{args.ids}", "--gen", str(args.gen)]
    print(f"Running engine (log: {log}) ...")
    try:
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    finally:
        stop_vram_sampler(sampler)
    if rc != 0:
        print(f"ERROR: engine exited with {rc}. Last 30 lines:", file=sys.stderr)
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]), file=sys.stderr)
        raise SystemExit(1)
    print_timing(log, args.gen)
    print(f"  log: {log}")


def run_http(args, stamp: str) -> None:
    print(f"\n=== HTTP benchmark: port={args.port} gen={args.gen} ===")
    if http_ok(args.port):
        raise SystemExit(f"ERROR: port {args.port} already in use (is a server running?)")

    log = args.out / f"http_{stamp}.log"
    err = args.out / f"http_{stamp}.err"
    sampler = start_vram_sampler(args.out / f"http_{stamp}_vram.csv")
    cmd = [sys.executable, "serve/server.py", "--engine", "inkling-custom-engine",
           "--config", str(args.config), "--host", "127.0.0.1", "--port", str(args.port)]
    lf, ef = log.open("w", encoding="utf-8"), err.open("w", encoding="utf-8")
    server = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=ef)
    try:
        print("Waiting for /health ...")
        ready = False
        for i in range(1, 301):
            if server.poll() is not None:
                raise SystemExit(f"Server died during startup:\n" + "\n".join(
                    log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
            if http_ok(args.port):
                ready = True
                print(f"  ready after {i}s")
                break
            time.sleep(1)
        if not ready:
            raise SystemExit("Timeout waiting for /health")

        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file.exists() else "Write a haiku about the sea."
        body = json.dumps({"model": "inkling-2-mini", "max_tokens": args.gen,
                           "reasoning_effort": "none",
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.load(r)
        dt = time.perf_counter() - t0
        usage = data.get("usage") or {}
        pt, ct = float(usage.get("prompt_tokens", 0)), float(usage.get("completion_tokens", 0))
        print(f"  prompt={pt:.0f} tokens  generated={ct:.0f} tokens  wall={dt:.2f}s")
        if dt > 0:
            print(f"  decode: {ct / dt:.1f} tok/s (wall, includes prefill)")
        answer = (data["choices"][0]["message"]["content"] or "").replace("\n", " ")
        print(f"  answer: {answer[:200]}")
    finally:
        stop_vram_sampler(sampler)
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        lf.close()
        ef.close()
    print(f"  log: {log}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["cli", "http", "both"], default="cli")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=163840)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--ids", type=Path, default=ROOT / "IDS.txt")
    ap.add_argument("--prompt-file", type=Path, default=ROOT / "prompt.txt")
    ap.add_argument("--bin", type=Path, default=ROOT / ("build/inkling-custom-engine.exe" if os.name == "nt"
                                                        else "build/inkling-custom-engine"))
    ap.add_argument("--model", type=Path, default=ROOT / "models/Inkling-2-Mini-UD-Q4_K_XL.gguf")
    ap.add_argument("--tokens-dir", type=Path, default=ROOT / "ik2/tokenizer")
    ap.add_argument("--config", type=Path, default=ROOT / "docs/config/ik2-mini-text.json")
    ap.add_argument("--vram-sample", action="store_true")
    return ap.parse_args()


def main() -> None:
    global args
    args = parse_args()
    args.out = ROOT / "bench-results"
    args.out.mkdir(parents=True, exist_ok=True)
    for f in (args.bin, args.model, args.config):
        if not f.exists():
            raise SystemExit(f"ERROR: missing {f}")
    if not (args.tokens_dir / "vocab.json").exists():
        raise SystemExit(f"ERROR: tokenizer missing at {args.tokens_dir} (run tools/strata_tokenizer.py)")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.mode in ("cli", "both"):
        run_cli(args, stamp + ("_cli" if args.mode == "both" else ""))
    if args.mode in ("http", "both"):
        run_http(args, stamp + ("_http" if args.mode == "both" else ""))
    print(f"\nResults in: {args.out}")


if __name__ == "__main__":
    main()
