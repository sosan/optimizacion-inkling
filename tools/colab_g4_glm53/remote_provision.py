#!/usr/bin/env python3
"""Runs ON the Colab G4 VM. Streamed to `ssh ... python3 -` by colab_g4.py.

Pure Python equivalent of the REMOTE_EOF bash block in upstream start.sh:
installs deps, builds Project Maya for sm_120, downloads + packs the EXL3
quant and DFlash2 drafter, writes the server config and waits for /v1/models.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DFASH_URL = ("https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF/resolve/"
             "f3569acb7f9cee95e66d24442d75536eb00ab842/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf")
ROOT = Path("/content")
MAYA = ROOT / "project-maya"
SERVER_CFG = ROOT / "maya-glm53.json"
SERVER_LOG = ROOT / "maya-server.log"
CF_LOG = ROOT / "cf_tunnel.log"
CF_STATE = ROOT / "colab-g4-cf.json"
REMOTE_PORT = int(os.environ.get("REMOTE_PORT", "8090"))
CF_URL_REGEX = re.compile(r"https://(?!api\.)[a-zA-Z0-9-]+\.trycloudflare\.com")


def env_flag(name: str, default: str = "0") -> bool:
    """Read and REMOVE a transient flag: `colab exec` cells share one kernel, so
    leaving WATCHDOG_ONLY set would put later cells into watchdog mode."""
    return os.environ.pop(name, default).strip().lower() in ("1", "true", "yes", "y")


def cf_url(text: str) -> str | None:
    m = CF_URL_REGEX.search(text or "")
    return m.group(0) if m else None


def say(*parts: object) -> None:
    print(*parts, flush=True)


def step(n: int, title: str) -> None:
    say(f"\n[{n}/6] {title}")


def ok(msg: str, seconds: float | None = None) -> None:
    say(f"  ✓ {msg}" + (f" ({int(seconds)//60:02d}:{int(seconds)%60:02d})" if seconds is not None else ""))


def shell(cmd: list[str], log: Path | None = None, env: dict | None = None) -> int:
    e = dict(os.environ, **(env or {}))
    if log:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=e)
    return subprocess.call(cmd, env=e)


def fail(msg: str, log: Path | None = None, lines: int = 50) -> None:
    print(f"  ✗ {msg}", file=sys.stderr, flush=True)
    if log and log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        print(f"Últimas {lines} líneas de {log}:", file=sys.stderr)
        print("\n".join(tail), file=sys.stderr)
    sys.exit(42)


def need_commands(names: list[str]) -> bool:
    return all(shutil.which(n) for n in names)


def need_modules(names: list[str]) -> bool:
    code = "import " + ", ".join(names)
    return subprocess.call([sys.executable, "-c", code],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def download_dflash(dst: Path) -> None:
    tmp = dst.with_suffix(".gguf.tmp")
    tmp.unlink(missing_ok=True)
    with urllib.request.urlopen(DFASH_URL, timeout=60) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, 1024 * 1024)
    tmp.replace(dst)


def download_model(repo: str, dst: str, tty: bool) -> None:
    env = dict(os.environ, HF_XET_HIGH_PERFORMANCE="1")
    if not tty:
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    code = (
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(repo_id=%r, local_dir=%r, max_workers=8)\n" % (repo, dst)
    )
    rc = subprocess.call([sys.executable, "-c", code], env=env)
    if rc:
        raise RuntimeError(f"snapshot_download falló con rc={rc}")


def build_maya() -> None:
    if shell([sys.executable, "-c", "import setup as S; S.get_llama_cpp()"], ROOT / "maya-build.log"):
        raise RuntimeError("get_llama_cpp falló")
    cmd = ["cmake", "-G", "Ninja", "-S", str(MAYA), "-B", str(MAYA / "build"),
           "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON",
           "-DCMAKE_CUDA_ARCHITECTURES=120", "-DSTRATA_NATIVE_EXPERTS=ON",
           "-DSTRATA_BUILD_TESTS=OFF", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
           "-DCUDAToolkit_ROOT=/usr/local/cuda",
           f"-DSTRATA_GGML_DIR={MAYA / 'third_party' / 'llama.cpp'}"]
    if shell(cmd, ROOT / "maya-build.log"):
        raise RuntimeError("configure de Maya falló")
    if shell(["cmake", "--build", str(MAYA / "build"), "--target", "maya", "-j", "24"], ROOT / "maya-build.log"):
        raise RuntimeError("build de Maya falló")


def pack_model(model_dir: Path) -> None:
    (model_dir / "pack").mkdir(parents=True, exist_ok=True)
    if shell([sys.executable, str(MAYA / "tools" / "exl3_pack.py"), "--model", str(model_dir)],
             ROOT / "maya-pack.log"):
        raise RuntimeError("exl3_pack falló")
    (model_dir / "pack" / ".packed_ok").write_text("ok\n", encoding="utf-8")


def write_config(model_dir: str, ctx: int, spec_p: str, tier: str) -> None:
    pack = f"/content/models/{model_dir}/pack"
    args = ["--glm-pack", pack, "--max-context", str(ctx), "--vram-reserve-mib", "200",
            "--glm-ram-resident", "--dflash",
            "/content/models/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf",
            "--spec-min-p", spec_p]
    if ctx <= 131072:
        args += ["--kv", "k8v4"]
        kv_desc = "k8v4"
    else:
        args += ["--kv", "int8", "--kv-resident", "20480"]
        kv_desc = "int8 (kv_resident=20480)"
    cfg = {
        "exe": str(MAYA / "build" / "maya"),
        "pack": pack,
        "tokenizer": f"{pack}/tokenizer",
        "model": f"glm-5.3-flash-uncensored-{tier}bpw",
        "model_name": f"glm-5.3-flash-uncensored-{tier}bpw",
        "host": "127.0.0.1", "port": 8090, "max_context": ctx, "args": args,
        "env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
                "MAYA_GLM_PCIE_SHARE": "0.00", "MAYA_GLM_CPU_LANE": "18",
                "MAYA_GLM_NGRAM_DRAFT": "1", "MAYA_GLM_DSA_GRAPHS": "1"},
    }
    SERVER_CFG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    ok(f"Config lista: {kv_desc}, DFlash2, spec_min_p={spec_p}, CPU_LANE=18, DSA_GRAPHS=1")


def wait_server(log: Path, timeout: int = 540) -> None:
    t0 = time.monotonic()
    last_report = 0
    while True:
        if subprocess.call(["pgrep", "-f", "serve/server.py"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) != 0:
            fail("El servidor Maya salió durante la carga.", log, 40)
        elapsed = int(time.monotonic() - t0)
        if elapsed >= timeout:
            fail("El servidor Maya no estuvo listo en 540s.", log, 40)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{REMOTE_PORT}/v1/models", timeout=1) as r:
                if r.status == 200:
                    ok("Modelo listo", elapsed)
                    return
        except Exception:
            pass
        if elapsed // 30 > last_report:
            last_report = elapsed // 30
            say(f"  cargando... {elapsed//60:02d}:{elapsed%60:02d}")
        time.sleep(2)


# ---- keep-alive / auto-heal -------------------------------------------------

def server_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{REMOTE_PORT}/v1/models", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def server_alive() -> bool:
    return subprocess.call(["pgrep", "-f", "serve/server.py"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


def start_server() -> None:
    """Restart the Maya server from the config written during provisioning."""
    if not SERVER_CFG.exists():
        raise RuntimeError(f"No existe {SERVER_CFG}; aprovisiona la sesión primero")
    for pattern in ("serve/server.py", "build/maya"):
        subprocess.call(["pkill", "-f", pattern], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    time.sleep(1)
    SERVER_LOG.write_text("", encoding="utf-8")
    with SERVER_LOG.open("a", encoding="utf-8") as f:
        subprocess.Popen([sys.executable, str(MAYA / "serve" / "server.py"), "--engine", "maya",
                          "--config", str(SERVER_CFG), "--host", "127.0.0.1", "--port", str(REMOTE_PORT)],
                         stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    wait_server(SERVER_LOG, 540)


def cloudflare_alive() -> bool:
    return subprocess.call(["pgrep", "-f", "[c]loudflared"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


def start_cloudflare_tunnel() -> str:
    """Open a Cloudflare Quick Tunnel to the VM's server and persist its URL."""
    if not shutil.which("cloudflared"):
        CF_LOG.parent.mkdir(parents=True, exist_ok=True)
        shell(["curl", "-fsSL",
               "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
               "-o", "/usr/local/bin/cloudflared"], CF_LOG)
        shell(["chmod", "+x", "/usr/local/bin/cloudflared"], CF_LOG)
    subprocess.call(["pkill", "-f", "[c]loudflared"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    CF_LOG.write_text("", encoding="utf-8")
    with CF_LOG.open("a", encoding="utf-8") as f:
        subprocess.Popen(["cloudflared", "--no-autoupdate", "tunnel", "--url",
                          f"http://localhost:{REMOTE_PORT}"],
                         stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    for _ in range(30):
        url = cf_url(CF_LOG.read_text(encoding="utf-8", errors="replace"))
        if url:
            base = url + "/v1"
            CF_STATE.write_text(json.dumps({"url": url, "base_url": base}) + "\n", encoding="utf-8")
            say(f"TUNNEL_BASE_URL={base}")
            return base
        time.sleep(1)
    raise RuntimeError("cloudflared no publicó su URL en 30s; revisa " + str(CF_LOG))


def run_watchdog(seconds: int, cf_mode: bool = False) -> None:
    """Keep the Jupyter cell busy with heartbeats and heal the server / tunnel."""
    say(f"[watchdog] Activo {max(0, seconds)}s (cloudflare={'on' if cf_mode else 'off'}).")
    t0 = time.monotonic()
    last_beat = 0.0
    failures = 0
    while time.monotonic() - t0 < seconds:
        if server_healthy():
            failures = 0
        else:
            failures += 1
            if failures >= 2 or not server_alive():
                say("[watchdog] Servidor no saludable; reiniciando Maya...")
                start_server()
                failures = 0
        if cf_mode and (not cloudflare_alive() or not CF_STATE.exists()):
            say("[watchdog] cloudflared caído; reabriendo túnel...")
            try:
                start_cloudflare_tunnel()
            except Exception as e:
                say(f"[watchdog] No se pudo reabrir el túnel: {e}")
        now = time.monotonic()
        if now - last_beat >= 60:
            say(f"[watchdog] heartbeat {int(now - t0):04d}s | servidor="
                f"{'OK' if server_healthy() else 'DEGRADED'} | túnel="
                f"{'cf' if cf_mode and cloudflare_alive() else ('ssh' if not cf_mode else 'down')}")
            last_beat = now
        time.sleep(min(20.0, max(0.0, seconds - (time.monotonic() - t0))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_repo")
    ap.add_argument("model_dir")
    ap.add_argument("ctx", type=int)
    ap.add_argument("spec_min_p")
    ap.add_argument("tier")
    ap.add_argument("is_tty", type=int, nargs="?", default=0)
    ap.add_argument("--cloudflare", action="store_true", help="abrir Cloudflare Quick Tunnel tras cargar el modelo")
    a = ap.parse_args()

    # Transient per-cell flags: pop them so a later cell in the same Jupyter
    # kernel does not accidentally run the watchdog again.
    watchdog_only = env_flag("WATCHDOG_ONLY")
    watchdog_seconds = int(os.environ.pop("COLAB_GLM_WATCHDOG_SECONDS", "900"))
    cf_mode = a.cloudflare or env_flag("COLAB_G4_ENABLE_CF")
    if watchdog_only:
        run_watchdog(watchdog_seconds, cf_mode)
        return 0

    os.environ["PATH"] = "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")
    model_dir = ROOT / "models" / a.model_dir

    step(1, "Comprobando herramientas y dependencias Python")
    mods = ["huggingface_hub", "jinja2", "regex", "yaml", "tqdm", "requests", "PIL", "psutil", "jsonschema"]
    if need_commands(["ninja", "cmake"]) and need_modules(mods):
        ok("En caché")
    else:
        t0 = time.monotonic()
        if not need_commands(["ninja", "cmake"]):
            for c in (["apt-get", "-qq", "update"],
                      ["apt-get", "-qq", "install", "-y", "aria2", "ninja-build", "cmake", "build-essential"]):
                if shell(c, ROOT / "maya-deps.log"):
                    fail("No se pudieron instalar las herramientas de build.", ROOT / "maya-deps.log")
        if not need_modules(mods):
            if shell([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "jinja2", "regex",
                      "pyyaml", "tqdm", "requests", "pillow", "psutil", "jsonschema>=4.18,<5"],
                     ROOT / "maya-deps.log"):
                fail("No se pudieron instalar los paquetes Python.", ROOT / "maya-deps.log")
        ok("Dependencias listas", time.monotonic() - t0)

    step(2, "Compilando Project Maya (CUDA sm_120)")
    if (MAYA / "build" / "maya").exists():
        ok("En caché")
    else:
        (ROOT / "maya-build.log").write_text("", encoding="utf-8")
        t0 = time.monotonic()
        if not (MAYA / "CMakeLists.txt").exists():
            shutil.rmtree(MAYA, ignore_errors=True)
            if shell(["git", "clone", "--depth", "1", "https://github.com/mw00/project-maya.git", str(MAYA)],
                     ROOT / "maya-build.log"):
                fail("No se pudo clonar Project Maya.", ROOT / "maya-build.log")
        try:
            build_maya()
        except RuntimeError as e:
            fail(str(e), ROOT / "maya-build.log")
        ok("Maya compilado", time.monotonic() - t0)

    step(3, "Descargando DFlash2 y el EXL3")
    dflash = ROOT / "models" / "dflash2" / "GLM-5.3-Flash-DFlash2-Q4_K_M.gguf"
    dflash.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    need_d, need_m = not dflash.exists(), not (model_dir / "pack" / ".packed_ok").exists()
    if not need_d and not need_m:
        ok("En caché")
    else:
        t0 = time.monotonic()
        if need_d:
            try:
                download_dflash(dflash)
            except Exception as e:
                fail(f"Falló la descarga de DFlash2: {e}")
            ok("DFlash2 descargado", time.monotonic() - t0)
        else:
            ok("DFlash2 en caché")
        if need_m:
            for d in (ROOT / "models").glob("MikeRoz-*"):
                if d.is_dir() and d != model_dir:
                    shutil.rmtree(d, ignore_errors=True)
            say(f"  ▸ Descargando {a.hf_repo} -> {model_dir}")
            try:
                download_model(a.hf_repo, str(model_dir), a.is_tty == 1)
            except Exception as e:
                fail(f"Falló la descarga del modelo: {e}")
            ok("Pesos descargados", time.monotonic() - t0)

    step(4, "Empaquetando pesos EXL3 (zero-copy)")
    if (model_dir / "pack" / ".packed_ok").exists():
        ok("En caché")
    else:
        (ROOT / "maya-pack.log").write_text("", encoding="utf-8")
        t0 = time.monotonic()
        try:
            pack_model(model_dir)
        except RuntimeError as e:
            fail(str(e), ROOT / "maya-pack.log")
        ok("Empaquetado", time.monotonic() - t0)

    step(5, "Escribiendo la configuración del servidor")
    write_config(a.model_dir, a.ctx, a.spec_min_p, a.tier)

    step(6, f"Cargando GLM-5.3-Flash ({a.tier} bpw) en VRAM")
    say("  Esperando a que cargue el modelo (frío ~3-5 min, caliente ~25s)...")
    try:
        start_server()
    except Exception as e:
        fail(str(e), SERVER_LOG)
    if cf_mode:
        say("[cloudflare] Abriendo túnel público opt-in...")
        start_cloudflare_tunnel()
    return 0


if __name__ == "__main__":
    sys.exit(main())
