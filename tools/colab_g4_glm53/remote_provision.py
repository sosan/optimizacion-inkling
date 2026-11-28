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
    (ROOT / "maya-glm53.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
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
            with urllib.request.urlopen("http://127.0.0.1:8090/v1/models", timeout=1) as r:
                if r.status == 200:
                    ok("Modelo listo", elapsed)
                    return
        except Exception:
            pass
        if elapsed // 30 > last_report:
            last_report = elapsed // 30
            say(f"  cargando... {elapsed//60:02d}:{elapsed%60:02d}")
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_repo")
    ap.add_argument("model_dir")
    ap.add_argument("ctx", type=int)
    ap.add_argument("spec_min_p")
    ap.add_argument("tier")
    ap.add_argument("is_tty", type=int, nargs="?", default=0)
    a = ap.parse_args()

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
    for pattern in ("serve/server.py", "build/maya"):
        subprocess.call(["pkill", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    log = ROOT / "maya-server.log"
    log.write_text("", encoding="utf-8")
    with log.open("a", encoding="utf-8") as f:
        subprocess.Popen([sys.executable, str(MAYA / "serve" / "server.py"), "--engine", "maya",
                          "--config", str(ROOT / "maya-glm53.json"), "--host", "127.0.0.1", "--port", "8090"],
                         stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    say("  Esperando a que cargue el modelo (frío ~3-5 min, caliente ~25s)...")
    wait_server(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
