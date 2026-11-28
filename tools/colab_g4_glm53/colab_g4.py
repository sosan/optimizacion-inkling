#!/usr/bin/env python3
"""One-command launcher for GLM-5.3-Flash EXL3 + DFlash2 on Google Colab G4.

Python port of alesha-pro/tools `colab-g4-glm53` (start.sh / tunnel.sh / stop.sh),
working on Linux, macOS and Windows. One CLI, three subcommands:

    python colab_g4.py start [2.51|3.05|4.05] [--ctx 131072] [--port 18100] [--bg]
    python colab_g4.py tunnel [glm53] [18100] [--bg]
    python colab_g4.py stop [glm53] [18100]

It provisions the G4 VM with `colab`, builds Project Maya for sm_120 on the VM,
downloads and packs the weights, starts the OpenAI/Anthropic server on remote
:8090 and opens a self-healing SSH port-forward to 127.0.0.1:<port>. Once
/v1/models answers 200 OK it prints the local endpoints, model name, env
snippets and the stop command.

Prerequisites: `google-colab-cli` installed and authenticated, and OpenSSH
(`ssh`, `ssh-keygen`) available. On Windows, OpenSSH Client is enough.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

IS_WIN = platform.system() == "Windows"
TIERS = {
    "2.51": {"repo": "MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3",
             "dir": "MikeRoz-2.51bpw-uncensored", "p": "0.95"},
    "3.05": {"repo": "MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3",
             "dir": "MikeRoz-3.05bpw-uncensored", "p": "0.90"},
    "4.05": {"repo": "MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3",
             "dir": "MikeRoz-4.05bpw-uncensored", "p": "0.95"},
}
DFASH_URL = ("https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF/resolve/"
             "f3569acb7f9cee95e66d24442d75536eb00ab842/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf")
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
C = {k: (("\033[" + v + "m") if COLOR else "") for k, v in {
    "reset": "0", "bold": "1", "dim": "2", "cyan": "36", "green": "32", "yellow": "33", "red": "31"}.items()}


def paint(text: str, color: str) -> str:
    return f"{C[color]}{text}{C['reset']}"


def key_path() -> Path:
    return Path(os.environ.get("COLAB_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519_colab"))).expanduser()


def colab_path() -> Path:
    env = os.environ.get("COLAB_BIN")
    if env:
        return Path(env).expanduser()
    found = shutil.which("colab")
    if found:
        return Path(found)
    name = "colab.exe" if IS_WIN else "colab"
    return Path.home() / ".local" / "bin" / name


def colab() -> str:
    p = colab_path()
    if not p.exists():
        raise SystemExit(paint(f"[colab] CLI no encontrado en {p}. Instala google-colab-cli o define COLAB_BIN.", "red"))
    return str(p)


def paths(session: str, port: int) -> dict[str, Path]:
    d = Path(tempfile.gettempdir())
    stem = f"colab-tunnel-{session}-{port}"
    return {"pid": d / f"{stem}.pid", "log": d / f"{stem}.log", "meta": d / f"{stem}.env"}


def run(cmd: list[str], *, capture: bool = False, env: dict | None = None,
        input_text: str | None = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=capture, env=env,
                          input=input_text, check=check)


def cmd_ok(cmd: list[str]) -> bool:
    return run(cmd, capture=True).returncode == 0


def ensure_key() -> Path:
    key = key_path()
    if key.exists():
        return key
    key.parent.mkdir(parents=True, exist_ok=True)
    print(paint("[ssh]", "cyan"), f"Generando clave ed25519 en {key} ...")
    r = run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "colab-cli", "-f", str(key)],
            capture=True)
    if r.returncode:
        raise SystemExit(paint(f"[ssh] Falló ssh-keygen: {r.stderr.strip()}", "red"))
    return key


def q(value: str) -> str:
    """Quote one token for an ssh ProxyCommand string (POSIX sh / Windows cmd)."""
    return '"' + value.replace('"', '\\"') + '"' if (" " in value or "(" in value) else value


def proxy_option(session: str, key: Path) -> str:
    inner = f"{q(colab())} ssh --proxy-mode -s {session} -i {q(str(key))}"
    return f"ProxyCommand={inner}"


def ssh_args(session: str, key: Path, *, forward: str | None = None, command: list[str] | None = None) -> list[str]:
    args = ["ssh", "-i", str(key), "-o", proxy_option(session, key),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]
    if forward:
        args += ["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=20", "-L", forward]
    else:
        args += ["-o", "ConnectTimeout=25"]
    args.append("root@colab")
    if command:
        args += command
    return args


HARD_ERROR = re.compile(
    r"Allocation refused|precondition failed|Backend rejected|TooManyAssignmentsError|"
    r"Too Many Requests|Unauthorized|Forbidden|not entitled|no quota|insufficient|"
    r"invalid_grant|token.*expired|HTTP[^\d]*(401|403|429)", re.I)
TRANSIENT = re.compile(
    r"Service\s*Unavailable|503|502|504|ReadTimeout|timed out|Temporary|capacity|Resource\s*Exhausted", re.I)
GONE = re.compile(r"not found|no active sessions|does not exist", re.I)


def session_active(session: str) -> bool:
    r = run([colab(), "status", "-s", session], capture=True)
    return r.returncode == 0 and not GONE.search(r.stdout + r.stderr)


def create_session(session: str, wait: bool = True) -> None:
    delays = [30, 45, 60, 90, 120]
    cap = int(os.environ.get("COLAB_WAIT_CAP", "120"))
    base = os.environ.get("COLAB_RETRY_BASE")
    max_attempts = int(os.environ.get("COLAB_MAX_ATTEMPTS", "0"))
    attempt = 1
    print()
    print(" ", paint("▸", "yellow"), f"Creando sesión Colab G4 {paint(session, 'bold')} (~10.05 CU/hr)...")
    while True:
        env = dict(os.environ, COLAB_CLI_ASSIGN_RETRIES="0")
        r = run([colab(), "new", "-s", session, "--gpu", "G4"], capture=True, env=env)
        out = (r.stdout + "\n" + r.stderr).strip()
        clean = " ".join(l for l in out.splitlines() if not l.startswith(("╭", "│", "╰")))
        if r.returncode == 0 and not re.search(r"Error:|Traceback|Service Unavailable|Allocation refused|Backend rejected", out, re.I):
            print(" ", paint("✓", "green"), f"Sesión {paint(session, 'bold')} asignada (intento #{attempt})")
            return
        if HARD_ERROR.search(clean):
            raise SystemExit(paint(f"\n  ✗ Colab rechazó la asignación (sin reintento):\n{out}", "yellow"))
        if wait and TRANSIENT.search(clean):
            if max_attempts and attempt >= max_attempts:
                raise SystemExit(paint(f"\n  ✗ COLAB_MAX_ATTEMPTS={max_attempts} sin G4 disponible.", "yellow"))
            delay = int(base) if base else delays[min(attempt - 1, len(delays) - 1)]
            if not base and delay >= 30:
                delay += random.randint(-4, 4)
            delay = max(1, min(delay, cap))
            attempt += 1
            print(f"  G4 ocupado (503). Próxima comprobación (#{attempt}) en {delay}s...", flush=True)
            time.sleep(delay)
            continue
        raise SystemExit(paint(f"\n  ✗ Falló la creación de la sesión G4:\n{out}", "yellow"))


def show_colab_status(session: str) -> None:
    for sub in (["status", "-s", session], ["usage"]):
        r = run([colab()] + sub, capture=True)
        for line in (r.stdout or "").splitlines():
            if line.strip():
                print("  " + paint(line, "dim"))


REMOTE_SCRIPT = r"""
set -euo pipefail
HF_REPO="$1"
MODEL_DIR="$2"
CTX="$3"
SPEC_MIN_P="$4"
TIER="$5"
IS_TTY="${6:-0}"
[ -n "${NO_COLOR:-}" ] && IS_TTY=0

if [ "$IS_TTY" -eq 1 ]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_CYAN=$'\033[36m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_RED=$'\033[31m'
else
  C_RESET=""
  C_BOLD=""
  C_DIM=""
  C_CYAN=""
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
fi

cleanup_line() {
  [ "$IS_TTY" -eq 1 ] && printf '\r\033[K' || true
}
trap 'cleanup_line; exit 130' INT TERM

fmt_duration() {
  local s="${1:-0}"
  printf '%02d:%02d' $((s / 60)) $((s % 60))
}

step_header() {
  printf '\n%s[%s/6]%s %s%s%s\n' "$C_CYAN$C_BOLD" "$1" "$C_RESET" "$C_BOLD" "$2" "$C_RESET"
}

step_ok() {
  printf '  %s✓%s %s\n' "$C_GREEN" "$C_RESET" "$1"
}

fail_with_log() {
  local msg="$1"
  local logfile="$2"
  local lines="${3:-50}"
  cleanup_line
  printf '  %s✗ %s%s\n' "$C_RED" "$msg" "$C_RESET" >&2
  if [ -f "$logfile" ]; then
    printf 'Last %s lines of %s:\n' "$lines" "$logfile" >&2
    tail -n "$lines" "$logfile" >&2 || true
  fi
  exit 42
}

LAST_DURATION=0
run_with_spinner() {
  local label="$1"
  local logfile="$2"
  shift 2
  local t0=$SECONDS
  "$@" > "$logfile" 2>&1 &
  local pid=$!
  local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
  local idx=0
  if [ "$IS_TTY" -eq 1 ]; then
    while kill -0 "$pid" 2>/dev/null; do
      local elapsed=$((SECONDS - t0))
      printf '\r\033[K  %s%s%s %s %s%s%s' \
        "$C_CYAN" "${frames[$idx]}" "$C_RESET" \
        "$label" \
        "$C_DIM" "$(fmt_duration "$elapsed")" "$C_RESET"
      idx=$(( (idx + 1) % ${#frames[@]} ))
      sleep 0.15
    done
    printf '\r\033[K'
  else
    printf '  %s\n' "$label"
  fi
  local rc=0
  wait "$pid" || rc=$?
  LAST_DURATION=$((SECONDS - t0))
  return "$rc"
}

export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

step_header 1 "Checking tools & Python dependencies"
install_deps() {
  if ! command -v ninja >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
    apt-get -qq update
    apt-get -qq install -y aria2 ninja-build cmake build-essential
  fi
  if ! python3 -c "import huggingface_hub, jinja2, regex, yaml, tqdm, requests, PIL, psutil, jsonschema" 2>/dev/null; then
    pip -q install -U huggingface_hub jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5"
  fi
}
if command -v ninja >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1 && \
   python3 -c "import huggingface_hub, jinja2, regex, yaml, tqdm, requests, PIL, psutil, jsonschema" 2>/dev/null; then
  step_ok "Cached"
else
  run_with_spinner "Installing build tools & Python packages..." /content/maya-deps.log install_deps || \
    fail_with_log "Failed to install tools/dependencies." /content/maya-deps.log 50
  step_ok "Dependencies ready in $(fmt_duration "$LAST_DURATION")"
fi

step_header 2 "Building Project Maya (CUDA sm_120)"
cd /content
if [ ! -f /content/project-maya/CMakeLists.txt ]; then
  rm -rf /content/project-maya
  run_with_spinner "Cloning Project Maya..." /content/maya-build.log \
    git clone --depth 1 https://github.com/mw00/project-maya.git /content/project-maya || \
    fail_with_log "Failed to clone Project Maya." /content/maya-build.log 50
fi
cd /content/project-maya

build_maya() {
  python3 -c 'import setup as S; S.get_llama_cpp()'
  cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DSTRATA_NATIVE_EXPERTS=ON \
    -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DSTRATA_GGML_DIR=/content/project-maya/third_party/llama.cpp
  cmake --build build --target maya -j 24
}

if [ -x /content/project-maya/build/maya ]; then
  step_ok "Cached"
else
  run_with_spinner "Building Maya (CUDA sm_120)..." /content/maya-build.log build_maya || \
    fail_with_log "Project Maya build failed." /content/maya-build.log 50
  step_ok "Built in $(fmt_duration "$LAST_DURATION")"
fi

step_header 3 "Downloading DFlash2 drafter (665 MB) & ${HF_REPO}"
mkdir -p /content/models/dflash2 "/content/models/${MODEL_DIR}"

need_dflash=0
need_model=0
[ ! -f /content/models/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf ] && need_dflash=1
[ ! -f "/content/models/${MODEL_DIR}/pack/.packed_ok" ] && need_model=1

download_dflash() {
  local tmp="/content/models/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf.tmp"
  rm -f "$tmp"
  curl -fSL -o "$tmp" \
    "https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF/resolve/f3569acb7f9cee95e66d24442d75536eb00ab842/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf"
  mv -f "$tmp" /content/models/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf
}

if [ "$need_dflash" -eq 0 ] && [ "$need_model" -eq 0 ]; then
  step_ok "Cached"
else
  if [ "$need_dflash" -eq 1 ]; then
    run_with_spinner "Downloading DFlash2 drafter (665 MB)..." /content/maya-dflash.log download_dflash || \
      fail_with_log "Failed to download DFlash2 drafter." /content/maya-dflash.log 50
    step_ok "DFlash2 drafter downloaded in $(fmt_duration "$LAST_DURATION")"
  else
    step_ok "DFlash2 drafter cached"
  fi

  if [ "$need_model" -eq 1 ]; then
    # Clean other model folders when switching tiers on Colab's ephemeral 236 GB disk
    for d in /content/models/MikeRoz-*; do
      [ -d "$d" ] && [ "$(basename "$d")" != "$MODEL_DIR" ] && rm -rf "$d"
    done
    dl_t0=$SECONDS
    printf '  %s▸%s Downloading %s -> /content/models/%s\n' "$C_CYAN" "$C_RESET" "$HF_REPO" "$MODEL_DIR"
    env HF_XET_HIGH_PERFORMANCE=1 python3 - "$HF_REPO" "/content/models/${MODEL_DIR}" "$IS_TTY" <<'PY'
import inspect, os, sys

repo, dst, is_tty = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
if not is_tty:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import snapshot_download

kwargs = {"repo_id": repo, "local_dir": dst, "max_workers": 8}
if is_tty:
    try:
        from huggingface_hub.utils import tqdm as hf_tqdm

        class LiveTqdm(hf_tqdm):
            def __init__(self, *args, **kw):
                if kw.get("disable") is None:
                    kw["disable"] = False
                super().__init__(*args, **kw)

        if "tqdm_class" in inspect.signature(snapshot_download).parameters:
            kwargs["tqdm_class"] = LiveTqdm
    except Exception:
        pass

snapshot_download(**kwargs)
PY
    step_ok "Model weights downloaded in $(fmt_duration "$((SECONDS - dl_t0))")"
  fi
fi

step_header 4 "Packing EXL3 weights (zero-copy)"
if [ -f "/content/models/${MODEL_DIR}/pack/.packed_ok" ]; then
  step_ok "Cached"
else
  rm -rf "/content/models/${MODEL_DIR}/pack"
  run_with_spinner "Packing EXL3 weights..." /content/maya-pack.log \
    python3 /content/project-maya/tools/exl3_pack.py --model "/content/models/${MODEL_DIR}" || \
    fail_with_log "Failed to pack EXL3 weights." /content/maya-pack.log 50
  touch "/content/models/${MODEL_DIR}/pack/.packed_ok"
  step_ok "Packed in $(fmt_duration "$LAST_DURATION")"
fi

step_header 5 "Writing Maya server config"
python3 - "$MODEL_DIR" "$CTX" "$SPEC_MIN_P" "$TIER" <<'PY'
import sys, json
model_dir, ctx, spec_p, tier = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
pack = f"/content/models/{model_dir}/pack"

args = [
    "--glm-pack", pack,
    "--max-context", str(ctx),
    "--vram-reserve-mib", "200",
    "--glm-ram-resident",
    "--dflash", "/content/models/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf",
    "--spec-min-p", spec_p,
]
if ctx <= 131072:
    args += ["--kv", "k8v4"]
else:
    # k8v4 cannot be combined with --kv-resident; use int8 KV streaming for 262K+
    args += ["--kv", "int8", "--kv-resident", "20480"]

cfg = {
    "exe": "/content/project-maya/build/maya",
    "pack": pack,
    "tokenizer": f"{pack}/tokenizer",
    "model": f"glm-5.3-flash-uncensored-{tier}bpw",
    "model_name": f"glm-5.3-flash-uncensored-{tier}bpw",
    "host": "127.0.0.1",
    "port": 8090,
    "max_context": ctx,
    "args": args,
    "env": {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "MAYA_GLM_PCIE_SHARE": "0.00",
        "MAYA_GLM_CPU_LANE": "18",
        "MAYA_GLM_NGRAM_DRAFT": "1",
        "MAYA_GLM_DSA_GRAPHS": "1",
    },
}
with open("/content/maya-glm53.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY
if [ "$CTX" -le 131072 ]; then
  KV_DESC="k8v4"
else
  KV_DESC="int8 (kv_resident=20480)"
fi
step_ok "Config ready: ${KV_DESC}, DFlash2, spec_min_p=${SPEC_MIN_P}, CPU_LANE=18, DSA_GRAPHS=1"

step_header 6 "Loading GLM-5.3-Flash (${TIER} bpw) into 96 GB VRAM"
pkill -f "serve/server.py" 2>/dev/null || true
pkill -f "build/maya" 2>/dev/null || true
sleep 1

nohup python3 /content/project-maya/serve/server.py \
  --engine maya \
  --config /content/maya-glm53.json \
  --host 127.0.0.1 \
  --port 8090 \
  > /content/maya-server.log 2>&1 < /dev/null &
srv_pid=$!

ready=0
t0=$SECONDS
last_probe=-1
last_log=0
frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
idx=0

if [ "$IS_TTY" -eq 0 ]; then
  printf '  Waiting for model load (cold ~3-5 min first time, ~25s warm)...\n'
fi

while true; do
  elapsed=$((SECONDS - t0))
  if ! kill -0 "$srv_pid" 2>/dev/null; then
    fail_with_log "Maya server exited unexpectedly during load." /content/maya-server.log 40
  fi
  if [ "$elapsed" -ge 540 ]; then
    break
  fi
  if [ "$last_probe" -lt 0 ] || [ $((elapsed - last_probe)) -ge 2 ]; then
    last_probe=$elapsed
    if curl -sf http://127.0.0.1:8090/v1/models >/dev/null 2>&1; then
      ready=1
      break
    fi
  fi
  if [ "$IS_TTY" -eq 1 ]; then
    printf '\r\033[K  %s%s%s Loading model into VRAM... %s%s%s' \
      "$C_CYAN" "${frames[$idx]}" "$C_RESET" \
      "$C_DIM" "$(fmt_duration "$elapsed")" "$C_RESET"
    idx=$(( (idx + 1) % ${#frames[@]} ))
    sleep 0.2
  else
    if [ "$elapsed" -gt 0 ] && [ $((elapsed / 30)) -gt "$last_log" ]; then
      last_log=$((elapsed / 30))
      printf '  still loading... (%s)\n' "$(fmt_duration "$elapsed")"
    fi
    sleep 2
  fi
done

cleanup_line
if [ "$ready" -ne 1 ]; then
  fail_with_log "Maya server did not become ready within 540s." /content/maya-server.log 40
fi
step_ok "Model ready in $(fmt_duration "$((SECONDS - t0))")"
"""


def provision(session: str, port: int, tier: str, ctx: int, bg: bool, wait: bool) -> None:
    key = ensure_key()
    t = TIERS[tier]
    args = [t["repo"], t["dir"], str(ctx), t["p"], tier, "1" if COLOR else "0"]
    ssh = ssh_args(session, key, command=["bash", "-s", "--"] + args)
    attempts, attempt = 4, 1
    while True:
        r = subprocess.run(ssh, input=REMOTE_SCRIPT, text=True)
        if r.returncode == 0:
            return
        if r.returncode in (42, 130):
            raise SystemExit(r.returncode)
        if attempt >= attempts:
            hint = f"python {Path(__file__).resolve()} stop {session} {port}"
            raise SystemExit(paint(f"\n  ✗ No se pudo aprovisionar tras {attempts} intentos. Para parar la facturación: {hint}", "yellow"))
        if not session_active(session):
            if wait:
                print(" ", paint("▸", "yellow"), "La sesión Colab ya no está activa; reasignando G4...")
                create_session(session, wait)
                attempt += 1
                continue
            raise SystemExit(paint(f"\n  ✗ La sesión {session} ya no está activa.", "yellow"))
        delay = attempt * 5
        print(f"\n  {paint('▸', 'yellow')} SSH/WebSocket interrumpido (rc={r.returncode}). Reconectando ({attempt}/{attempts}) en {delay}s...")
        time.sleep(delay)
        attempt += 1


def write_meta(session: str, port: int, tier: str, ctx: int) -> None:
    p = paths(session, port)["meta"]
    p.write_text(f'MODEL_TIER="{tier}"\nMODEL_CTX="{ctx}"\nMODEL_NAME="glm-5.3-flash-uncensored-{tier}bpw"\n',
                 encoding="utf-8")


def load_meta(session: str, port: int) -> dict[str, str]:
    meta = paths(session, port)["meta"]
    out: dict[str, str] = {}
    overrides = {k: os.environ.get(k) for k in ("MODEL_TIER", "MODEL_CTX", "MODEL_NAME")}
    if meta.exists():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    for k, v in overrides.items():
        if v:
            out[k] = v
    return out


def probe(port: int, timeout: float = 2.0) -> str | None:
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


def wait_tunnel(proc: subprocess.Popen, port: int, seconds: int, label: str = "") -> int:
    start = time.monotonic()
    last = -1.0
    if label and not COLOR:
        print(label, flush=True)
    while True:
        elapsed = time.monotonic() - start
        if proc.poll() is not None:
            return 2
        if elapsed >= seconds:
            return 1
        if last < 0 or elapsed - last >= 1:
            last = elapsed
            if probe(port) is not None:
                return 0
        if label and COLOR:
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            ch = frames[int(elapsed * 5) % len(frames)]
            print(f"\r\033[K  {paint(ch, 'cyan')} {label} {paint(f'{int(elapsed)//60:02d}:{int(elapsed)%60:02d}', 'dim')}", end="", flush=True)
        time.sleep(0.2 if COLOR else 1)


def model_name(meta: dict[str, str], port: int) -> str:
    if meta.get("MODEL_NAME"):
        return meta["MODEL_NAME"]
    if meta.get("MODEL_TIER"):
        return f"glm-5.3-flash-uncensored-{meta['MODEL_TIER']}bpw"
    body = probe(port)
    if body:
        try:
            ids = json.loads(body).get("data") or []
            if ids and ids[0].get("id"):
                return ids[0]["id"]
        except (ValueError, AttributeError, IndexError):
            pass
    return "glm-5.3-flash-uncensored-2.51bpw"


def ctx_label(raw: str) -> str:
    return {"131072": "131K (131072)", "262144": "262K (262144)"}.get(raw, raw or "131K / 262K (server active)")


def stop_command(session: str, port: int) -> str:
    script = Path(__file__).resolve()
    base = f"python {script} stop"
    return base if (session, port) == ("glm53", 18100) else f"{base} {session} {port}"


def ready_summary(session: str, port: int, mode: str) -> None:
    meta = load_meta(session, port)
    name = model_name(meta, port)
    print()
    print(paint("╭" + "─" * 72 + "╮", "green"))
    print(paint("│", "green"), paint("● GLM-5.3-Flash Uncensored — LISTO (200 OK)", "green bold"))
    print(paint("╰" + "─" * 72 + "╯", "green"))
    for k, v in (("Session", f"{session} (~10.05 CU/hr)"), ("GPU", "RTX PRO 6000 Blackwell (96 GB)"),
                 ("Tier / Contexto", f"{meta.get('MODEL_TIER', 'auto')} bpw / {ctx_label(meta.get('MODEL_CTX',''))}"),
                 ("Túnel", mode), ("Host local", "127.0.0.1"), ("Puerto local", str(port)),
                 ("Modelo", name), ("OpenAI Base URL", f"http://127.0.0.1:{port}/v1"),
                 ("Anthropic Base URL", f"http://127.0.0.1:{port}"), ("Web UI", f"http://127.0.0.1:{port}/"),
                 ("API Key", "sk-local (cualquier cadena no vacía)")):
        print(f"  {paint(f'{k:<19}', 'dim')} {v}")
    print()
    print(" ", paint("OpenAI env", "bold"))
    for line in (f'OPENAI_BASE_URL="http://127.0.0.1:{port}/v1"',
                 'OPENAI_API_KEY="sk-local"', f'OPENAI_MODEL="{name}"'):
        print("   ", paint(("set " + line) if IS_WIN else ("export " + line), "cyan"))
    print()
    print(" ", paint("Claude Code", "bold"))
    for line in (f'ANTHROPIC_BASE_URL="http://127.0.0.1:{port}"',
                 'ANTHROPIC_AUTH_TOKEN="sk-local"', f'ANTHROPIC_MODEL="{name}"'):
        print("   ", paint(("set " + line) if IS_WIN else (line + " \\"), "cyan"))
    print("   ", paint("claude" if IS_WIN else "    claude", "cyan"))
    print()
    print(" ", paint("Parar y cortar la facturación", "bold"))
    print("   ", paint(stop_command(session, port), "yellow bold"))
    print(paint("─" * 74, "dim"), "\n")


def terminate_pid(pid: int) -> None:
    if IS_WIN:
        r = run(["taskkill", "/PID", str(pid), "/T", "/F"], capture=True)
        if r.returncode:
            print(paint(f"[stop] No se pudo terminar PID {pid}: {r.stderr.strip()}", "yellow"))
    else:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass


def tunnel(session: str, port: int, bg: bool = False, child: bool = False) -> None:
    key = ensure_key()
    p = paths(session, port)
    remote_port = int(os.environ.get("REMOTE_PORT", "8090"))
    meta = load_meta(session, port)

    if bg and not child:
        if p["pid"].exists():
            try:
                old = int(p["pid"].read_text().strip())
            except ValueError:
                old = 0
            alive = False
            if old:
                if IS_WIN:
                    alive = cmd_ok(["tasklist", "/FI", f"PID eq {old}"])
                else:
                    try:
                        os.kill(old, 0)
                        alive = True
                    except OSError:
                        alive = False
            if alive:
                print(paint("[tunnel]", "cyan"), f"Ya hay un túnel en PID {old} (log: {p['log']})")
                if probe(port):
                    ready_summary(session, port, f"background (PID {old})")
                return
        log = p["log"].open("w", encoding="utf-8")
        kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT}
        if IS_WIN:
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "tunnel", session, str(port), "--bg-child"], **kwargs)
        p["pid"].write_text(str(proc.pid), encoding="utf-8")
        rc = wait_tunnel(proc, port, 30, "Abriendo túnel SSH...")
        if rc == 0:
            ready_summary(session, port, f"background (PID {proc.pid}, log: {p['log']})")
        elif rc == 2:
            print(paint("[tunnel] El túnel en segundo plano salió inesperadamente. Log:", "red"), p["log"])
            p["pid"].unlink(missing_ok=True)
            raise SystemExit(1)
        else:
            print(paint("[tunnel]", "yellow"), f"Iniciado en segundo plano (PID {proc.pid}); esperando 200 OK. Log: {p['log']}")
        return

    printed = False
    first_label = "" if child or os.environ.get("TUNNEL_QUIET_SUMMARY") == "1" else f"Abriendo túnel SSH a 127.0.0.1:{port}..."
    try:
        while True:
            forward = f"127.0.0.1:{port}:127.0.0.1:{remote_port}"
            proc = subprocess.Popen(ssh_args(session, key, forward=forward),
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL if child else None)
            label = first_label if not printed else ""
            rc = wait_tunnel(proc, port, 60, label)
            if rc == 0:
                if not printed and not child and not os.environ.get("TUNNEL_QUIET_SUMMARY"):
                    ready_summary(session, port, "foreground (Ctrl+C cierra el túnel; la VM sigue activa)")
                    printed = True
                elif printed:
                    print(paint("[tunnel]", "green"), f"Reconectado {time.strftime('%H:%M:%S')}: http://127.0.0.1:{port}/v1 (200 OK)")
            proc.wait()
            print(paint("[tunnel]", "yellow"), f"Desconectado {time.strftime('%H:%M:%S')}. Reconectando en 3s (Ctrl+C para parar)...")
            time.sleep(3)
    except KeyboardInterrupt:
        if proc.poll() is None:
            terminate_pid(proc.pid)
        p["pid"].unlink(missing_ok=True)
        print("\n" + paint("[tunnel] Detenido.", "yellow"))


def start(tier: str, ctx: int, port: int, session: str, bg: bool, wait: bool) -> None:
    if tier not in TIERS:
        raise SystemExit(f"Tier desconocido: {tier}")
    if not colab_path().exists():
        raise SystemExit(paint(f"[start] CLI de Colab no encontrado en {colab_path()}. Define COLAB_BIN.", "red"))
    ensure_key()
    print()
    print(paint("╭" + "─" * 72 + "╮", "cyan"))
    print(paint("│", "cyan"), paint("GLM-5.3-Flash Uncensored · Colab G4 Launcher", "bold"))
    print(paint("╰" + "─" * 72 + "╯", "cyan"))
    for k, v in (("Session", session), ("GPU", "Colab G4 (RTX PRO 6000 Blackwell 96 GB)"),
                 ("Quant", f"{tier} bpw"), ("Context", str(ctx)), ("Local Port", str(port))):
        print(f"  {paint(f'{k:<12}', 'dim')} {paint(v, 'bold') if k in ('Session','Quant') else v}")
    if not session_active(session):
        create_session(session, wait)
    show_colab_status(session)
    provision(session, port, tier, ctx, bg, wait)
    write_meta(session, port, tier, ctx)
    tunnel(session, port, bg=bg)


def stop(session: str, port: int) -> None:
    p = paths(session, port)
    if p["pid"].exists():
        try:
            pid = int(p["pid"].read_text().strip())
        except ValueError:
            pid = 0
        if pid:
            print(paint("[stop]", "cyan"), f"Deteniendo túnel en segundo plano PID {pid}...")
            terminate_pid(pid)
        p["pid"].unlink(missing_ok=True)
    p["meta"].unlink(missing_ok=True)
    if not IS_WIN:
        for pattern in (f"127.0.0.1:{port}:127.0.0.1:", f"{port}:127.0.0.1:"):
            try:
                run(["pkill", "-f", pattern], capture=True)
            except FileNotFoundError:
                break
    else:
        print(paint("[stop]", "yellow"), "En Windows los reenvíos huérfanos se cierran al terminar el PID del túnel.")
    cli = colab_path()
    if not cli.exists():
        print(paint(f"[stop] Aviso: colab no encontrado en {cli}; solo se limpió el túnel local.", "yellow"))
        return
    print(paint("[stop]", "cyan"), f"Deteniendo sesión Colab '{session}'...")
    run([str(cli), "stop", "-s", session], capture=True)
    for title, sub in (("=== Active sessions ===", ["sessions"]), ("=== Compute unit usage ===", ["usage"])):
        print("\n" + title)
        r = run([str(cli)] + sub, capture=True)
        print(r.stdout or "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="provisionar G4 + cargar el modelo + abrir el túnel")
    s.add_argument("tier", nargs="?", default="2.51", choices=sorted(TIERS))
    s.add_argument("--ctx", type=int, default=131072, choices=(131072, 262144))
    s.add_argument("--port", type=int, default=18100)
    s.add_argument("--session", "-s", default="glm53")
    s.add_argument("--bg", action="store_true", help="túnel en segundo plano")
    s.add_argument("--no-wait", action="store_true", help="no esperar a que haya G4 libre")
    t = sub.add_parser("tunnel", help="reconectar el túnel (VM ya en marcha)")
    t.add_argument("session", nargs="?", default="glm53")
    t.add_argument("port", nargs="?", type=int, default=18100)
    t.add_argument("--session", "-s", dest="session_opt")
    t.add_argument("--port", "-p", dest="port_opt", type=int)
    t.add_argument("--bg", action="store_true")
    t.add_argument("--bg-child", action="store_true", help=argparse.SUPPRESS)
    x = sub.add_parser("stop", help="cerrar túnel y sesión facturable")
    x.add_argument("session", nargs="?", default="glm53")
    x.add_argument("port", nargs="?", type=int, default=18100)
    x.add_argument("--session", "-s", dest="session_opt")
    x.add_argument("--port", "-p", dest="port_opt", type=int)
    a = ap.parse_args()
    if a.cmd == "start":
        start(a.tier, a.ctx, a.port, a.session, a.bg, not a.no_wait)
    elif a.cmd == "tunnel":
        tunnel(a.session_opt or a.session, a.port_opt or a.port, bg=a.bg or a.bg_child, child=a.bg_child)
    else:
        stop(a.session_opt or a.session, a.port_opt or a.port)


if __name__ == "__main__":
    main()
