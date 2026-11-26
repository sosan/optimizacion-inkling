# Inkling 2 Mini — motor `inkling-custom-engine`

Motor autocontenido para [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) en una GPU NVIDIA (CUDA 12.8).

Todo se ejecuta desde la raíz del repo. Los binarios van en `build/` y los pesos en `models/`; ambas carpetas están en `.gitignore` (los modelos pesan ~17 GB y no se versionan).

### Qué es este motor

Inkling 2 Mini es un **MoE en bucle de 26B** de Thinking Machines: 24 capas (20 deslizantes), 128 expertos de los que se usan 8 por token, anchura 2304 y vocabulario 200058. Las capas 6 a 17 se ejecutan 3 veces (48 visitas por token). Los 3072 expertos (14,7 GiB) y los pesos densos (1,1 GiB) caben en una GPU de 24 GB; con caché KV en FP8 (5,48 GiB) el contexto llega a 344064 sin drafter o 262144 con él. Los expertos de las capas 0 a 19 son bits NVFP4 propios de Thinking Machines: la 3090 no tiene hardware FP4, así que el motor decodifica los códigos de 4 bits en registros y los alimenta a los tensor cores.

Además incluye *continuous batching* (`--parallel`), caché de conversaciones en RAM (`IK2_SLOT_KEEP`), el drafter oficial DSpark, y APIs compatibles con OpenAI y Anthropic en `/v1` con imágenes, herramientas, control de razonamiento y streaming.

### Rendimiento

Medido en este repo con `bench_perf.sh` (RTX 3090 24 GB, `UD-Q4_K_XL`, KV FP8, `--ctx 344064`):

| Métrica |  |
|---|---:|
| Decode (119.417 tokens) | **94.8 tok/s** |
| Prefill (119.417 tokens) | **1764 tok/s** |
| Modelo residente | 3072/3072 expertos en VRAM |
| llama.cpp b12590 (referencia) | 57,0 tok/s |

El drafter añade ~1,2 GiB y solo compensa en código/ediciones; la config conservadora es `ik2-mini-text-nodraft.json`.

## 0. Clonar el repo

```bash
git clone https://github.com/sosan/optimizacion-inkling
cd optimizacion-inkling
```

---

## 1 Dependencias Python (Python 3.10 o superior):

```bash
pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub
```

---

## 2. Obtener el binario

### Opción A: compilar en local

Requisitos

- Linux (Ubuntu 24.04) o Windows, con CUDA 12.8 y `nvcc` en el `PATH`.
- CMake ≥ 3.21, Ninja, compilador C++20 (GCC 13+ / MSVC).

```bash
# 80=Ampere, 86=RTX 30, 89=RTX 40, 90a=Hopper, 100=Blackwell B200/GB200, 103=B300/GB300, 120=Blackwell RTX 50, 121=DGX Spark (GB10) / RTX Spark
cmake -S engine-inkling-custom -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86  # cambia 86 por la de tu GPU, o añade todas: 80;86;89;90a;100;103;120;121
cmake --build build --target inkling-custom-engine -j "$(nproc)"
```

El ejecutable queda en `build/inkling-custom-engine` (o `.exe` en Windows), que es la ruta que usan los JSON configuracion en `docs/config/`.

En **Windows con MSVC** la línea que funcionó (RTX 5060 Ti 16 GB, CUDA 13.5) es:

```bat
cmake -S engine-inkling-custom -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF
cmake --build build --target inkling-custom-engine
```

`STRATA_NATIVE_EXPERTS=OFF` evita los decodificadores nativos IQ que no se usan con `UD-Q4_K_XL` (ver issue [#19](https://github.com/alesha-pro/maya-inkling-mini/issues/19)).

### Opción B: bajarlo de GitHub Actions / Releases

Los workflows `Build Inkling 2 Custom Engine (Ubuntu|Windows)` suben el artifact `inkling-custom-engine-cuda<arch>-(ubuntu|windows)`, y `publish-release.yml` publica el binario en los Releases del repo.

> Si construyes con varias arquitecturas (`cuda_arch=all`), el binario crece (~16 MB de zip) por el código device de cada arquitectura. Es normal.

**Builds nightly**: los workflows tratan `main` (y los tags `v*`) como canal *stable*. Si se lanzan desde cualquier otra rama, el artifact lleva el sufijo `-nightly-<rama>` (p. ej. `inkling-custom-engine-cuda86-ubuntu-nightly-pr-rdna4-wmma`) y `release.yml` crea un **prerelease** con el tag `<versión>-nightly-<rama>` en vez de un release estable. `build-info.txt` incluye el campo `channel` indicándolo. Para publicar una versión estable, haz el bump de `VERSION`, commit y tag `vX.Y.Z` desde `main`.

---

## 3. Descargar los modelos

**Importante (23 noviembre 2026):** el repo de Unsloth cambió el layout de los archivos de la raíz. Los archivos raíz son ahora el formato *mainline* (`inkling2_loop`, para llama.cpp ≥ b12611) y **no sirven para este motor**. El layout que entiende `maya-ik2` está en `Old_Support/` y lleva las claves `inkling2.*` (35 claves, incluida `inkling2.block_count`).

### 3.1 GGUF del modelo (layout antiguo)

```bash
hf download unsloth/Inkling-2-Mini-GGUF \
  Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --local-dir models

mv models/Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
   models/Inkling-2-Mini-UD-Q4_K_XL.gguf
```

Solo funciona el quant `UD-Q4_K_XL` (17,6 GB). `UD-Q8_K_XL` guarda los expertos de las últimas capas en BF16 y el motor no los carga; otros quants fallan en tipos de tensor que el motor no decodifica (p. ej. `UD-IQ3_XXS` se detiene en `type IQ2_XXS not decoded here`, y `UD-Q2_K_XL` usa además IQ1_S, MXFP4 y un tipo de experto más nuevo). Si el motor dice `no key inkling2.block_count`, has descargado el fichero de la raíz (layout `inkling2_loop` del 23-nov) en vez del de `Old_Support/`.

### 3.2 Drafter (opcional, para `--draft`)

```bash
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --local-dir models
```

Este archivo es de **otro repo** (Thinking Machines), no tiene versión en `Old_Support/`, y se deja tal cual.

### 3.3 Codificador de imágenes (opcional, solo para configs con visión)

```bash
hf download unsloth/Inkling-2-Mini-GGUF \
  Old_Support/mmproj-BF16.gguf_file --local-dir models
mv models/Old_Support/mmproj-BF16.gguf_file models/mmproj-BF16.gguf
```

Además necesitas `build-vision/bin/inkling-custom-engine-vision`, que **no** construye el workflow. Para uso solo de texto, usa `ik2-mini-text.json`, `ik2-mini-text-long.json` o `ik2-mini-8gb.json` y saltate este apartado.

### 3.4 Tokenizer

```bash
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
  --chat-template tools/ik2/chat_template.jinja --check
```

Genera `ik2/tokenizer`, que es lo que lee el servidor.

---

## 4. Arrancar el servidor

```bash
python serve/server.py --engine inkling-custom-engine \
  --config docs/config/ik2-mini-text.json \
  --host 127.0.0.1 --port 18000
```

- `--engine` acepta `mock`, `inkling-custom-engine` o `strata`.
- El servidor lanza el binario del campo `"exe"` del JSON y le añade `--serve` automáticamente.
- Usa `0.0.0.0` en vez de `127.0.0.1` si quieres acceder desde otros dispositivos (configura una API key).
- Endpoint OpenAI-compatible: `http://127.0.0.1:18000/v1`.

Ejemplo de petición:

```bash
curl -s http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","max_tokens":300,"reasoning_effort":"none",
       "messages":[{"role":"user","content":"Write a haiku about a lighthouse."}]}'
```

### Configs disponibles (`docs/config/`)

**Solo texto** (no necesitan el codificador de imágenes). Matrix completa con/sin drafter:

| Contexto | Sin drafter | Con drafter (DSpark) |
|---:|---|---|
| 262144 | **`ik2-mini-text-nodraft.json`** (conservadora, recomendada) | `ik2-mini-text.json` (más rápida en código, +~1,2 GiB) |
| 344064 | `ik2-mini-text-long.json` | `ik2-mini-text-long-draft.json` (experimental, ver nota) |
| 16384 | `ik2-mini-8gb.json` (~41 tok/s en 8 GB), `ik2-mini-16gb.json` (~43 tok/s en 16 GB) | `ik2-mini-8gb-draft.json` |

Para tarjetas de **~22 GB** (RTX 3090 Ti, donde `cuMemGetInfo` reporta solo ~22,8 GiB libres) hay dos configs específicas: `ik2-mini-text-22gb.json` (233472 **con** drafter) e `ik2-mini-text-long-22gb.json` (307200 **sin** drafter). Con 262144 + drafter faltan 0,15 GiB y con 344064 tampoco cabe. Comprueba siempre `resident=1` en `INFO` o en `/api/report`: si no está, los expertos se quedan en RAM y el decode va a la mitad.

**Con visión** (requieren `mmproj-BF16.gguf` y el binario `build-vision/`, apartado 3.3; no trabajadas aquí):

| Config | Contexto | Codificador |
|---|---:|---|
| `ik2-mini.json` | 262144 | CPU (default del upstream) |
| `ik2-mini-gpu-vision.json` | 262144 | GPU |
| `ik2-mini-long-vision.json` | 344064 | CPU |

El drafter oficial de Thinking Machines es `mtp.safetensors` (0,6 GB, apartado 3.2). Se activa con `"--draft", "models/mtp.safetensors"` dentro del array `args` y `"IK2_DRAFT_CTX"` en `env`. El motor verifica 2 a 8 tokens por ventana y acelera código y ediciones; en prosa apenas aporta.

**Por qué la recomendada es la de 262144 sin drafter**: es la que deja más margen de VRAM en una tarjeta de 24 GB y mantiene todos los expertos residentes (decode ~125-130 tok/s). El drafter suma ~1,2 GiB y solo compensa en código/ediciones; 344064 llena la KV cache (~5,5 GiB) y el decode cae con la profundidad del contexto. Si tu GPU es de 8 GB, usa `ik2-mini-8gb.json`; para 16 GB, `ik2-mini-16gb.json`.

**Tarjetas de 16 GB** (RTX 5060 Ti 16 GB + i5-14600KF sin AVX-512, Windows/MSVC/CUDA 13.5, contexto 16384, 300 tokens):

| `--cache-gb` | `IK2_CPU` | Decode |
|---:|---:|---:|
| 7 | 1 | 30 tok/s |
| 10 | 1 | 39 tok/s |
| 12 | 1 | 39 tok/s |
| 12 | 0 | **43 tok/s** CLI (40 en servidor) |

La caché es la palanca (de 7 a 12 GB, ~+30 %); más de 12 GB se topa con lo que cabe (2446 slots / 11,7 GiB) y 14,5 GB fue más lento (30,8 tok/s). `--threads` de 6 a 16 solo cambió ~5 %. Con una caché tan grande el carril CPU no ayudó: arrancaba, pero `IK2_CPU=0` fue ligeramente más rápido, por eso la config de 16 GB lo desactiva. Son medidas de una sola pasada en una máquina.

**Memoria que cuesta el drafter** (`ik2_main.cu:4756-4772`): ~0,7 GiB de pesos + 4 KB por posición de su capa global + ~0,25 GiB de buffers. Con `IK2_DRAFT_CTX=65536` son ~1,2 GiB extra; si no se pone la variable, usa el contexto completo (~2 GiB a 262K). El motor lo incluye en su cálculo de `fits`: si no caben los expertos en VRAM, los deja en RAM y el decode va a la mitad. Por eso el upstream usa 262144 con drafter: esos 81920 tokens menos de KV FP8 liberan ~1,25 GiB y compensan el coste. Las variantes `long-draft` (344064) y `8gb-draft` reducen `IK2_DRAFT_CTX` a 16384 / 2048 para intentar que quepan; **no están medidas** y en hardware justo pueden caer a la ruta lenta.

El motor también soporta **continuous batching** y **caché de conversaciones**:

- Añade `"--parallel", "4"` a `args` en un config para servir peticiones concurrentes (2 a 8, una sola GPU, KV no gestionado).
- En la ruta por defecto (`--parallel 1`), la variable `IK2_SLOT_KEEP` (por defecto `8`, `0` = off) guarda en RAM las conversaciones desplazadas y las restaura en ~70 ms en vez de volver a prefill. Se pone en `"env"` del config, por ejemplo `"IK2_SLOT_KEEP": "8"` y opcionalmente `"IK2_SLOT_RAM_GB"`.

---

## 5. Comprobación rápida del motor

Puedes arrancar el ejecutable directamente, sin el servidor:

```bash
./build/inkling-custom-engine --serve \
  --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --max-context 344064
```

Debe imprimir `INFO ...` y `READY <ctx> stop multi=...`.

También puedes verificar los metadatos del GGUF sin cargar los pesos:

```bash
python3 -c "
import sys; sys.path.insert(0,'tools')
from gguf_reader import GGUFFile
g = GGUFFile('models/Inkling-2-Mini-UD-Q4_K_XL.gguf')
ks = [k for k in g.metadata if k.startswith('inkling2.')]
print('tensores:', len(g.tensors), '| claves inkling2:', len(ks))
print('block_count =', g.metadata.get('inkling2.block_count'))
"
```

Debe dar `claves inkling2: 35` y un `block_count` numérico.

---

## 6. Solución de problemas

**`maya-ik2: no key inkling2.block_count`**
El GGUF es el layout mainline de la raíz de Unsloth, no el de `Old_Support/`. Vuelve al apartado 3.1.

**`RuntimeError: the engine exited before it was ready`**
El motor murió antes de `READY` y su `stderr` se descarta. Se añade una entrada al engine.log:

```json
{ "exe": "build/inkling-custom-engine", "log": "engine.log", ... }
```

comprueba el log, por ejemplo, con `tail -50 engine.log`.

**`no se encuentra el binario`**
Comprueba que `build/inkling-custom-engine` exista y que el campo `exe` del JSON coincida (los JSON de este repo ya apuntan a `build/inkling-custom-engine`).

**CUDA: `no kernel image is available`**
El binario se compiló para otra arquitectura. Recompila con `-DCMAKE_CUDA_ARCHITECTURES=<tu_sm>` o descarga el artifact de la arquitectura correcta.

**`23.3 GiB of VRAM free, 28.3 GiB needed: the experts stay in host memory`**
No es un error: el motor avisa de que no caben todos los expertos en VRAM y el decode irá ~a la mitad de velocidad. Al arrancar **a mano** con `--serve` no se aplican las variables del JSON, así que la KV cache sale en FP16 (~11 GiB). Al arrancar con `serve/server.py` y un config que incluye `"IK2_KV": "fp8"` en `"env"`, la KV cache baja a ~5,5 GiB y lo normal es que quepan todos los expertos. Si aun así no caben, baja `--max-context` a 163840 o menos (el propio aviso lo sugiere).

---

## 7. Medir rendimiento

Un único script Python sirve para Linux y Windows:

```bash
python3 tests/bench_perf.py cli                        # motor directo: prefill y decode tok/s
python3 tests/bench_perf.py http                       # a través del servidor (/v1/chat/completions)
python3 tests/bench_perf.py both --gen 256 --ctx 163840
python3 tests/bench_perf.py cli --vram-sample          # añade muestreo de VRAM con nvidia-smi
```

En Windows cambia `python3` por `python` (o `py`). Los launchers `launch_bench.sh` / `launch_bench.ps1` siguen funcionando y llaman a este script con `--ctx 344064`.

El modo `cli` usa la línea `@END` que imprime el motor (prompt, segundos de prefill y decode) y necesita `ik2/tokenizer` para generar `IDS.txt` desde `prompt.txt` si no existe. El modo `http` arranca el servidor con el config, espera `/health`, hace una petición y para el servidor. Los logs y CSV quedan en `bench-results/` (ignorado por git).

Opciones ajustables: `--bin`, `--model`, `--tokens-dir`, `--config`, `--ids`, `--prompt-file`, `--gen`, `--ctx`, `--port`, `--vram-sample`.

> Con `--ctx` bajo (o sin él) el motor no mantiene todos los expertos en VRAM y el decode va más lento.

---

## 8. Estructura y ramas

```
engine-inkling-custom/   CMake autocontenido
include/                 headers (gguf_reader, strata, hip_compat)
src/ik2/                 fuentes del motor
serve/                   servidor HTTP y dashboard
tools/                   tokenizer, lector GGUF, nombres/quants
docs/config/             configs JSON
.github/workflows/       builds Linux/Windows y publicación de releases
build/                   binarios (ignorado por git)
models/                  pesos (ignorado por git)
```

Ramas: `main` (motor autocontenido, la que usa el workflow), `dev` (árbol completo para experimentar con el CMake).

Créditos y proyectos relacionados: [Project Maya](https://github.com/mw00/project-maya) (base de `maya-ik2` y del motor DS4), [Strata](https://github.com/Niko1221/Strata) (origen de Maya), [DS4](https://github.com/antirez/ds4) (motor DeepSeek 4 Flash), Thinking Machines y Unsloth.
