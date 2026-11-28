# GLM-5.3-Flash en Colab G4 — launcher Python

Script **Python multiplataforma** (Linux, macOS y Windows) que aprovisiona una
VM **Google Colab G4** (RTX PRO 6000 Blackwell 96 GB), compila Project Maya para
`sm_120`, descarga y empaqueta un EXL3 "abliterated" de GLM-5.3-Flash (321B-A18B)
junto al drafter DFlash2, arranca el servidor compatible OpenAI/Anthropic 
en `:8090` de la VM y abre un port-forward SSH **autoreparable** hacia 
`http://127.0.0.1:18100`.

Cuando `/v1/models` responde `200 OK`, imprime los endpoints locales, el nombre
del modelo, los snippets `OPENAI_*` / Claude Code y el comando de parada.

> **Aviso de uso**: esta herramienta está pensada para uso personal y local con
> tu propia cuenta de Colab. El **alojamiento público** del endpoint (exponerlo
> a terceros, compartirlo o revender el acceso) podría **infringir las reglas de
> uso de Google** y las condiciones del servicio de Colab. Úsalo bajo tu propia
> responsabilidad y no lo ofrezcas como servicio público.
>
> Tampoco elimina las limitaciones de Colab: la **disponibilidad de las tarjetas
> gráficas** (las G4 pueden agotarse y el launcher reintenta con backoff) y los
> **límites de duración de las sesiones** siguen existiendo. La sesión puede
> caducar o ser recuperada por la plataforma; reconecta con `tunnel` si la VM sigue
> vivo y para siempre la sesión con `stop` cuando termines.

## Requisitos

- Cuenta de Google Colab con Compute Units (Pro / AI Pro / AI Ultra).
- `google-colab-cli` (`colab`) instalado y autenticado. Se busca en `COLAB_BIN`,
  luego en el `PATH`, y por defecto en `~/.local/bin/colab` (`colab.exe` en Windows).
- OpenSSH: `ssh` y `ssh-keygen` (en Windows, el cliente OpenSSH incluido).
- Si no existe `~/.ssh/id_ed25519_colab`, se genera automáticamente
  (`COLAB_SSH_KEY` permite cambiar la ruta).

## Uso

Desde esta carpeta (o desde cualquier directorio, indicando la ruta al script):

```bash
# Linux / macOS
python3 colab_g4.py start                       # 2.51 bpw, 131K, puerto 18100
python3 colab_g4.py start 3.05                  # equilibrado
python3 colab_g4.py start 4.05 --ctx 262144    # calidad Q4, contexto largo
python3 colab_g4.py start 2.51 --bg             # túnel en segundo plano
python3 colab_g4.py tunnel glm53 18100 --bg     # reconectar tras dormir el portátil
python3 colab_g4.py stop                        # cerrar túnel y cortar facturación
```

```bat
:: Windows
py colab_g4.py start
py colab_g4.py start 3.05 --ctx 262144
py colab_g4.py tunnel glm53 18100 --bg
py colab_g4.py stop
```

Opciones comunes: `--ctx 131072|262144`, `--port N`, `--session NAME`,
`--no-wait` (no esperar a que haya G4 libre). `REMOTE_PORT` cambia el puerto
remoto (por defecto `8090`); `COLAB_MAX_ATTEMPTS`, `COLAB_RETRY_BASE` y
`COLAB_WAIT_CAP` controlan los reintentos de asignación G4.

## Endpoints tras el `200 OK`

| Servicio | URL |
|---|---|
| OpenAI | `http://127.0.0.1:18100/v1` |
| Anthropic | `http://127.0.0.1:18100` (`/v1/messages`) |
| Web UI + monitor Maya | `http://127.0.0.1:18100/` |
| API key | `sk-local` (cualquier cadena no vacía) |
| Modelo | `glm-5.3-flash-uncensored-<tier>bpw` |

El estado del PID, log y metadatos del túnel viven en el directorio temporal
del sistema con el nombre `colab-tunnel-<sesion>-<puerto>`.

## Diferencias respecto a los scripts bash

- Un único CLI Python en vez de `start.sh` / `tunnel.sh` / `stop.sh`. No queda código bash incrustado: el aprovisionamiento de la VM es `remote_provision.py`, un segundo archivo Python que se envía por `stdin` a `ssh ... python3 -`. `apt-get`, `cmake`, `pip` y `pkill` se lanzan con `subprocess` desde Python.
- Rutas y procesos multiplataforma (`pathlib`, `tempfile`, `shutil.which`,
  `subprocess` sin `shell=True`; `taskkill` en Windows, `pkill` en Unix).
- El aprovisionamiento remoto se envía por `stdin` a `ssh ... bash -s --`, sin
  archivos temporales locales.
- Los snippets de entorno usan `set VAR=...` en Windows y `export VAR=...` en Unix.
- La lógica remota (compilación, descarga, pack, configuración y carga) es la
  misma del patch upstream; los `configs/` se copian tal cual. Archivos: `colab_g4.py` (CLI local), `remote_provision.py` (VM), `test_colab_g4.py` (smoke tests) y `configs/`.

## Rendimiento de referencia (upstream, G4)

| Quant | Contexto | Decode | Prefill 2K / 8K |
|---|---|---:|---:|
| 2.51 bpw | 131K | **102.3 tok/s** | 1190 / 2255 tok/s |
| 3.05 bpw | 131K | ~89–95 tok/s | 1139 / 2258 tok/s |
| 4.05 bpw | 131K | 71.8 tok/s | 819 / 2164 tok/s |

Consumo medido: **10.05 CU/hora**. Carga en frío: 10–20 min según el quant.
Detalles y comparativas completas en el [README upstream](https://github.com/alesha-pro/tools/blob/main/colab-g4-glm53/README.md).
