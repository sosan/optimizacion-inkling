# Ideas de Basalt aplicables a este motor

[Basalt](https://github.com/jesdga95/basalt) es un fork de **Strata / Project Maya**
—la misma base de la que deriva `inkling-custom-engine`— especializado en
**NVIDIA Blackwell (sm_120)** con CUDA 13 y pesos en formato `.basalt`
(NVFP4 / MXFP8 nativos del release de Inkling 2 Mini). En una RTX PRO 4000
Blackwell reporta 80–120 tok/s de decode y >4000 tok/s de prefill con
Qwen Flash Next.

## Por qué va más rápido

1. **Códigos NVFP4 nativos**: los bloques del release se alimentan directamente
   a los tensor cores con escala por sub-bloque (`mma.sync` s8 / FP4
   block-scaled), en vez de decodificar cada code a FP16 con dots escalares
   como hacemos aquí.
2. **Pack `.basalt`**: pesos densos Q8_0, head Q8_0, expertos NVFP4/MX8 y
   drafter DSpark empaquetados; no hay conversión GGUF en caliente.
3. **Prefill agrupado con TMA** (`cp.async.bulk` + mbarrier) y planes Q8_1
   de activaciones reutilizados por todos los expertos del grupo.
4. **Spill de expertos a RAM** con perfil de uso real y *prompt loan*, para
   tarjetas pequeñas o contextos largos.

## Candidatos a portar (ordenados por impacto / esfuerzo)

| # | Idea | Archivo en Basalt | Encaje aquí | Esfuerzo |
|---|---|---|---|---|
| 1 | Dot NVFP4 exacto con `mma.sync.aligned.m16n8k16.s32.s8.s8.s32` para gate/up | `src/kernels/cuda/nvfp4_grouped_mma.cuh` (`nvfp4_frag_s8`, `nv_tile_mma`) | **Implementado** en `pr/basalt-ideas` (ver abajo). Nuestro GGUF guarda los mismos códigos E2M1 + escala UE4M3 por 16 valores; ya tenemos `q8_1` (PR #30) para las activaciones | Alto (hecho) |
| 2 | Caché KV en FP4 (E2M1 + E4M3 por 16 valores) | `include/basalt/kernels/kv_fp4.cuh` | Bajaría la KV de 16 KiB a 8 KiB por posición; requiere revisar precisión en atención larga | Medio |
| 3 | Spill de expertos a RAM con perfil y *prompt loan* | `include/basalt/program/expert_spill.hpp`, `src/core/expert_tier.cpp` | Muy útil en tarjetas de 16 GB y contextos largos (nuestro `IK2_DENSE=q8` / `IK2_LAZY_*` es más simple) | Alto |
| 4 | Planes Q8_0 para proyecciones densas (`native_q8_planes_mmvq`) | `include/basalt/kernels/nvfp4_planes.hpp` | Similar a nuestro `IK2_DENSE=q8` + `decode_q8p_f16`; posible mejora de prefill | Medio |
| 5 | Convertidor `.basalt` (`tools/convert/convert.py`) | `tools/convert/convert.py` | Permitiría usar los packs oficiales, pero cambia el formato de modelo de este repo | Muy alto |

**No portable directamente**: el soporte gfx906/AMD del issue #36 (Basalt es
solo NVIDIA) y el lector de `.basalt`.

## Estado de la idea #1 (rama `pr/basalt-ideas`)

Ya portada y compilada en **sm_89 y sm_120** (CUDA Linux/Windows) y en
**gfx1201/gfx1151** (HIP, con un stub que devuelve `false` para que el motor
siga su ruta actual):

- `include/strata/kernels/nvfp4_s8.hpp`: `nvfp4_frag_s8` + `mma_s8_k16`.
  Comprueba bit a bit los 4096 fragmentos contra la tabla real `kvalues_fp4`
  de ggml (`tests/ik2_nvfp4_s8_test.cu`).
- `src/kernels/cuda/nvfp4_s8_gate_up.cu`: GEMM de gate/up de 16 filas × 8
  tokens con `mma.sync.s8`, reparto de K en 8 warps, reducción en memoria
  compartida en el mismo orden que Basalt y escritura directa en `h16`.
  Usa cargas globales (no TMA), así que funciona también en **sm_89**.
- `tests/ik2_nvfp4_s8_gate_up_test.cu`: compara la salida del kernel MMA con
  una referencia escalar independiente sobre los mismos pesos GGUF y las
  mismas activaciones `q8_1`. CI solo lo compila (no hay GPU en los runners)
  y lo publica como artefacto.
- `tests/ik2_expert_bench.cu`: modo `./ik2_expert_bench gates8 20 - nvfp4`
  para comparar tok/s contra la ruta escalar `gate 40`.

**Pendiente**: ejecutar el test de corrección y el bench en la GPU del usuario
(sm_120 y/o sm_89). Solo después de verificar bits y speeds se cableará al
dispatch de `ik2_main.cu` como camino opt-in. Las variantes FP4
*block-scaled* (`tcgen05`) se dejarían tras `__CUDA_ARCH__ >= 1000`; aquí no
se han portado.

Antes de conectar al engine conviene confirmar que la GPU de destino es
realmente **sm_120** (RTX PRO 4000 / 5000 Blackwell). Una RTX 4000 Ada
(sm_89) no tiene los tensor cores de bloque FP4 y solo usa la variante s8.

## Referencias

- Código: <https://github.com/jesdga95/basalt>
- Precisión de Inkling 2 Mini: <https://github.com/jesdga95/basalt/blob/main/docs/precision/inkling2.md>
- Guía del modelo: <https://github.com/jesdga95/basalt/blob/main/docs/models/inkling2.md>
