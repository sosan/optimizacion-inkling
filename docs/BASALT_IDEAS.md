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
| 1 | Dot NVFP4 exacto con `mma.sync.aligned.m16n8k16.s32.s8.s8.s32` para gate/up | `src/kernels/cuda/nvfp4_grouped_mma.cuh` (`nvfp4_frag_s8`, `nv_tile_mma`) | Nuestro GGUF guarda los mismos códigos E2M1 + escala UE4M3 por 16 valores; ya tenemos `q8_1` (PR #30) para las activaciones | Alto |
| 2 | Caché KV en FP4 (E2M1 + E4M3 por 16 valores) | `include/basalt/kernels/kv_fp4.cuh` | Bajaría la KV de 16 KiB a 8 KiB por posición; requiere revisar precisión en atención larga | Medio |
| 3 | Spill de expertos a RAM con perfil y *prompt loan* | `include/basalt/program/expert_spill.hpp`, `src/core/expert_tier.cpp` | Muy útil en tarjetas de 16 GB y contextos largos (nuestro `IK2_DENSE=q8` / `IK2_LAZY_*` es más simple) | Alto |
| 4 | Planes Q8_0 para proyecciones densas (`native_q8_planes_mmvq`) | `include/basalt/kernels/nvfp4_planes.hpp` | Similar a nuestro `IK2_DENSE=q8` + `decode_q8p_f16`; posible mejora de prefill | Medio |
| 5 | Convertidor `.basalt` (`tools/convert/convert.py`) | `tools/convert/convert.py` | Permitiría usar los packs oficiales, pero cambia el formato de modelo de este repo | Muy alto |

**No portable directamente**: el soporte gfx906/AMD del issue #36 (Basalt es
solo NVIDIA) y el lector de `.basalt`.

## Primer paso recomendado

Empezar por **#1**: es el que explica la mayor parte de la diferencia de decode.
Se puede incorporar como un kernel opt-in (`IK2_NVFP4_S8=1`) junto a los
actuales `experts_pf_nvfp4` / `nvfp4_sub_dot`, con un modo de benchmark que
compare bits y tok/s contra la ruta actual en sm_120. Requiere activaciones
`q8_1`, que ya están disponibles en `src/kernels/cuda/iq_kernels.cu`.

Antes de portar conviene confirmar que la GPU de destino es realmente
**sm_120** (RTX PRO 4000 / 5000 Blackwell). Una RTX 4000 Ada (sm_89) no tiene
los tensor cores de bloque FP4 y solo podría usar la variante int8 s8.

## Referencias

- Código: <https://github.com/jesdga95/basalt>
- Precisión de Inkling 2 Mini: <https://github.com/jesdga95/basalt/blob/main/docs/precision/inkling2.md>
- Guía del modelo: <https://github.com/jesdga95/basalt/blob/main/docs/models/inkling2.md>
