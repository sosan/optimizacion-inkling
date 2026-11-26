# Layout GGUF de los quants pequeños (issue #19)

`tools/dump_gguf_layout.py` lee solo el **header** de un GGUF por HTTP Range (sin descargar los pesos) y vuelca metadata, tipos y forma de cada tensor. Funciona sin GPU:

```bash
python3 tools/dump_gguf_layout.py UD-IQ2_XXS UD-Q2_K_XL --out /tmp/layout.json
```

Informe completo subido a GitHub: <https://gist.github.com/sosan/64d3cffc0d51b37efd5bbd63a5228170>

## Resultado

| archivo | arquitectura | tensores | keys `inkling2.*` |
|---|---|---:|---:|
| `UD-IQ2_XXS.gguf` | `inkling2_loop` | 744 | 0 |
| `UD-Q2_K_XL.gguf` | `inkling2_loop` | 744 | 0 |
| `Old_Support/UD-Q2_K_XL.gguf_file` | `inkling2` | 734 | 35 |
| `UD-Q4_K_XL.gguf` (control) | `inkling2_loop` | 744 | 0 |
| `Old_Support/UD-Q4_K_XL.gguf_file` (control) | `inkling2` | 734 | 35 |

`UD-Q4_K_XL` funciona porque en realidad solo contiene F32 / Q8_0 / NVFP4 / BF16. Los quants pequeños añaden formatos que el engine ik2 no decodifica:

- `IQ1_S` (19) — 6 tensores gate/up de expertos; falta en `block_geometry` → error de carga.
- `IQ2_XXS` (16), `IQ2_XS` (17), `IQ2_S` (22) — gate/up de expertos; geometría conocida pero sin `decode_value()` → pesos a cero.
- `MXFP4` (39) — 1 tensor down de experto; sin geometría → error de carga.
- `Q4_K` (12) — 14 tensores de atención/output; sin `decode_value()` en el camino ik2.

El engine cubría NVFP4, Q8_0, Q5_K, Q6_K, IQ3_XXS e IQ4_XS (más F32/F16/BF16 y Q2_0 con kernels de experto dedicados); el cableado de abajo añade Q4_K, IQ2_XXS, IQ2_XS, IQ1_S, IQ2_S y MXFP4.

Los kernels reutilizables ya están declarados en `include/strata/kernels/iq_kernels.hpp` (`iq_supported`, `iq_mmvq`, `iq_dequant_f16`) y `native_mmvq.hpp`, pero sus `.cu` viven en `local/src/kernels/cuda/` (ignorado por git) y no están cableados en `engine-inkling-custom/CMakeLists.txt`.

## Cableado en esta rama (`pr/issue19-iq-decoders`)

En vez de arrastrar esos `.cu` (que usan activaciones q8_1 y `__dp4a`, incompatibles con el camino FP16 de ik2 y con HIP), los decodificadores se han implementado **dentro del camino existente** de ik2:

- `src/ik2/ik2_iq2_tables.h`: códigos IQ2_XXS / IQ2_XS / IQ2_S / IQ1_S transcritos de `third_party/ggml/ggml-common.h`. `tools/check_iq_tables.py` verifica que sean byte a byte iguales.
- `src/ik2/ik2_kernels.cu`: nuevos casos en `decode_value()` para Q4_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, MXFP4 y Q2_0 (este último lo usan los expertos `down` de los quants pequeños; el control Q4_K_XL usaba NVFP4), y ampliación de `block_values()` / `block_bytes()`. Las rutas de expertos (`row_dot`, `row_dot_2`, `row_dot_multi`, `decode_host`) caen en el fallback genérico y ya los usan.
- `src/ik2/ik2_main.cu`: la comprobación de filas de expertos ahora exige bloques enteros solo en la dimensión que cada proyección lee (`C` para gate/up, `F` para down). Antes exigía ambas, y `F=1152` no es múltiplo de 256, lo que rechazaba los bloques IQ de dos bits aunque la fila gate/up mide 2304.
- `include/strata/artifact/gguf_reader.hpp`: geometría de bloque para IQ1_S (256/50) y MXFP4 (32/17), que antes no tenían.

Verificación local (sin GPU): las tablas coinciden con ggml, la geometría compila con `g++`, y `decode_value` se extrajo y compiló como código host contra 28 valores sintéticos con 0 discrepancias. Falta la compilación CUDA/HIP en CI y la prueba real en GPU.</think:6124c78e>
