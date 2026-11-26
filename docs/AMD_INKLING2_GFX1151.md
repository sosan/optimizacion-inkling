# Inkling 2 Mini on AMD gfx1151 (Strix Halo)

What the AMD build of `maya-ik2` adds for Ryzen AI Max (gfx1151, Radeon 8060S, one RAM for CPU and GPU), and how it
was checked. Everything is under `#if defined(STRATA_USE_HIP)`: the CUDA build reads the same sources it read before
(`python tools/nv_path_check.py` proves it as text).

## Build

    cmake -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_HIP_IK2_SPIKE=ON \
          -DCMAKE_HIP_ARCHITECTURES=gfx1151
    cmake --build build-hip --target maya-ik2 ik2_kernels_test ik2_fuse_test ik2_experts_test

The tensor-core (PTX) paths are not in the AMD build: their settings are refused, and so is `--parallel > 1`
(its batched attention is one of them).

## What is on by default, and where

The tuned paths are on by default only on **gfx1151 with one GPU** (where they were measured). Every other AMD
device keeps the plain path unless the settings ask; the prefill kernels (`IK2_HBPF`) need gfx11 (WMMA).

| class | setting (gfx1151 default) | exact? | the check |
| --- | --- | --- | --- |
| load | (always) pinned bounce ring for the experts' copy-in | yes | the same tokens |
| fused decode | `IK2_FUSE_MASK=0xf14f`, `IK2_GRAPH=0` | yes, bit for bit | `ik2_fuse_test`, `ik2_experts_test` (memcmp); greedy ids on vs off |
| prefill | `IK2_HBPF=439` (183 + 256; 256 acts only on a turbo6 cache) | no (another sum order) | KL divergence vs an fp64 reference, as near as the old path |
| memory | `IK2_MEMTRIM=0x13` | yes | greedy ids on vs off; GTT peak |
| DSpark window | `IK2_DSC` (off) | yes | `IK2_DSC_PARITY`, `ik2_dsc_test` |
| turbo6 KV | `IK2_KV=turbo6` (off) | no against FP16 KV (lossy); its readers yes against each other | `ik2_kernels_test` (`IK2_TEST_T6=1`: the turbo6 tests alone); KL vs FP16 KV |

## turbo6 KV cache (`IK2_KV=turbo6`)

Off by default. One GPU, AMD build only (the CUDA build refuses the value; so do `--gpus 2`, `IK2_KV_MANAGED` and
`IK2_DSC` bit 1).

- **Format** (`src/ik2/ik2_turbo6.h`): a 128-value head is one 98-byte block - its L2 norm as FP16, then 6-bit codes
  (64 bytes of low nibbles, 32 bytes of top 2 bits). The values are divided by their norm and rotated (random signs,
  the normalized 128-point Walsh-Hadamard transform, signs again); a code picks one of 64 Lloyd-Max centroids of
  N(0, 1/128). The stored norm is the input's L2 norm over the norm of the centroids picked, so a decoded block keeps
  the input's norm.
- **Memory**: the full layers' K / V 32 KB a token in FP16 -> 12.25 KB. Inkling-2-Mini's 262,144 tokens of context
  need 3.06 GiB of KV instead of 8: with UD-Q4_K_XL (~17.8 GiB of weights and work buffers on the GPU) ~21 GiB in
  all against ~26, out of the 32 GB the CPU and GPU share.
- **Kernels** (`src/ik2/ik2_turbo6.inc`): the encoder writes exactly the CPU reference's bytes
  (`ik2::t6::QuantizeBlock`: the same order of sums, no fused multiply-add); the token's attention reads the codes
  in its loop (the query rotated once - q.k = (Rq).(Rk) - V summed in the rotated space, the head's output rotated
  back in the combine); the prompt path decodes a visit's blocks to FP16 rows for its GEMMs, and with `IK2_HBPF` 256
  runs bit 4's attention kernel over them (in passes when a full visit is longer than the scratch rows).
- **Exact where it can be**: the three turbo6 tests in `ik2_kernels_test` compare bit for bit - the encoder against the
  CPU reference, the token reader against the calls it replaces (`IK2_FUSE_MASK` bit 22 keeps the older reader), the
  prompt-path attention against the FP16 kernel over the CPU-decoded cache, passes against one call.
- **Quality**: lossy against an FP16 cache. UD-Q4_K_XL, the last 512 positions of three 4096-token texts: KL(FP16 KV
  || turbo6) 0.008-0.010 nats a position, top-1 the same in 95.3-96.5 % (it grows with depth). HumanEval (164, greedy,
  thinking off): 142 with turbo6 KV (on UD-IQ4_XS weights), 142 with FP16 KV (UD-Q4_K_XL). FP16 stays the default.
- **Speed** on gfx1151: prefill 2-3.5 % behind FP16 at 4,096-16,384 tokens (the encode and the FP16 unpack; measured
  on our standalone HIP build of the same kernels); decode: see Measured.

Credits: the codec is **tubo**'s `src/models/qwen/hip/turbo4_codec.hpp`, its 6-bit path
(https://github.com/BonkusCheemus/tubo, MIT; tubo is a gufo-based engine, its license is gufo's, below): the tables
and functions are copied with that file's line numbers in `ik2_turbo6.h`. tubo's turbo6 block and codebooks come
from **TheTom/llama-cpp-turboquant** PR #381 ("turbo5 and turbo6 KV cache types", JakeATX; MIT, the ggml authors'
license as in `third_party/ggml/LICENSE`), an implementation of **TurboQuant** (Zandieh, Daliri, Hadian, Mirrokni:
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", arXiv 2504.19874).

## Measured

`--bench`'s measurement (3 questions x 256 tokens, greedy, thinking off, one warm-up; prefill at 2048 / 8192), driven
over `--serve`, context 16,384, `Inkling-2-Mini-UD-Q4_K_XL.gguf`, gfx1151 (Radeon 8060S, 32 GB shared), ROCm 7.2.4.
`main` does not build on AMD, so "before" is this branch with every tuned path off (`IK2_FUSE_MASK=0 IK2_GRAPH=1
IK2_HBPF=0 IK2_MEMTRIM=0`). Two runs each (night, idle / day, desktop in use):

| | before (tuned paths off) | after (defaults) |
| --- | --- | --- |
| decode, mean of 3 | 38.7 / 39.9 tok/s | 46.2 / 46.2 tok/s |
| prefill 2048 | 243 / 314 tok/s | 1,356 / 1,616 tok/s |
| prefill 8192 | 232 / 299 tok/s | 1,357 / 1,506 tok/s |

Prefill answers: KL divergence against an fp64 reference (4096-token prompt, last 512 positions) 2.764e-3 with
`IK2_HBPF=183` vs 2.913e-3 on the old path; paired difference -1.5e-4, SE 2.0e-4.

turbo6 KV (`IK2_KV=turbo6`), the same measurement on this branch's defaults, one run each, laptop warm: decode 44.5
tok/s (45.6 with FP16 KV), prefill 2048 1,356 tok/s (1,219), prefill 8192 1,275 tok/s (1,237). Single runs: the
prefill gap is within run-to-run noise here.

## Credits and third-party code

- `src/ik2/ik2_hbpf.inc` ports prefill kernels from **headbouyJB/inkling2-gufo** (commit 8bb6cd5, `src/hip/fast.hip`):
  `k_gemm4` and its dequant helpers, `k_sum_pairs`, `k_hist` / `k_scan_items` / `k_scatter`, `attn_prefill`, and
  the host wrappers' launch shapes. The changes are listed at the top of that file.
- `k_rg` (`IK2_HBPF` bit 7) follows the shape of **gufo**'s routed expert GEMMs (gufo-org/gufo v0.35
  `RoutedDownF16Kernel` / `RoutedPairF16Kernel`, `SwapHalves`), with the NVFP4 type and 192-token tile of
  mikealanni/gufo's `RoutedF16GEMMKernel<kNVFP4>`.
- `src/ik2/ik2_turbo6.h` / `ik2_turbo6.inc` (`IK2_KV=turbo6`): tubo's codec (MIT), from TheTom/llama-cpp-turboquant
  PR #381 (MIT), after the TurboQuant paper (arXiv 2504.19874): details in the turbo6 section above.
- The engine itself is Maya's (and Strata's `hip_compat` layer).
- Made by Claude (Anthropic) working for Cold-Serve Labs.

headbouyJB/inkling2-gufo's license:

```
MIT License

Copyright (c) 2026 headbouyJB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


gufo's license:

```
MIT License

Copyright (c) 2026 gufo contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

