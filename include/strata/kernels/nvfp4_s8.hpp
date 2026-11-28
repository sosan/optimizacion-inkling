// include/strata/kernels/nvfp4_s8.hpp - NVFP4 -> int8 fragments for mma.sync.s8.
// Adapted from Basalt (jesdga95/basalt, src/kernels/cuda/nvfp4_grouped_mma.cuh,
// Strata MIT lineage). GGUF's NVFP4 block keeps its E2M1 codes as 0..7 (half
// the real magnitude) and a halved UE4M3 scale; the doubled int8 codes here
// therefore reproduce the same products as our scalar decode_value() path.
// Works on sm_80+ (int8 mma.sync), including sm_89 and sm_120.
#pragma once

#include <cstdint>

namespace strata::kernels::nvfp4s8 {

// One ggml NVFP4 block: 4 UE4M3 sub-block scales then 32 code bytes (64 values).
struct Block {
    uint8_t d[4];
    uint8_t qs[32];
};

__device__ __forceinline__ uint32_t load_b4(const uint8_t* p, int i) {
    return reinterpret_cast<const uint32_t*>(p)[i];
}

// The four int8 codes lane group t contributes to an m16n8k16 s8 A operand.
__device__ __forceinline__ uint32_t frag(const Block* b, int iqs, int t) {
    const uint32_t q4 = load_b4(b->qs, 2 * iqs + (t & 1));
    const uint32_t nibs = (t & 2) ? ((q4 >> 4) & 0x0F0F0F0Fu) : (q4 & 0x0F0F0F0Fu);
    const uint32_t sel = (nibs & 0x0Fu) | ((nibs >> 4) & 0xF0u) |
                         ((nibs >> 8) & 0xF00u) | ((nibs >> 12) & 0xF000u);
    const uint32_t lo = __byte_perm(0x03020100u, 0x0c080604u, sel);
    const uint32_t hi = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, sel);
    return __byte_perm(lo, hi, 0x3210u | ((sel & 0x8888u) >> 1));
}

__device__ __forceinline__ void mma_k16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

}  // namespace strata::kernels::nvfp4s8
