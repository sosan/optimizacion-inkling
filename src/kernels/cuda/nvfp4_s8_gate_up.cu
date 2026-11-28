// NVFP4 gate/up GEMM with mma.sync.s8, adapted from Basalt's
// src/kernels/cuda/nvfp4_grouped_mma.cuh (Strata MIT lineage). Simplified for
// this engine: plain global loads instead of TMA, one 16-row x 8-token tile per
// CTA and an 8-warp split over K, feeding ik2's h16 layout directly. The
// fragment mapping is Basalt's exactly (see docs/BASALT_IDEAS.md):
//   a0 = codes of weight row g,     k = 4t .. 4t+3
//   a1 = codes of weight row g + 8, same k positions
//   b0 = q8_1 word of token g (lane>>2), same k positions
//   d0/d1 = q8_1 scales of the *output* tokens 2t and 2t+1
//   si[0..3] = (row g, col 2t), (row g, col 2t+1), (row g+8, col 2t), (row g+8, col 2t+1)
// The MMA crossbar routes B's g column to C's 2t column; only the scale pairing
// above is the kernel's responsibility. sm_80+ (sm_89 and sm_120 both covered).
#include "strata/kernels/nvfp4_s8_gate_up.hpp"

#if !defined(STRATA_USE_HIP)
#include "strata/kernels/nvfp4_s8.hpp"
#include <cuda_fp16.h>

namespace strata::kernels::nvfp4s8 {
namespace {

// Same arithmetic as ik2's ue4m3() / Basalt's nvfp4_scale_dev(): ggml stores
// half the UE4M3 scale and doubles the E2M1 codes, so d * (2*E2M1) is exact.
__device__ __forceinline__ float nv_scale(uint8_t x) {
    const uint32_t v = x & 0x7Fu;
    const uint32_t e = v >> 3, m = v & 7u;
    const float r = e == 0 ? float(m) * 0x1.0p-10f : __uint_as_float(((e + 119u) << 23) | (m << 20));
    return v == 0x7Fu ? 0.0f : r;
}

__device__ __forceinline__ uint16_t half_bits(float f) { return __half_as_ushort(__float2half_rn(f)); }

// One CTA: expert e (blockIdx.y), 16 gate/up rows starting at r0 (blockIdx.x).
// 8 warps split K=2304 into 18 steps of 16 values: kk = wp + 8*j.
__global__ void gate_up_s8_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ tok,
                                  const int* __restrict__ row, size_t eb0, int C, int F,
                                  const uint32_t* __restrict__ xq, const float* __restrict__ sg,
                                  const float* __restrict__ su, float gcap, float ucap,
                                  uint16_t* __restrict__ h) {
    const int tid = threadIdx.x;
    const int wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int e = blockIdx.y, r0 = blockIdx.x * 16;
    __shared__ float s_g[8][16][9], s_u[8][16][9];
    const size_t rb = (size_t) (C / 64) * 36;
    const uint8_t* wg = blob[e] + (size_t) r0 * rb;
    const uint8_t* wu = blob[e] + eb0 + (size_t) r0 * rb;
    const int xb_tok = (C / 32) * 9;   // uint32 words per token's q8_1 activations

    float ag[4][4] = {}, au[4][4] = {};
#pragma unroll 1
    for (int j = 0; j < C / 128; ++j) {
        const int s = j & 3, kk = wp + 8 * j;
        const int kbx = kk >> 2, iqs = kk & 3;
        const auto* bga = reinterpret_cast<const Block*>(wg + (size_t) g * rb) + kbx;
        const auto* bgb = reinterpret_cast<const Block*>(wg + (size_t) (g + 8) * rb) + kbx;
        const auto* bua = reinterpret_cast<const Block*>(wu + (size_t) g * rb) + kbx;
        const auto* bub = reinterpret_cast<const Block*>(wu + (size_t) (g + 8) * rb) + kbx;

        const uint32_t ag0 = frag(bga, iqs, t), ag1 = frag(bgb, iqs, t);
        const uint32_t au0 = frag(bua, iqs, t), au1 = frag(bub, iqs, t);
        const float dga = nv_scale(bga->d[iqs]), dgb = nv_scale(bgb->d[iqs]);
        const float dua = nv_scale(bua->d[iqs]), dub = nv_scale(bub->d[iqs]);

        const int yb = kk >> 1;
        const uint32_t* xb = xq + (size_t) tok[e * 8 + g] * xb_tok;
        const uint32_t b0 = xb[9 * yb + 1 + 4 * ((kk & 3) & 1) + t];
        const float d8_0 = __low2float(*reinterpret_cast<const half2*>(xq + (size_t) tok[e * 8 + 2 * t] * xb_tok + 9 * yb));
        const float d8_1 = __low2float(*reinterpret_cast<const half2*>(xq + (size_t) tok[e * 8 + 2 * t + 1] * xb_tok + 9 * yb));

        int sgi[4], sui[4];
        mma_k16(sgi, ag0, ag1, b0);
        mma_k16(sui, au0, au1, b0);
        ag[s][0] += d8_0 * dga * float(sgi[0]);
        ag[s][1] += d8_1 * dga * float(sgi[1]);
        ag[s][2] += d8_0 * dgb * float(sgi[2]);
        ag[s][3] += d8_1 * dgb * float(sgi[3]);
        au[s][0] += d8_0 * dua * float(sui[0]);
        au[s][1] += d8_1 * dua * float(sui[1]);
        au[s][2] += d8_0 * dub * float(sui[2]);
        au[s][3] += d8_1 * dub * float(sui[3]);
    }

    // Same tree order as Basalt's per-tile stage: 4 k groups, then 8 warps.
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int col = 2 * t + (q & 1), row = 8 * (q >> 1) + g;
        s_g[col][row][wp] = (ag[0][q] + ag[2][q]) + (ag[1][q] + ag[3][q]);
        s_u[col][row][wp] = (au[0][q] + au[2][q]) + (au[1][q] + au[3][q]);
    }
    __syncthreads();
    for (int idx = tid; idx < 8 * 16; idx += 256) {
        const int row = idx & 15, col = idx >> 4;
        const float* pg = s_g[col][row];
        const float* pu = s_u[col][row];
        const float g0 = pg[0] + pg[4], g1 = pg[1] + pg[5], g2 = pg[2] + pg[6], g3 = pg[3] + pg[7];
        const float u0 = pu[0] + pu[4], u1 = pu[1] + pu[5], u2 = pu[2] + pu[6], u3 = pu[3] + pu[7];
        const float gv = (g0 + g2) + (g1 + g3), uv = (u0 + u2) + (u1 + u3);
        const int slot = e * 8 + col, ro = row[slot];
        const float g = gv * (sg ? sg[ro] : 1.0f), u = uv * (su ? su[ro] : 1.0f);
        h[(size_t) ro * F + r0 + row] =
            half_bits(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
    }
}

}  // namespace

bool gate_up_s8(const uint8_t* const* blob, const int* tok, const int* row, int G, size_t eb0, int C, int F,
                const void* x_q8_1, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                void* stream) {
    if (G <= 0 || C % 128 || F % 16 || !blob || !tok || !row || !x_q8_1) return false;
    gate_up_s8_kernel<<<dim3((unsigned) (F / 16), (unsigned) G), 256, 0, (cudaStream_t) stream>>>(
        blob, tok, row, eb0, C, F, (const uint32_t*) x_q8_1, sg, su, gcap, ucap, h);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels::nvfp4s8

#else  // HIP: no mma.sync.s8; the caller keeps the existing path.

namespace strata::kernels::nvfp4s8 {

bool gate_up_s8(const uint8_t*, const int*, const int*, int, size_t, int, int, const void*, const float*,
                const float*, float, float, uint16_t*, void*) {
    return false;
}

}  // namespace strata::kernels::nvfp4s8

#endif
