// src/ik2/ik2_g906.inc - gfx906 (MI50/MI60/Radeon VII) native NVFP4 expert kernels.
// Ported from the patch in upstream issue #36 (G0ldd1/mi50-playground, gfx906-kernels.patch
// against 27daec5). HIP only; CUDA untouched. IK2_G906=0 restores the stock path.
#if defined(STRATA_USE_HIP)

// Only the gfx906 target has v_dot2_f32_f16 / the 64-wide LDS layout these kernels use.
// Other AMD targets keep the stock path; the host gate below reports that.
bool ik2_g906_ok() {
    static const bool ok = [] {
        if (getenv("IK2_G906") && !std::atoi(getenv("IK2_G906"))) return false;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return std::strncmp(p.gcnArchName, "gfx906", 6) == 0;
    }();
    return ok;
}

#if defined(__gfx906__)

typedef _Float16 g906_h2 __attribute__((ext_vector_type(2)));
constexpr int G906_RPH = 4;
__device__ __forceinline__ float g906_dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(g906_h2, a), __builtin_bit_cast(g906_h2, b), c, false);
__device__ __forceinline__ uint32_t g906_pair_lo(uint32_t s) {   // FP16 bits of e2m1 low nibbles of bytes 0, 1
    const uint32_t sel = (s << 4) & 0x7070u;
    const uint32_t sgn = ((s & 8u) << 12) | ((s & 0x800u) << 20);
    return __byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn;
__device__ __forceinline__ uint32_t g906_pair_hi(uint32_t s) {   // ... high nibbles
    const uint32_t sel = s & 0x7070u;
    const uint32_t sgn = ((s & 0x80u) << 8) | ((s & 0x8000u) << 16);
    return __byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn;
}
__device__ __forceinline__ uint32_t g906_f2h2(float a, float b) {
    const g906_h2 v = {(_Float16) a, (_Float16) b};
    return __builtin_bit_cast(uint32_t, v);
}
// one sub-block: codes q0 (bytes 0-3), q1 (bytes 4-7); pairs p = 0..7 are values (2p, 2p + 1)
template <bool TwoLimb>
__device__ __forceinline__ float g906_sub(uint32_t q0, uint32_t q1, const uint32_t* xh, const uint32_t* xl) {
    const uint32_t w[8] = {g906_pair_lo(q0), g906_pair_lo(q0 >> 16), g906_pair_lo(q1), g906_pair_lo(q1 >> 16),
                           g906_pair_hi(q0), g906_pair_hi(q0 >> 16), g906_pair_hi(q1), g906_pair_hi(q1 >> 16)};
    float a = 0.0f;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        a = g906_dot2(w[p], xh[p], a);
        if (TwoLimb) a = g906_dot2(w[p], xl[p], a);
    }
    return a;
}
__device__ __forceinline__ float g906_hsum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor(v, m, 32);
    return v;
}

template <int NSB>
__global__ void __launch_bounds__(256) g906_gate_up_nv_kernel(const uint8_t* const* blob, const int* tok, size_t eb0,
                                                              const float* x, int C, int F, size_t rb,
                                                              const float* sg, const float* su, float gcap, float ucap,
                                                              uint16_t* h, const int* orow, int h_ld) {
    const int lane = threadIdx.x & 31, hw = threadIdx.x >> 5, s = blockIdx.y;
    const int nsub = C >> 4;
    uint32_t xh[NSB][8], xl[NSB][8];
    const float2* xr = (const float2*) (x + (size_t) tok[s] * C);
#pragma unroll
    for (int j = 0; j < NSB; ++j) {
        const int sb = lane + 32 * j;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            float2 f = sb < nsub ? xr[8 * sb + p] : make_float2(0.0f, 0.0f);
            const g906_h2 hi = {(_Float16) f.x, (_Float16) f.y};
            xh[j][p] = __builtin_bit_cast(uint32_t, hi);
            xl[j][p] = g906_f2h2(f.x - (float) hi.x, f.y - (float) hi.y);
        }
    }
    const uint8_t* bl = blob[s];
    const float sgs = sg ? sg[s] : 1.0f, sus = su ? su[s] : 1.0f;
    const int r0 = (blockIdx.x * 8 + hw) * G906_RPH;
#pragma unroll 2
    for (int i = 0; i < G906_RPH; ++i) {
        const int r = r0 + i;
        if (r >= F) break;
        const uint8_t* wg = bl + (size_t) r * rb;
        const uint8_t* wu = bl + eb0 + (size_t) r * rb;
        float ag = 0.0f, au = 0.0f;
#pragma unroll
        for (int j = 0; j < NSB; ++j) {
            const int sb = lane + 32 * j;
            if (sb < nsub) {
                const uint8_t* bg = wg + (size_t) (sb >> 2) * 36;
                const uint8_t* bu = wu + (size_t) (sb >> 2) * 36;
                const int sub = sb & 3;
                const uint32_t sgw = __ldg((const uint32_t*) bg), suw = __ldg((const uint32_t*) bu);
                const uint32_t g0 = __ldg((const uint32_t*) (bg + 4 + sub * 8)), g1 = __ldg((const uint32_t*) (bg + 8 + sub * 8));
                const uint32_t u0 = __ldg((const uint32_t*) (bu + 4 + sub * 8)), u1 = __ldg((const uint32_t*) (bu + 8 + sub * 8));
                ag += ue4m3((uint8_t) (sgw >> (8 * sub))) * g906_sub<true>(g0, g1, xh[j], xl[j]);
                au += ue4m3((uint8_t) (suw >> (8 * sub))) * g906_sub<true>(u0, u1, xh[j], xl[j]);
            }
        }
        ag = g906_hsum(ag);
        au = g906_hsum(au);
        if (lane == 0) {
            const float g = ag * sgs, u = au * sus;
            h[(size_t) (orow ? orow[s] : s) * h_ld + r] =
                f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
        }
    }
}

template <int NSB>
__global__ void __launch_bounds__(256) g906_down_nv_kernel(const uint8_t* const* blob, size_t off_down,
                                                           const uint16_t* hin, int C, int F, size_t rb, float* y) {
    const int lane = threadIdx.x & 31, hw = threadIdx.x >> 5, s = blockIdx.y;
    const int nsub = F >> 4;
    uint32_t xh[NSB][8];
    const uint32_t* hr = (const uint32_t*) (hin + (size_t) s * F);
#pragma unroll
    for (int j = 0; j < NSB; ++j) {
        const int sb = lane + 32 * j;
#pragma unroll
        for (int p = 0; p < 8; ++p) xh[j][p] = sb < nsub ? hr[8 * sb + p] : 0u;
    }
    const uint8_t* w = blob[s] + off_down;
    const int r0 = (blockIdx.x * 8 + hw) * G906_RPH;
#pragma unroll 2
    for (int i = 0; i < G906_RPH; ++i) {
        const int r = r0 + i;
        if (r >= C) break;
        const uint8_t* wr = w + (size_t) r * rb;
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < NSB; ++j) {
            const int sb = lane + 32 * j;
            if (sb < nsub) {
                const uint8_t* b = wr + (size_t) (sb >> 2) * 36;
                const int sub = sb & 3;
                const uint32_t sw = __ldg((const uint32_t*) b);
                const uint32_t q0 = __ldg((const uint32_t*) (b + 4 + sub * 8)), q1 = __ldg((const uint32_t*) (b + 8 + sub * 8));
                a += ue4m3((uint8_t) (sw >> (8 * sub))) * g906_sub<false>(q0, q1, xh[j], nullptr);
            }
        }
        a = g906_hsum(a);
        if (lane == 0) y[(size_t) s * C + r] = a;
    }
}

// ---- gfx906 v2: streamed NVFP4 rows.  A 32-lane half-wavefront reads a row as contiguous CW-dword chunks (16 B or 8 B a
// lane, coalesced); a dword d of the row is word d % 9 of block d / 9 (word 0: the four ue4m3 scales, words 1-8: the
// codes of sub-block (w - 1) / 2, half (w - 1) & 1).  The activations sit in LDS arranged per (sub-block, half) - the
// 8 values a code dword multiplies, as four FP16 pairs (two limbs for gate / up) - so a dword is 4 v_perm decodes and
// 4 (8) v_dot2_f32_f16.  Low VGPR use: many waves per SIMD to keep HBM busy.
// the 8 value indices (in pair order) that code dword half hf multiplies: q0 -> 0,1,2,3,8,9,10,11; q1 -> 4..7,12..15
__device__ __forceinline__ int g906b_vidx(int hf, int k) { return (k < 4 ? k : k + 4) + 4 * hf; }

template <bool TwoLimb>
__device__ __forceinline__ void g906b_fill(const float* xf, const uint16_t* xh16, int K, uint4* xs) {
    const int ne = (K >> 4) * 2;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        const int sb = e >> 1, hf = e & 1;
        uint32_t hi[4], lo[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int i0 = sb * 16 + g906b_vidx(hf, 2 * p), i1 = sb * 16 + g906b_vidx(hf, 2 * p + 1);
            if (TwoLimb) {
                const float a = xf[i0], b = xf[i1];
                const g906_h2 h = {(_Float16) a, (_Float16) b};
                hi[p] = __builtin_bit_cast(uint32_t, h);
                lo[p] = g906_f2h2(a - (float) h.x, b - (float) h.y);
            } else {
                hi[p] = (uint32_t) xh16[i0] | ((uint32_t) xh16[i1] << 16);
            }
        }
        if (TwoLimb) {
            xs[2 * e] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
            xs[2 * e + 1] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
        } else {
            xs[e] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        }
    }
}
// one code dword q (word w >= 1 of block blk, the scale dword sw) against its LDS entry; w == 0 contributes 0
template <bool TwoLimb>
__device__ __forceinline__ float g906b_word(uint32_t q, int blk, int w, uint32_t sw, const uint4* xs) {
    const int j = w > 0 ? w - 1 : 0;
    const int e = (4 * blk + (j >> 1)) * 2 + (j & 1);
    q = w > 0 ? q : 0u;
    const uint32_t w0 = g906_pair_lo(q), w1 = g906_pair_lo(q >> 16), w2 = g906_pair_hi(q), w3 = g906_pair_hi(q >> 16);
    float a;
    if (TwoLimb) {
        const uint4 h = xs[2 * e], l = xs[2 * e + 1];
        a = g906_dot2(w0, h.x, 0.0f); a = g906_dot2(w1, h.y, a); a = g906_dot2(w2, h.z, a); a = g906_dot2(w3, h.w, a);
        a = g906_dot2(w0, l.x, a); a = g906_dot2(w1, l.y, a); a = g906_dot2(w2, l.z, a); a = g906_dot2(w3, l.w, a);
    } else {
        const uint4 h = xs[e];
        a = g906_dot2(w0, h.x, 0.0f); a = g906_dot2(w1, h.y, a); a = g906_dot2(w2, h.z, a); a = g906_dot2(w3, h.w, a);
    }
    return a * ue4m3((uint8_t) (sw >> (8 * (j >> 1))));
}
template <int CW>
struct G906Chunk { uint32_t v[CW]; };
template <int CW>
__device__ __forceinline__ G906Chunk<CW> g906b_load(const uint8_t* p) {
    G906Chunk<CW> c;
    if (CW == 4) {
        const uint4 u = __ldg((const uint4*) p);
        c.v[0] = u.x, c.v[1] = u.y, c.v[2] = u.z, c.v[3] = u.w;
    } else {
        const uint2 u = __ldg((const uint2*) p);
        c.v[0] = u.x, c.v[1] = u.y;
    }
    return c;
}
// a row's dot: chunks c = lane + 32 it; the chunk spans blocks blk0 and maybe blk0 + 1, their scale dwords loaded
template <int CW, bool TwoLimb>
__device__ __forceinline__ float g906b_row(const uint8_t* wr, int nd, int lane, const uint4* xs) {
    const int nchunk = nd / CW;
    float acc = 0.0f;
    for (int c = lane; c < nchunk; c += 32) {
        const G906Chunk<CW> ch = g906b_load<CW>(wr + (size_t) c * CW * 4);
        const int d0 = c * CW, blk0 = d0 / 9;
        const uint32_t s0 = __ldg((const uint32_t*) (wr + (size_t) blk0 * 36));
        const int last = (d0 + CW - 1) / 9;
        const uint32_t s1 = last != blk0 ? __ldg((const uint32_t*) (wr + (size_t) last * 36)) : s0;
#pragma unroll
        for (int k = 0; k < CW; ++k) {
            const int d = d0 + k, blk = d / 9, w = d - 9 * blk;
            acc += g906b_word<TwoLimb>(ch.v[k], blk, w, blk == blk0 ? s0 : s1, xs);
        }
    }
    return acc;
}

template <int CW>
__global__ void __launch_bounds__(256) g906b_gate_up_kernel(const uint8_t* const* blob, const int* tok, size_t eb0,
                                                            const float* x, int C, int F, size_t rb, const float* sg,
                                                            const float* su, float gcap, float ucap, uint16_t* h,
                                                            const int* orow, int h_ld) {
    extern __shared__ uint4 g906_xs[];
    const int s = blockIdx.y;
    g906b_fill<true>(x + (size_t) tok[s] * C, nullptr, C, g906_xs);
    __syncthreads();
    const int lane = threadIdx.x & 31, hw = threadIdx.x >> 5;
    const uint8_t* bl = blob[s];
    const float sgs = sg ? sg[s] : 1.0f, sus = su ? su[s] : 1.0f;
    const int nd = (int) (rb / 4);
    const int r0 = (blockIdx.x * 8 + hw) * G906_RPH;
    for (int i = 0; i < G906_RPH; ++i) {
        const int r = r0 + i;
        if (r >= F) break;
        float ag = g906b_row<CW, true>(bl + (size_t) r * rb, nd, lane, g906_xs);
        float au = g906b_row<CW, true>(bl + eb0 + (size_t) r * rb, nd, lane, g906_xs);
        ag = g906_hsum(ag);
        au = g906_hsum(au);
        if (lane == 0) {
            const float g = ag * sgs, u = au * sus;
            h[(size_t) (orow ? orow[s] : s) * h_ld + r] =
                f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
        }
    }
}
template <int CW>
__global__ void __launch_bounds__(256) g906b_down_kernel(const uint8_t* const* blob, size_t off_down,
                                                         const uint16_t* hin, int C, int F, size_t rb, float* y) {
    extern __shared__ uint4 g906_xs[];
    const int s = blockIdx.y;
    g906b_fill<false>(nullptr, hin + (size_t) s * F, F, g906_xs);
    __syncthreads();
    const int lane = threadIdx.x & 31, hw = threadIdx.x >> 5;
    const uint8_t* w = blob[s] + off_down;
    const int nd = (int) (rb / 4);
    const int r0 = (blockIdx.x * 8 + hw) * G906_RPH;
    for (int i = 0; i < G906_RPH; ++i) {
        const int r = r0 + i;
        if (r >= C) break;
        float a = g906b_row<CW, false>(w + (size_t) r * rb, nd, lane, g906_xs);
        a = g906_hsum(a);
        if (lane == 0) y[(size_t) s * C + r] = a;
    }
}
static int g906_ver() {
    static const int v = getenv("IK2_G906") ? atoi(getenv("IK2_G906")) : 6;
    return v;
}
static bool g906b_gate_up(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C, int F,
                          const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                          const int* orow, int h_ld) {
    const size_t rb = mv_row_bytes(kNVFP4, C);
    const dim3 g((unsigned) ((F + 8 * G906_RPH - 1) / (8 * G906_RPH)), (unsigned) S);
    const size_t lds = (size_t) (C / 16) * 2 * 2 * sizeof(uint4);
    const int ld = h_ld > 0 ? h_ld : F;
    if (lds > 60000) return false;
    if (rb % 16 == 0 && eb0 % 16 == 0)
        g906b_gate_up_kernel<4><<<g, 256, lds, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld);
    else if (rb % 8 == 0 && eb0 % 8 == 0)
        g906b_gate_up_kernel<2><<<g, 256, lds, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld);
    else
        return false;
    check("g906b_gate_up");
    return true;
}
static bool g906b_down(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                       cudaStream_t s) {
    const size_t rb = mv_row_bytes(kNVFP4, F);
    const dim3 g((unsigned) ((C + 8 * G906_RPH - 1) / (8 * G906_RPH)), (unsigned) S);
    const size_t lds = (size_t) (F / 16) * 2 * sizeof(uint4);
    if (lds > 60000) return false;
    if (rb % 16 == 0 && off_down % 16 == 0)
        g906b_down_kernel<4><<<g, 256, lds, s>>>(blob, off_down, h, C, F, rb, y);
    else if (rb % 8 == 0 && off_down % 8 == 0)
        g906b_down_kernel<2><<<g, 256, lds, s>>>(blob, off_down, h, C, F, rb, y);
    else
        return false;
    check("g906b_down");
    return true;
}

// ---- gfx906 v3: as v2, but every chunk of the row (and its scale dwords) is loaded before any is used - up to
// MAXIT x CW dwords in flight a lane per matrix - and 512-thread blocks, one row per half-wavefront.

#else   // not gfx906: no symbol conflict, the engine's gate keeps the stock path
static bool g906b_gate_up(const uint8_t* const*, const int*, int, size_t, const float*, int, int, const float*,
                         const float*, float, float, uint16_t*, cudaStream_t, const int*, int) { return false; }
static bool g906b_down(const uint8_t* const*, int, size_t, const uint16_t*, int, int, float*, cudaStream_t) {
    return false;
}
#endif  // __gfx906__

// One visit's gate/up + down through the gfx906 kernels. False: the caller must take the stock path.
bool ik2_g906_experts(const uint8_t* const* blob, const int* tok, int S, size_t eb0, size_t off_down, const float* x,
                      int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                      cudaStream_t s) {
    if (!ik2_g906_ok()) return false;
    if (!g906b_gate_up(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s, nullptr, 0)) return false;
    if (!g906b_down(blob, S, off_down, h, C, F, y, s)) return false;
    return true;
}

#endif   // STRATA_USE_HIP
