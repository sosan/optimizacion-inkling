// src/ik2/ik2_g906_full.inc - complete gfx906 kernel set from upstream issue #36
// (G0ldd1/mi50-playground, gfx906-kernels.patch against 27daec5). Only built
// when CMake sets IK2_G906_TARGET (gfx906 kernel instructions); other targets
// get declining stubs and the engine keeps the stock path. CUDA is untouched.
#if defined(STRATA_USE_HIP)
bool ik2_g906_ok();
#if defined(IK2_G906_TARGET)

__device__ float4 g906lg_part4[3 * kLgcChunks * 8 * 33];
__device__ unsigned g906lg_cnt[3];
// a transpose step over lanes xor O: 2 O values -> O (the half the lane's bit O picks)
template <int O>
__device__ __forceinline__ void g906_tr32(float* v, int lane) {
    const bool up = (lane & O) != 0;
#pragma unroll
    for (int k = 0; k < O; ++k) {
        const float snd = up ? v[k] : v[k + O], keep = up ? v[k + O] : v[k];
        v[k] = keep + __shfl_xor(snd, O, 64);
    }
}
            a.y[j][(size_t) t * C + c] = (1.0f - tanhf(f)) * xv[u] + tanhf(gn) * mv[u];
        }
    }
}
static bool g906lg_launch(const LoopGateArgs& a, bool compact, int n, int T, int C, int R, float eps, float* sc,
                          cudaStream_t s) {
    static const bool on = !getenv("IK2_G906_LG") || atoi(getenv("IK2_G906_LG"));
    if (!on || compact || C % 4 || C / 4 > 192 * 4 || R > 128) return false;
    for (int j = 0; j < n; ++j)
        if (((uintptr_t) a.x[j] | (uintptr_t) a.m[j] | (uintptr_t) a.D[j]) & 15) return false;
    const dim3 dots(R, n, T);
    const int mi = (C / 4 + 191) / 192;
    static const bool lgb = !getenv("IK2_G906_LGB") || atoi(getenv("IK2_G906_LGB"));
    if (lgb && T >= 2 && T <= 8 && R == 64 && n <= 3 && C % 128 == 0 && C / 128 <= kLgcChunks) {
        const dim3 gd((unsigned) (C / 128), (unsigned) n), ga((unsigned) ((C + 63) / 64), (unsigned) n);
#define G906LGB_L(NT) g906lgc_dots_kernel<NT><<<gd, 512, 0, s>>>(a, C, sc); \
        g906lgb_apply_kernel<16, NT><<<ga, 256, 0, s>>>(a, C, R, eps, sc)
        switch (T) {
            case 2: G906LGB_L(2); break; case 3: G906LGB_L(3); break; case 4: G906LGB_L(4); break;
            case 5: G906LGB_L(5); break; case 6: G906LGB_L(6); break; case 7: G906LGB_L(7); break;
            default: G906LGB_L(8); break;
        }
#undef G906LGB_L
        check("g906lgb");
        return true;
    }
    if (mi <= 1) g906lg_dots_kernel<1><<<dots, 192, 0, s>>>(a, T, C, R, sc);
    else if (mi == 2) g906lg_dots_kernel<2><<<dots, 192, 0, s>>>(a, T, C, R, sc);
    else if (mi == 3) g906lg_dots_kernel<3><<<dots, 192, 0, s>>>(a, T, C, R, sc);
    else g906lg_dots_kernel<4><<<dots, 192, 0, s>>>(a, T, C, R, sc);
    check("g906lg (dots)");
    g906lg_apply_kernel<<<dim3((C + 63) / 64, n, T), 256, 0, s>>>(a, T, C, R, eps, sc);
    check("g906lg (apply)");
    return true;
}
// A block: 64 queries of one head, 4 waves; keys in tiles of 64.  Q, K [key][dim] and V^T [dim][key] in LDS as FP16
// (an FP8 tile made FP16 as it is stored; the next tile's bytes in registers meanwhile); P (FP16) over K's space.
// S = Q K^T: a thread 4 queries x 4 keys; O = P V: 4 queries x 8 dims; v_dot2_f32_f16, FP32 sums.
typedef _Float16 g906fa_h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float g906fa_dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(g906fa_h2, a), __builtin_bit_cast(g906fa_h2, b), c, false);
}
        for (int c = 0; c < 8; ++c) op[16 * c] = o[a][c];
        if (tk == 0) ml_part[((size_t) split * Tq + i) * H + h] = make_float2(m[a], la);
    }
}
static bool g906fa_on() {
    static const int on = getenv("IK2_G906_FA") ? atoi(getenv("IK2_G906_FA")) : 1;
    return on != 0;
}
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
struct G906Chunk { uint32_t v[CW]; };
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
struct G906Row {
    G906Chunk<CW> ch[MAXIT];
    uint32_t s0[MAXIT], s1[MAXIT];
};
template <int CW, int MAXIT>
__device__ __forceinline__ void g906c_load(const uint8_t* wr, int nchunk, int lane, G906Row<CW, MAXIT>& R) {
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 32 * it;
        const int cc = c < nchunk ? c : nchunk - 1;   // (clamped: a valid address; zeroed below)
        R.ch[it] = g906b_load<CW>(wr + (size_t) cc * CW * 4);
        const int d0 = cc * CW, blk0 = d0 / 9, last = (d0 + CW - 1) / 9;
        R.s0[it] = __ldg((const uint32_t*) (wr + (size_t) blk0 * 36));
        R.s1[it] = __ldg((const uint32_t*) (wr + (size_t) last * 36));
    }
}
template <int CW, int MAXIT, bool TwoLimb>
__device__ __forceinline__ float g906c_dot(const G906Row<CW, MAXIT>& R, int nchunk, int lane, const uint4* xs) {
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 32 * it;
        if (c < nchunk) {
            const int d0 = c * CW, blk0 = d0 / 9;
#pragma unroll
            for (int k = 0; k < CW; ++k) {
                const int d = d0 + k, blk = d / 9, w = d - 9 * blk;
                acc += g906b_word<TwoLimb>(R.ch[it].v[k], blk, w, blk == blk0 ? R.s0[it] : R.s1[it], xs);
            }
        }
    }
    return acc;
}
    a = g906_hsum(a);
    if (lane == 0) y[(size_t) s * C + r] = a;
}
#define G906C_GU(CW, MI) g906c_gate_up_kernel<CW, MI><<<g, 512, lds, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld)
static bool g906c_gate_up(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C, int F,
                          const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                          const int* orow, int h_ld) {
    const size_t rb = mv_row_bytes(kNVFP4, C);
    const dim3 g((unsigned) ((F + 15) / 16), (unsigned) S);
    const size_t lds = (size_t) (C / 16) * 2 * 2 * sizeof(uint4);
    const int ld = h_ld > 0 ? h_ld : F;
    if (lds > 60000) return false;
    const bool c16 = rb % 16 == 0 && eb0 % 16 == 0;
    if (!c16 && (rb % 8 || eb0 % 8)) return false;
    const int nchunk = (int) (rb / 4) / (c16 ? 4 : 2), mi = (nchunk + 31) / 32;
    if (c16 && mi == 1) G906C_GU(4, 1); else if (c16 && mi == 2) G906C_GU(4, 2); else if (c16 && mi == 3) G906C_GU(4, 3);
    else if (c16 && mi == 4) G906C_GU(4, 4); else if (!c16 && mi <= 2) G906C_GU(2, 2); else if (!c16 && mi <= 4) G906C_GU(2, 4);
    else if (!c16 && mi <= 6) G906C_GU(2, 6); else return false;
    check("g906c_gate_up");
    return true;
}
#define G906C_DN(CW, MI) g906c_down_kernel<CW, MI><<<g, 512, lds, s>>>(blob, off_down, h, C, F, rb, y)
static bool g906c_down(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                       cudaStream_t s) {
    const size_t rb = mv_row_bytes(kNVFP4, F);
    const dim3 g((unsigned) ((C + 15) / 16), (unsigned) S);
    const size_t lds = (size_t) (F / 16) * 2 * sizeof(uint4);
    if (lds > 60000) return false;
    const bool c16 = rb % 16 == 0 && off_down % 16 == 0;
    if (!c16 && (rb % 8 || off_down % 8)) return false;
    const int nchunk = (int) (rb / 4) / (c16 ? 4 : 2), mi = (nchunk + 31) / 32;
    if (c16 && mi == 1) G906C_DN(4, 1); else if (c16 && mi == 2) G906C_DN(4, 2); else if (c16 && mi == 3) G906C_DN(4, 3);
    else if (c16 && mi == 4) G906C_DN(4, 4); else if (!c16 && mi <= 2) G906C_DN(2, 2); else if (!c16 && mi <= 4) G906C_DN(2, 4);
    else if (!c16 && mi <= 6) G906C_DN(2, 6); else return false;
    check("g906c_down");
    return true;
}
// v_perm into a byte table (+ one for the negatives, a v_bfi on the sign bits) and v_dot4_i32_i8 against the
// activations quantized per 16-value sub-block to two int8 limbs (q = round(x * 16256 / max|x|), q = 128 hi + lo;
// the scheme of nv_input / pack_limbs).  LDS: per (sub-block, half) a uint4 {hi of values A, hi of B, lo A, lo B}
// (A = the 4 values a code dword's low nibbles multiply, B = the high nibbles') + one float scale per sub-block.
__device__ __forceinline__ uint32_t g906d_dec(uint32_t n) {   // n: four nibble codes in the low nibbles of 4 bytes
    const uint32_t sel = n & 0x07070707u;
    const uint32_t p = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel);
    const uint32_t m = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel);
    const uint32_t neg = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (p & ~neg) | (m & neg);
}
// LDS fill: entries e = 2 sb + hf (threads loop); scale per sub-block
template <bool F32In>
__device__ __forceinline__ void g906d_fill(const float* xf, const uint16_t* xh16, int K, uint4* xs, float* xsc) {
    const int ne = (K >> 4) * 2;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        const int sb = e >> 1, hf = e & 1;
        float v[16], mx = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = F32In ? xf[sb * 16 + j] : h2f(xh16[sb * 16 + j]);
            mx = fmaxf(mx, fabsf(v[j]));
        }
        const float inv = mx > 0.0f ? 16256.0f / mx : 0.0f;
        uint32_t hA = 0, hB = 0, lA = 0, lB = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int qa = __float2int_rn(v[4 * hf + t] * inv), qb = __float2int_rn(v[8 + 4 * hf + t] * inv);
            const int ha = (qa + 64) >> 7, hb = (qb + 64) >> 7;
            hA |= (uint32_t) (uint8_t) ha << (8 * t);
            hB |= (uint32_t) (uint8_t) hb << (8 * t);
            lA |= (uint32_t) (uint8_t) (qa - 128 * ha) << (8 * t);
            lB |= (uint32_t) (uint8_t) (qb - 128 * hb) << (8 * t);
        }
        xs[e] = make_uint4(hA, hB, lA, lB);
        if (hf == 0) xsc[sb] = mx / 16256.0f;
    }
}
__device__ __forceinline__ float g906d_word(uint32_t q, int blk, int w, uint32_t sw, const uint4* xs, const float* xsc) {
    const int j = w > 0 ? w - 1 : 0;
    const int sb = 4 * blk + (j >> 1);
    const uint4 x = xs[2 * sb + (j & 1)];
    q = w > 0 ? q : 0u;
    const int wa = (int) g906d_dec(q & 0x0F0F0F0Fu), wb = (int) g906d_dec((q >> 4) & 0x0F0F0F0Fu);
    int ih = __builtin_amdgcn_sdot4(wa, (int) x.x, 0, false);
    ih = __builtin_amdgcn_sdot4(wb, (int) x.y, ih, false);
    int il = __builtin_amdgcn_sdot4(wa, (int) x.z, 0, false);
    il = __builtin_amdgcn_sdot4(wb, (int) x.w, il, false);
    return (float) (ih * 128 + il) * (xsc[sb] * ue4m3((uint8_t) (sw >> (8 * (j >> 1)))));
}
template <int CW, int MAXIT>
__device__ __forceinline__ float g906d_dot(const G906Row<CW, MAXIT>& R, int nchunk, int lane, const uint4* xs,
                                           const float* xsc) {
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 32 * it;
        if (c < nchunk) {
            const int d0 = c * CW, blk0 = d0 / 9;
#pragma unroll
            for (int k = 0; k < CW; ++k) {
                const int d = d0 + k, blk = d / 9, w = d - 9 * blk;
                acc += g906d_word(R.ch[it].v[k], blk, w, blk == blk0 ? R.s0[it] : R.s1[it], xs, xsc);
            }
        }
    }
    return acc;
}
    a = g906_hsum(a);
    if (lane == 0) y[(size_t) s * C + r] = a;
}
#define G906D_GU(CW, MI) g906d_gate_up_kernel<CW, MI><<<g, 512, lds, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld)
static bool g906d_gate_up(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C, int F,
                          const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                          const int* orow, int h_ld) {
    const size_t rb = mv_row_bytes(kNVFP4, C);
    const dim3 g((unsigned) ((F + 15) / 16), (unsigned) S);
    const size_t lds = (size_t) (C / 16) * (2 * sizeof(uint4) + sizeof(float));
    const int ld = h_ld > 0 ? h_ld : F;
    if (lds > 60000) return false;
    const bool c16 = rb % 16 == 0 && eb0 % 16 == 0;
    if (!c16 && (rb % 8 || eb0 % 8)) return false;
    const int nchunk = (int) (rb / 4) / (c16 ? 4 : 2), mi = (nchunk + 31) / 32;
    if (c16 && mi == 1) G906D_GU(4, 1); else if (c16 && mi == 2) G906D_GU(4, 2); else if (c16 && mi == 3) G906D_GU(4, 3);
    else if (c16 && mi == 4) G906D_GU(4, 4); else if (!c16 && mi <= 2) G906D_GU(2, 2); else if (!c16 && mi <= 4) G906D_GU(2, 4);
    else if (!c16 && mi <= 6) G906D_GU(2, 6); else return false;
    check("g906d_gate_up");
    return true;
}
#define G906D_DN(CW, MI) g906d_down_kernel<CW, MI><<<g, 512, lds, s>>>(blob, off_down, h, C, F, rb, y)
static bool g906d_down(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                       cudaStream_t s) {
    const size_t rb = mv_row_bytes(kNVFP4, F);
    const dim3 g((unsigned) ((C + 15) / 16), (unsigned) S);
    const size_t lds = (size_t) (F / 16) * (2 * sizeof(uint4) + sizeof(float));
    if (lds > 60000) return false;
    const bool c16 = rb % 16 == 0 && off_down % 16 == 0;
    if (!c16 && (rb % 8 || off_down % 8)) return false;
    const int nchunk = (int) (rb / 4) / (c16 ? 4 : 2), mi = (nchunk + 31) / 32;
    if (c16 && mi == 1) G906D_DN(4, 1); else if (c16 && mi == 2) G906D_DN(4, 2); else if (c16 && mi == 3) G906D_DN(4, 3);
    else if (c16 && mi == 4) G906D_DN(4, 4); else if (!c16 && mi <= 2) G906D_DN(2, 2); else if (!c16 && mi <= 4) G906D_DN(2, 4);
    else if (!c16 && mi <= 6) G906D_DN(2, 6); else return false;
    check("g906d_down");
    return true;
}

static int g906_wg() {
    static const int v = getenv("IK2_G906_WG") ? atoi(getenv("IK2_G906_WG")) : 256;
    return v;
}
#define G906_X4 1
// int8 with v_perm (positive table, negative table, sign mask from v_perm's 0x0C/0x0D constant selectors) and
// v_dot4_i32_i8 against two int8 limbs.  LDS a block: 8 uint4 {hiA, hiB, loA, loB} (word w = 1..8) + 1 float4 of
// the sub-block activation scales x 2^119 (ue4m3 = as_float(b << 20) x 2^119, denormal-exact) = 9 uint4 = 36
// dwords (stride 4 mod 32 banks).
__device__ __forceinline__ int g906e_dec(uint32_t sel7, uint32_t sgn) {   // sel7: codes & 7 a byte; sgn: bit0 a byte
    const uint32_t p = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel7);
    const uint32_t m = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel7);
    const uint32_t neg = __builtin_amdgcn_perm(0u, 0u, sgn | 0x0C0C0C0Cu);   // 0x0C -> 0x00, 0x0D -> 0xFF
    return (int) ((p & ~neg) | (m & neg));
}
template <bool F32In>
__device__ __forceinline__ void g906e_fill(const float* xf, const uint16_t* xh16, int K, uint4* xs) {
    const int nsb = K >> 4;
    for (int sb = threadIdx.x; sb < nsb; sb += blockDim.x) {
        float v[16], mx = 0.0f;
        if (F32In) {
#pragma unroll
            for (int j = 0; j < 16; j += 4) {
                const float4 t = *(const float4*) (xf + sb * 16 + j);
                v[j] = t.x; v[j + 1] = t.y; v[j + 2] = t.z; v[j + 3] = t.w;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                const uint32_t t = *(const uint32_t*) (xh16 + sb * 16 + j);
                v[j] = h2f((uint16_t) t); v[j + 1] = h2f((uint16_t) (t >> 16));
            }
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) mx = fmaxf(mx, fabsf(v[j]));
        const float inv = mx > 0.0f ? 16256.0f / mx : 0.0f;
        const int blk = sb >> 2, sub = sb & 3;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            uint32_t hA = 0, hB = 0, lA = 0, lB = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int qa = __float2int_rn(v[4 * hf + t] * inv), qb = __float2int_rn(v[8 + 4 * hf + t] * inv);
                const int ha = (qa + 64) >> 7, hb = (qb + 64) >> 7;
                hA |= (uint32_t) (uint8_t) ha << (8 * t);
                hB |= (uint32_t) (uint8_t) hb << (8 * t);
                lA |= (uint32_t) (uint8_t) (qa - 128 * ha) << (8 * t);
                lB |= (uint32_t) (uint8_t) (qb - 128 * hb) << (8 * t);
            }
            xs[blk * 9 + 2 * sub + hf] = make_uint4(hA, hB, lA, lB);
        }
        ((float*) (xs + blk * 9 + 8))[sub] = mx * (1.0f / 16256.0f);
    }
}
struct G906X16 {
    float v[16];
};
template <bool F32In>
__device__ __forceinline__ void g906e_xld(const float* xf, const uint16_t* xh16, int K, G906X16& X) {
    const int sb = threadIdx.x;
    if (sb >= (K >> 4)) return;
    if (F32In) {
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
            const float4 t = *(const float4*) (xf + sb * 16 + j);
            X.v[j] = t.x; X.v[j + 1] = t.y; X.v[j + 2] = t.z; X.v[j + 3] = t.w;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            const uint32_t t = *(const uint32_t*) (xh16 + sb * 16 + j);
            X.v[j] = h2f((uint16_t) t); X.v[j + 1] = h2f((uint16_t) (t >> 16));
        }
    }
}
__device__ __forceinline__ void g906e_xst(const G906X16& X, int K, uint4* xs) {
    const int sb = threadIdx.x;
    if (sb >= (K >> 4)) return;
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) mx = fmaxf(mx, fabsf(X.v[j]));
    const float inv = mx > 0.0f ? 16256.0f / mx : 0.0f;
    const int blk = sb >> 2, sub = sb & 3;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        uint32_t hA = 0, hB = 0, lA = 0, lB = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int qa = __float2int_rn(X.v[4 * hf + t] * inv), qb = __float2int_rn(X.v[8 + 4 * hf + t] * inv);
            const int ha = (qa + 64) >> 7, hb = (qb + 64) >> 7;
            hA |= (uint32_t) (uint8_t) ha << (8 * t);
            hB |= (uint32_t) (uint8_t) hb << (8 * t);
            lA |= (uint32_t) (uint8_t) (qa - 128 * ha) << (8 * t);
            lB |= (uint32_t) (uint8_t) (qb - 128 * hb) << (8 * t);
        }
        xs[blk * 9 + 2 * sub + hf] = make_uint4(hA, hB, lA, lB);
    }
    ((float*) (xs + blk * 9 + 8))[sub] = mx * (1.0f / 16256.0f);
}
struct G906Blk {
    uint32_t d[9];
};
__device__ __forceinline__ void g906e_ld(const uint8_t* p, G906Blk& B) {
#if G906_X4
    // (dwordx4 at a dword-aligned address: gfx9's unaligned mode; 3 loads a block, not 9)
    const g906_u4 a = *(const g906_u4*) p, b = *(const g906_u4*) (p + 16);
    B.d[0] = a.x, B.d[1] = a.y, B.d[2] = a.z, B.d[3] = a.w, B.d[4] = b.x, B.d[5] = b.y, B.d[6] = b.z, B.d[7] = b.w;
    B.d[8] = __ldg((const uint32_t*) (p + 32));
#else
    const uint32_t* q = (const uint32_t*) p;
#pragma unroll
    for (int i = 0; i < 9; ++i) B.d[i] = __ldg(q + i);
#endif
}
__device__ __forceinline__ float g906e_blk(const G906Blk& B, const uint4* xb) {
    const uint4 sc = xb[8];
    const float xsc[4] = {__uint_as_float(sc.x), __uint_as_float(sc.y), __uint_as_float(sc.z), __uint_as_float(sc.w)};
    float acc = 0.0f;
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        int ih = 0, il = 0;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const uint32_t q = B.d[1 + 2 * sub + hf];
            const uint4 x = xb[2 * sub + hf];
            const int wa = g906e_dec(q & 0x07070707u, (q >> 3) & 0x01010101u);
            const int wb = g906e_dec((q >> 4) & 0x07070707u, (q >> 7) & 0x01010101u);
            ih = __builtin_amdgcn_sdot4(wa, (int) x.x, ih, false);
            ih = __builtin_amdgcn_sdot4(wb, (int) x.y, ih, false);
            il = __builtin_amdgcn_sdot4(wa, (int) x.z, il, false);
            il = __builtin_amdgcn_sdot4(wb, (int) x.w, il, false);
        }
        const float ws = ue4m3((uint8_t) (B.d[0] >> (8 * sub)));
        acc = fmaf((float) (ih * 128 + il), ws * xsc[sub], acc);
    }
    return acc;
}
// a row's G partial sums -> lane (row's first): LDS scratch red[512]
__device__ __forceinline__ float g906e_rsum(float v, float* red, int G, int part) {
    red[threadIdx.x] = v;
    __syncthreads();
    float s = 0.0f;
    if (part == 0)
        for (int i = 0; i < G; ++i) s += red[threadIdx.x + i];
    return s;
}
    a = g906e_rsum(a, red, G, part);
    if (live && part == 0 && r < C) y[(size_t) s * C + r] = a;
}
// (G, NB) for nblk blocks a row: G * NB == nblk, G <= 16, the fewest idle lanes
static bool g906e_pick(int nblk, int& G, int& NB) {
    static const int cand[][2] = {{12, 3}, {9, 2}, {16, 2}, {16, 1}, {8, 1}, {12, 1}, {9, 1}, {16, 3}, {16, 4},
                                  {12, 2}, {12, 4}, {9, 3}, {9, 4}, {8, 2}, {8, 3}, {8, 4}};
    for (auto& c : cand)
        if (c[0] * c[1] == nblk) { G = c[0]; NB = c[1]; return true; }
    return false;
}
#define G906E_GU(GG, NN) g906e_gate_up_kernel<GG, NN><<<g, g906_wg(), lds, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld)
static bool g906e_gate_up(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C, int F,
                          const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                          const int* orow, int h_ld) {
    const size_t rb = mv_row_bytes(kNVFP4, C);
    int G, NB;
    if (!g906e_pick(C / 64, G, NB) || rb % 4 || eb0 % 4 || ((uintptr_t) x & 15)) return false;
    const int rpb = (g906_wg() / 64) * (64 / G);
    const dim3 g((unsigned) ((F + rpb - 1) / rpb), (unsigned) S);
    const size_t lds = (size_t) (C / 64) * 9 * sizeof(uint4) + 512 * sizeof(float);
    const int ld = h_ld > 0 ? h_ld : F;
    if (lds > 60000) return false;
    if (G == 12 && NB == 3) G906E_GU(12, 3); else if (G == 9 && NB == 2) G906E_GU(9, 2);
    else if (G == 16 && NB == 2) G906E_GU(16, 2); else if (G == 16 && NB == 1) G906E_GU(16, 1);
    else return false;
    check("g906e_gate_up");
    return true;
}
#define G906E_DN(GG, NN) g906e_down_kernel<GG, NN><<<g, g906_wg(), lds, s>>>(blob, off_down, h, C, F, rb, y)
static bool g906e_down(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                       cudaStream_t s) {
    const size_t rb = mv_row_bytes(kNVFP4, F);
    int G, NB;
    if (!g906e_pick(F / 64, G, NB) || rb % 4 || off_down % 4 || ((uintptr_t) h & 3) || F / 16 > g906_wg()) return false;
    const int rpb = (g906_wg() / 64) * (64 / G);
    const dim3 g((unsigned) ((C + rpb - 1) / rpb), (unsigned) S);
    const size_t lds = (size_t) (F / 64) * 9 * sizeof(uint4) + 512 * sizeof(float);
    if (lds > 60000) return false;
    if (G == 12 && NB == 3) G906E_DN(12, 3); else if (G == 9 && NB == 2) G906E_DN(9, 2);
    else if (G == 16 && NB == 2) G906E_DN(16, 2); else if (G == 16 && NB == 1) G906E_DN(16, 1);
    else return false;
    check("g906e_down");
    return true;
}
// codes are dwords 9..16, block A's are 2 bytes off (v_alignbit), the scales are d0's low and d8's high half.
// Activations: two int8 limbs a 32 in LDS, a pair = 9 uint4 {hiA0, hiA1, loA0, loA1, hiB0, hiB1, loB0, loB1,
// {sA, sB}} (stride 36 dwords: 4 mod 32 banks).
template <bool F32In>
__device__ __forceinline__ void g906k_fill(const float* xf, const uint16_t* xh16, int K, uint4* xs) {
    const int ng = K >> 5;
    for (int gi = threadIdx.x; gi < ng; gi += blockDim.x) {
        float v[32], mx = 0.0f;
        if (F32In) {
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const float4 t = *(const float4*) (xf + gi * 32 + j);
                v[j] = t.x; v[j + 1] = t.y; v[j + 2] = t.z; v[j + 3] = t.w;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 32; j += 2) {
                const uint32_t t = *(const uint32_t*) (xh16 + gi * 32 + j);
                v[j] = h2f((uint16_t) t); v[j + 1] = h2f((uint16_t) (t >> 16));
            }
        }
#pragma unroll
        for (int j = 0; j < 32; ++j) mx = fmaxf(mx, fabsf(v[j]));
        const float inv = mx > 0.0f ? 16256.0f / mx : 0.0f;
        uint32_t hi[8], lo[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t h = 0, l = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int q = __float2int_rn(v[4 * k + t] * inv), hh = (q + 64) >> 7;
                h |= (uint32_t) (uint8_t) hh << (8 * t);
                l |= (uint32_t) (uint8_t) (q - 128 * hh) << (8 * t);
            }
            hi[k] = h, lo[k] = l;
        }
        uint4* e = xs + (gi >> 1) * 9 + (gi & 1) * 4;
        e[0] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        e[1] = make_uint4(hi[4], hi[5], hi[6], hi[7]);
        e[2] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
        e[3] = make_uint4(lo[4], lo[5], lo[6], lo[7]);
        ((float*) (xs + (gi >> 1) * 9 + 8))[gi & 1] = mx * (1.0f / 16256.0f);
    }
}
struct G906Pair {
    uint32_t d[17];
};
__device__ __forceinline__ void g906k_ld(const uint8_t* p, G906Pair& B) {
#if G906_X4
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const g906_u4 a = *(const g906_u4*) (p + 16 * k);
        B.d[4 * k] = a.x, B.d[4 * k + 1] = a.y, B.d[4 * k + 2] = a.z, B.d[4 * k + 3] = a.w;
    }
    B.d[16] = __ldg((const uint32_t*) (p + 64));
#else
    const uint32_t* q = (const uint32_t*) p;
#pragma unroll
    for (int i = 0; i < 17; ++i) B.d[i] = __ldg(q + i);
#endif
}
__device__ __forceinline__ int g906k_dot8(const uint32_t* w, uint4 x0, uint4 x1) {
    int a = __builtin_amdgcn_sdot4((int) w[0], (int) x0.x, 0, false);
    a = __builtin_amdgcn_sdot4((int) w[1], (int) x0.y, a, false);
    a = __builtin_amdgcn_sdot4((int) w[2], (int) x0.z, a, false);
    a = __builtin_amdgcn_sdot4((int) w[3], (int) x0.w, a, false);
    a = __builtin_amdgcn_sdot4((int) w[4], (int) x1.x, a, false);
    a = __builtin_amdgcn_sdot4((int) w[5], (int) x1.y, a, false);
    a = __builtin_amdgcn_sdot4((int) w[6], (int) x1.z, a, false);
    return __builtin_amdgcn_sdot4((int) w[7], (int) x1.w, a, false);
}
__device__ __forceinline__ float g906k_pair(const G906Pair& B, const uint4* xb) {
    uint32_t qa[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qa[i] = __builtin_amdgcn_alignbit(B.d[i + 1], B.d[i], 16);
    const uint4 sc = xb[8];
    const int ia = g906k_dot8(qa, xb[0], xb[1]) * 128 + g906k_dot8(qa, xb[2], xb[3]);
    const int ib = g906k_dot8(B.d + 9, xb[4], xb[5]) * 128 + g906k_dot8(B.d + 9, xb[6], xb[7]);
    const float da = h2f((uint16_t) B.d[0]), db = h2f((uint16_t) (B.d[8] >> 16));
    return (float) ia * (da * __uint_as_float(sc.x)) + (float) ib * (db * __uint_as_float(sc.y));
}
        }
    }
}
#define G906F_GU(GG, NN) g906f_gate_up_kernel<GG, NN><<<g, wg, lds, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld)
static bool g906f_gate_up(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C, int F,
                          const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                          const int* orow, int h_ld) {
    static const int wg = getenv("IK2_G906_FWG") ? atoi(getenv("IK2_G906_FWG")) : 256;
    const size_t rb = mv_row_bytes(kNVFP4, C);
    int G, NB;
    if (!g906e_pick(C / 64, G, NB) || rb % 4 || eb0 % 4 || ((uintptr_t) x & 15) || C / 16 > wg) return false;
    const int vpb = (wg / 64) * (64 / G);              // virtual rows a block
    if (vpb % 2) return false;
    const int rpb = vpb / 2;
    const dim3 g((unsigned) ((F + rpb - 1) / rpb), (unsigned) S);
    const size_t lds = (size_t) (C / 64) * 9 * sizeof(uint4) + (512 + 64) * sizeof(float);
    const int ld = h_ld > 0 ? h_ld : F;
    if (lds > 60000) return false;
    if (G == 12 && NB == 3) G906F_GU(12, 3); else if (G == 9 && NB == 2) G906F_GU(9, 2);
    else if (G == 16 && NB == 2) G906F_GU(16, 2); else if (G == 16 && NB == 1) G906F_GU(16, 1);
    else return false;
    check("g906f_gate_up");
    return true;
}
        a = g906e_rsum(a, red, G, part);
        if (live && part == 0 && r < C) y[(size_t) rw[j] * C + r] = a;
    }
}
static bool g906kg_gate_up(const uint8_t* const* blob, const int* gn, const int* gtok, const int* grow, int NG,
                           size_t eb0, const float* x, int C, int F, const float* sg, const float* su, float gcap,
                           float ucap, uint16_t* h, cudaStream_t s) {
    const size_t rb = (size_t) (C / 32) * 34;
    int G, NB;
    if (C % 64 || !g906k_pick(C / 64, G, NB) || rb % 4 || eb0 % 4 || ((uintptr_t) x & 15)) return false;
    const int rpb = (g906_wg() / 64) * (64 / G);
    const dim3 g((unsigned) ((F + rpb - 1) / rpb), (unsigned) NG);
    const size_t lds = (size_t) (C / 64) * 9 * sizeof(uint4) + 512 * sizeof(float);
    if (lds > 60000) return false;
#define G906KG_GU(GG, NN) g906kg_gate_up_kernel<GG, NN><<<g, g906_wg(), lds, s>>>(blob, gn, gtok, grow, eb0, x, C, F, rb, sg, su, gcap, ucap, h, F)
    if (G == 18 && NB == 2) G906KG_GU(18, 2); else if (G == 9 && NB == 2) G906KG_GU(9, 2);
    else if (G == 16 && NB == 2) G906KG_GU(16, 2); else if (G == 16 && NB == 1) G906KG_GU(16, 1);
    else return false;
#undef G906KG_GU
    check("g906kg_gate_up");
    return true;
}
static bool g906kg_down(const uint8_t* const* blob, const int* gn, const int* grow, int NG, size_t off_down,
                        const uint16_t* h, int C, int F, float* y, cudaStream_t s) {
    const size_t rb = (size_t) (F / 32) * 34;
    int G, NB;
    if (F % 64 || !g906k_pick(F / 64, G, NB) || rb % 4 || off_down % 4 || ((uintptr_t) h & 3)) return false;
    const int rpb = (g906_wg() / 64) * (64 / G);
    const dim3 g((unsigned) ((C + rpb - 1) / rpb), (unsigned) NG);
    const size_t lds = (size_t) (F / 64) * 9 * sizeof(uint4) + 512 * sizeof(float);
    if (lds > 60000) return false;
#define G906KG_DN(GG, NN) g906kg_down_kernel<GG, NN><<<g, g906_wg(), lds, s>>>(blob, gn, grow, off_down, h, C, F, rb, y)
    if (G == 18 && NB == 2) G906KG_DN(18, 2); else if (G == 9 && NB == 2) G906KG_DN(9, 2);
    else if (G == 16 && NB == 2) G906KG_DN(16, 2); else if (G == 16 && NB == 1) G906KG_DN(16, 1);
    else return false;
#undef G906KG_DN
    check("g906kg_down");
    return true;
}
        a = g906e_rsum(a, red, G, part);
        if (live && part == 0 && r < C) y[(size_t) rw[j] * C + r] = a;
    }
}
static bool g906g_on() {
    static const bool on = !getenv("IK2_G906_GROUP") || atoi(getenv("IK2_G906_GROUP"));
    return on;
}
#define G906FG_GU(GG, NN) g906fg_gate_up_kernel<GG, NN><<<g, wg, lds, s>>>(blob, gn, gtok, grow, eb0, x, C, F, rb, sg, su, gcap, ucap, h, F)
bool g906_group_gate_up(const uint8_t* const* blob, const int* gn, const int* gtok, const int* grow, int NG, int ty_gate,
                        int ty_up, size_t eb0, const float* x, int C, int F, const float* sg, const float* su,
                        float gcap, float ucap, uint16_t* h, cudaStream_t s) {
    // (Q8 gate | up grouped is slower than per pair - 2 matrices of registers a lane, members one after another;
    //  IK2_G906_KG_GU=1 to take it)
    static const bool kg_gu = getenv("IK2_G906_KG_GU") && atoi(getenv("IK2_G906_KG_GU"));
    if (ty_gate == kQ8_0 && ty_up == kQ8_0 && !kg_gu) return false;
    if (g906g_on() && ty_gate == kQ8_0 && ty_up == kQ8_0)
        return g906kg_gate_up(blob, gn, gtok, grow, NG, eb0, x, C, F, sg, su, gcap, ucap, h, s);
    if (!g906g_on() || ty_gate != kNVFP4 || ty_up != kNVFP4 || C % 64 || ((uintptr_t) x & 15)) return false;
    static const int wg = getenv("IK2_G906_FWG") ? atoi(getenv("IK2_G906_FWG")) : 256;
    const size_t rb = mv_row_bytes(kNVFP4, C);
    int G, NB;
    if (!g906e_pick(C / 64, G, NB) || rb % 4 || eb0 % 4 || C / 16 > wg) return false;
    const int vpb = (wg / 64) * (64 / G);
    if (vpb % 2) return false;
    const int rpb = vpb / 2;
    const dim3 g((unsigned) ((F + rpb - 1) / rpb), (unsigned) NG);
    const size_t lds = (size_t) (C / 64) * 9 * sizeof(uint4) + (512 + 64) * sizeof(float);
    if (lds > 60000) return false;
    if (G == 12 && NB == 3) G906FG_GU(12, 3); else if (G == 9 && NB == 2) G906FG_GU(9, 2);
    else if (G == 16 && NB == 2) G906FG_GU(16, 2); else if (G == 16 && NB == 1) G906FG_GU(16, 1);
    else return false;
    check("g906fg_gate_up");
    return true;
}
#undef G906FG_GU
#define G906EG_DN(GG, NN) g906eg_down_kernel<GG, NN><<<g, g906_wg(), lds, s>>>(blob, gn, grow, off_down, h, C, F, rb, y)
bool g906_group_down(const uint8_t* const* blob, const int* gn, const int* grow, int NG, int ty_down, size_t off_down,
                     const uint16_t* h, int C, int F, float* y, cudaStream_t s) {
    if (g906g_on() && ty_down == kQ8_0) return g906kg_down(blob, gn, grow, NG, off_down, h, C, F, y, s);
    if (!g906g_on() || ty_down != kNVFP4 || F % 64 || ((uintptr_t) h & 3)) return false;
    const size_t rb = mv_row_bytes(kNVFP4, F);
    int G, NB;
    if (!g906e_pick(F / 64, G, NB) || rb % 4 || off_down % 4 || F / 16 > g906_wg()) return false;
    const int rpb = (g906_wg() / 64) * (64 / G);
    const dim3 g((unsigned) ((C + rpb - 1) / rpb), (unsigned) NG);
    const size_t lds = (size_t) (F / 64) * 9 * sizeof(uint4) + 512 * sizeof(float);
    if (lds > 60000) return false;
    if (G == 12 && NB == 3) G906EG_DN(12, 3); else if (G == 9 && NB == 2) G906EG_DN(9, 2);
    else if (G == 16 && NB == 2) G906EG_DN(16, 2); else if (G == 16 && NB == 1) G906EG_DN(16, 1);
    else return false;
    check("g906eg_down");
    return true;
}
#undef G906EG_DN
static bool g906k_pick(int npair, int& G, int& NP) {
    static const int cand[][2] = {{18, 2}, {9, 2}, {16, 2}, {16, 1}, {8, 1}, {12, 1}, {9, 1}, {12, 2}, {8, 2}};
    for (auto& c : cand)
        if (c[0] * c[1] == npair) { G = c[0]; NP = c[1]; return true; }
    return false;
}
#define G906K_GU(GG, NN) g906k_gate_up_kernel<GG, NN><<<g, g906_wg(), lds, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld)
static bool g906k_gate_up(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C, int F,
                          const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                          const int* orow, int h_ld) {
    const size_t rb = (size_t) (C / 32) * 34;
    int G, NB;
    if (C % 64 || !g906k_pick(C / 64, G, NB) || rb % 4 || eb0 % 4 || ((uintptr_t) x & 15)) return false;
    const int rpb = (g906_wg() / 64) * (64 / G);
    const dim3 g((unsigned) ((F + rpb - 1) / rpb), (unsigned) S);
    const size_t lds = (size_t) (C / 64) * 9 * sizeof(uint4) + 512 * sizeof(float);
    const int ld = h_ld > 0 ? h_ld : F;
    if (lds > 60000) return false;
    if (G == 18 && NB == 2) G906K_GU(18, 2); else if (G == 9 && NB == 2) G906K_GU(9, 2);
    else if (G == 16 && NB == 2) G906K_GU(16, 2); else if (G == 16 && NB == 1) G906K_GU(16, 1);
    else return false;
    check("g906k_gate_up");
    return true;
}
#define G906K_DN(GG, NN) g906k_down_kernel<GG, NN><<<g, g906_wg(), lds, s>>>(blob, off_down, h, C, F, rb, y)
static bool g906k_down(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                       cudaStream_t s) {
    const size_t rb = (size_t) (F / 32) * 34;
    int G, NB;
    if (F % 64 || !g906k_pick(F / 64, G, NB) || rb % 4 || off_down % 4 || ((uintptr_t) h & 3)) return false;
    const int rpb = (g906_wg() / 64) * (64 / G);
    const dim3 g((unsigned) ((C + rpb - 1) / rpb), (unsigned) S);
    const size_t lds = (size_t) (F / 64) * 9 * sizeof(uint4) + 512 * sizeof(float);
    if (lds > 60000) return false;
    if (G == 18 && NB == 2) G906K_DN(18, 2); else if (G == 9 && NB == 2) G906K_DN(9, 2);
    else if (G == 16 && NB == 2) G906K_DN(16, 2); else if (G == 16 && NB == 1) G906K_DN(16, 1);
    else return false;
    check("g906k_down");
    return true;
}

static bool g906_on() {
    static const int on = !(getenv("IK2_G906") && !atoi(getenv("IK2_G906")));
    return on != 0;
}
// true when it ran
static bool g906_gate_up(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                         const float* x, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                         uint16_t* h, cudaStream_t s, const int* orow, int h_ld) {
    if (g906_on() && ty_gate == kQ8_0 && ty_up == kQ8_0)
        return g906k_gate_up(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld);
    if (!g906_on() || ty_gate != kNVFP4 || ty_up != kNVFP4 || C % 64 || ((uintptr_t) x & 7)) return false;
    if (g906_ver() >= 6 && g906f_gate_up(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld)) return true;
    if (g906_ver() >= 5 && g906e_gate_up(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld)) return true;
    if (g906_ver() >= 4 && g906d_gate_up(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld)) return true;
    if (g906_ver() >= 3 && g906c_gate_up(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld)) return true;
    if (g906_ver() >= 2 && g906b_gate_up(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld)) return true;
    const int nsb = (C / 16 + 31) / 32;
    const dim3 g((unsigned) ((F + 8 * G906_RPH - 1) / (8 * G906_RPH)), (unsigned) S);
    const size_t rb = mv_row_bytes(kNVFP4, C);
    const int ld = h_ld > 0 ? h_ld : F;
    switch (nsb) {
    case 1: g906_gate_up_nv_kernel<1><<<g, 256, 0, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld); break;
    case 2: g906_gate_up_nv_kernel<2><<<g, 256, 0, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld); break;
    case 3: g906_gate_up_nv_kernel<3><<<g, 256, 0, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld); break;
    case 4: g906_gate_up_nv_kernel<4><<<g, 256, 0, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld); break;
    case 5: g906_gate_up_nv_kernel<5><<<g, 256, 0, s>>>(blob, tok, eb0, x, C, F, rb, sg, su, gcap, ucap, h, orow, ld); break;
    default: return false;
    }
    check("g906_gate_up_nv");
    return true;
}
static bool g906_down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C, int F,
                      float* y, cudaStream_t s) {
    if (g906_on() && ty_down == kQ8_0) return g906k_down(blob, S, off_down, h, C, F, y, s);
    if (!g906_on() || ty_down != kNVFP4 || F % 64 || ((uintptr_t) h & 3)) return false;
    if (g906_ver() >= 5 && g906e_down(blob, S, off_down, h, C, F, y, s)) return true;
    if (g906_ver() >= 4 && g906d_down(blob, S, off_down, h, C, F, y, s)) return true;
    if (g906_ver() >= 3 && g906c_down(blob, S, off_down, h, C, F, y, s)) return true;
    if (g906_ver() >= 2 && g906b_down(blob, S, off_down, h, C, F, y, s)) return true;
    const int nsb = (F / 16 + 31) / 32;
    const dim3 g((unsigned) ((C + 8 * G906_RPH - 1) / (8 * G906_RPH)), (unsigned) S);
    const size_t rb = mv_row_bytes(kNVFP4, F);
    switch (nsb) {
    case 1: g906_down_nv_kernel<1><<<g, 256, 0, s>>>(blob, off_down, h, C, F, rb, y); break;
    case 2: g906_down_nv_kernel<2><<<g, 256, 0, s>>>(blob, off_down, h, C, F, rb, y); break;
    case 3: g906_down_nv_kernel<3><<<g, 256, 0, s>>>(blob, off_down, h, C, F, rb, y); break;
    case 4: g906_down_nv_kernel<4><<<g, 256, 0, s>>>(blob, off_down, h, C, F, rb, y); break;
    case 5: g906_down_nv_kernel<5><<<g, 256, 0, s>>>(blob, off_down, h, C, F, rb, y); break;
    default: return false;
    }
    check("g906_down_nv");
    return true;
}
// R q's (LDS, broadcast), the relative bias, the mask; (2) the block's max per head, P = exp(l - m) into LDS;
// (3) a lane a pair of dims x a quarter of the keys (a wave a quarter): o += P V, the V rows read coalesced.
constexpr int G906DA_R = 4;
template <class KV>
__device__ __forceinline__ void g906da_row16(const KV* p, float* v) {   // 16 values of a cache row
    if constexpr (sizeof(KV) == 1) {
        const uint4 w = __ldg((const uint4*) p);
        const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 a = e4m3fnx2_to_f2((uint16_t) (u[i] & 0xFFFFu)), b = e4m3fnx2_to_f2((uint16_t) (u[i] >> 16));
            v[4 * i] = a.x, v[4 * i + 1] = a.y, v[4 * i + 2] = b.x, v[4 * i + 3] = b.y;
        }
    } else {
        const uint4 w0 = __ldg((const uint4*) p), w1 = __ldg((const uint4*) p + 1);
        const uint32_t u[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 a = u32_to_f2(u[i]);
            v[2 * i] = a.x, v[2 * i + 1] = a.y;
        }
    }
}
}
// v5: e4m3 -> FP16 by a shift (FP16 = e4m3 bits placed in the FP16 fields = value / 256, subnormals exact; the 256
// goes into the scales).  4 waves a block, a wave a tile of 64 keys: Q K^T a lane a key (v_dot2 against q pairs in
// LDS), the wave's online softmax, P as FP16 in the wave's LDS, P V a lane a pair of dims with v_dot2 over key pairs.
__device__ __forceinline__ void g906_e8h(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t m = w & 0x7F7F7F7Fu, s = w & 0x80808080u;
    lo = (__builtin_amdgcn_perm(0u, m, 0x04010400u) << 7) | __builtin_amdgcn_perm(0u, s, 0x01040004u);
    hi = (__builtin_amdgcn_perm(0u, m, 0x04030402u) << 7) | __builtin_amdgcn_perm(0u, s, 0x03040204u);
}
typedef uint32_t g906_u4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void g906_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#define G906DA_L(RR) g906da_kernel<KV, RR><<<g, 256, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj,                                                            d_rel, extent, window, part, pos_d, page_table, P)
#undef G906DA_L
#if defined(STRATA_USE_HIP)
// ---- gfx906 dense: Q8_0 (int8 [N][K], fp16 scale a 32) by x quantized in the block's LDS to two int8 limbs a 32
// (q = round(x * 16256 / max), 128 hi + lo), v_dot4_i32_i8; a row a half-wavefront, 16 bytes a lane a chunk, every
// chunk loaded before any is used.
__device__ __forceinline__ float g906_wsum64(float v) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}
}
struct G906XQ {
    float4 v[3];
};
__device__ __forceinline__ void g906q_xld(const float* x, int K, G906XQ& X) {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int t = threadIdx.x + k * blockDim.x;
        X.v[k] = t < (K >> 2) ? *(const float4*) (x + 4 * t) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}
__device__ __forceinline__ void g906q_xst(const G906XQ& X, int K, uint32_t* hi, uint32_t* lo, float* xsc) {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int t = threadIdx.x + k * blockDim.x;
        if (k * (int) blockDim.x >= (K >> 2)) break;   // (whole waves: K / 4 a multiple of 32)
        const float4 v = X.v[k];
        float mx = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        mx = fmaxf(mx, __shfl_xor(mx, 1, 64));
        mx = fmaxf(mx, __shfl_xor(mx, 2, 64));
        mx = fmaxf(mx, __shfl_xor(mx, 4, 64));
        if (t >= (K >> 2)) continue;
        const float inv = mx > 0.0f ? 16256.0f / mx : 0.0f;
        const float vv[4] = {v.x, v.y, v.z, v.w};
        uint32_t h = 0, l = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = __float2int_rn(vv[i] * inv), hh = (q + 64) >> 7;
            h |= (uint32_t) (uint8_t) hh << (8 * i);
            l |= (uint32_t) (uint8_t) (q - 128 * hh) << (8 * i);
        }
        hi[t] = h;
        lo[t] = l;
        if ((t & 7) == 0) xsc[t >> 3] = mx * (1.0f / 16256.0f);
    }
}
        acc = g906q2_dot<MAXIT>(wb, db, K, lane, hi, lo, xsc);
        if (lane == 0 && !mv_conv_row(c0, r1, acc) && !mv_conv_row(c1, r1, acc)) y[r1] = acc;
    }
}
static int g906_ncu() {
    static int n = 0;
    if (!n) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        if (n <= 0) n = 60;
    }
    return n;
}
    check("g906q2_mv");
    return true;
// ---- gfx906 dense Q8_0, a wavefront a row, small blocks (the measured best read pattern: ~800 GB/s), FP32 math
// against x straight from the cache (no LDS fill, no quantized input): a lane's 16-byte chunks, all loaded first
__device__ __forceinline__ float g906r_dot16(uint4 w, const float4* xv) {
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float4 x = xv[k];
        a = fmaf((float) (int8_t) u[k], x.x, a);
        a = fmaf((float) (int8_t) (u[k] >> 8), x.y, a);
        a = fmaf((float) (int8_t) (u[k] >> 16), x.z, a);
        a = fmaf((float) ((int) u[k] >> 24), x.w, a);
    }
    return a;
}
struct G906Xq {
#define G906Q_L(MI, L) g906q_kernel<MI, L><<<g, qwg, lds, s>>>(q, d, N, K, x, y, c0, c1, a0, a1)
#undef G906Q_L
#define G906B_WPE 1
struct G906TStep {
struct G906TStep<1, M_> {
        }
    }
};
template <int V>
__device__ __forceinline__ int g906_treduce(float* acc, int lane) {
    int idx = 0;
    G906TStep<V, 32>::run(acc, lane, idx);
    return idx;
}
struct G906Pow2 { static constexpr int v = N_ <= 1 ? 1 : N_ <= 2 ? 2 : N_ <= 4 ? 4 : N_ <= 8 ? 8 : N_ <= 16 ? 16 : 32; };
    }
static bool g906b_on() {
    static const bool on = !getenv("IK2_G906_B") || atoi(getenv("IK2_G906_B"));
    return on;
static bool g906b_mv(const int8_t* q, const uint16_t* d, int N, int K, const uint16_t* x, int T, float* y, int ldy,
                     cudaStream_t s) {
    if (!g906b_on() || T < 1 || T > kG906bMaxT || K % 128 || K > 4096 || ((uintptr_t) q & 15) || ((uintptr_t) x & 7))
        return false;
    const int mi = (K / 16 + 63) / 64;
    const int ntot = 2 * (K / 16) + K / 128;
    const int tg0 = T < kG906bTG ? T : kG906bTG;
    const int ncp = (tg0 * ntot + 511) / 512;
    if (mi > 4 || ncp > 5) return false;
    static const int b1l = getenv("IK2_G906_B1L") ? atoi(getenv("IK2_G906_B1L")) : 3;
    static const int b1l_nmax = getenv("IK2_G906_B1L_NMAX") ? atoi(getenv("IK2_G906_B1L_NMAX")) : 65536;
    if (b1l > 0 && T >= b1l && N <= b1l_nmax) {   // single-limb x (not the head: its logits): half the dot4s (rel. error ~4e-3, as q8_1 activations)
        const int nt4 = K / 16 + K / 128, nc4 = (T * nt4 + 511) / 512;
        if (nc4 <= 5) {
            g906b4_quant_kernel<<<dim3((unsigned) ((K / 4 + 255) / 256), (unsigned) T), 256, 0, s>>>(x, K);
            check("g906b4_quant");
            const unsigned g4 = (unsigned) ((N + 15) / 16);
            const size_t l4 = (size_t) T * nt4 * 16;
#define G906B4_L(MI, NT, NC) g906b4_kernel<MI, NT, NC><<<g4, 512, l4, s>>>(q, d, N, K, y, ldy)
#define G906B4_NC(MI, NT) switch (nc4) { case 1: G906B4_L(MI, NT, 1); break; case 2: G906B4_L(MI, NT, 2); break; \
    case 3: G906B4_L(MI, NT, 3); break; case 4: G906B4_L(MI, NT, 4); break; default: G906B4_L(MI, NT, 5); break; }
#define G906B4_T(MI) switch (T) { case 3: G906B4_NC(MI, 3); break; case 4: G906B4_NC(MI, 4); break; \
    case 5: G906B4_NC(MI, 5); break; case 6: G906B4_NC(MI, 6); break; case 7: G906B4_NC(MI, 7); break; \
    case 8: G906B4_NC(MI, 8); break; case 1: G906B4_NC(MI, 1); break; default: G906B4_NC(MI, 2); break; }
            switch (mi) {
                case 1: G906B4_T(1); break;
                case 2: G906B4_T(2); break;
                case 3: G906B4_T(3); break;
                default: G906B4_T(4); break;
            }
#undef G906B4_T
#undef G906B4_NC
#undef G906B4_L
            check("g906b4_mv");
            return true;
        }
    }
    g906b_quant_kernel<<<dim3((unsigned) ((K / 4 + 255) / 256), (unsigned) T), 256, 0, s>>>(x, K);
    check("g906b_quant");
    const unsigned g = (unsigned) ((N + 15) / 16);
    const size_t lds = (size_t) tg0 * ntot * 16;
#define G906B_L(MI, NT, NC) g906b_kernel<MI, NT, NC><<<g, 512, lds, s>>>(q, d, N, K, y, ldy)
#define G906B_NC(MI, NT) switch (ncp) { case 1: G906B_L(MI, NT, 1); break; case 2: G906B_L(MI, NT, 2); break; \
    case 3: G906B_L(MI, NT, 3); break; case 4: G906B_L(MI, NT, 4); break; default: G906B_L(MI, NT, 5); break; }
#define G906B_T(MI) switch (T) { case 1: G906B_NC(MI, 1); break; case 2: G906B_NC(MI, 2); break; \
    case 3: G906B_NC(MI, 3); break; case 4: G906B_NC(MI, 4); break; case 5: G906B_NC(MI, 5); break; \
    case 6: G906B_NC(MI, 6); break; case 7: G906B_NC(MI, 7); break; default: G906B_NC(MI, 8); break; }
    switch (mi) {
        case 1: G906B_T(1); break;
        case 2: G906B_T(2); break;
        case 3: G906B_T(3); break;
        default: G906B_T(4); break;
    }
#undef G906B_T
#undef G906B_NC
#undef G906B_L
    check("g906b_mv");
    return true;
}
    }
    const int idx = g906_treduce<V>(acc, lane);
    if ((lane & (64 / V - 1)) == 0 && idx < NT) y[(size_t) idx * y_ld + row] = acc[0];
}
static bool g906hr_mv(const uint16_t* W, int N, int K, const float* x, int x_ld, int T, float* y, int y_ld,
                      cudaStream_t s) {
    static const bool on = !getenv("IK2_G906_HR") || atoi(getenv("IK2_G906_HR"));
    if (!on || T < 1 || T > 8 || K % 8 || x_ld % 4 || ((uintptr_t) W & 15) || ((uintptr_t) x & 15)) return false;
    const int mi = (K / 8 + 63) / 64;
    if (mi > 5) return false;
#define G906HR_L(MI, NT) g906hr_kernel<MI, NT><<<(unsigned) N, 64, 0, s>>>(W, N, K, x, x_ld, y, y_ld)
#define G906HR_T(MI) switch (T) { case 1: G906HR_L(MI, 1); break; case 2: G906HR_L(MI, 2); break; \
    case 3: G906HR_L(MI, 3); break; case 4: G906HR_L(MI, 4); break; case 5: G906HR_L(MI, 5); break; \
    case 6: G906HR_L(MI, 6); break; case 7: G906HR_L(MI, 7); break; default: G906HR_L(MI, 8); break; }
    switch (mi) {
        case 1: G906HR_T(1); break;
        case 2: G906HR_T(2); break;
        case 3: G906HR_T(3); break;
        case 4: G906HR_T(4); break;
        default: G906HR_T(5); break;
    }
#undef G906HR_T
#undef G906HR_L
    check("g906hr_mv");
    return true;
}
#endif
template <bool BF>
__device__ __forceinline__ void g906_h8(const uint4 u, float* f) {
    const uint32_t a[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (BF) {
            f[2 * i] = __uint_as_float(a[i] << 16), f[2 * i + 1] = __uint_as_float(a[i] & 0xFFFF0000u);
        } else {
            f[2 * i] = h2f((uint16_t) a[i]), f[2 * i + 1] = h2f((uint16_t) (a[i] >> 16));
        }
    }
}
        const int r = r0 + idx / NT, t = idx % NT;
        if (r < N) Y[(size_t) t * ldy + r] = acc[0];
    }
}
bool g906_gemm16(bool bf, const uint16_t* W, int N, int K, const uint16_t* X, int T, float* Y, int ldy, cudaStream_t s) {
    static const bool on = !getenv("IK2_G906_BF") || atoi(getenv("IK2_G906_BF"));
    if (!on || T < 1 || T > 8 || K % 8 || ((uintptr_t) W & 15) || ((uintptr_t) X & 15)) return false;
    const unsigned g = (unsigned) ((N + 7) / 8);
#define G906BF_L(B_, NT) g906bf_kernel<B_, NT, 4><<<g, 256, 0, s>>>(W, N, K, X, Y, ldy)
#define G906BF_T(B_) switch (T) { case 1: G906BF_L(B_, 1); break; case 2: G906BF_L(B_, 2); break; \
    case 3: G906BF_L(B_, 3); break; case 4: G906BF_L(B_, 4); break; case 5: G906BF_L(B_, 5); break; \
    case 6: G906BF_L(B_, 6); break; case 7: G906BF_L(B_, 7); break; default: G906BF_L(B_, 8); break; }
    if (bf) { G906BF_T(true); } else { G906BF_T(false); }
#undef G906BF_T
#undef G906BF_L
    check("g906bf");
    return true;
}
struct G906PfArgs {
            for (int i = 0; i < RM; ++i) yr[16 * i] = acc[i][j];
        }
    }
}
bool g906pf_on() {
    static const int on = getenv("IK2_G906_PF") ? atoi(getenv("IK2_G906_PF")) : 1;
    return on != 0;
}
bool g906pf_mm(int ty, int n_exp, const int* exps, const uint8_t* const* blob, const int* start, const int* cnt,
               size_t off_a, size_t off_b, int split, const uint16_t* x, int K, int N, float* y, cudaStream_t s) {
    static const int cfg = getenv("IK2_G906_PFT") ? atoi(getenv("IK2_G906_PFT")) : 0;
    const int BM = cfg == 2 ? 256 : 128, BN = cfg == 1 ? 128 : 64;
    if (!g906pf_on() || (ty != kNVFP4 && ty != kQ8_0) || K % 64 || N % BM || ((uintptr_t) x & 15) ||
        ((uintptr_t) y & 3))
        return false;
    for (int b0 = 0; b0 < n_exp; b0 += kG906PfMaxE) {
        G906PfArgs A;
        A.ne = 0;
        int nt = 0;
        for (int m = b0; m < n_exp && m < b0 + kG906PfMaxE; ++m) {
            const int e = exps[m];
            if (cnt[e] <= 0) continue;
            if (((uintptr_t) blob[e] & 3) || ((off_a | off_b) & 3)) return false;
            A.blob[A.ne] = blob[e];
            A.start[A.ne] = start[e];
            A.cnt[A.ne] = cnt[e];
            A.tile0[A.ne] = nt;
            nt += (cnt[e] + BN - 1) / BN;
            ++A.ne;
        }
        if (!A.ne) continue;
        A.tile0[A.ne] = nt;
        const dim3 g((unsigned) (N / BM), (unsigned) nt);
#define G906PF_L(T, R, C) g906pf_kernel<T, R, C><<<g, 256, 0, s>>>(A, off_a, off_b, split, x, K, N, y)
        if (ty == kNVFP4) {
            if (cfg == 1) G906PF_L(0, 8, 8); else if (cfg == 2) G906PF_L(0, 16, 4); else G906PF_L(0, 8, 4);
        } else {
            if (cfg == 1) G906PF_L(1, 8, 8); else if (cfg == 2) G906PF_L(1, 16, 4); else G906PF_L(1, 8, 4);
        }
#undef G906PF_L
        check("g906pf_mm");
    }
    return true;
}
bool ik2_g906_experts(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, int ty_down,
                      size_t eb0, size_t off_down, const float* x, const uint16_t* h, int C, int F, const float* sg,
                      const float* su, float gcap, float ucap, uint16_t* hout, float* y, cudaStream_t s) {
    if (!ik2_g906_ok()) return false;
    if (!g906_gate_up(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, hout, s, nullptr, 0)) return false;
    if (!g906_down(blob, S, ty_down, off_down, hout ? hout : h, C, F, y, s)) return false;
    return true;
}
#else   // not gfx906: decline every entry point
bool g906pf_on() { return false; }
bool g906pf_mm(int, int, const int*, const uint8_t* const*, const int*, const int*, size_t, size_t, int,
               const uint16_t*, int, int, float*, cudaStream_t) { return false; }
bool g906_group_gate_up(const uint8_t* const*, const int*, const int*, const int*, int, int, int, size_t,
                        const float*, int, int, const float*, const float*, float, float, uint16_t*, cudaStream_t) {
    return false;
}
bool g906_group_down(const uint8_t* const*, const int*, const int*, int, int, size_t, const uint16_t*, int, int,
                     float*, cudaStream_t) { return false; }
bool g906_gemm16(bool, const uint16_t*, int, int, const uint16_t*, int, float*, int, cudaStream_t) { return false; }
bool ik2_g906_ok() { return false; }
bool ik2_g906_experts(const uint8_t* const*, const int*, int, int, int, int, size_t, size_t, const float*,
                      const uint16_t*, int, int, const float*, const float*, float, float, uint16_t*, float*,
                      cudaStream_t) { return false; }
#endif  // IK2_G906_TARGET
#endif   // STRATA_USE_HIP
