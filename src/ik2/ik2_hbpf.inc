// ik2_hbpf.inc - prefill kernels ported from headbouyJB/inkling2-gufo (MIT, Copyright (c) 2026 headbouyJB; the
// notice: docs/AMD_INKLING2_GFX1151.md) at commit 8bb6cd5, src/hip/fast.hip.
// Behind env IK2_HBPF (g_hbpf in ik2_main.cu), a bit a class; IK2_HBPF unset / 0 = none of this runs.
//   bit 0 experts : hb::k_hist / k_scan_items / k_scatter (fast.hip:1527-1570, GPU-side expert grouping), k_gemm4
//                   EPI 1 (gate_up + fused activation) and EPI 2 (down, a row per (token, slot)), k_sum_pairs
//                   (fast.hip:1375-1525)
//   bit 1 dense   : k_gemm4 EPI 0 over our PLANAR Q8_0 matrices (GemmDense, fast.hip:2037-2060)
//   bit 2 attn    : attn_prefill (fast.hip:1636-1790) over OUR cache / prep (prep_qkv_r_f16 stays)
// Adapters : expert weights through our resident pointer table (tab_blob[e]
// + offset) instead of W + e * wexp; our activation (swiglu_cap = their situ with beta = gate_cap, lb = up_cap) with
// our separate gate / up / down expert scales; a third weight format FMT_Q8P (int8 [N][K] + f16 d [N][K / 32], our
// IK2_DENSE q8 layout); attention: qk_scale on the F32 WMMA sums (our q is unscaled F16, as the old path's GEMM
// input), the relative bias an F32 dot with our F32 table (extent and window runtime, not WIN), gate left to our
// gate_mul_f16.  Dequantized weights are the same F16 values as decode_f16 / decode_q8p_f16 (a code times its scale
// is exact in F16); the sums' order is not hipBLAS's, so the results are close, not equal, to the old path.  Run to
// run they are equal: no float atomics (the expert bucket order from k_scatter's atomic counter does not change any
// output element's sum).

namespace hb {

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v16h_a16 __attribute__((ext_vector_type(16), aligned(16)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));
typedef uint32_t v4u_a2 __attribute__((ext_vector_type(4), aligned(2)));
typedef uint32_t v4u_a4 __attribute__((ext_vector_type(4), aligned(4)));
typedef uint32_t v4u_a16 __attribute__((ext_vector_type(4), aligned(16)));
typedef uint32_t v8u __attribute__((ext_vector_type(8)));

// The WMMA and v_permlanex16 instructions here are RDNA 3 / 3.5's (gfx11).  Other targets compile these kernels as
// traps, and hb_arch_ok() keeps IK2_HBPF off there (the host side's *_fits answer false).
#if defined(__HIP_DEVICE_COMPILE__) && !(defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
                                         defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__))
#define IK2_HB_GFX11 0
#else
#define IK2_HB_GFX11 1
#endif
__device__ __forceinline__ v8f wmma_f16(v16h a, v16h b, v8f c) {
#if IK2_HB_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

constexpr int FMT_Q8 = 0, FMT_NVFP4 = 1, FMT_Q8P = 2;
constexpr int kF = 1152;   // the experts' F (gate rows, then up rows at +kF)

struct GP {
    const uint8_t* const* Wtab;   // experts: Wtab[e] + woff = the expert's first row
    size_t woff;
    const int8_t* q8;             // FMT_Q8P (dense): int8 [N][K] and F16 scales [N][K / 32]
    const uint16_t* q8d;
    const float *sg, *su, *sd;    // the experts' gate / up / down scales [E] (null: 1)
    float gcap, ucap;
    const _Float16* ah;           // the activations, F16 rows of K
    const int* actrow;            // row of ah for a sorted position (null: the position)
    const int *nitems, *item_e, *item_start, *item_n;   // null item_e: dense, T tokens in BN-token tiles
    int T;
    float* out;                   // EPI 0 [pos][ldo], EPI 2 [pair][ldo]
    _Float16* outh;               // EPI 1 [pos][kF]
    int ldo;
    const int* pair;              // EPI 2: token * 8 + slot of each sorted position
    const float* posw;            // EPI 2: the route weight of each sorted position
};

// e2m1 nibbles -> F16 bits (fast.hip:1344), codes {0,1,2,3,4,6,8,12} as ours (kv x ggml's half-scale ue4m3)
__device__ __forceinline__ void nib4_to_half(uint32_t n, uint32_t& a, uint32_t& b) {
    uint32_t m = n & 0x07070707u, t = (n << 4) & 0x80808080u;
    uint32_t sA = __builtin_amdgcn_perm(m, 0x0C0C0C0Cu, 0x05000400u),
             sB = __builtin_amdgcn_perm(m, 0x0C0C0C0Cu, 0x07000600u);
    a = __builtin_amdgcn_perm(0x4A484644u, 0x42403C00u, sA) | __builtin_amdgcn_perm(t, 0u, 0x050C040Cu);
    b = __builtin_amdgcn_perm(0x4A484644u, 0x42403C00u, sB) | __builtin_amdgcn_perm(t, 0u, 0x070C060Cu);
}
// a 16-value NVFP4 sub-block (8 bytes) times its scale (fast.hip:1351)
__device__ __forceinline__ v16h dq_nv16(uint32_t w0, uint32_t w1, float ds) {
    v8u r;
    uint32_t a, b;
    nib4_to_half(w0 & 0x0F0F0F0Fu, a, b); r[0] = a; r[1] = b;
    nib4_to_half(w1 & 0x0F0F0F0Fu, a, b); r[2] = a; r[3] = b;
    nib4_to_half((w0 >> 4) & 0x0F0F0F0Fu, a, b); r[4] = a; r[5] = b;
    nib4_to_half((w1 >> 4) & 0x0F0F0F0Fu, a, b); r[6] = a; r[7] = b;
    return __builtin_bit_cast(v16h, r) * (_Float16) ds;
}
// 16 int8 (4 words) times d: the 0x6400 magic bias (fast.hip:1358)
__device__ __forceinline__ v16h dq_q8_16(v4u w, float d) {
    v8u r;
    uint32_t x[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t u = x[i] ^ 0x80808080u;
        r[2 * i] = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
        r[2 * i + 1] = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
    }
    return (__builtin_bit_cast(v16h, r) - (_Float16) 1152.0f) * (_Float16) d;
}

// A load; G: through the global address space (global_load) - a pointer read from memory (p.Wtab[e]) is generic and
// loads FLAT, which counts in LGKMcnt too, so while one is in flight every wait on an LDS read drains it (RDNA3.5 ISA,
// 'the only sensible S_WAITCNT value to use after Flat instructions is zero').  Not G: the generic load as before.
template <bool G, class T> __device__ __forceinline__ T ldg(const void* q) {
    if constexpr (G) return *(const __attribute__((address_space(1))) T*) q;
    else return *(const T*) q;
}

// k_gemm4 (fast.hip:1375): BM = 32 RG weight rows x BN tokens a tile, 2 RG waves, K steps of 64 through a double LDS
// buffer, f16 WMMA 16x16x16 with F32 sums.  EPI 0: out[pos][row] (dense); EPI 1: gate row j and up row kF + j in one
// wave, the activation in registers, F16 h[pos][j]; EPI 2: down, w x sum into the row of the position's (token, slot).
// PIPE (IK2_HBPF bit 6): the step in the order the source states, held by sched
// barriers - step kb + 1's loads (global, not FLAT) in flight under step kb's fragment reads + WMMA, then the wait, the
// dequant and the stores ('late dequant'); the compiler otherwise hoists the dequant above the WMMA and waits for the
// loads at once.  A wave whose half of the tile holds no token skips its reads + WMMA.  Same WMMAs in the same K order
// for every kept output: bit for bit the PIPE-off result.  PROBE (IK2_HBPF_PROBE, a price on the GPU, wrong values on
// purpose, launched beside the real kernels): 1 no fragment reads in the K loop (step 0's ks 0 fragments kept in
// registers), 2 no dequant (the raw words stored).
template <int FMT, int EPI, int BN, int K, int RG, bool PIPE = false, int PROBE = 0>
__global__ void __launch_bounds__(64 * RG) k_gemm4(GP p, int rgn) {
    constexpr int STR = 72, NT = 64 * RG, BM = 32 * RG, TTW = BN / 32, NK = K / 64, NCH = (BN * 8 + NT - 1) / NT;
    constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
    __shared__ _Float16 Ws[2][BM * STR];
    __shared__ _Float16 As[2][BN * STR];
    const bool dense = p.item_e == nullptr;
    const int nit = dense ? (p.T + BN - 1) / BN : *p.nitems;
    const int total = nit * rgn;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, ln = lane & 15, hi = lane >> 4, rg = wave % RG,
              th = wave / RG;
    for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
        const int item = unit / rgn, rgi = unit % rgn;
        const int e = dense ? 0 : p.item_e[item];
        const int start = dense ? item * BN : p.item_start[item];
        const int n = dense ? min(BN, p.T - start) : p.item_n[item];
        int wr = tid >> 1, wh = tid & 1, srow;
        if (EPI == 1) {
            int h2 = wr >> 5, sub = (wr >> 4) & 1, r = wr & 15;
            srow = (sub ? kF : 0) + rgi * (16 * RG) + h2 * 16 + r;
        } else {
            srow = rgi * BM + wr;
        }
        const uint8_t* wp = nullptr;
        const int8_t* qp = nullptr;
        const uint16_t* dp = nullptr;
        if (FMT == FMT_Q8P) {
            qp = p.q8 + (size_t) srow * K;
            dp = p.q8d + (size_t) srow * (K / 32);
        } else {
            wp = p.Wtab[e] + p.woff + (size_t) srow * RB;
        }
        uint32_t raw[9];
        v4u areg[NCH];
        const _Float16* ap[NCH];
        int arow_l[NCH];
        bool aok[NCH];
#pragma unroll
        for (int i = 0; i < NCH; i++) {
            int idx = tid + NT * i, row = idx >> 3, ch = idx & 7;
            aok[i] = idx < BN * 8;
            arow_l[i] = row * STR + ch * 8;
            int pos = row < n ? start + row : start;
            int ar = p.actrow ? p.actrow[pos] : pos;
            ap[i] = p.ah + (size_t) ar * K + ch * 8;
        }
        auto gload = [&](int kb) {
            if (FMT == FMT_NVFP4) {
                const uint8_t* b = wp + (size_t) kb * 36;
                raw[0] = ldg<PIPE, uint32_t>(b);
                v4u q = ldg<PIPE, v4u_a4>(b + 4 + 16 * wh);
                raw[1] = q.x; raw[2] = q.y; raw[3] = q.z; raw[4] = q.w;
            } else if (FMT == FMT_Q8) {
                const uint8_t* b = wp + (size_t) (2 * kb + wh) * 34;
                raw[0] = ldg<PIPE, uint16_t>(b);
                v4u a = ldg<PIPE, v4u_a2>(b + 2), c = ldg<PIPE, v4u_a2>(b + 18);
                raw[1] = a.x; raw[2] = a.y; raw[3] = a.z; raw[4] = a.w;
                raw[5] = c.x; raw[6] = c.y; raw[7] = c.z; raw[8] = c.w;
            } else {   // FMT_Q8P: one Q8_0 block = 32 int8 (16-B aligned: K % 16 == 0) and its scale
                raw[0] = dp[2 * kb + wh];
                const v4u* q = (const v4u*) (qp + (size_t) kb * 64 + 32 * wh);
                v4u a = q[0], c = q[1];
                raw[1] = a.x; raw[2] = a.y; raw[3] = a.z; raw[4] = a.w;
                raw[5] = c.x; raw[6] = c.y; raw[7] = c.z; raw[8] = c.w;
            }
#pragma unroll
            for (int i = 0; i < NCH; i++)
                if (aok[i]) areg[i] = *(const v4u*) (ap[i] + kb * 64);
        };
        auto sstore = [&](int buf) {
            _Float16* dst = &Ws[buf][wr * STR + 32 * wh];
            if constexpr ((PROBE & 2) != 0) {
                const v4u w = {raw[1], raw[2], raw[3], raw[4]};
#pragma unroll
                for (int i = 0; i < 4; i++) *(v4u_a16*) (dst + 8 * i) = w;
            } else if (FMT == FMT_NVFP4) {
#pragma unroll
                for (int s2 = 0; s2 < 2; s2++) {
                    float ds = ue4m3((uint8_t) ((raw[0] >> (16 * wh + 8 * s2)) & 0xFFu));
                    *(v16h_a16*) (dst + 16 * s2) = dq_nv16(raw[1 + 2 * s2], raw[2 + 2 * s2], ds);
                }
            } else {
                float d = __half2float(__ushort_as_half((unsigned short) raw[0]));
                *(v16h_a16*) dst = dq_q8_16((v4u){raw[1], raw[2], raw[3], raw[4]}, d);
                *(v16h_a16*) (dst + 16) = dq_q8_16((v4u){raw[5], raw[6], raw[7], raw[8]}, d);
            }
#pragma unroll
            for (int i = 0; i < NCH; i++)
                if (aok[i]) *(v4u_a16*) &As[buf][arow_l[i]] = areg[i];
        };
        v8f facc[TTW][2];
#pragma unroll
        for (int t = 0; t < TTW; t++) {
            facc[t][0] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            facc[t][1] = facc[t][0];
        }
        v16h bfc[2], ac[TTW];   // (PROBE 1)
        auto mma = [&](int cur) {   // one K step's fragment reads + WMMA
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                v16h bf[2];
#pragma unroll
                for (int r = 0; r < 2; r++)
                    bf[r] = PROBE & 1 ? bfc[r] : *(const v16h_a16*) &Ws[cur][(rg * 32 + 16 * r + ln) * STR + ks * 16];
#pragma unroll
                for (int t = 0; t < TTW; t++) {
                    v16h a = PROBE & 1 ? ac[t] : *(const v16h_a16*) &As[cur][(th * (BN / 2) + 16 * t + ln) * STR + ks * 16];
#pragma unroll
                    for (int r = 0; r < 2; r++)
                        facc[t][r] = wmma_f16(a, bf[r], facc[t][r]);
                }
            }
        };
        // (PIPE: this wave's token half of the tile holds a token; wave-uniform, the same every step)
        const bool live = !PIPE || th * (BN / 2) < n;
        gload(0);
        sstore(0);
        __syncthreads();
        if constexpr ((PROBE & 1) != 0) {
#pragma unroll
            for (int r = 0; r < 2; r++) bfc[r] = *(const v16h_a16*) &Ws[0][(rg * 32 + 16 * r + ln) * STR];
#pragma unroll
            for (int t = 0; t < TTW; t++) ac[t] = *(const v16h_a16*) &As[0][(th * (BN / 2) + 16 * t + ln) * STR];
        }
        for (int kb = 0; kb < NK; kb++) {
            int cur = kb & 1;
            if (kb + 1 < NK) gload(kb + 1);
            if (PIPE) __builtin_amdgcn_sched_barrier(0);
            if (live) mma(cur);
            if (PIPE) __builtin_amdgcn_sched_barrier(0);
            if (kb + 1 < NK) sstore(cur ^ 1);
            __syncthreads();
        }
        const float sg = EPI == 1 && p.sg ? p.sg[e] : 1.0f, su = EPI == 1 && p.su ? p.su[e] : 1.0f;
        const float sd = EPI == 2 && p.sd ? p.sd[e] : 1.0f;
#pragma unroll
        for (int t = 0; t < TTW; t++)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                int pl = th * (BN / 2) + 16 * t + 2 * i + hi;
                if (pl >= n) continue;
                int pd = start + pl;
                if (EPI == 0) {
                    int row = rgi * BM + rg * 32 + ln;
                    p.out[(size_t) pd * p.ldo + row] = facc[t][0][i];
                    p.out[(size_t) pd * p.ldo + row + 16] = facc[t][1][i];
                } else if (EPI == 1) {   // swiglu_cap_f32_kernel's expression (ik2_kernels.cu), F32 in, F16 out
                    int j = rgi * (16 * RG) + rg * 16 + ln;
                    const float g = facc[t][0][i] * sg, u = facc[t][1][i] * su;
                    p.outh[(size_t) pd * kF + j] = (_Float16) (p.gcap * tanhf(g / p.gcap) * (1.0f / (1.0f + expf(-g))) *
                                                               p.ucap * tanhf(u / p.ucap));
                } else {
                    // one row per (token, slot); k_sum_pairs adds the 8 slots in a fixed order (fast.hip:1488: an F32
                    // atomicAdd made two identical prefills differ in the last bits).  w = route weight x the
                    // expert's down scale, multiplied first, as run_prefill_moe's wt.
                    float w = p.posw[pd] * sd;
                    size_t o = (size_t) p.pair[pd] * p.ldo;
                    int row = rgi * BM + rg * 32 + ln;
                    p.out[o + row] = w * facc[t][0][i];
                    p.out[o + row + 16] = w * facc[t][1][i];
                }
            }
        __syncthreads();
    }
}

// ---- bit 7 (0x80): the expert GEMMs in upstream gufo's routed shape - gufo-org/gufo
// v0.35 RoutedDownF16Kernel / RoutedPairF16Kernel (src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp, MIT,
// Copyright (c) 2026 gufo contributors; notice: docs/AMD_INKLING2_GFX1151.md), with NVFP4 and the 192-token tile after
// mikealanni/gufo's RoutedF16GEMMKernel<kNVFP4> (499cd1a, MIT).  A block: 8 waves, 128 weight rows x BN tokens of one
// item; a wave owns one 16-row WMMA tile over all BN tokens (BN / 16 accumulators).  Weights straight from the GGUF
// rows into registers (two sets: stage s + 1 loads in flight under stage s), never LDS; the activations the only LDS
// stage ([kb][quarter][token][16 B], stride BN + 1).  Decoded once a wave: the gfx1151 WMMA reads row r's A operand
// from lane r (even r) or r + 16 (odd r) only (upstream EXPERIMENTS.md, measured), so a lane decodes one K16 half of
// its row's K32 block (the one the reading lane needs first) and v_permlanex16 hands its partner the other.
// Ours: our items (hb_moe_route at tile BN, no 16-row padding: an item is one expert's contiguous positions), the
// resident blobs (no repack: a stage = one NVFP4 block / two Q8_0 blocks as stored), k_gemm4's dequant helpers and its
// F32 epilogues (EPI 1: gate tile rows 0-7, their up rows 8-15 - acc[i] and acc[i + 4] in one lane - swiglu_cap in F32,
// F16 h; EPI 2: posw x sd x acc into the (token, slot) row), stored from registers through permlanex16 pairs.  Exact:
// the same F16 operands and the same WMMA chain in K order an output as k_gemm4 (A / B roles swapped: the argument is
// in docs/AMD_INKLING2_GFX1151.md; IK2_HBPF_CHECK compares bit 7 to bit 7 off to the bit).
typedef uint32_t v2u_a4 __attribute__((ext_vector_type(2), aligned(4)));
typedef uint32_t u32_a2 __attribute__((aligned(2)));
__device__ __forceinline__ uint32_t swap16(uint32_t v) {   // lane i gets lane i ^ 16's v (v0.35 SwapHalves)
    uint32_t old;
    asm("" : "=v"(old));   // (all lanes active: the old operand is never read)
#if IK2_HB_GFX11
    return (uint32_t) __builtin_amdgcn_permlanex16((int) old, (int) v, 0x76543210, 0xfedcba98, false, false);
#else
    __builtin_trap();
    return old;
#endif
}
__device__ __forceinline__ float swap16f(float v) { return __uint_as_float(swap16(__float_as_uint(v))); }

template <int FMT, int EPI, int BN>
__global__ void __launch_bounds__(256) k_rg(GP p) {
    constexpr int K = EPI == 1 ? 2304 : kF, NS = K / 64, NT = BN / 16, AS = BN + 1, NA = BN / 32;
    constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
    static_assert(NS % 2 == 0 && BN % 32 == 0 && NT <= 16, "k_rg shape");
    __shared__ v4u_a16 s_act[8 * AS];
    const int item = blockIdx.y;
    if (item >= *p.nitems) return;
    const int e = p.item_e[item], start = p.item_start[item], n = p.item_n[item], live = (n + 15) >> 4;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, sl = lane & 15, hh = lane >> 4;
    const int fc = hh ^ (sl & 1);   // this lane's K16 half: 0 in the lane the WMMA reads (row sl), 1 in its partner
    const int srow = EPI == 1 ? (sl < 8 ? 0 : kF) + (int) blockIdx.x * 64 + wave * 8 + (sl & 7)
                              : (int) blockIdx.x * 128 + wave * 16 + sl;
    // (addresses = a block-uniform base + a 32-bit lane offset in bytes: global loads in SADDR form, one VGPR an
    // address instead of two)
    const uint8_t* wbase = p.Wtab[e] + p.woff;
    const uint32_t woffb = (uint32_t) srow * (uint32_t) RB + (FMT == FMT_NVFP4 ? 8u * fc : 2u + 16u * fc);
    // a thread's 16-B activation chunks a stage: token t, K quarter sub (8 values).  A token past n reads the item's
    // last row (its column is never stored; WMMA columns do not mix): straight-line loads, no exec branches
    const uint8_t* abase = (const uint8_t*) p.ah;
    uint32_t aoff[NA];
    int aslot[NA];
#pragma unroll
    for (int i = 0; i < NA; i++) {
        const int c = tid + 256 * i, t = min(c >> 3, n - 1), sub = c & 7;
        aoff[i] = (uint32_t) ((EPI == 1 ? p.actrow[start + t] : start + t) * K + sub * 8) * 2u;
        aslot[i] = sub * AS + (c >> 3);
    }
    uint32_t hd[2][2];   // NVFP4: [set][0] the block's 4 scale bytes; Q8_0: [set][kb] d in the low 16 bits
    v2u_a4 nv[2][2];     // NVFP4: [set][kb] the 8 nibble bytes of sub-block 2 kb + fc
    v4u_a2 cq[2][2];     // Q8_0: [set][kb] code bytes 16 fc .. 16 fc + 15 of block 2 s + kb
    v4u adata[NA];
    auto fetch = [&]<int S>(int s) {
        if constexpr (FMT == FMT_NVFP4) {   // one block: scale word at 0, sub-block 2 kb + fc's nibbles at 4 + 8 (2 kb + fc)
            const uint32_t o = woffb + (uint32_t) s * 36u;   // (woffb holds + 8 fc)
            hd[S][0] = ldg<true, uint32_t>(wbase + (o - 8u * fc));
            nv[S][0] = ldg<true, v2u_a4>(wbase + (o + 4u));
            nv[S][1] = ldg<true, v2u_a4>(wbase + (o + 20u));
        } else {   // two blocks: d at 0, code bytes 16 fc .. 16 fc + 15 at 2 + 16 fc
#pragma unroll
            for (int kb = 0; kb < 2; kb++) {
                const uint32_t o = woffb + (uint32_t) (2 * s + kb) * 34u;   // (woffb holds 2 + 16 fc)
                // (d as 32 bits: a 16-bit load packed both sets' d into one register, and the decode then waited for
                // the next set's load)
                hd[S][kb] = ldg<true, u32_a2>(wbase + (o - 2u - 16u * fc));
                cq[S][kb] = ldg<true, v4u_a2>(wbase + o);
            }
        }
#pragma unroll
        for (int i = 0; i < NA; i++) adata[i] = ldg<true, v4u_a16>(abase + (aoff[i] + (uint32_t) s * 128u));
    };
    v8f acc[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    auto compute = [&]<int S>() {
#pragma unroll
        for (int kb = 0; kb < 2; kb++) {
            v16h own;
            if constexpr (FMT == FMT_NVFP4)
                own = dq_nv16(nv[S][kb].x, nv[S][kb].y, ue4m3((uint8_t) ((hd[S][0] >> (8 * (2 * kb + fc))) & 0xFFu)));
            else
                own = dq_q8_16((v4u) cq[S][kb], __half2float(__ushort_as_half((unsigned short) hd[S][kb])));
            const v8u o = __builtin_bit_cast(v8u, own);
            v8u q;
#pragma unroll
            for (int i = 0; i < 8; i++) q[i] = swap16(o[i]);
            const v16h alo = own, ahi = __builtin_bit_cast(v16h, q);
#pragma unroll
            for (int j = 0; j < NT; j++) {
                asm volatile("" ::: "memory");   // (one token tile's fragments live at a time, as upstream)
                if (j >= live) continue;
                const v4u_a16* f = s_act + kb * 4 * AS + j * 16 + sl;   // token j * 16 + sl, K quarters 0-3
                const v4u b0 = f[0], b1 = f[AS], b2 = f[2 * AS], b3 = f[3 * AS];
                const v16h blo = __builtin_bit_cast(v16h, __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7));
                const v16h bhi = __builtin_bit_cast(v16h, __builtin_shufflevector(b2, b3, 0, 1, 2, 3, 4, 5, 6, 7));
                acc[j] = wmma_f16(alo, blo, acc[j]);
                acc[j] = wmma_f16(ahi, bhi, acc[j]);
            }
        }
    };
    auto stage = [&]<int S>(int s) {
#pragma unroll
        for (int i = 0; i < NA; i++) s_act[aslot[i]] = adata[i];
        __syncthreads();
        if (s + 1 < NS) fetch.template operator()<1 - S>(s + 1);
        compute.template operator()<S>();
        __syncthreads();
    };
    fetch.template operator()<0>(0);
    for (int s = 0; s < NS; s += 2) {
        stage.template operator()<0>(s);
        stage.template operator()<1>(s + 1);
    }
    // lane (token sl, half hh) holds tile rows 2 i + hh in acc[j][i].  The token tiles through a pack expansion, not a
    // loop: at BN 192 the compiler left the epilogue loop rolled, acc[j] took a runtime index and the whole acc array
    // went to scratch (a store after every WMMA pair; first build, 416 B scratch)
    const auto tiles = [&](auto&& f) {
        [&]<int... J>(std::integer_sequence<int, J...>) { (f.template operator()<J>(), ...); }(std::make_integer_sequence<int, NT>{});
    };
    if constexpr (EPI == 1) {
        const float sg = p.sg ? p.sg[e] : 1.0f, su = p.su ? p.su[e] : 1.0f;
        const int j0 = (int) blockIdx.x * 64 + wave * 8 + 4 * hh;   // this lane stores h of j0 .. j0 + 3
        tiles([&]<int j>() {
            if (j >= live) return;
            uint32_t hv[4];   // h of gate row 2 i + hh (its up row in acc[i + 4]), k_gemm4 EPI 1's expression
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float g = acc[j][i] * sg, u = acc[j][i + 4] * su;
                hv[i] = __builtin_bit_cast(uint16_t, (_Float16) (p.gcap * tanhf(g / p.gcap) * (1.0f / (1.0f + expf(-g))) *
                                                                 p.ucap * tanhf(u / p.ucap)));
            }
            // half 0 keeps rows 0..3 (its 0, 2 + the partner's 1, 3), half 1 rows 4..7 (the partner's 4, 6 + its 5, 7)
            const uint32_t r = swap16(hh ? hv[0] | hv[1] << 16 : hv[2] | hv[3] << 16);
            const uint32_t d0 = hh ? (r & 0xFFFFu) | hv[2] << 16 : hv[0] | r << 16;
            const uint32_t d1 = hh ? (r >> 16) | hv[3] << 16 : hv[1] | (r & 0xFFFF0000u);
            const int t = j * 16 + sl;
            if (t < n) *(v2u_a4*) (p.outh + (size_t) (start + t) * kF + j0) = (v2u_a4){d0, d1};
        });
    } else {
        const float sd = p.sd ? p.sd[e] : 1.0f;
        const int r0 = (int) blockIdx.x * 128 + wave * 16 + 8 * hh;   // this lane stores rows r0 .. r0 + 7
        tiles([&]<int j>() {
            if (j >= live) return;
            float rc[4];   // half 0 gets the partner's rows 1, 3, 5, 7; half 1 the partner's rows 8, 10, 12, 14
#pragma unroll
            for (int k = 0; k < 4; k++) rc[k] = swap16f(hh ? acc[j][k] : acc[j][k + 4]);
            const int t = j * 16 + sl;
            if (t < n) {
                const int pd = start + t;
                const float w = p.posw[pd] * sd;   // (run_prefill_moe's wt, then w x sum as k_gemm4 EPI 2)
                float4* o = (float4*) (p.out + (size_t) p.pair[pd] * p.ldo + r0);
                if (hh == 0) {
                    o[0] = make_float4(w * acc[j][0], w * rc[0], w * acc[j][1], w * rc[1]);
                    o[1] = make_float4(w * acc[j][2], w * rc[2], w * acc[j][3], w * rc[3]);
                } else {
                    o[0] = make_float4(w * rc[0], w * acc[j][4], w * rc[1], w * acc[j][5]);
                    o[1] = make_float4(w * rc[2], w * acc[j][6], w * rc[3], w * acc[j][7]);
                }
            }
        });
    }
}

// y[t][r] = sum over the 8 slots of pairs[t * 8 + k][r], in slot order (fast.hip:1500)
__global__ void k_sum_pairs(const float* __restrict__ pairs, float* __restrict__ y, int d) {
    const size_t t = blockIdx.x;
    for (int r = threadIdx.x * 4; r < d; r += blockDim.x * 4) {
        float4 a = *(const float4*) (pairs + (t * 8) * d + r);
#pragma unroll
        for (int k = 1; k < 8; k++) {
            float4 b = *(const float4*) (pairs + (t * 8 + k) * d + r);
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        *(float4*) (y + t * d + r) = a;
    }
}

// GPU-side expert grouping, E = 128 (fast.hip:1527-1570)
__global__ void k_hist(const int* __restrict__ ids, int P, int* cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P) atomicAdd(&cnt[ids[i]], 1);
}
__global__ void k_scan_items(const int* cnt, int tile, int* goff, int* fill, int* nitems, int* ie, int* is, int* in) {
    __shared__ int sc[128], st[128];
    int e = threadIdx.x;
    int c = cnt[e], tiles = (c + tile - 1) / tile;
    sc[e] = c;
    st[e] = tiles;
    __syncthreads();
    for (int o = 1; o < 128; o <<= 1) {
        int a = e >= o ? sc[e - o] : 0, b = e >= o ? st[e - o] : 0;
        __syncthreads();
        sc[e] += a;
        st[e] += b;
        __syncthreads();
    }
    int ec = sc[e] - c, et = st[e] - tiles;
    goff[e] = ec;
    fill[e] = 0;
    for (int t = 0; t < tiles; t++) {
        int idx = et + t;
        ie[idx] = e;
        is[idx] = ec + t * tile;
        in[idx] = min(tile, c - t * tile);
    }
    if (e == 127) {
        *nitems = st[127];
        goff[128] = sc[127];
    }
}
__global__ void k_scatter(const int* __restrict__ ids, const float* __restrict__ w, int P, const int* goff, int* fill,
                          int* postok, int* pair, float* posw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    int e = ids[i];
    int pos = goff[e] + atomicAdd(&fill[e], 1);
    postok[pos] = i / 8;
    pair[pos] = i;
    posw[pos] = w[i];
}

// ---- prefill flash attention (attn_prefill, fast.hip:1636), on our cache: FP16 [cap][8 x 128], slot = pos % cap
constexpr int HD = 128, NH = 24, NKV = 8, RD = 16, CELL = NKV * HD, PF_KT = 64, PF_KS = HD + 8, PF_VS = PF_KT + 8;
struct PfArgs {
    const __half *kc, *vc;
    int cap, window, extent, U, p0;
    float qk_scale;
    const __half* q16;   // our q (prep_qkv_r_f16's, tau folded), to F16, unscaled [U][NH][HD]
    const float* r;      // our r [U][NH][RD] (tau folded)
    const float* sink;   // [U][NH]
    const float* rel;    // F32 [extent][RD]
    float* out;          // [U][NH][HD]
    // PASS only (hb_attn_prefill_t6): the keys [klo, khi) - on a full visit tiles start at key 0, so
    // a boundary at a multiple of PF_KT is a tile boundary of the single call -, the online softmax state st
    // [U][NH][kPfSt] (acc [HD], m, l of half 0, l of half 1) read first when st_in, written in place of out when st_out
    int klo, khi, st_in, st_out;
    float* st;
};
constexpr int kPfSt = HD + 4;
__device__ __forceinline__ v16h cat16(uint4 a, uint4 b) {
    union { uint4 u[2]; v16h v; } x;
    x.u[0] = a;
    x.u[1] = b;
    return x.v;
}
__device__ __forceinline__ uint32_t as_u32(v2h v) { return __builtin_bit_cast(uint32_t, v); }
__device__ __forceinline__ v2h as_v2h(uint32_t u) { return __builtin_bit_cast(v2h, u); }

// A block: NTT x 16 queries x the 3 q-heads of one kv-head (blockIdx.y), 12 waves (NTT 4); 64-key K and transposed
// V tiles in LDS; S^T = K Q^T on WMMA, online softmax in registers (initial state = the sink: m = sink, l = 1), P
// repacked into a WMMA operand by one shfl_xor 16, PV a second WMMA.  PASS: a pass of a full visit
// (PfArgs klo .. st); the same tiles, in the same order, as the single call's that fall in [klo, khi), the state
// between passes in F32: the single call's bits.  PASS false: the code as before (bit 2's FP16 call).
template <int NTT, bool PASS = false>
__global__ __launch_bounds__(NTT * 96) void attn_prefill(PfArgs a) {
    constexpr int NT = NTT * 96;
    __shared__ __attribute__((aligned(16))) __half sK[PF_KT][PF_KS];
    __shared__ __attribute__((aligned(16))) __half sVt[HD][PF_VS];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int kvh = blockIdx.y;
    const int hh = w % 3, tt = w / 3, h = kvh * 3 + hh;
    const int q16 = lane & 15, half = lane >> 4;
    const int tbase = blockIdx.x * 16 * NTT;
    const int tb = tbase + tt * 16;
    const int t = tb + q16;
    const bool rowvalid = t < a.U;
    const int tcl = min(t, a.U - 1);
    const int posq = rowvalid ? a.p0 + t : -1;

    v16h bq[8];
    {
        const uint4* qp = (const uint4*) (a.q16 + ((size_t) tcl * NH + h) * HD);
#pragma unroll
        for (int s = 0; s < 8; s++) bq[s] = cat16(qp[2 * s], qp[2 * s + 1]);
    }
    float rf[RD];   // (the relative bias in F32, as the old path's GEMM; theirs: F16 fdot2)
    {
        const float4* rq = (const float4*) (a.r + ((size_t) tcl * NH + h) * RD);
#pragma unroll
        for (int c = 0; c < RD / 4; c++) {
            const float4 f = rq[c];
            rf[4 * c] = f.x, rf[4 * c + 1] = f.y, rf[4 * c + 2] = f.z, rf[4 * c + 3] = f.w;
        }
    }
    float m = rowvalid ? a.sink[(size_t) t * NH + h] : 0.f;
    float l = (half == 0) ? 1.f : 0.f;
    v8f acc[8];
#pragma unroll
    for (int d = 0; d < 8; d++) acc[d] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (PASS) {
        if (a.st_in && rowvalid) {   // (where the previous pass stopped; m is the same in both halves)
            const float* sp = a.st + ((size_t) t * NH + h) * kPfSt;
#pragma unroll
            for (int dt = 0; dt < 8; dt++)
#pragma unroll
                for (int i = 0; i < 8; i++) acc[dt][i] = sp[dt * 16 + 2 * i + half];
            m = sp[HD];
            l = sp[HD + 1 + half];
        }
    }

    const int pfirst = a.p0 + tbase;
    const int plast = min(a.p0 + a.U - 1, a.p0 + tbase + 16 * NTT - 1);
    int kfirst = max(0, pfirst - (a.window - 1)), kend = plast;   // (kend: the last key a tile loads)
    if constexpr (PASS) kfirst = max(kfirst, a.klo), kend = min(kend, a.khi - 1);
    const int ntiles = (kend - kfirst + PF_KT) / PF_KT;
    const int wmin = a.p0 + tb, wmax = a.p0 + tb + 15;

    for (int tile = 0; tile < ntiles; tile++) {
        const int kbase = kfirst + tile * PF_KT;
        __syncthreads();
        for (int idx = tid; idx < PF_KT * 16; idx += NT) {
            const int key = idx >> 4, c = idx & 15;
            const int pos = min(kbase + key, kend);
            *(uint4*) &sK[key][c * 8] = *(const uint4*) (a.kc + (size_t) (pos % a.cap) * CELL + kvh * HD + c * 8);
        }
        for (int idx = tid; idx < (PF_KT / 2) * 16; idx += NT) {
            const int kp = idx >> 4, c = idx & 15;
            const int p0_ = min(kbase + 2 * kp, kend), p1_ = min(kbase + 2 * kp + 1, kend);
            uint4 v0 = *(const uint4*) (a.vc + (size_t) (p0_ % a.cap) * CELL + kvh * HD + c * 8);
            uint4 v1 = *(const uint4*) (a.vc + (size_t) (p1_ % a.cap) * CELL + kvh * HD + c * 8);
            uint32_t* dst = (uint32_t*) &sVt[c * 8][2 * kp];
            constexpr int RS = PF_VS / 2;
            uint32_t x0[4] = {v0.x, v0.y, v0.z, v0.w}, x1[4] = {v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                dst[(2 * e) * RS] = (x0[e] & 0xffffu) | (x1[e] << 16);
                dst[(2 * e + 1) * RS] = (x0[e] >> 16) | (x1[e] & 0xffff0000u);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < PF_KT / 16; kt++) {
            const int pos0 = kbase + kt * 16;
            if (pos0 > wmax) continue;
            if ((wmin - (pos0 + 15)) >= a.window) continue;
            v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; s++)
                c = wmma_f16(
                    cat16(*(const uint4*) &sK[kt * 16 + q16][s * 16], *(const uint4*) &sK[kt * 16 + q16][s * 16 + 8]),
                    bq[s], c);
            float sv[8], mx = -INFINITY;
            const bool needb = rowvalid && (posq - (pos0 + 15) < a.extent);
            const bool anyb = __any(needb);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int pos = pos0 + 2 * i + half, d = posq - pos;
                const bool ok = (d >= 0) && (pos <= plast) && d < a.window;
                float s = c[i] * a.qk_scale;
                if (anyb && ok && d < a.extent) {
                    // ponytail: the 64-B table row from L1/L2 each element; an LDS copy of rel if the bias shows in a profile
                    const float4* br = (const float4*) (a.rel + (size_t) d * RD);
                    float b = 0.0f;
#pragma unroll
                    for (int j = 0; j < RD / 4; j++) {
                        const float4 v = br[j];
                        b += v.x * rf[4 * j] + v.y * rf[4 * j + 1] + v.z * rf[4 * j + 2] + v.w * rf[4 * j + 3];
                    }
                    s += b;
                }
                sv[i] = ok ? s : -INFINITY;
                mx = fmaxf(mx, sv[i]);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            const float mn = fmaxf(m, mx);
            const float alpha = __expf(m - mn);
            float p[8], ps = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                p[i] = __expf(sv[i] - mn);
                ps += p[i];
            }
            l = l * alpha + ps;
            m = mn;
            uint32_t mine[4], theirs[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                v2h pr;
                pr[0] = (_Float16) p[2 * i];
                pr[1] = (_Float16) p[2 * i + 1];
                mine[i] = as_u32(pr);
                theirs[i] = __shfl_xor(mine[i], 16, 32);
            }
            v16h pf;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                v2h mv = as_v2h(mine[i]), tv = as_v2h(theirs[i]);
                pf[4 * i + 0] = half ? tv[0] : mv[0];
                pf[4 * i + 1] = half ? mv[0] : tv[0];
                pf[4 * i + 2] = half ? tv[1] : mv[1];
                pf[4 * i + 3] = half ? mv[1] : tv[1];
            }
#pragma unroll
            for (int dt = 0; dt < 8; dt++) {
                acc[dt] *= alpha;
                const __half* vp = &sVt[dt * 16 + q16][kt * 16];
                acc[dt] = wmma_f16(cat16(*(const uint4*) vp, *(const uint4*) (vp + 8)),
                                                                     pf, acc[dt]);
            }
        }
    }
    if (!rowvalid) return;
    if constexpr (PASS) {
        if (a.st_out) {   // (the next pass goes on from here)
            float* sp = a.st + ((size_t) t * NH + h) * kPfSt;
#pragma unroll
            for (int dt = 0; dt < 8; dt++)
#pragma unroll
                for (int i = 0; i < 8; i++) sp[dt * 16 + 2 * i + half] = acc[dt][i];
            if (half == 0) sp[HD] = m;
            sp[HD + 1 + half] = l;
            return;
        }
    }
    const float inv = 1.0f / (l + __shfl_xor(l, 16, 32));
    float* op = a.out + ((size_t) t * NH + h) * HD;
#pragma unroll
    for (int dt = 0; dt < 8; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) op[dt * 16 + 2 * i + half] = acc[dt][i] * inv;
}

// ---- bit 4 (ours): a looped visit's loop controller over a chunk in ONE kernel a gate - in place
// of loop_gates_chunk's 4 hipBLAS F32 SGEMMs (a1 = x D_x^T, a2 = m D_m^T, f = g F, gn = g G) + loop_gate_mix +
// loop_gate_blend and the rms_f16 after it.  f and gn ([T][C] F32 each) are never stored; x and m are read once
// from DRAM, y16 written once.  Prior art: headbouy CtlPrefill = k_gate_p + k_ctl_apply_p (fast.hip:880-990, read
// only, not copied), Loom's g_ink_ctrl_apply_norm_f16_pre (apply + the next norm).  The expressions are the old
// kernels' (g = rx a1 + rm a2; (1 - tanh f) x + tanh(gn) m; y rs w), the sums' order is not hipBLAS's: close, not
// equal.  A block: kCtlT tokens of one gate; (1) rx, rm a wave a token; (2) g [kCtlT][64] through LDS tiles of D;
// (3) a thread 4 channels (576 threads), its F and G columns in registers 16 ranks at a time, y of its 4 x kCtlT values kept;
// (4) the row's RMS and y16 = y rs w (y32 too when asked).
constexpr int kCtlT = 8, kCtlR = 64, kCtlC = 2304, kCtlKC = 64, kCtlNT = 576, kCtlCh = kCtlC / kCtlNT;
struct CtlArgs {
    CtlJob j[3];
    int n, T;
    float eps_g, eps_n;
};
__global__ void __launch_bounds__(kCtlNT) k_ctl(CtlArgs a) {
    constexpr int C = kCtlC, R = kCtlR, NW = kCtlNT / 32;
    __shared__ float sD[R][kCtlKC + 1];
    __shared__ __attribute__((aligned(16))) float sX[kCtlT][kCtlKC];
    __shared__ __attribute__((aligned(16))) float sG[kCtlT][R];
    __shared__ float sR[2][kCtlT], sQ[kCtlT][NW], sRs[kCtlT];
    const int j = blockIdx.x % a.n, t0 = (int) (blockIdx.x / a.n) * kCtlT;
    const CtlJob p = j == 0 ? a.j[0] : j == 1 ? a.j[1] : a.j[2];   // (uniform selects: no dynamic kernarg index)
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (wave < kCtlT) {   // (1) rx, rm (loop_gate_mix_kernel's rsqrt(sum / C + eps)); rows past T: the last row
        const int t = min(t0 + wave, a.T - 1);
        const float4* xr = (const float4*) (p.x + (size_t) t * C);
        const float4* mr = (const float4*) (p.m + (size_t) t * C);
        float sx = 0.0f, sm = 0.0f;
        for (int c = lane; c < C / 4; c += 32) {
            const float4 u = xr[c], v = mr[c];
            sx += u.x * u.x + u.y * u.y + u.z * u.z + u.w * u.w;
            sm += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        sx = wsum(sx), sm = wsum(sm);
        if (lane == 0) sR[0][wave] = rsqrtf(sx / (float) C + a.eps_g), sR[1][wave] = rsqrtf(sm / (float) C + a.eps_g);
    }
    {   // (2) g = rx a1 + rm a2: threads below 512 a (rank r, token tt); D's x half against x, its m half against m
        const int r = tid & 63, tt = min(tid >> 6, kCtlT - 1);
        float acc[2] = {0.0f, 0.0f};
        for (int seg = 0; seg < 2; ++seg) {
            const float* src = seg ? p.m : p.x;
            for (int k0 = 0; k0 < C; k0 += kCtlKC) {
                __syncthreads();
                for (int i = tid; i < R * kCtlKC / 4; i += kCtlNT) {
                    const int rr = i / (kCtlKC / 4), c4 = i % (kCtlKC / 4);
                    const float4 v = *(const float4*) (p.D + (size_t) rr * 2 * C + seg * C + k0 + c4 * 4);
                    sD[rr][c4 * 4] = v.x, sD[rr][c4 * 4 + 1] = v.y, sD[rr][c4 * 4 + 2] = v.z, sD[rr][c4 * 4 + 3] = v.w;
                }
                if (tid < kCtlT * kCtlKC / 4) {
                    const int t1 = tid / (kCtlKC / 4), c4 = tid % (kCtlKC / 4), t = min(t0 + t1, a.T - 1);
                    *(float4*) &sX[t1][c4 * 4] = *(const float4*) (src + (size_t) t * C + k0 + c4 * 4);
                }
                __syncthreads();
                float s = 0.0f;
#pragma unroll 16
                for (int k = 0; k < kCtlKC; ++k) s += sD[r][k] * sX[tt][k];
                acc[seg] += s;
            }
        }
        if (tid < R * kCtlT) sG[tt][r] = sR[0][tt] * acc[0] + sR[1][tt] * acc[1];
    }
    __syncthreads();
    // (3) y = (1 - tanh(F g)) x + tanh(G g) m: a thread channels c = ch * kCtlNT + tid, F and G columns in registers
    // 16 ranks at a time (the sums in rank order)
    float y[kCtlCh][kCtlT], sq[kCtlT];
#pragma unroll
    for (int t = 0; t < kCtlT; ++t) sq[t] = 0.0f;
#pragma unroll
    for (int ch = 0; ch < kCtlCh; ++ch) {
        const int c = ch * kCtlNT + tid;
        float f[kCtlT], gn[kCtlT];
#pragma unroll
        for (int t = 0; t < kCtlT; ++t) f[t] = 0.0f, gn[t] = 0.0f;
#pragma unroll 1
        for (int h = 0; h < 4; ++h) {   // (16 ranks a step, a runtime loop: the weights' registers stay few)
            float fw[16], gw[16];
#pragma unroll
            for (int r = 0; r < 16; ++r) fw[r] = p.F[(size_t) (16 * h + r) * C + c], gw[r] = p.G[(size_t) (16 * h + r) * C + c];
#pragma unroll
            for (int t = 0; t < kCtlT; ++t)
#pragma unroll
                for (int r4 = 0; r4 < 4; ++r4) {
                    const float4 g = *(const float4*) &sG[t][16 * h + 4 * r4];
                    f[t] += fw[4 * r4] * g.x + fw[4 * r4 + 1] * g.y + fw[4 * r4 + 2] * g.z + fw[4 * r4 + 3] * g.w;
                    gn[t] += gw[4 * r4] * g.x + gw[4 * r4 + 1] * g.y + gw[4 * r4 + 2] * g.z + gw[4 * r4 + 3] * g.w;
                }
        }
#pragma unroll
        for (int t = 0; t < kCtlT; ++t) {
            const size_t i = (size_t) min(t0 + t, a.T - 1) * C + c;
            const float v = (1.0f - tanhf(f[t])) * p.x[i] + tanhf(gn[t]) * p.m[i];
            y[ch][t] = v;
            sq[t] += v * v;
        }
    }
    // (4) the row's RMS (rms_impl_kernel's rsqrt(sum / C + eps)), then y rs w
#pragma unroll
    for (int t = 0; t < kCtlT; ++t) {
        const float s = wsum(sq[t]);
        if (lane == 0) sQ[t][wave] = s;
    }
    __syncthreads();
    if (tid < kCtlT) {
        float s = 0.0f;
        for (int w = 0; w < NW; ++w) s += sQ[tid][w];
        sRs[tid] = rsqrtf(s / (float) C + a.eps_n);
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < kCtlT; ++t) {
        if (t0 + t >= a.T) break;
        const float rs = sRs[t];
#pragma unroll
        for (int ch = 0; ch < kCtlCh; ++ch) {
            const int c = ch * kCtlNT + tid;
            const size_t i = (size_t) (t0 + t) * C + c;
            const float o = y[ch][t] * rs * p.w[c];
            p.y16[i] = f2h(o);
            if (p.y32) p.y32[i] = o;
        }
    }
}

}  // namespace hb

// ---- host side

bool hb_arch_ok() {   // (the device a gfx11: the kernels above are its instructions)
    static const bool ok = [] {
        int d = 0;
        cudaDeviceProp pr{};
        return cudaGetDevice(&d) == cudaSuccess && cudaGetDeviceProperties(&pr, d) == cudaSuccess &&
               std::strncmp(pr.gcnArchName, "gfx11", 5) == 0;
    }();
    return ok;
}

bool hb_ctl_fits(int C, int R) { return hb_arch_ok() && C == hb::kCtlC && R == hb::kCtlR; }

void hb_ctl_prefill(const CtlJob* jobs, int n, int T, float eps_gate, float eps_norm, cudaStream_t s) {
    hb::CtlArgs a{};
    for (int i = 0; i < n; ++i) a.j[i] = jobs[i];
    a.n = n, a.T = T, a.eps_g = eps_gate, a.eps_n = eps_norm;
    // (a gate's blocks next to each other's: the 3 gates of a sliding visit share x, read while it is in L2)
    hb::k_ctl<<<(unsigned) (((T + hb::kCtlT - 1) / hb::kCtlT) * n), hb::kCtlNT, 0, s>>>(a);
    check("hb_ctl_prefill");
}

bool hb_dense_fits(int N, int K) { return hb_arch_ok() && N % 64 == 0 && (K == 2304 || K == 3072); }

void hb_dense(const int8_t* q8, const uint16_t* q8d, int N, int K, const uint16_t* x16, int T, float* Y, int ldy,
              cudaStream_t s) {
    hb::GP p{};
    p.q8 = q8;
    p.q8d = q8d;
    p.ah = (const _Float16*) x16;
    p.T = T;
    p.out = Y;
    p.ldo = ldy;
    constexpr int RG = 2, BM = 32 * RG, BN = 64;
    const int rgn = N / BM;
    const dim3 grid((unsigned) (((T + BN - 1) / BN) * rgn));
    if (K == 2304) hb::k_gemm4<hb::FMT_Q8P, 0, BN, 2304, RG><<<grid, 64 * RG, 0, s>>>(p, rgn);
    else hb::k_gemm4<hb::FMT_Q8P, 0, BN, 3072, RG><<<grid, 64 * RG, 0, s>>>(p, rgn);
    check("hb_dense");
}

bool hb_moe_fits(int ty_gate, int ty_up, int ty_down, int C, int F, int E, int K) {
    return hb_arch_ok() && ty_gate == ty_up && ty_up == ty_down && (ty_gate == kNVFP4 || ty_gate == kQ8_0) && C == 2304 && F == hb::kF &&
           E == 128 && K == 8;
}

void hb_moe_route(const int* ids, const float* w, int S, int tile, const HbMoeWs& ws, cudaStream_t s) {
    (void) cudaMemsetAsync(ws.cnt, 0, 128 * 4, s);   // (an error shows in check below)
    hb::k_hist<<<(S + 255) / 256, 256, 0, s>>>(ids, S, ws.cnt);
    hb::k_scan_items<<<1, 128, 0, s>>>(ws.cnt, tile, ws.goff, ws.fill, ws.nitems, ws.item_e, ws.item_start, ws.item_n);
    hb::k_scatter<<<(S + 255) / 256, 256, 0, s>>>(ids, w, S, ws.goff, ws.fill, ws.postok, ws.pair, ws.posw);
    check("hb_moe_route");
}

template <int FMT, int BN, int RG, bool PIPE, int PROBE = 0>
static void hb_moe_launch(hb::GP gu, hb::GP dn, int max_items, cudaStream_t s) {
    constexpr int rgn_gu = hb::kF / (16 * RG), rgn_dn = 2304 / (32 * RG);
    hb::k_gemm4<FMT, 1, BN, 2304, RG, PIPE, PROBE><<<dim3((unsigned) (max_items * rgn_gu)), 64 * RG, 0, s>>>(gu, rgn_gu);
    hb::k_gemm4<FMT, 2, BN, hb::kF, RG, PIPE, PROBE><<<dim3((unsigned) (max_items * rgn_dn)), 64 * RG, 0, s>>>(dn, rgn_dn);
}
template <int FMT, int BN, int RG>
static void hb_moe_launch(hb::GP gu, hb::GP dn, int max_items, bool pipe, cudaStream_t s) {
    if (pipe) hb_moe_launch<FMT, BN, RG, true>(gu, dn, max_items, s);
    else hb_moe_launch<FMT, BN, RG, false>(gu, dn, max_items, s);
}

template <int FMT, int BN>
static void hb_rg_launch(const hb::GP& gu, const hb::GP& dn, int max_items, cudaStream_t s) {   // (bit 7)
    hb::k_rg<FMT, 1, BN><<<dim3(hb::kF / 64, (unsigned) max_items), 256, 0, s>>>(gu);
    hb::k_rg<FMT, 2, BN><<<dim3(2304 / 128, (unsigned) max_items), 256, 0, s>>>(dn);
}

void hb_moe_prefill(int type, int tile, const uint8_t* const* tab, size_t off_down, const float* sg, const float* su,
                    const float* sd, float gcap, float ucap, const uint16_t* x16, uint16_t* h16, float* pairs, float* y,
                    int T, const HbMoeWs& ws, bool pipe, bool rg, int probe, cudaStream_t s) {
    hb::GP gu{};
    gu.Wtab = tab;
    gu.woff = 0;
    gu.sg = sg;
    gu.su = su;
    gu.gcap = gcap;
    gu.ucap = ucap;
    gu.ah = (const _Float16*) x16;
    gu.actrow = ws.postok;
    gu.nitems = ws.nitems;
    gu.item_e = ws.item_e;
    gu.item_start = ws.item_start;
    gu.item_n = ws.item_n;
    gu.outh = (_Float16*) h16;
    hb::GP dn = gu;
    dn.woff = off_down;
    dn.sg = dn.su = nullptr;
    dn.sd = sd;
    dn.ah = (const _Float16*) h16;
    dn.actrow = nullptr;
    dn.outh = nullptr;
    dn.out = pairs;
    dn.ldo = 2304;
    dn.pair = ws.pair;
    dn.posw = ws.posw;
    const int max_items = T * 8 / tile + 128;
    if (rg) {   // (bit 7: tile = BN, 192 or 128 - hb_moe_route made the items with it)
        if (type == kNVFP4) tile == 192 ? hb_rg_launch<hb::FMT_NVFP4, 192>(gu, dn, max_items, s)
                                        : hb_rg_launch<hb::FMT_NVFP4, 128>(gu, dn, max_items, s);
        else tile == 192 ? hb_rg_launch<hb::FMT_Q8, 192>(gu, dn, max_items, s) : hb_rg_launch<hb::FMT_Q8, 128>(gu, dn, max_items, s);
        hb::k_sum_pairs<<<T, 256, 0, s>>>(pairs, y, 2304);
        check("hb_moe_prefill");
        return;
    }
    if (pipe && type == kNVFP4 && tile == 64) {   // (the probes first: the real kernels below overwrite their output)
        if (probe & 1) hb_moe_launch<hb::FMT_NVFP4, 64, 4, true, 1>(gu, dn, max_items, s);
        if (probe & 2) hb_moe_launch<hb::FMT_NVFP4, 64, 4, true, 2>(gu, dn, max_items, s);
    }
    if (type == kNVFP4) {
        if (tile == 32) hb_moe_launch<hb::FMT_NVFP4, 32, 2>(gu, dn, max_items, pipe, s);
        else hb_moe_launch<hb::FMT_NVFP4, 64, 4>(gu, dn, max_items, pipe, s);
    } else {
        if (tile == 32) hb_moe_launch<hb::FMT_Q8, 32, 2>(gu, dn, max_items, pipe, s);
        else hb_moe_launch<hb::FMT_Q8, 64, 4>(gu, dn, max_items, pipe, s);
    }
    hb::k_sum_pairs<<<T, 256, 0, s>>>(pairs, y, 2304);
    check("hb_moe_prefill");
}

bool hb_attn_fits(int H, int KV, int D, int d_rel) { return hb_arch_ok() && H == hb::NH && KV == hb::NKV && D == hb::HD && d_rel == hb::RD; }

void hb_attn_prefill(const uint16_t* q16, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                     int cap, int T, int pos0, int window, int extent, const float* rel, float qk_scale, float* out,
                     cudaStream_t s) {
    hb::PfArgs a{};
    a.kc = (const __half*) kc;
    a.vc = (const __half*) vc;
    a.cap = cap;
    a.window = window;
    a.extent = extent;
    a.U = T;
    a.p0 = pos0;
    a.qk_scale = qk_scale;
    a.q16 = (const __half*) q16;
    a.r = r;
    a.sink = sink;
    a.rel = rel;
    a.out = out;
    hb::attn_prefill<4><<<dim3((unsigned) ((T + 63) / 64), hb::NKV), 384, 0, s>>>(a);
    check("hb_attn_prefill");
}

// IK2_HBPF bit 8 (with bit 2): attn_prefill over a turbo6 cache.  The keys
// [k0, pos0 + T) (k0 the window's first, 0 on a full visit) decoded ONCE by unpack_kv_t6_to_f16 - the old path's own
// FP16 values (the CPU decode rounded: test_turbo6) - into k16 / v16 at row pos % rows, then the FP16 kernel over them:
// bit 2's arithmetic on those values.  A full visit of more than `pass` keys: passes of `pass` keys (row pos - lo), the
// online softmax state in st [T][NH][kPfSt] F32 between them - the single call's tiles, so its bits.
// plan: 1 one call, n > 1 that many passes, 0 neither (a window's keys past the rows, or no room: the old path).
int hb_attn_t6_plan(int T, int pos0, int window, int rows, int pass, size_t st_floats) {
    const bool full = window >= (1 << 30);
    const int k0 = full ? 0 : std::max(0, pos0 - window + 1), span = pos0 + T - k0;
    if (rows <= 0 || T <= 0) return 0;
    if (!full) return span <= rows ? 1 : 0;   // (passes need tiles from key 0: a window's start per block is not)
    if (span <= std::min(pass, rows)) return 1;
    if (pass < hb::PF_KT || pass % hb::PF_KT || pass > rows || (size_t) T * hb::NH * hb::kPfSt > st_floats) return 0;
    return (span + pass - 1) / pass;
}
void hb_attn_prefill_t6(const uint16_t* q16, const float* r, const float* sink, const uint8_t* kc6, const uint8_t* vc6,
                        int kv_cap, int T, int pos0, int window, int extent, const float* rel, float qk_scale,
                        float* out, uint16_t* k16, uint16_t* v16, int rows, int pass, float* st, size_t st_floats,
                        cudaStream_t s) {
    const int n = hb_attn_t6_plan(T, pos0, window, rows, pass, st_floats);
    const int hi = pos0 + T, k0 = window >= (1 << 30) ? 0 : std::max(0, pos0 - window + 1);
    if (!n) {
        std::fprintf(stderr, "ik2::k::hb_attn_prefill_t6: keys %d .. %d, %d rows, pass %d: no plan\n", k0, hi, rows, pass);
        std::abort();
    }
    if (n == 1) {   // (row pos % rows: two pieces at most)
        for (int p = k0; p < hi;) {
            const int row = p % rows, m = std::min(hi - p, rows - row);
            unpack_kv_t6_to_f16(kc6, vc6, kv_cap, p, m, hb::NKV, k16 + (size_t) row * hb::CELL,
                                v16 + (size_t) row * hb::CELL, s);
            p += m;
        }
        hb_attn_prefill(q16, r, sink, k16, v16, rows, T, pos0, window, extent, rel, qk_scale, out, s);
        return;
    }
    hb::PfArgs a{};
    a.kc = (const __half*) k16;
    a.vc = (const __half*) v16;
    a.cap = pass;   // (lo is a multiple of pass: row pos % pass = pos - lo)
    a.window = window;
    a.extent = extent;
    a.U = T;
    a.p0 = pos0;
    a.qk_scale = qk_scale;
    a.q16 = (const __half*) q16;
    a.r = r;
    a.sink = sink;
    a.rel = rel;
    a.out = out;
    a.st = st;
    for (int lo = 0; lo < hi; lo += pass) {
        const int m = std::min(pass, hi - lo);
        unpack_kv_t6_to_f16(kc6, vc6, kv_cap, lo, m, hb::NKV, k16, v16, s);
        a.klo = lo, a.khi = lo + m, a.st_in = lo > 0, a.st_out = lo + m < hi;
        hb::attn_prefill<4, true><<<dim3((unsigned) ((T + 63) / 64), hb::NKV), 384, 0, s>>>(a);
        check("hb_attn_prefill_t6");
    }
}
