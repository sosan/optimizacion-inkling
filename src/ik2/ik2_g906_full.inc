// src/ik2/ik2_g906_full.inc - complete gfx906 kernel set from upstream issue #36
// (G0ldd1/mi50-playground, gfx906-kernels.patch against 27daec5). Every kernel is
// gfx906-only; on other targets the host entry points below are stubs that decline,
// so the engine takes the stock path. CUDA is untouched.
#if defined(STRATA_USE_HIP)

// host gate (shared with ik2_g906.inc)
bool ik2_g906_ok();

#if defined(__gfx906__)

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
static bool g906fa_on() {
    static const int on = getenv("IK2_G906_FA") ? atoi(getenv("IK2_G906_FA")) : 1;
    return on != 0;
}
static bool g906fa_run(const uint16_t* q16, int q_ld, const KVT* kc, const KVT* vc, int kv_cap, int H, int KV, int Tq,
                       int qpos0, float qk_scale, float v_inv, const float* bias, int extent, int window,
                       const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats, int n_split,
                       cudaStream_t s, const int* page_table) {
    if (!g906fa_on() || (q_ld & 7) || H % KV) return false;
    static const int ns_env = getenv("IK2_G906_FA_SPLIT") ? atoi(getenv("IK2_G906_FA_SPLIT")) : 0;
    if (ns_env > 0) n_split = ns_env;
    const size_t per_split = (size_t) Tq * H * (kFaD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFaD);
    const dim3 grid((unsigned) ((Tq + G906FA_BM - 1) / G906FA_BM), (unsigned) H, (unsigned) n_split);
    g906fa_kernel<KVT><<<grid, 256, 0, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent,
                                            window, n_split, o_part, ml_part, page_table);
    check("g906fa");
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFaD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, v_inv);
    check("g906fa_combine");
    return true;
}
struct G906Chunk { uint32_t v[CW]; };
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
typedef unsigned int g906_u4 __attribute__((ext_vector_type(4)));
// ---- gfx906 v5: int8, a lane owns whole 36-byte blocks (G lanes a row, NB blocks a lane, nblk = G * NB), so the
// block's scale word, its 4 sub-block scales and the word -> activation mapping are all compile-time.  Codes ->
// int8 with v_perm (positive table, negative table, sign mask from v_perm's 0x0C/0x0D constant selectors) and
// v_dot4_i32_i8 against two int8 limbs.  LDS a block: 8 uint4 {hiA, hiB, loA, loB} (word w = 1..8) + 1 float4 of
struct G906X16 {
    float v[16];
};
struct G906Blk {
    uint32_t d[9];
};
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
struct G906Pair {
    uint32_t d[17];
};
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
static bool g906k_pick(int npair, int& G, int& NP);
template <int G, int NP>
__global__ void __launch_bounds__(512) g906kg_gate_up_kernel(const uint8_t* const* blob, const int* gn, const int* gtok,
                                                             const int* grow, size_t eb0, const float* x, int C, int F,
                                                             size_t rb, const float* sg, const float* su, float gcap,
                                                             float ucap, uint16_t* h, int h_ld) {
    extern __shared__ uint4 g906_xs[];
    constexpr int RW = 64 / G;
    const int gi = blockIdx.y, ng = gn[gi];
    if (ng <= 0) return;
    float* red = (float*) (g906_xs + (C >> 6) * 9);
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    const int grp = lane / G, part = lane - grp * G;
    const bool live = grp < RW;
    const int r = (blockIdx.x * (int) (blockDim.x >> 6) + wv) * RW + grp;
    const int rr = (live && r < F) ? r : F - 1;
    const uint8_t* bg = blob[gi] + (size_t) rr * rb;
    const int* tk = gtok + gi * kGroupMax;
    const int* rw = grow + gi * kGroupMax;
    G906Pair Bg[NP], Bu[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        g906k_ld(bg + (size_t) (part + G * i) * 68, Bg[i]);
        g906k_ld(bg + eb0 + (size_t) (part + G * i) * 68, Bu[i]);
    }
#pragma unroll 1
    for (int j = 0; j < ng; ++j) {
        if (j > 0) __syncthreads();   // (the previous member's x and red read)
        g906k_fill<true>(x + (size_t) tk[j] * C, nullptr, C, g906_xs);
        __syncthreads();
        float ag = 0.0f, au = 0.0f;
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const uint4* xb = g906_xs + (part + G * i) * 9;
            ag += g906k_pair(Bg[i], xb);
            au += g906k_pair(Bu[i], xb);
        }
        ag = g906e_rsum(ag, red, G, part);
        __syncthreads();
        au = g906e_rsum(au, red, G, part);
        const int orow = rw[j];
        if (live && part == 0 && r < F) {
            const float g = ag * (sg ? sg[orow] : 1.0f), u = au * (su ? su[orow] : 1.0f);
            h[(size_t) orow * h_ld + r] = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
        }
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
template <int G, int NB>
__global__ void __launch_bounds__(512) g906fg_gate_up_kernel(const uint8_t* const* blob, const int* gn, const int* gtok,
                                                             const int* grow, size_t eb0, const float* x, int C, int F,
                                                             size_t rb, const float* sg, const float* su, float gcap,
                                                             float ucap, uint16_t* h, int h_ld) {
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
static bool g906da_on() {
    static const int on = getenv("IK2_G906_DA") ? atoi(getenv("IK2_G906_DA")) : 1;
    return on != 0;
template <class KV>
static bool g906da_launch(const float* q, const float* r, const KV* kc, const KV* vc, int kv_cap, int H, int Hkv,
                          int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent,
                          int window, float* part, int P, const int* pos_d, const int* page_table, cudaStream_t s,
                          const float* sink, float v_inv_scale, float* out) {
    if (!g906da_on() || H % Hkv || H / Hkv > G906DA_R || d_rel > 32 || (((uintptr_t) q) & 15)) return false;
    // (a sliding layer never sees more than window keys: its splits sized for that; the layout keeps stride P)
    const int Pg = window < (1 << 30) ? std::max(1, std::min(P, (window + 63) / 64)) : P;
    const dim3 g((unsigned) Hkv, (unsigned) Pg);
    static const int de_on = getenv("IK2_G906_DE") ? atoi(getenv("IK2_G906_DE")) : 1;
    if constexpr (sizeof(KV) == 1) {
        if (de_on && H / Hkv == 3 && Pg <= 128 && (page_table || kv_cap == 0 || kv_cap >= 64)) {
            g906de_kernel<KV, 3><<<g, 256, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj, d_rel,
                                                    extent, window, part, pos_d, page_table, P);
            check("g906de");
            g906dc_kernel<<<(unsigned) H, 128, 0, s>>>(sink, part, H, P, v_inv_scale, out, k0, nk, window, pos_d, Pg,
                                                        256);
            check("g906dc");
            return true;
        }
    }
#define G906DA_L(RR) g906da_kernel<KV, RR><<<g, 256, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj,                                                            d_rel, extent, window, part, pos_d, page_table, P)
    switch (H / Hkv) {
        case 1: G906DA_L(1); break;
        case 2: G906DA_L(2); break;
        case 3: G906DA_L(3); break;
        default: G906DA_L(4); break;
    }
#undef G906DA_L
    check("g906da");
    g906dc_kernel<<<(unsigned) H, 128, 0, s>>>(sink, part, H, P, v_inv_scale, out, k0, nk, window, pos_d, Pg, 64);
    check("g906dc");
    return true;
#endif
template <class KV>
void attn_decode_impl(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc, int kv_cap, int H,
                      int Hkv, int D, int k0, int nk, int qpos, float qk_scale, float v_inv_scale, const float* proj,
                      int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s,
                      const int* page_table = nullptr) {
    if (D != 128 || d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_decode: D %d, d_rel %d\n", D, d_rel);
        std::abort();
    // Flash-Decode splits over chunks of ~64 keys, up to 128 blocks per head (partials written directly to scratch)
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    if (P == 1) {
        attn_decode_kernel<KV><<<H, 128, 0, s>>>(q, r, sink, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale,
                                                   v_inv_scale, proj, d_rel, extent, window, out, page_table);
        check("attn_decode");
    float* part = scratch;
#if defined(STRATA_USE_HIP)
    if (g906da_launch<KV>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj, d_rel, extent, window, part, P,
                          nullptr, page_table, s, sink, v_inv_scale, out))
        return;
#endif
    attn_split_kernel<KV><<<dim3(H, P), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj, d_rel,
                                                      extent, window, part, nullptr, page_table);
    check("attn_decode_split");
    attn_combine_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, part, H, P, v_inv_scale, out);
    check("attn_decode_combine");
void attn_decode(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int H, int Hkv,
                 int D, int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                 float* scratch, float* out, cudaStream_t s, const int* page_table) {
    attn_decode_impl(q, r, sink, kc, vc, 0, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s, page_table);
void attn_decode_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale, const float* proj,
                     int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s,
                     const int* page_table) {
    attn_decode_impl(q, r, sink, kc, vc, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s, page_table);
void attn_decode_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale,
                     float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                     float* out, cudaStream_t s, const int* page_table) {
    attn_decode_impl(q, r, sink, kc, vc, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, v_inv_scale, proj, d_rel, extent,
                     window, scratch, out, s, page_table);
#if defined(STRATA_USE_HIP)
// ---- gfx906 dense: Q8_0 (int8 [N][K], fp16 scale a 32) by x quantized in the block's LDS to two int8 limbs a 32
// (q = round(x * 16256 / max), 128 hi + lo), v_dot4_i32_i8; a row a half-wavefront, 16 bytes a lane a chunk, every
// chunk loaded before any is used.
__device__ __forceinline__ float g906_wsum64(float v) {
#pragma unroll
struct G906XQ {
    float4 v[3];
};
static bool g906q_on() {
    static const int v = getenv("IK2_G906_DENSE") ? atoi(getenv("IK2_G906_DENSE")) : 1;
    return v != 0;
// ---- gfx906 dense Q8_0, persistent: a block fills its LDS once, then walks rows (blocks side by side on
// neighbouring rows), the next row's chunks loaded while the current row is summed
template <int MAXIT>
__device__ __forceinline__ void g906q2_ld(const int8_t* q, const uint16_t* d, int K, int row, int lane, uint4* w,
                                          uint16_t* ds) {
    const int nch = K >> 4;
    const uint4* qr = (const uint4*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 32 * it, cc = c < nch ? c : nch - 1;
        w[it] = __ldg(qr + cc);
        ds[it] = __ldg(dr + (cc >> 1));
    }
template <int MAXIT>
__device__ __forceinline__ float g906q2_dot(const uint4* w, const uint16_t* ds, int K, int lane, const uint32_t* hi,
                                            const uint32_t* lo, const float* xsc) {
    const int nch = K >> 4;
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 32 * it;
        if (c < nch) {
            const uint4 xh = ((const uint4*) hi)[c], xl = ((const uint4*) lo)[c];
            int ih = __builtin_amdgcn_sdot4((int) w[it].x, (int) xh.x, 0, false);
            ih = __builtin_amdgcn_sdot4((int) w[it].y, (int) xh.y, ih, false);
            ih = __builtin_amdgcn_sdot4((int) w[it].z, (int) xh.z, ih, false);
            ih = __builtin_amdgcn_sdot4((int) w[it].w, (int) xh.w, ih, false);
            int il = __builtin_amdgcn_sdot4((int) w[it].x, (int) xl.x, 0, false);
            il = __builtin_amdgcn_sdot4((int) w[it].y, (int) xl.y, il, false);
            il = __builtin_amdgcn_sdot4((int) w[it].z, (int) xl.z, il, false);
            il = __builtin_amdgcn_sdot4((int) w[it].w, (int) xl.w, il, false);
            acc = fmaf((float) (ih * 128 + il), xsc[c >> 1] * h2f(ds[it]), acc);
        }
    return g906_hsum(acc);
}
template <int MAXIT>
__global__ void __launch_bounds__(256) g906q2_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d,
                                                     int N, int K, const float* __restrict__ x, float* y, MvConv c0,
                                                     MvConv c1) {
    extern __shared__ uint32_t g906q_sm[];
    uint32_t* hi = g906q_sm;
    uint32_t* lo = hi + (K >> 2);
    float* xsc = (float*) (lo + (K >> 2));
    const int lane = threadIdx.x & 31, hw = threadIdx.x >> 5;
    const int step = gridDim.x * 8;
    int row = blockIdx.x * 8 + hw;
    uint4 wa[MAXIT], wb[MAXIT];
    uint16_t da[MAXIT], db[MAXIT];
    g906q2_ld<MAXIT>(q, d, K, row < N ? row : N - 1, lane, wa, da);
    g906q_fill(x, K, hi, lo, xsc);
    __syncthreads();
    for (; row < N; row += 2 * step) {
        const int r1 = row + step;
        g906q2_ld<MAXIT>(q, d, K, r1 < N ? r1 : N - 1, lane, wb, db);
        float acc = g906q2_dot<MAXIT>(wa, da, K, lane, hi, lo, xsc);
        if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
        if (r1 >= N) break;
        const int r2 = r1 + step;
        g906q2_ld<MAXIT>(q, d, K, r2 < N ? r2 : N - 1, lane, wa, da);
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
static bool g906q2_mv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                      const MvConv& c1, cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    static const int bpc = getenv("IK2_G906_QBPC") ? atoi(getenv("IK2_G906_QBPC")) : 0;
    if (bpc <= 0 || a0.n || a1.n || K % 128 || ((uintptr_t) q & 15) || ((uintptr_t) x & 15)) return false;
    const int mi = (K / 16 + 31) / 32;
    const size_t lds = (size_t) K * 2 + (size_t) K / 32 * 4;
    const int nb = std::min((N + 7) / 8, g906_ncu() * bpc);
    switch (mi) {
        case 4: g906q2_kernel<4><<<nb, 256, lds, s>>>(q, d, N, K, x, y, c0, c1); break;
        case 5: g906q2_kernel<5><<<nb, 256, lds, s>>>(q, d, N, K, x, y, c0, c1); break;
        case 6: g906q2_kernel<6><<<nb, 256, lds, s>>>(q, d, N, K, x, y, c0, c1); break;
        default: return false;
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
template <int MAXIT>
__global__ void __launch_bounds__(256) g906r_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d,
                                                    int N, int K, const float* __restrict__ x, float* y, MvConv c0,
                                                    MvConv c1, MvAlt a0, MvAlt a1) {
    const int lane = threadIdx.x & 63;
    const int row = blockIdx.x * (int) (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= N) return;
    const int nch = K >> 4;
    const uint4* qr = (const uint4*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    uint4 w[MAXIT];
    uint16_t ds[MAXIT];
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 64 * it, cc = c < nch ? c : nch - 1;
        w[it] = __ldg(qr + cc);
        ds[it] = __ldg(dr + (cc >> 1));
    }
    const float4* x4 = (const float4*) ((unsigned) (row - a0.r0) < (unsigned) a0.n   ? a0.x
                                        : (unsigned) (row - a1.r0) < (unsigned) a1.n ? a1.x
                                                                                     : x);
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 64 * it;
        if (c < nch) {
            float4 xv[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) xv[k] = __ldg(x4 + 4 * c + k);
            acc = fmaf(g906r_dot16(w[it], xv), h2f(ds[it]), acc);
        }
    }
    acc = g906_wsum64(acc);
    if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
}
static bool g906r_mv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                     const MvConv& c1, cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    static const int wg = getenv("IK2_G906_RWG") ? atoi(getenv("IK2_G906_RWG")) : 0;
    if (wg <= 0 || K % 32 || ((uintptr_t) q & 15) || (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15))
        return false;
    const int mi = (K / 16 + 63) / 64;
    const unsigned g = (unsigned) ((N + wg / 64 - 1) / (wg / 64));
    switch (mi) {
        case 1: g906r_kernel<1><<<g, wg, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1); break;
        case 2: g906r_kernel<2><<<g, wg, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1); break;
        case 3: g906r_kernel<3><<<g, wg, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1); break;
        case 4: g906r_kernel<4><<<g, wg, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1); break;
        default: return false;
    }
    check("g906r_mv");
    return true;
// ---- gfx906 dense Q8_0 v3: x quantized once (a small kernel: two int8 limbs a 32, into a device buffer: 3 slots for
// a launch's inputs), then a wavefront a row in small blocks reading the limbs from L2 - no per-block LDS fill
constexpr int kG906XqMaxK = 4096;
struct G906Xq {
    uint32_t hi[kG906XqMaxK / 4], lo[kG906XqMaxK / 4];
    float sc[kG906XqMaxK / 32];
};
__device__ G906Xq g906_xq_buf[3];
__global__ void __launch_bounds__(256) g906s_quant_kernel(const float* x0, const float* x1, const float* x2, int K) {
    const float* x = blockIdx.y == 0 ? x0 : blockIdx.y == 1 ? x1 : x2;
    if (!x) return;
    G906Xq& o = g906_xq_buf[blockIdx.y];
    const int t = blockIdx.x * 256 + threadIdx.x;   // (K / 4 a multiple of 8: whole groups of 8 lanes)
    const float4 v = t < (K >> 2) ? *(const float4*) (x + 4 * t) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float mx = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    mx = fmaxf(mx, __shfl_xor(mx, 1, 64));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 64));
    mx = fmaxf(mx, __shfl_xor(mx, 4, 64));
    if (t >= (K >> 2)) return;
    const float inv = mx > 0.0f ? 16256.0f / mx : 0.0f;
    const float vv[4] = {v.x, v.y, v.z, v.w};
    uint32_t h = 0, l = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = __float2int_rn(vv[i] * inv), hh = (q + 64) >> 7;
        h |= (uint32_t) (uint8_t) hh << (8 * i);
        l |= (uint32_t) (uint8_t) (q - 128 * hh) << (8 * i);
    }
    o.hi[t] = h;
    o.lo[t] = l;
    if ((t & 7) == 0) o.sc[t >> 3] = mx * (1.0f / 16256.0f);
}
template <int MAXIT>
__global__ void __launch_bounds__(256) g906s_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d,
                                                    int N, int K, float* y, MvConv c0, MvConv c1, MvAlt a0, MvAlt a1) {
    const int lane = threadIdx.x & 63;
    const int row = blockIdx.x * (int) (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= N) return;
    const int nch = K >> 4;
    const uint4* qr = (const uint4*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    uint4 w[MAXIT];
    uint16_t ds[MAXIT];
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 64 * it, cc = c < nch ? c : nch - 1;
        w[it] = __ldg(qr + cc);
        ds[it] = __ldg(dr + (cc >> 1));
    }
    const int slot = (unsigned) (row - a0.r0) < (unsigned) a0.n ? 1 : (unsigned) (row - a1.r0) < (unsigned) a1.n ? 2 : 0;
    const G906Xq& X = g906_xq_buf[slot];
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 64 * it;
        if (c < nch) {
            const uint4 xh = ((const uint4*) X.hi)[c], xl = ((const uint4*) X.lo)[c];
            int ih = __builtin_amdgcn_sdot4((int) w[it].x, (int) xh.x, 0, false);
            ih = __builtin_amdgcn_sdot4((int) w[it].y, (int) xh.y, ih, false);
            ih = __builtin_amdgcn_sdot4((int) w[it].z, (int) xh.z, ih, false);
            ih = __builtin_amdgcn_sdot4((int) w[it].w, (int) xh.w, ih, false);
            int il = __builtin_amdgcn_sdot4((int) w[it].x, (int) xl.x, 0, false);
            il = __builtin_amdgcn_sdot4((int) w[it].y, (int) xl.y, il, false);
            il = __builtin_amdgcn_sdot4((int) w[it].z, (int) xl.z, il, false);
            il = __builtin_amdgcn_sdot4((int) w[it].w, (int) xl.w, il, false);
static bool g906s_mv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                     const MvConv& c1, cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    static const int wg = getenv("IK2_G906_SWG") ? atoi(getenv("IK2_G906_SWG")) : 128;
    if (wg <= 0 || K % 128 || K > kG906XqMaxK || ((uintptr_t) q & 15) ||
        (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15))
        return false;
    const int mi = (K / 16 + 63) / 64;
    if (mi > 4) return false;
    g906s_quant_kernel<<<dim3((unsigned) ((K / 4 + 255) / 256), 3), 256, 0, s>>>(x, a0.n ? a0.x : nullptr,
                                                                                 a1.n ? a1.x : nullptr, K);
    check("g906s_quant");
    static const int twg = getenv("IK2_G906_TWG") ? atoi(getenv("IK2_G906_TWG")) : 256;
    const int rpb = twg / 64;
    const int ntot = 2 * (K / 16) + K / 128;
    if (twg >= 128 && ntot <= 2 * twg && (!a0.n || (a0.r0 % rpb == 0 && a0.n % rpb == 0)) &&
        (!a1.n || (a1.r0 % rpb == 0 && a1.n % rpb == 0))) {
        const unsigned gt = (unsigned) ((N + rpb - 1) / rpb);
        const size_t lds = (size_t) ntot * 16;
        switch (mi) {
            case 1: g906t_kernel<1><<<gt, twg, lds, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
            case 2: g906t_kernel<2><<<gt, twg, lds, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
            case 3: g906t_kernel<3><<<gt, twg, lds, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
            case 4: g906t_kernel<4><<<gt, twg, lds, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
        }
        check("g906t_mv");
        return true;
    }
    const unsigned g = (unsigned) ((N + wg / 64 - 1) / (wg / 64));
    switch (mi) {
        case 1: g906s_kernel<1><<<g, wg, 0, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
        case 2: g906s_kernel<2><<<g, wg, 0, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
        case 3: g906s_kernel<3><<<g, wg, 0, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
        case 4: g906s_kernel<4><<<g, wg, 0, s>>>(q, d, N, K, y, c0, c1, a0, a1); break;
    }
    check("g906s_mv");
    return true;
static bool g906q_mv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                     const MvConv& c1, cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    static const int qwg = getenv("IK2_G906_QWG") ? atoi(getenv("IK2_G906_QWG")) : 256;
    static const int lpr = getenv("IK2_G906_QLPR") ? atoi(getenv("IK2_G906_QLPR")) : 64;
    if (!g906q_on() || K % 128 || ((uintptr_t) q & 15) || K / 4 > 3 * qwg || (lpr != 32 && lpr != 64)) return false;
    const int rpb = qwg / lpr;
    if ((a0.n && (a0.r0 % rpb || a0.n % rpb)) || (a1.n && (a1.r0 % rpb || a1.n % rpb))) return false;
    if (getenv("IK2_SHAPES")) { static int nshow = 0; if (nshow++ < 200) std::fprintf(stderr, "Q8SHAPE %d %d %d %d\n", N, K, a0.n, a1.n); }
    const int mi = (K / 16 + lpr - 1) / lpr;
    const size_t lds = (size_t) K * 2 + (size_t) K / 32 * 4;
    const unsigned g = (unsigned) ((N + rpb - 1) / rpb);
#define G906Q_L(MI, L) g906q_kernel<MI, L><<<g, qwg, lds, s>>>(q, d, N, K, x, y, c0, c1, a0, a1)
    if (lpr == 64) {
        switch (mi) {
            case 1: G906Q_L(1, 64); break;
            case 2: G906Q_L(2, 64); break;
            case 3: G906Q_L(3, 64); break;
            case 4: G906Q_L(4, 64); break;
            default: return false;
        }
    } else {
        switch (mi) {
            case 1: G906Q_L(1, 32); break;
            case 2: G906Q_L(2, 32); break;
            case 3: G906Q_L(3, 32); break;
            case 4: G906Q_L(4, 32); break;
            case 5: G906Q_L(5, 32); break;
            case 6: G906Q_L(6, 32); break;
            case 7: case 8: G906Q_L(8, 32); break;
            default: return false;
        }
#undef G906Q_L
    check("g906q_mv");
    return true;
// ---- gfx906 dense FP16 (router, dyn conv): a row a wavefront, 8 halves a lane a chunk, all loaded first, FP32 math
template <int MAXIT>
__global__ void __launch_bounds__(256) g906h_kernel(const uint16_t* __restrict__ W, int N, int K,
                                                    const float* __restrict__ x, float* __restrict__ y) {
    const int lane = threadIdx.x & 63, row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= N) return;
    const int nch = K >> 3;
    const uint4* wr = (const uint4*) (W + (size_t) row * K);
    uint4 w[MAXIT];
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 64 * it;
        w[it] = __ldg(wr + (c < nch ? c : nch - 1));
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < MAXIT; ++it) {
        const int c = lane + 64 * it;
        if (c < nch) {
            const float4 x0 = __ldg((const float4*) x + 2 * c), x1 = __ldg((const float4*) x + 2 * c + 1);
            const uint32_t u[4] = {w[it].x, w[it].y, w[it].z, w[it].w};
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc = fmaf(h2f((uint16_t) u[i]), xv[2 * i], acc);
                acc = fmaf(h2f((uint16_t) (u[i] >> 16)), xv[2 * i + 1], acc);
            }
        }
    acc = g906_wsum64(acc);
    if (lane == 0) y[row] = acc;
}
static bool g906h_mv(const uint16_t* W, int N, int K, const float* x, float* y, cudaStream_t s) {
    if (!g906q_on() || K % 8 || ((uintptr_t) W & 15) || ((uintptr_t) x & 15)) return false;
    const int mi = (K / 8 + 63) / 64;
    const unsigned g = (unsigned) ((N + 3) / 4);
    switch (mi) {
        case 1: g906h_kernel<1><<<g, 256, 0, s>>>(W, N, K, x, y); break;
        case 2: g906h_kernel<2><<<g, 256, 0, s>>>(W, N, K, x, y); break;
        case 3: g906h_kernel<3><<<g, 256, 0, s>>>(W, N, K, x, y); break;
        case 4: g906h_kernel<4><<<g, 256, 0, s>>>(W, N, K, x, y); break;
        case 5: g906h_kernel<5><<<g, 256, 0, s>>>(W, N, K, x, y); break;
        case 6: g906h_kernel<6><<<g, 256, 0, s>>>(W, N, K, x, y); break;
        case 7: case 8: g906h_kernel<8><<<g, 256, 0, s>>>(W, N, K, x, y); break;
        default: return false;
    }
    check("g906h_mv");
    return true;
#endif
#if defined(STRATA_USE_HIP)
// ---- gfx906 batched dense Q8_0 (g906b): q8_tile's T <= 8 tokens (multi-slot decode, short verify windows)
constexpr int kG906bMaxT = 8, kG906bMaxNtot = 2 * (4096 / 16) + 4096 / 128;
__device__ uint4 g906b_xq[kG906bMaxT * kG906bMaxNtot];   // per token: hi [K/16] | lo [K/16] | sc [K/32 floats]
__global__ void __launch_bounds__(256) g906b_quant_kernel(const uint16_t* __restrict__ x, int K) {
    const int tk = blockIdx.y, ntot = 2 * (K >> 4) + (K >> 7);
    uint32_t* hi = (uint32_t*) (g906b_xq + (size_t) tk * ntot);
    uint32_t* lo = hi + (K >> 2);
    float* sc = (float*) (lo + (K >> 2));
    const int t = blockIdx.x * 256 + threadIdx.x;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (t < (K >> 2)) {
        const uint2 u = *(const uint2*) (x + (size_t) tk * K + 4 * t);
        v[0] = h2f((uint16_t) u.x), v[1] = h2f((uint16_t) (u.x >> 16)), v[2] = h2f((uint16_t) u.y),
        v[3] = h2f((uint16_t) (u.y >> 16));
    }
    float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    mx = fmaxf(mx, __shfl_xor(mx, 1, 64));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 64));
    mx = fmaxf(mx, __shfl_xor(mx, 4, 64));
    if (t >= (K >> 2)) return;
    const float inv = mx > 0.0f ? 16256.0f / mx : 0.0f;
    uint32_t h = 0, l = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = __float2int_rn(v[i] * inv), hh = (q + 64) >> 7;
        h |= (uint32_t) (uint8_t) hh << (8 * i);
        l |= (uint32_t) (uint8_t) (q - 128 * hh) << (8 * i);
    }
    hi[t] = h, lo[t] = l;
    if ((t & 7) == 0) sc[t >> 3] = mx * (1.0f / 16256.0f);
}
constexpr int kG906bTG = 4;   // tokens an LDS fill
#ifndef G906B_WPE
#define G906B_WPE 1
#endif
// lane-transpose reduction of V (a power of 2) values: lane L ends with the full sum of value idx in acc[0]
template <int N_, int M_>
struct G906TStep {
    static __device__ __forceinline__ void run(float* acc, int lane, int& idx) {
        const bool up = (lane & M_) != 0;
#pragma unroll
        for (int i = 0; i < N_ / 2; ++i) {
            const float keep = up ? acc[i + N_ / 2] : acc[i], send = up ? acc[i] : acc[i + N_ / 2];
            acc[i] = keep + __shfl_xor(send, M_, 64);
        }
        if (up) idx += N_ / 2;
        G906TStep<N_ / 2, M_ / 2>::run(acc, lane, idx);
    }
};
template <int M_>
struct G906TStep<1, M_> {
    static __device__ __forceinline__ void run(float* acc, int lane, int& idx) {
        if constexpr (M_ > 0) {
            acc[0] += __shfl_xor(acc[0], M_, 64);
            G906TStep<1, M_ / 2>::run(acc, lane, idx);
        }
    }
};
template <int V>
__device__ __forceinline__ int g906_treduce(float* acc, int lane) {
    int idx = 0;
    G906TStep<V, 32>::run(acc, lane, idx);
    return idx;
struct G906Pow2 { static constexpr int v = N_ <= 1 ? 1 : N_ <= 2 ? 2 : N_ <= 4 ? 4 : N_ <= 8 ? 8 : N_ <= 16 ? 16 : 32; };
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
#endif
#if defined(STRATA_USE_HIP)
    if (g906b_mv(q, d, N, K, x, T, y, ldy, s)) return;
#endif
#if defined(STRATA_USE_HIP)
    if (g906q_on() && g906s_mv(q, d, N, K, x, y, c0, c1, s, a0, a1)) return;
    if (g906q_on() && g906r_mv(q, d, N, K, x, y, c0, c1, s, a0, a1)) return;
    if (g906q_on() && g906q2_mv(q, d, N, K, x, y, c0, c1, s, a0, a1)) return;
    if (g906q_mv(q, d, N, K, x, y, c0, c1, s, a0, a1)) return;
#endif
#if defined(STRATA_USE_HIP)
    if (N > 0 && g906h_mv(W, N, K, x, y, s)) return;
#endif
    unsigned nt = (unsigned) ((E + 31) / 32 * 32);
#if defined(STRATA_USE_HIP)
    if (E <= 128 && E % 32 == 0) nt = (unsigned) (8 * E);   // (route_token_kernel's 8-thread ranks)
#endif
    #if defined(STRATA_USE_HIP)
    if (g906da_launch<uint16_t>(q, r, kc, vc, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj, d_rel, extent, window, scratch, P,
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
    const uint8_t* blob[kG906PfMaxE];
    int start[kG906PfMaxE], cnt[kG906PfMaxE], tile0[kG906PfMaxE + 1];
    int ne;
};
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

// Public host gate and one-visit expert entry (used by the engine's decode path).
bool ik2_g906_ok() {
    static const bool ok = [] {
        if (!g906_on()) return false;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
        return std::strncmp(prop.gcnArchName, "gfx906", 6) == 0;
    }();
    return ok;
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
#endif  // __gfx906__

#endif   // STRATA_USE_HIP
