// ik2_dsc.inc - the exact verify window: a speculative step's T tokens (2 .. 8) through
// the resident path in one go, every row the same float bits as the single-token decode gives that token.  Included
// at the end of ik2_kernels.cu (it uses that file's helpers and the expert tiles of ik2_experts2.inc); reached only
// from the block path (IK2_DSC=1), so nothing here is on the default path.  The rule each kernel keeps: a row's sums
// are the decode kernel's - the same lane owns the same values, the same iteration order, the same expressions (so
// -ffp-contract=on contracts them the same way), the same butterfly - only the weights are read once for the rows.

namespace {

// 1a: y [t][row] (rows ldy apart) = W [N][K] . x[t] (rows x_ld apart) for T tokens, a Q8_0 matrix kept planar.
// mv_q8p_kernel's sums for every token: lane l owns the values it*128 + 4 l .. + 3 of every 128-value step, steps in
// order, the step's expression as written there, then wsum.  A warp takes R rows (x's loads shared by them) and the
// T tokens (each weight dword read once for all).  The rows of a0 / a1 take those inputs: a warp's R rows one input
// (mv_q8p_rows checks that the parts' bounds are multiples of R).
template <int T, int R>
__global__ void __launch_bounds__(128) dsc_mv_q8p_rows_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d,
                                                              int N, int K, const float* __restrict__ x, int x_ld,
                                                              float* __restrict__ y, int ldy, MvAlt a0, MvAlt a1) {
    const int row0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const float* xb = (unsigned) (row0 - a0.r0) < (unsigned) a0.n   ? a0.x
                      : (unsigned) (row0 - a1.r0) < (unsigned) a1.n ? a1.x
                                                                    : x;
    const uint32_t* qr[R];
    const uint16_t* dr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = min(row0 + r, N - 1);
        qr[r] = (const uint32_t*) (q + (size_t) rr * K);
        dr[r] = d + (size_t) rr * (K >> 5);
    }
    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.0f;
#pragma unroll 2
    for (int it = 0, n_it = K >> 7; it < n_it; ++it) {
        float4 xv[T];
#pragma unroll
        for (int t = 0; t < T; ++t) xv[t] = __ldg((const float4*) (xb + (size_t) t * x_ld) + it * 32 + lane);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t w = __ldg(qr[r] + it * 32 + lane);
            const float dv = h2f(__ldg(dr[r] + it * 4 + (lane >> 3)));
#pragma unroll
            for (int t = 0; t < T; ++t)
                acc[r][t] += dv * ((float) (int8_t) (w & 0xFFu) * xv[t].x + (float) (int8_t) ((w >> 8) & 0xFFu) * xv[t].y +
                                   (float) (int8_t) ((w >> 16) & 0xFFu) * xv[t].z + (float) (int8_t) (w >> 24) * xv[t].w);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float a = wsum(acc[r][t]);
            if (lane == 0 && row0 + r < N) y[(size_t) t * ldy + row0 + r] = a;
        }
}

// 1a, x staged in LDS (IK2_DSC bit 4): a block of 8 warps x R rows loads x [T][KC] a chunk at a time (KC 512 values =
// 4 steps; the last chunk what is left) and every warp reads its steps from there - the same values, the same steps
// in the same order, the same expression: the same bits as dsc_mv_q8p_rows_kernel.  x's L2 reads a weight byte drop
// from T x 16 / (R x 4) to T x 4 / (8 R x ~1.06) (a block shares one x: the parts' bounds multiples of 8 R).
template <int T, int R>
__global__ void __launch_bounds__(256) dsc_mv_q8p_rows_lds_kernel(const int8_t* __restrict__ q,
                                                                  const uint16_t* __restrict__ d, int N, int K,
                                                                  const float* __restrict__ x, int x_ld,
                                                                  float* __restrict__ y, int ldy, MvAlt a0, MvAlt a1) {
    constexpr int KC = 512;
    __shared__ float4 xs[T][KC / 4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int blk0 = blockIdx.x * 8 * R, row0 = blk0 + warp * R;
    const float* xb = (unsigned) (blk0 - a0.r0) < (unsigned) a0.n   ? a0.x
                      : (unsigned) (blk0 - a1.r0) < (unsigned) a1.n ? a1.x
                                                                    : x;
    const uint32_t* qr[R];
    const uint16_t* dr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = min(row0 + r, N - 1);
        qr[r] = (const uint32_t*) (q + (size_t) rr * K);
        dr[r] = d + (size_t) rr * (K >> 5);
    }
    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.0f;
    for (int c0 = 0; c0 < K; c0 += KC) {
        const int n4 = min(KC, K - c0) >> 2;
        __syncthreads();
        for (int i = threadIdx.x; i < T * n4; i += 256) {
            const int t = i / n4, j = i - t * n4;
            xs[t][j] = __ldg((const float4*) (xb + (size_t) t * x_ld + c0) + j);
        }
        __syncthreads();
        if (row0 >= N) continue;
        for (int s = 0, n_s = n4 >> 5; s < n_s; ++s) {
            const int it = (c0 >> 7) + s;
            float4 xv[T];
#pragma unroll
            for (int t = 0; t < T; ++t) xv[t] = xs[t][s * 32 + lane];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t w = __ldg(qr[r] + it * 32 + lane);
                const float dv = h2f(__ldg(dr[r] + it * 4 + (lane >> 3)));
#pragma unroll
                for (int t = 0; t < T; ++t)
                    acc[r][t] += dv * ((float) (int8_t) (w & 0xFFu) * xv[t].x + (float) (int8_t) ((w >> 8) & 0xFFu) * xv[t].y +
                                       (float) (int8_t) ((w >> 16) & 0xFFu) * xv[t].z + (float) (int8_t) (w >> 24) * xv[t].w);
            }
        }
    }
    if (row0 >= N) return;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float a = wsum(acc[r][t]);
            if (lane == 0 && row0 + r < N) y[(size_t) t * ldy + row0 + r] = a;
        }
}

// 1c: the window's T queries, each split as attn_decode_f16_at's attn_split_kernel splits ONE token: query t at
// pos_d[t] sees the keys k0 = max(0, qpos - window + 1) .. qpos (a full layer: 0 .. qpos), P splits of
// len = ceil(nk / P) - the decode's k0, nk and len, so the same keys in the same warp's online softmax; the range and
// the split's merge are attn_split_kernel's (attn_flash_range, then its loop verbatim).  q and r rows q_ld / r_ld apart.
template <class KV>
__global__ void dsc_attn_split_kernel(const float* q, int q_ld, const float* r, int r_ld, const KV* kc, const KV* vc,
                                      int kv_cap, int H, int Hkv, float qk_scale, const float* proj, int d_rel,
                                      int extent, int window, float* part, const int* pos_d) {
    const int qi = blockIdx.z;
    const int qpos = pos_d[qi];
    const int k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
    const int nk = qpos + 1 - k0;
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, g = h / (H / Hkv), tid = threadIdx.x;
    const int KD = Hkv * 128;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    const float* qh = q + (size_t) qi * q_ld + (size_t) h * 128;
    const float* rh = r + (size_t) qi * r_ld + (size_t) h * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, j0, j1, qpos, qk_scale, proj, d_rel, extent, window, sm_m,
                     sm_s, sm_o);
    const float M = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
    float sum = 0.0f, o = 0.0f;
    if (M > -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (sm_m[w] == -INFINITY) continue;
            const float f = expf(sm_m[w] - M);
            sum += sm_s[w] * f;
            o += sm_o[w][tid] * f;
        }
    }
    float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
    if (tid == 0) pp[0] = M, pp[1] = sum;
    pp[2 + tid] = o;
}
// ... and the combine with the output gate: attn_combine_kernel's loop (the sink first in M, the splits in order, an
// empty split skipped), out = (o / den) * v_inv_scale, then gate_mul_f16's y = out * (1 / (1 + exp(-gate))) (the
// decode's fused combine + gate kernels claim and test these values); out [t] rows o_ld apart, gate rows gate_ld.
__global__ void dsc_attn_combine_gate_kernel(const float* sink, const float* part, int P, float v_inv_scale,
                                             const float* gate, int gate_ld, float* out, int o_ld) {
    const int h = blockIdx.x, qi = blockIdx.y, tid = threadIdx.x, H = gridDim.x;
    const float* pp = part + ((size_t) qi * H + h) * P * 130;
    const float sk = sink[(size_t) qi * H + h];
    float M = sk;
    for (int p = 0; p < P; ++p) M = fmaxf(M, pp[(size_t) p * 130]);
    float den = expf(sk - M), o = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = pp[(size_t) p * 130];
        if (mp == -INFINITY) continue;
        const float f = expf(mp - M);
        den += pp[(size_t) p * 130 + 1] * f;
        o += pp[(size_t) p * 130 + 2 + tid] * f;
    }
    const float res = (o / den) * v_inv_scale;
    const float g = gate[(size_t) qi * gate_ld + (size_t) h * 128 + tid];
    out[(size_t) qi * o_ld + (size_t) h * 128 + tid] = res * (1.0f / (1.0f + expf(-g)));
}

// 1b (lazy): the window's S = T K picks as slots sorted by (expert, pick): slot r of pick i = the picks ahead of it,
// so an expert's slots sit together and its repeat reads come from L2 / the Infinity Cache.  A slot's tables are
// route_gather's (blob, its token, sg, su) and wt / inv by pick (moe_combine_f32's): wt[i] = w[i] * down's scale.
__global__ void dsc_slots_kernel(const int* __restrict__ top, const float* __restrict__ w, int S, int K,
                                 const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                                 const float* __restrict__ su_t, const float* __restrict__ sd_t, const uint8_t** blob,
                                 int* tok, int* inv, float* sg, float* su, float* wt) {
    __shared__ int ex[64];
    const int i = threadIdx.x;
    const int e = i < S ? top[i] : 0;
    ex[i] = e;
    __syncthreads();
    if (i >= S) return;
    int r = 0;
    for (int j = 0; j < S; ++j) r += (ex[j] < e || (ex[j] == e && j < i)) ? 1 : 0;
    blob[r] = tab[e];
    tok[r] = i / K;
    sg[r] = sg_t ? sg_t[e] : 1.0f;
    su[r] = su_t ? su_t[e] : 1.0f;
    wt[i] = w[i] * (sd_t ? sd_t[e] : 1.0f);
    inv[i] = r;
}
// set 13 / 14's tiles (set 16's: gu_nv_tile<2304, 4, 2>, dn_nv_tile<1152, 4, 4>) with the slot the fast grid index:
// the blocks of one tile of an expert's slots run side by side (L2), a slot's sums untouched
template <int K, int WARPS, int RPW>
__global__ void __launch_bounds__(WARPS * 32) dsc_gu_nv_kernel(const uint8_t* const* blob, const int* tok, size_t eb0,
                                                               const float* x, int F, const float* sg, const float* su,
                                                               float gcap, float ucap, uint16_t* h) {
    gu_nv_tile<K, WARPS, RPW, false>(blockIdx.y, blockIdx.x, blob, tok, eb0, x, F, sg, su, gcap, ucap, h);
}
template <int K, int WARPS, int RPW>
__global__ void __launch_bounds__(WARPS * 32) dsc_dn_nv_kernel(const uint8_t* const* blob, size_t off_down,
                                                               const uint16_t* h, int C, float* y) {
    dn_nv_tile<K, WARPS, RPW, false>(blockIdx.y, blockIdx.x, blob, off_down, h, C, y);
}

}  // namespace

int g_dsc_lds = -1;   // mv_q8p_rows' x through LDS: -1 = IK2_DSC bit 4; 0 / 1 (tests/ik2_dsc_test.cu sets it)
void mv_q8p_rows(const int8_t* q, const uint16_t* d, int N, int K, const float* x, int x_ld, int T, float* y, int ldy,
                 cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    constexpr int R = 4;
    if (N <= 0) return;
    if (T < 1 || T > 8 || K % 128 || x_ld % 4 || ((uintptr_t) q & 3) ||
        (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15) || a0.r0 % R || a0.n % R || a1.r0 % R ||
        a1.n % R) {
        std::fprintf(stderr, "ik2::k::mv_q8p_rows: T %d, K %d, x_ld %d, alignment or a part's bounds\n", T, K, x_ld);
        std::abort();
    }
    // (IK2_DSC bit 4: x through LDS, 8 warps x R rows a block - where every part's bounds are multiples of 8 R)
    if (g_dsc_lds < 0) g_dsc_lds = getenv("IK2_DSC") && (strtol(getenv("IK2_DSC"), nullptr, 0) & 4) ? 1 : 0;
    if (g_dsc_lds && a0.r0 % (8 * R) == 0 && a0.n % (8 * R) == 0 && a1.r0 % (8 * R) == 0 && a1.n % (8 * R) == 0) {
        const unsigned nb8 = (unsigned) ((N + 8 * R - 1) / (8 * R));
        switch (T) {
#define DSC_MVL(TT) \
    case TT: dsc_mv_q8p_rows_lds_kernel<TT, R><<<nb8, 256, 0, s>>>(q, d, N, K, x, x_ld, y, ldy, a0, a1); break;
            DSC_MVL(1) DSC_MVL(2) DSC_MVL(3) DSC_MVL(4) DSC_MVL(5) DSC_MVL(6) DSC_MVL(7) DSC_MVL(8)
#undef DSC_MVL
        }
        check("mv_q8p_rows (lds)");
        return;
    }
    const unsigned nb = (unsigned) ((N + 4 * R - 1) / (4 * R));
    switch (T) {
#define DSC_MVR(TT) \
    case TT: dsc_mv_q8p_rows_kernel<TT, R><<<nb, 128, 0, s>>>(q, d, N, K, x, x_ld, y, ldy, a0, a1); break;
        DSC_MVR(1) DSC_MVR(2) DSC_MVR(3) DSC_MVR(4) DSC_MVR(5) DSC_MVR(6) DSC_MVR(7) DSC_MVR(8)
#undef DSC_MVR
    }
    check("mv_q8p_rows");
}

void attn_rows_f16(const float* q, int q_ld, const float* r, int r_ld, const float* sink, const uint16_t* kc,
                   const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int P, float qk_scale,
                   const float* proj, int d_rel, int extent, int window, float* scratch, const float* gate,
                   int gate_ld, float* out, int o_ld, cudaStream_t s) {
    dsc_attn_split_kernel<uint16_t><<<dim3(H, P, T), 128, 0, s>>>(q, q_ld, r, r_ld, kc, vc, kv_cap, H, Hkv, qk_scale,
                                                                   proj, d_rel, extent, window, scratch, pos);
    check("attn_rows_f16 (split)");
    dsc_attn_combine_gate_kernel<<<dim3(H, T), 128, 0, s>>>(sink, scratch, P, 1.0f, gate, gate_ld, out, o_ld);
    check("attn_rows_f16 (combine)");
}

void dsc_slots(const int* top, const float* w, int S, int K, const uint8_t* const* tab, const float* sg_t,
               const float* su_t, const float* sd_t, const uint8_t** blob, int* tok, int* inv, float* sg, float* su,
               float* wt, cudaStream_t s) {
    if (S < 1 || S > 64) {
        std::fprintf(stderr, "ik2::k::dsc_slots: %d picks (1 .. 64)\n", S);
        std::abort();
    }
    dsc_slots_kernel<<<1, 64, 0, s>>>(top, w, S, K, tab, sg_t, su_t, sd_t, blob, tok, inv, sg, su, wt);
    check("dsc_slots");
}

void experts_rows_nv(const uint8_t* const* blob, const int* tok, int S, size_t eb0, size_t off_down, const float* x,
                     int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                     cudaStream_t s) {
    if (S <= 0) return;
    if (!experts_nv_fits(kNVFP4, kNVFP4, kNVFP4, C, F)) {
        std::fprintf(stderr, "ik2::k::experts_rows_nv: C %d, F %d (2304, 1152 only)\n", C, F);
        std::abort();
    }
    dsc_gu_nv_kernel<2304, 4, 2><<<dim3(S, (F + 7) / 8), 128, 0, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h);
    check("experts_rows_nv (gate_up)");
    dsc_dn_nv_kernel<1152, 4, 4><<<dim3(S, (C + 15) / 16), 128, 0, s>>>(blob, off_down, h, C, y);
    check("experts_rows_nv (down)");
}
