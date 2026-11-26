// ik2_fuse.inc - the resident decode path's small kernels, fewer and shorter (IK2_FUSE_MASK).  Included at the end of ik2_kernels.cu (it uses that file's helpers); nothing here is on the
// default path.  On gfx1151 these kernels were not bound by their launches but by a lone warp's chain of loads
// (measured): a warp a row reading 72 values a lane one after the other.  So each one here issues all of a
// lane's loads first, then does the same sums in the same order as the kernel it replaces - the same values to the
// bit (tests/ik2_fuse_test.cu) - and neighbours share a launch where the data allow it.

namespace {

// lane's part of a row . x over K = 32 NPL values (FP16 row, F32 x), then the warp's sum: mv_f16_kernel's and the
// sinks' sums (lane j takes j, j + 32, ... in order; wsum), every load asked for before the first sum
template <int NPL>
__device__ __forceinline__ float dot_f16_pf(const uint16_t* __restrict__ w, const float* __restrict__ x, int lane) {
    uint16_t wv[NPL];
    float xv[NPL];
#pragma unroll
    for (int j = 0; j < NPL; ++j) wv[j] = w[lane + 32 * j];
#pragma unroll
    for (int j = 0; j < NPL; ++j) xv[j] = x[lane + 32 * j];
    float a = 0.0f;
#pragma unroll
    for (int j = 0; j < NPL; ++j) a += h2f(wv[j]) * xv[j];
    return wsum(a);
}

// up to two FP16 products in one launch (a full layer's two dynamic taps): blocks [0, nb0) job 0, the rest job 1
struct MvF16Job {
    const uint16_t* W;
    const float* x;
    float* y;
    int N;
};
template <int NPL>
__global__ void __launch_bounds__(128) mv_f16_pf_kernel(MvF16Job j0, MvF16Job j1, int nb0) {
    const bool second = (int) blockIdx.x >= nb0;
    const MvF16Job j = second ? j1 : j0;
    const int row = (int) (second ? blockIdx.x - nb0 : blockIdx.x) * 4 + (int) (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= j.N) return;
    const float a = dot_f16_pf<NPL>(j.W + (size_t) row * (32 * NPL), j.x, lane);
    if (lane == 0) j.y[row] = a;
}

// last-block counters (a kernel's slot; its last block sets it back to 0, and the stream orders the launches)
__device__ unsigned int g_fuse_cnt[64] = {};
constexpr int kFuseCntRouter = 0;

// the router's product and route_token in one launch: blocks of 4 warps, a warp an expert's logit (mv_f16's sums);
// the last block to finish routes (route_token_kernel's block 0, the same code: E <= 128 = the block's threads)
// LDS (set 6): the block's 4 rows and x come in 16-byte loads (a lane's 9 of its row, a thread's <= 5 of x; were 72 +
// 72 loads of 2 and 4 bytes a lane) into LDS, and each lane then sums its strided values from there - the same values
// in the same order (dot_f16_pf's)
template <int NPL, bool LDS>
__global__ void __launch_bounds__(128)
    router_route_kernel(const uint16_t* __restrict__ Wr, const float* __restrict__ x, float* logits,
                        const float* __restrict__ bias, int E, int K, float scale, const uint8_t* const* __restrict__ tab,
                        const float* __restrict__ sg_t, const float* __restrict__ su_t, const float* __restrict__ sd_t,
                        int* top, float* w, const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = (int) blockIdx.x * 4 + warp;
    if constexpr (LDS) {
        constexpr int KX = 32 * NPL, NX = (KX / 4 + 127) / 128, NW = (KX / 8 + 31) / 32;
        __shared__ __align__(16) float sx[KX];
        __shared__ __align__(16) uint16_t sw[4][KX];
        float4 xr[NX];
        uint4 wr[NW];
#pragma unroll
        for (int i = 0; i < NX; ++i) {
            const int c = (int) threadIdx.x + 128 * i;
            if (c < KX / 4) xr[i] = ((const float4*) x)[c];
        }
#pragma unroll
        for (int i = 0; i < NW; ++i) {
            const int c = lane + 32 * i;
            if (row < E && c < KX / 8) wr[i] = ((const uint4*) (Wr + (size_t) row * KX))[c];
        }
#pragma unroll
        for (int i = 0; i < NX; ++i) {
            const int c = (int) threadIdx.x + 128 * i;
            if (c < KX / 4) ((float4*) sx)[c] = xr[i];
        }
#pragma unroll
        for (int i = 0; i < NW; ++i) {
            const int c = lane + 32 * i;
            if (row < E && c < KX / 8) ((uint4*) sw[warp])[c] = wr[i];
        }
        __syncthreads();
        if (row < E) {
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < NPL; ++j) a += h2f(sw[warp][lane + 32 * j]) * sx[lane + 32 * j];
            a = wsum(a);
            if (lane == 0) logits[row] = a;
        }
    } else if (row < E) {
        const float a = dot_f16_pf<NPL>(Wr + (size_t) row * (32 * NPL), x, lane);
        if (lane == 0) logits[row] = a;
    }
    __shared__ unsigned int s_ticket;
    __threadfence();   // (this block's logits seen by the last block)
    __syncthreads();
    if (threadIdx.x == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntRouter], 1u);
    __syncthreads();
    if (s_ticket != gridDim.x - 1) return;
    __threadfence();   // (every block's logits seen here)
    __shared__ float val[128];       // sigmoid + bias
    __shared__ uint32_t picked[4];   // a warp's picks, a bit a lane
    __shared__ float sig[17];        // the picks' sigmoids in order, [K]: their sum
    const int e = threadIdx.x;
    const float lg = e < E ? ((volatile const float*) logits)[e] : 0.0f;
    const float sl = e < E ? 1.0f / (1.0f + expf(-lg)) : 0.0f;
    const float v = e < E ? sl + __ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) picked[warp] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(picked[ww]);
    if (pick) sig[pos] = sl;
    __syncthreads();
    if (e == 0) {   // (the sum in route's order: the same weights to the bit)
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
        g_fuse_cnt[kFuseCntRouter] = 0;
    }
    __syncthreads();
    if (pick) {
        const float wk = sl / (sig[K] + 1e-20f) * scale;
        top[pos] = e;
        w[pos] = wk;
        blob[pos] = tab[e];
        tok[pos] = 0;
        inv[pos] = pos;
        sg[pos] = sg_t ? sg_t[e] : 1.0f;
        su[pos] = su_t ? su_t[e] : 1.0f;
        wt[pos] = wk * (sd_t ? sd_t[e] : 1.0f);
    }
}

// prep_qkv_r_impl_kernel for one token (T 1, the position on the device) with the sinks' sums' loads asked for at
// once: the clip warps are prep_qkv_r_impl_kernel's code; a sink warp was its launch's last 15 us
template <class KV, int NPL>
__global__ void __launch_bounds__(32)
    prep_qkv_r_pf_kernel(float* q, const float* q_clip, const float* k_in, KV* kc, const float* v_in, KV* vc,
                         float* r, const float* r_clip, int H, int KVh, int D, int d_rel, float qk_clip, float rel_clip,
                         float eps, const int* pos, float alpha, float n_floor, int kv_cap, float k_scale,
                         float v_scale, const int* pos0_d, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                         float* sink_y) {
    const int pos0 = *pos0_d;
    const int gid = blockIdx.x;
    const int per_tok = 2 * H + 2 * KVh;
    if (gid >= per_tok) {
        const int h = gid - per_tok, lane = threadIdx.x;
        if (!sink_w || h >= H) return;
        const float a = dot_f16_pf<NPL>(sink_w + (size_t) h * (32 * NPL), sink_x, lane);
        if (lane == 0) sink_y[h] = a;
        return;
    }
    const int rem = gid;
    const float tau = pos ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[0] + 1.0f) / n_floor))) : 1.0f;
    if (rem < H) {
        const int h = rem;
        float* qp = q + (size_t) h * D;
        l2clip_warp(qp, q_clip ? q_clip + (size_t) h * D : nullptr, qp, q16 ? q16 + (size_t) h * D : nullptr, D,
                    qk_clip, tau);
    } else if (rem < H + KVh) {
        const int kv = rem - H;
        const int slot = kv_cap > 0 ? pos0 % kv_cap : pos0;
        store_k_warp(k_in + (size_t) kv * D, kc + ((size_t) slot * KVh + kv) * D, D, qk_clip, k_scale);
    } else if (rem < H + 2 * KVh) {
        const int kv = rem - H - KVh;
        const int slot = kv_cap > 0 ? pos0 % kv_cap : pos0;
        store_v_warp(v_in + (size_t) kv * D, vc + ((size_t) slot * KVh + kv) * D, D, eps, v_scale);
    } else {
        const int h = rem - H - 2 * KVh;
        float* rp = r + (size_t) h * d_rel;
        l2clip_warp(rp, r_clip ? r_clip + (size_t) h * d_rel : nullptr, rp, nullptr, d_rel, rel_clip, tau);
    }
}

// attn_combine_kernel and the output gate after it (gate_mul_f16: o sigmoid(gate), F32 and FP16) in one launch, a
// split's values asked for at once (a full layer's 128 splits were 36 us of loads one after the other)
// SKIP (set 7, one token whose position is at pos_d): the splits attn_split_kernel leaves without keys (p >= its
// ceil(nk / len): M -INF, sum 0) are not read but taken as such, and each split's factor expf(m - M) is computed once,
// by one thread, into LDS (every thread computed all NP) - the same values to the bit
template <int NP, bool SKIP = false>
__global__ void __launch_bounds__(128)
    attn_combine_gate_kernel(const float* __restrict__ sink, const float* __restrict__ part, float v_inv_scale,
                             const float* __restrict__ gate, float* out, uint16_t* out16, const int* pos_d = nullptr,
                             int window = 0) {
    const int h = blockIdx.x, tid = threadIdx.x;
    const float* pp = part + (size_t) h * NP * 130;
    __shared__ float sm_m[NP], sm_s[NP], sm_f[SKIP ? NP : 1];
    int used = NP;
    if constexpr (SKIP) {
        const int qpos = *pos_d, k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0, nk = qpos + 1 - k0;
        const int len = (nk + NP - 1) / NP;
        used = (nk + len - 1) / len;
    }
    for (int p = tid; p < NP; p += 128) {
        sm_m[p] = p < used ? pp[(size_t) p * 130] : -INFINITY;
        sm_s[p] = p < used ? pp[(size_t) p * 130 + 1] : 0.0f;
    }
    float ov[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) ov[p] = p < used ? pp[(size_t) p * 130 + 2 + tid] : 0.0f;
    const float g = gate[(size_t) h * 128 + tid];
    const float sk = sink[h];
    __syncthreads();
    float M = sk;
    for (int p = 0; p < NP; ++p) M = fmaxf(M, sm_m[p]);
    if constexpr (SKIP) {
        for (int p = tid; p < NP; p += 128) sm_f[p] = sm_m[p] == -INFINITY ? 0.0f : expf(sm_m[p] - M);
        __syncthreads();
    }
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const float mp = sm_m[p];
        if (mp == -INFINITY) continue;
        const float f = SKIP ? sm_f[p] : expf(mp - M);
        den += sm_s[p] * f;
        o += ov[p] * f;
    }
    const float res = (o / den) * v_inv_scale;
    const float y = res * (1.0f / (1.0f + expf(-g)));
    out[(size_t) h * 128 + tid] = y;
    out16[(size_t) h * 128 + tid] = f2h(y);
}

constexpr int kFuseNpl = 72;   // a lane's values of a 2304-wide row (Inkling-2-Mini's width)
constexpr int kFuseCntGate = 1;    // [1, 4): a loop gate's last apply block
constexpr int kFuseCntAttn = 8;    // [8, 8 + H): a head's last split block

// loop_gate_dots_kernel<false> for one token, a thread's NPT values of x, m and D's two halves asked for at once
// (C = 128 NPT); the sums in its order
template <int NPT>
__global__ void __launch_bounds__(128) loop_gate_dots_pf_kernel(LoopGateArgs a, int C, int R, float* sc) {
    __shared__ float sh[4][4];
    const int r = blockIdx.x, j = blockIdx.y, tid = threadIdx.x, warp = tid >> 5;
    const float* xr = a.x[j];
    const float* mr = a.m[j];
    const float* Dr = (const float*) a.D[j] + (size_t) r * 2 * C;
    float xv[NPT], mv[NPT], dx[NPT], dm[NPT];
#pragma unroll
    for (int i = 0; i < NPT; ++i) {
        const int c = tid + 128 * i;
        xv[i] = xr[c], mv[i] = mr[c], dx[i] = Dr[c], dm[i] = Dr[C + c];
    }
    float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
#pragma unroll
    for (int i = 0; i < NPT; ++i) {
        ax += dx[i] * xv[i], am += dm[i] * mv[i];
        sx += xv[i] * xv[i], sm += mv[i] * mv[i];
    }
    ax = wsum(ax), am = wsum(am), sx = wsum(sx), sm = wsum(sm);
    if ((tid & 31) == 0) sh[warp][0] = ax, sh[warp][1] = am, sh[warp][2] = sx, sh[warp][3] = sm;
    __syncthreads();
    if (tid == 0) {
        float* o = sc + (size_t) j * (2 * R + 2);
        o[2 * r] = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
        o[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
        if (r == 0) {
            o[2 * R] = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
            o[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
        }
    }
}

// loop_gate_apply_kernel<64, false> for one token at 256 threads (its rank loop unrolled: a thread's 128 loads at
// once), then the last block of each gate takes the rms after it - rms_multi_kernel's / rms_impl_kernel<false,
// true>'s code at their 256 threads (job j of rj: x = the gate's y) - a launch less a gate
__global__ void __launch_bounds__(256)
    loop_gate_apply_rms_kernel(LoopGateArgs a, int C, int R, float eps, const float* __restrict__ sc, RmsJobs rj,
                               float rms_eps) {
    constexpr int Rank = 64;
    __shared__ float sh_g[kLoopRankMax];
    const int j = blockIdx.y, tid = threadIdx.x;
    const float* o = sc + (size_t) j * (2 * R + 2);
    for (int i = tid; i < R; i += blockDim.x) {
        const float rx = rsqrtf(o[2 * R] / (float) C + eps), rm = rsqrtf(o[2 * R + 1] / (float) C + eps);
        sh_g[i] = rx * o[2 * i] + rm * o[2 * i + 1];
    }
    __syncthreads();
    const int c = (int) blockIdx.x * blockDim.x + tid;
    if (c < C) {
        const float* F = (const float*) a.F[j];
        const float* G = (const float*) a.G[j];
        float f = 0.0f, gn = 0.0f;
#pragma unroll
        for (int r = 0; r < Rank; ++r) {
            const float gv = sh_g[r];
            f += F[(size_t) r * C + c] * gv, gn += G[(size_t) r * C + c] * gv;
        }
        a.y[j][c] = (1.0f - tanhf(f)) * a.x[j][c] + tanhf(gn) * a.m[j][c];
    }
    __shared__ unsigned int s_ticket;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntGate + j], 1u);
    __syncthreads();
    if (s_ticket != gridDim.x - 1) return;
    __threadfence();
    if (tid == 0) g_fuse_cnt[kFuseCntGate + j] = 0;
    // (rms_multi_kernel's vec4 path, job j; the gate's y read after the fence, not through __ldg)
    __shared__ float sh[256];
    const float* xr = rj.x[j];
    const float* w = rj.w[j];
    float* y32r = rj.y[j];
    float a2 = 0.0f;
    const int n4 = C >> 2;
    const float4* x4 = (const float4*) xr;
    for (int d4 = tid; d4 < n4; d4 += 256) {
        const float4 v = x4[d4];
        a2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float rs = rsqrtf(bsum(a2, sh) / (float) C + rms_eps);
    const float4* w4 = (const float4*) w;
    float4* y32_4 = (float4*) y32r;
    for (int d4 = tid; d4 < n4; d4 += 256) {
        const float4 v = x4[d4];
        const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
        y32_4[d4] = make_float4(r0, r1, r2, r3);
    }
}

// attn_split_kernel (one token) and, in the last of a head's NP split blocks, attn_combine_gate_kernel's code: a
// launch less a layer
template <class KV, int NP>
__global__ void __launch_bounds__(128)
    attn_split_combine_gate_kernel(const float* q, const float* r, const KV* kc, const KV* vc, int kv_cap, int H,
                                   int Hkv, float qk_scale, const float* proj, int d_rel, int extent, int window,
                                   float* part, const int* pos_d, const float* __restrict__ sink, float v_inv_scale,
                                   const float* __restrict__ gate, float* out, uint16_t* out16) {
    // == attn_split_kernel, gridDim.z 1 ==
    const int qpos0 = *pos_d;
    const int k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
    const int nk = qpos0 + 1 - k0;
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, p = blockIdx.y, g = h / (H / Hkv), tid = threadIdx.x;
    const int KD = Hkv * 128, qpos = qpos0;
    const int len = (nk + NP - 1) / NP, j0 = p * len, j1 = min(nk, j0 + len);
    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, j0, j1, qpos, qk_scale, proj, d_rel, extent, window, sm_m,
                     sm_s, sm_o);
    {
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
        float* pp = part + ((size_t) h * NP + p) * 130;
        if (tid == 0) pp[0] = M, pp[1] = sum;
        pp[2 + tid] = o;
    }
    // == the head's last block: attn_combine_gate_kernel ==
    __shared__ unsigned int s_ticket;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntAttn + h], 1u);
    __syncthreads();
    if (s_ticket != NP - 1) return;
    __threadfence();
    if (tid == 0) g_fuse_cnt[kFuseCntAttn + h] = 0;
    const float* pp = part + (size_t) h * NP * 130;
    __shared__ float cm[NP], cs[NP];
    for (int i = tid; i < NP; i += 128) cm[i] = pp[(size_t) i * 130], cs[i] = pp[(size_t) i * 130 + 1];
    float ov[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) ov[i] = pp[(size_t) i * 130 + 2 + tid];
    const float gt = gate[(size_t) h * 128 + tid];
    const float sk = sink[h];
    __syncthreads();
    float M = sk;
    for (int i = 0; i < NP; ++i) M = fmaxf(M, cm[i]);
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const float mp = cm[i];
        if (mp == -INFINITY) continue;
        const float f = expf(mp - M);
        den += cs[i] * f;
        o += ov[i] * f;
    }
    const float res = (o / den) * v_inv_scale;
    const float y = res * (1.0f / (1.0f + expf(-gt)));
    out[(size_t) h * 128 + tid] = y;
    out16[(size_t) h * 128 + tid] = f2h(y);
}

// moe_combine_conv_kernel (one token) and then attn_res_vec4_kernel<FUSED_RMS, 512> over states whose last is the
// combine's output, in one block of 512 threads: a thread's float4s of the attention residual (d4 = tid, tid + 512)
// are the combine's channels it computed, kept in registers for that state; every other value is loaded as there
struct MoeCombine {
    const float* y;
    const int* inv;
    const float* wt;
    float* hist;
    const float* w;
    const float* res;
    float* out;
};
// moe_combine_conv_kernel's arithmetic for the 4 channels of float4 d4 (K = 8 slots), its loads at once
__device__ __forceinline__ float4 moe_combine_conv4(const MoeCombine& mc, int C, int d4) {
    constexpr int K = 8;
    float4 yv[K];
    float wt[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        wt[k] = mc.wt[k];
        yv[k] = ((const float4*) (mc.y + (size_t) mc.inv[k] * C))[d4];
    }
    const float4 h0 = ((const float4*) mc.hist)[d4], h1 = ((const float4*) (mc.hist + C))[d4],
                 h2 = ((const float4*) (mc.hist + 2 * (size_t) C))[d4];
    float4 wc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) wc[i] = ((const float4*) mc.w)[4 * d4 + i];
    const float4 rv = mc.res ? ((const float4*) mc.res)[d4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float o[4], av[4];
    const float h0v[4] = {h0.x, h0.y, h0.z, h0.w}, h1v[4] = {h1.x, h1.y, h1.z, h1.w}, h2v[4] = {h2.x, h2.y, h2.z, h2.w};
    const float rvv[4] = {rv.x, rv.y, rv.z, rv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float yk = i == 0 ? yv[k].x : i == 1 ? yv[k].y : i == 2 ? yv[k].z : yv[k].w;
            a += wt[k] * yk;
        }
        const float w0 = wc[i].x, w1 = wc[i].y, w2 = wc[i].z, w3 = wc[i].w;
        float b = w0 * h0v[i] + w1 * h1v[i] + w2 * h2v[i] + w3 * a;
        if (mc.res) b += rvv[i];
        o[i] = b, av[i] = a;
    }
    const float4 out = make_float4(o[0], o[1], o[2], o[3]);
    ((float4*) mc.out)[d4] = out;
    ((float4*) mc.hist)[d4] = h1;
    ((float4*) (mc.hist + C))[d4] = h2;
    ((float4*) (mc.hist + 2 * (size_t) C))[d4] = make_float4(av[0], av[1], av[2], av[3]);
    return out;
}

template <bool FUSED_RMS>
__global__ void __launch_bounds__(512)
    moe_attn_res_kernel(MoeCombine mc, ResStates st, int n, const float* q, int nq, int C, float res_eps,
                        float rms_eps, float* out, RmsOuts outs) {
    constexpr int BLOCK_THREADS = 512, NUM_WARPS = 16;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float sc[3][kResMax];
    __shared__ float rs_rms[3];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c4 = C >> 2, d4_0 = tid, d4_1 = tid + BLOCK_THREADS;
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4;
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // == the combine: state n - 1's float4s of this thread ==
    const float4 hA = in0 ? moe_combine_conv4(mc, C, d4_0) : zero4;
    const float4 hB = in1 ? moe_combine_conv4(mc, C, d4_1) : zero4;
    // == attn_res_vec4_kernel (T 1) ==
    float4 q0[3] = {}, q1[3] = {};
    if (in0) {
        q0[0] = __ldg((const float4*) q + d4_0);
        if (nq > 1) q0[1] = __ldg((const float4*) (q + C) + d4_0);
        if (nq > 2) q0[2] = __ldg((const float4*) (q + 2 * C) + d4_0);
    }
    if (in1) {
        q1[0] = __ldg((const float4*) q + d4_1);
        if (nq > 1) q1[1] = __ldg((const float4*) (q + C) + d4_1);
        if (nq > 2) q1[2] = __ldg((const float4*) (q + 2 * C) + d4_1);
    }
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int ii = min(i0 + j, n - 1);
            const float4* x4 = (const float4*) st.st[ii];
            xa[j] = in0 ? (ii == n - 1 ? hA : x4[d4_0]) : zero4;
            xb[j] = in1 ? (ii == n - 1 ? hB : x4[d4_1]) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
            float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
            if (in0) {
                const float4 xv0 = xa[j];
                sum_sq += xv0.x * xv0.x + xv0.y * xv0.y + xv0.z * xv0.z + xv0.w * xv0.w;
                qd0 += xv0.x * q0[0].x + xv0.y * q0[0].y + xv0.z * q0[0].z + xv0.w * q0[0].w;
                if (nq > 1) qd1 += xv0.x * q0[1].x + xv0.y * q0[1].y + xv0.z * q0[1].z + xv0.w * q0[1].w;
                if (nq > 2) qd2 += xv0.x * q0[2].x + xv0.y * q0[2].y + xv0.z * q0[2].z + xv0.w * q0[2].w;
            }
            if (in1) {
                const float4 xv1 = xb[j];
                sum_sq += xv1.x * xv1.x + xv1.y * xv1.y + xv1.z * xv1.z + xv1.w * xv1.w;
                qd0 += xv1.x * q1[0].x + xv1.y * q1[0].y + xv1.z * q1[0].z + xv1.w * q1[0].w;
                if (nq > 1) qd1 += xv1.x * q1[1].x + xv1.y * q1[1].y + xv1.z * q1[1].z + xv1.w * q1[1].w;
                if (nq > 2) qd2 += xv1.x * q1[2].x + xv1.y * q1[2].y + xv1.z * q1[2].z + xv1.w * q1[2].w;
            }
            sum_sq = wsum(sum_sq);
            qd0 = wsum(qd0);
            if (nq > 1) qd1 = wsum(qd1);
            if (nq > 2) qd2 = wsum(qd2);
            if (lane == 0) {
                sh_red[warp][i * 4 + 0] = sum_sq;
                sh_red[warp][i * 4 + 1] = qd0;
                sh_red[warp][i * 4 + 2] = qd1;
                sh_red[warp][i * 4 + 3] = qd2;
            }
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    __syncthreads();

    float4 o0[3] = {}, o1[3] = {};
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int ii = min(i0 + j, n - 1);
            const float4* x4 = (const float4*) st.st[ii];
            xa[j] = in0 ? (ii == n - 1 ? hA : x4[d4_0]) : zero4;
            xb[j] = in1 ? (ii == n - 1 ? hB : x4[d4_1]) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                const float w_qi = sc[qi][i];
                if (in0) {
                    o0[qi].x += w_qi * xa[j].x;
                    o0[qi].y += w_qi * xa[j].y;
                    o0[qi].z += w_qi * xa[j].z;
                    o0[qi].w += w_qi * xa[j].w;
                }
                if (in1) {
                    o1[qi].x += w_qi * xb[j].x;
                    o1[qi].y += w_qi * xb[j].y;
                    o1[qi].z += w_qi * xb[j].z;
                    o1[qi].w += w_qi * xb[j].w;
                }
            }
        }
    }
    if constexpr (!FUSED_RMS) {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float4* y4 = (float4*) (out + (size_t) qi * C);
            if (d4_0 < c4) y4[d4_0] = o0[qi];
            if (d4_1 < c4) y4[d4_1] = o1[qi];
        }
    } else {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float sq = 0.0f;
            if (d4_0 < c4) sq += o0[qi].x * o0[qi].x + o0[qi].y * o0[qi].y + o0[qi].z * o0[qi].z + o0[qi].w * o0[qi].w;
            if (d4_1 < c4) sq += o1[qi].x * o1[qi].x + o1[qi].y * o1[qi].y + o1[qi].z * o1[qi].z + o1[qi].w * o1[qi].w;
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] = rsqrtf(s_tot / (float) C + rms_eps);
            }
        }
        __syncthreads();
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            const float rs = rs_rms[qi];
            const float4* w4 = (const float4*) outs.w[qi];
            float4* y32_4 = (float4*) outs.y32[qi];
            uint2* y16_4 = (uint2*) outs.y16[qi];
            if (d4_0 < c4) {
                const float4 ww = w4 ? __ldg(w4 + d4_0) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o0[qi].x * rs * ww.x, r1 = o0[qi].y * rs * ww.y;
                const float r2 = o0[qi].z * rs * ww.z, r3 = o0[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_0] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_0] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
            if (d4_1 < c4) {
                const float4 ww = w4 ? __ldg(w4 + d4_1) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o1[qi].x * rs * ww.x, r1 = o1[qi].y * rs * ww.y;
                const float r2 = o1[qi].z * rs * ww.z, r3 = o1[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_1] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_1] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
        }
    }
}

// prep_qkv_r (one token, FP16 cache) folded into the one-launch attention (P 16): a split block clips its head's q and
// r itself (l2clip_warp's arithmetic), and the block whose keys hold the token's own position makes that key and
// value as prep stores them (clip / rms, FP16) and uses them - the head group's first head's block writes them to the
// cache; block NP of a head takes the head's sink (prep's sum); the head's last block of the NP + 1 combines
template <int NP>
__global__ void __launch_bounds__(128)
    attn_prep_one_kernel(const float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                         uint16_t* vc, const float* r, const float* r_clip, int H, int Hkv, int d_rel, float qk_clip,
                         float rel_clip, float eps, const int* pos_d, bool log_scale, float alpha, float n_floor,
                         int kv_cap, const uint16_t* sink_w, const float* sink_x, float* sink, float qk_scale,
                         const float* proj, int extent, int window, float* part, float v_inv_scale,
                         const float* __restrict__ gate, float* out, uint16_t* out16) {
    const int h = blockIdx.x, p = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (p == NP) {   // == the head's sink: prep_qkv_r_pf_kernel's sink warp ==
        if (warp == 0) {
            const float a = dot_f16_pf<kFuseNpl>(sink_w + (size_t) h * (32 * kFuseNpl), sink_x, lane);
            if (lane == 0) sink[h] = a;
        }
    } else {
        const int qpos = *pos_d;
        const int k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
        const int nk = qpos + 1 - k0;
        const int g = h / (H / Hkv), KD = Hkv * 128;
        const int len = (nk + NP - 1) / NP, j0 = p * len, j1 = min(nk, j0 + len);
        const float tau = log_scale ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) qpos + 1.0f) / n_floor))) : 1.0f;
        // q: l2clip_warp (vec4 path) of q's head with its scale
        float4 q4;
        {
            const float4 xv = ((const float4*) (q + (size_t) h * 128))[lane];
            const float4 sv = q_clip ? __ldg((const float4*) (q_clip + (size_t) h * 128) + lane)
                                     : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
            const float nrm = sqrtf(wsum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3));
            const float f = fminf(1.0f, qk_clip / (nrm + 1e-6f)) * tau;
            q4 = make_float4(v0 * f, v1 * f, v2 * f, v3 * f);
        }
        // r: l2clip_warp (the loop path: d_rel < 128) of r's head
        float r_lane = 0.0f;
        {
            const float* src = r + (size_t) h * d_rel;
            const float* sc = r_clip ? r_clip + (size_t) h * d_rel : nullptr;
            float a = 0.0f;
            for (int d = lane; d < d_rel; d += 32) {
                const float v = src[d] * (sc ? sc[d] : 1.0f);
                a += v * v;
            }
            const float nrm = sqrtf(wsum(a));
            const float f = fminf(1.0f, rel_clip / (nrm + 1e-6f)) * tau;
            if (lane < d_rel) r_lane = src[lane] * (sc ? sc[lane] : 1.0f) * f;
        }
        // the token's own key and value (store_k_warp / store_v_warp: FP16), if this block's keys hold its position
        float4 kcur = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vcur = kcur;
        if (j0 <= nk - 1 && nk - 1 < j1) {
            const float4 kx = ((const float4*) (k_in + (size_t) g * 128))[lane];
            const float k0v = kx.x * 1.0f, k1v = kx.y * 1.0f, k2v = kx.z * 1.0f, k3v = kx.w * 1.0f;
            const float knrm = sqrtf(wsum(k0v * k0v + k1v * k1v + k2v * k2v + k3v * k3v));
            const float kf = fminf(1.0f, qk_clip / (knrm + 1e-6f)) * 1.0f;
            const uint32_t kp0 = f2_to_u32(k0v * kf, k1v * kf), kp1 = f2_to_u32(k2v * kf, k3v * kf);
            const float4 vx = ((const float4*) (v_in + (size_t) g * 128))[lane];
            const float rs = rsqrtf(wsum(vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w) / (float) 128 + eps);
            const uint32_t vp0 = f2_to_u32(vx.x * rs, vx.y * rs), vp1 = f2_to_u32(vx.z * rs, vx.w * rs);
            const float2 a0 = u32_to_f2(kp0), a1 = u32_to_f2(kp1), b0 = u32_to_f2(vp0), b1 = u32_to_f2(vp1);
            kcur = make_float4(a0.x, a0.y, a1.x, a1.y), vcur = make_float4(b0.x, b0.y, b1.x, b1.y);
            if (h % (H / Hkv) == 0 && warp == 0) {
                const int slot = kv_cap > 0 ? qpos % kv_cap : qpos;
                ((uint2*) (kc + ((size_t) slot * Hkv + g) * 128))[lane] = make_uint2(kp0, kp1);
                ((uint2*) (vc + ((size_t) slot * Hkv + g) * 128))[lane] = make_uint2(vp0, vp1);
            }
        }
        // == attn_flash_range, the token's own key and value from registers ==
        __shared__ float sm_m[4], sm_s[4];
        __shared__ __align__(16) float sm_o[4][128];
        {
            float m_w = -INFINITY, s_w = 0.0f;
            float4 o_w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int j = j0 + warp; j < j1; j += 4) {
                const int kpos = k0 + j;
                const int dist = qpos - kpos;
                if (dist < 0 || dist >= window) continue;
                const int slot = kv_cap > 0 ? kpos % kv_cap : kpos;
                const size_t head_off = (size_t) slot * KD + (size_t) g * 128;
                const float4 k4 = kpos == qpos ? kcur : v_load4(kc + head_off, lane);
                float a = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                float l = wsum(a) * qk_scale;
                if (dist < extent) {
                    const float b = (lane < d_rel) ? r_lane * __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
                    l += wsum(b);
                }
                if (l > -INFINITY) {
                    const float4 v4 = kpos == qpos ? vcur : v_load4(vc + head_off, lane);
                    const float m_new = fmaxf(m_w, l);
                    const float alpha_ = (m_w == -INFINITY) ? 0.0f : expf(m_w - m_new);
                    const float beta = expf(l - m_new);
                    m_w = m_new;
                    s_w = s_w * alpha_ + beta;
                    o_w.x = o_w.x * alpha_ + v4.x * beta;
                    o_w.y = o_w.y * alpha_ + v4.y * beta;
                    o_w.z = o_w.z * alpha_ + v4.z * beta;
                    o_w.w = o_w.w * alpha_ + v4.w * beta;
                }
            }
            if (lane == 0) {
                sm_m[warp] = m_w;
                sm_s[warp] = s_w;
            }
            *(float4*) (&sm_o[warp][lane * 4]) = o_w;
            __syncthreads();
        }
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
        float* pp = part + ((size_t) h * NP + p) * 130;
        if (tid == 0) pp[0] = M, pp[1] = sum;
        pp[2 + tid] = o;
    }
    // == the head's last block of NP + 1: attn_combine_gate_kernel ==
    __shared__ unsigned int s_ticket;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntAttn + h], 1u);
    __syncthreads();
    if (s_ticket != NP) return;
    __threadfence();
    if (tid == 0) g_fuse_cnt[kFuseCntAttn + h] = 0;
    const float* pp = part + (size_t) h * NP * 130;
    __shared__ float cm[NP], cs[NP];
    for (int i = tid; i < NP; i += 128) cm[i] = pp[(size_t) i * 130], cs[i] = pp[(size_t) i * 130 + 1];
    float ov[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) ov[i] = pp[(size_t) i * 130 + 2 + tid];
    const float gt = gate[(size_t) h * 128 + tid];
    const float sk = ((volatile const float*) sink)[h];
    __syncthreads();
    float M = sk;
    for (int i = 0; i < NP; ++i) M = fmaxf(M, cm[i]);
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const float mp = cm[i];
        if (mp == -INFINITY) continue;
        const float f = expf(mp - M);
        den += cs[i] * f;
        o += ov[i] * f;
    }
    const float res = (o / den) * v_inv_scale;
    const float y = res * (1.0f / (1.0f + expf(-gt)));
    out[(size_t) h * 128 + tid] = y;
    out16[(size_t) h * 128 + tid] = f2h(y);
}

// == set 5: the attention residuals with every load at once ==
// attn_res_vec4_kernel<FUSED_RMS, 512> for one token - and with MOE, moe_attn_res_kernel's combine as its last state -
// with each thread's loads issued before its first sum: every state's float4s (kept in registers for the weighted sum;
// the old kernels read the states twice, 4 a round, and in moe_attn_res_kernel the combine's stores held back every
// load after them), q's, the rms weights and the combine's inputs.  The same sums in the same order: the same values
// to the bit (tests/ik2_fuse_test.cu).
constexpr int kResPf = 6;   // states at most (Inkling-2-Mini's residuals hold <= 5: blocks at 0, 6, 12, 18 + the stream)

// moe_combine_conv4's arithmetic and stores for float4 d4, from inputs already loaded (y: the K = 8 rows' float4s)
__device__ __forceinline__ float4 moe_apply4(const MoeCombine& mc, int C, int d4, const float* wt, const float4* yv,
                                             const float4* hv, const float4* wc, float4 rv4) {
    constexpr int K = 8;
    const float4 h0 = hv[0], h1 = hv[1], h2 = hv[2];
    float o[4], av[4];
    const float h0v[4] = {h0.x, h0.y, h0.z, h0.w}, h1v[4] = {h1.x, h1.y, h1.z, h1.w}, h2v[4] = {h2.x, h2.y, h2.z, h2.w};
    const float rvv[4] = {rv4.x, rv4.y, rv4.z, rv4.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float yk = i == 0 ? yv[k].x : i == 1 ? yv[k].y : i == 2 ? yv[k].z : yv[k].w;
            a += wt[k] * yk;
        }
        const float w0 = wc[i].x, w1 = wc[i].y, w2 = wc[i].z, w3 = wc[i].w;
        float b = w0 * h0v[i] + w1 * h1v[i] + w2 * h2v[i] + w3 * a;
        if (mc.res) b += rvv[i];
        o[i] = b, av[i] = a;
    }
    const float4 out = make_float4(o[0], o[1], o[2], o[3]);
    ((float4*) mc.out)[d4] = out;
    ((float4*) mc.hist)[d4] = h1;
    ((float4*) (mc.hist + C))[d4] = h2;
    ((float4*) (mc.hist + 2 * (size_t) C))[d4] = make_float4(av[0], av[1], av[2], av[3]);
    return out;
}

template <bool FUSED_RMS, bool MOE>
__global__ void __launch_bounds__(512)
    attn_res_pf_kernel(MoeCombine mc, ResStates st, int n, const float* q, int nq, int C, float res_eps,
                       float rms_eps, float* out, RmsOuts outs) {
    constexpr int BLOCK_THREADS = 512, NUM_WARPS = 16, K = 8;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float sc[3][kResMax];
    __shared__ float rs_rms[3];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c4 = C >> 2, d4_0 = tid, d4_1 = tid + BLOCK_THREADS;
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4;
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), one4 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const int ns = MOE ? n - 1 : n;   // the states read as they are (MOE: state n - 1 is the combine's, made here)
    // == every load ==
    float wt[K];
    float4 ya[K], yb[K], ha[3], hb[3], wca[4], wcb[4], ra = zero4, rb = zero4;
    if constexpr (MOE) {
        int iv[K];
#pragma unroll
        for (int k = 0; k < K; ++k) wt[k] = mc.wt[k], iv[k] = mc.inv[k];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float4* y4 = (const float4*) (mc.y + (size_t) iv[k] * C);
            ya[k] = in0 ? y4[d4_0] : zero4;
            yb[k] = in1 ? y4[d4_1] : zero4;
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const float4* h4 = (const float4*) (mc.hist + (size_t) i * C);
            ha[i] = in0 ? h4[d4_0] : zero4;
            hb[i] = in1 ? h4[d4_1] : zero4;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wca[i] = in0 ? ((const float4*) mc.w)[4 * d4_0 + i] : zero4;
            wcb[i] = in1 ? ((const float4*) mc.w)[4 * d4_1 + i] : zero4;
        }
        if (mc.res) {
            ra = in0 ? ((const float4*) mc.res)[d4_0] : zero4;
            rb = in1 ? ((const float4*) mc.res)[d4_1] : zero4;
        }
    }
    float4 xa[kResPf], xb[kResPf];
#pragma unroll
    for (int j = 0; j < kResPf; ++j) {
        const float4* x4 = (const float4*) st.st[j < ns ? j : 0];
        xa[j] = in0 && j < ns ? x4[d4_0] : zero4;
        xb[j] = in1 && j < ns ? x4[d4_1] : zero4;
    }
    float4 q0[3] = {}, q1[3] = {}, w0[3] = {}, w1[3] = {};
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) break;
        if (in0) q0[qi] = ((const float4*) (q + (size_t) qi * C))[d4_0];
        if (in1) q1[qi] = ((const float4*) (q + (size_t) qi * C))[d4_1];
        if constexpr (FUSED_RMS) {
            const float4* w4 = (const float4*) outs.w[qi];
            w0[qi] = w4 && in0 ? w4[d4_0] : one4;
            w1[qi] = w4 && in1 ? w4[d4_1] : one4;
        }
    }
    // == the combine (MOE): the stream's float4s of this thread, stored as moe_combine_conv stores them ==
    float4 hA = zero4, hB = zero4;
    if constexpr (MOE) {
        if (in0) hA = moe_apply4(mc, C, d4_0, wt, ya, ha, wca, ra);
        if (in1) hB = moe_apply4(mc, C, d4_1, wt, yb, hb, wcb, rb);
    }
    // == attn_res_vec4_kernel's sums, from registers ==
#pragma unroll
    for (int i = 0; i < kResPf; ++i) {
        if (i >= n) break;
        const float4 xv0 = MOE && i == ns ? hA : xa[i], xv1 = MOE && i == ns ? hB : xb[i];
        float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
        if (in0) {
            sum_sq += xv0.x * xv0.x + xv0.y * xv0.y + xv0.z * xv0.z + xv0.w * xv0.w;
            qd0 += xv0.x * q0[0].x + xv0.y * q0[0].y + xv0.z * q0[0].z + xv0.w * q0[0].w;
            if (nq > 1) qd1 += xv0.x * q0[1].x + xv0.y * q0[1].y + xv0.z * q0[1].z + xv0.w * q0[1].w;
            if (nq > 2) qd2 += xv0.x * q0[2].x + xv0.y * q0[2].y + xv0.z * q0[2].z + xv0.w * q0[2].w;
        }
        if (in1) {
            sum_sq += xv1.x * xv1.x + xv1.y * xv1.y + xv1.z * xv1.z + xv1.w * xv1.w;
            qd0 += xv1.x * q1[0].x + xv1.y * q1[0].y + xv1.z * q1[0].z + xv1.w * q1[0].w;
            if (nq > 1) qd1 += xv1.x * q1[1].x + xv1.y * q1[1].y + xv1.z * q1[1].z + xv1.w * q1[1].w;
            if (nq > 2) qd2 += xv1.x * q1[2].x + xv1.y * q1[2].y + xv1.z * q1[2].z + xv1.w * q1[2].w;
        }
        sum_sq = wsum(sum_sq);
        qd0 = wsum(qd0);
        if (nq > 1) qd1 = wsum(qd1);
        if (nq > 2) qd2 = wsum(qd2);
        if (lane == 0) {
            sh_red[warp][i * 4 + 0] = sum_sq;
            sh_red[warp][i * 4 + 1] = qd0;
            sh_red[warp][i * 4 + 2] = qd1;
            sh_red[warp][i * 4 + 3] = qd2;
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    __syncthreads();
    float4 o0[3] = {}, o1[3] = {};
#pragma unroll
    for (int i = 0; i < kResPf; ++i) {
        if (i >= n) break;
        const float4 xv0 = MOE && i == ns ? hA : xa[i], xv1 = MOE && i == ns ? hB : xb[i];
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            const float w_qi = sc[qi][i];
            if (in0) {
                o0[qi].x += w_qi * xv0.x;
                o0[qi].y += w_qi * xv0.y;
                o0[qi].z += w_qi * xv0.z;
                o0[qi].w += w_qi * xv0.w;
            }
            if (in1) {
                o1[qi].x += w_qi * xv1.x;
                o1[qi].y += w_qi * xv1.y;
                o1[qi].z += w_qi * xv1.z;
                o1[qi].w += w_qi * xv1.w;
            }
        }
    }
    if constexpr (!FUSED_RMS) {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float4* y4 = (float4*) (out + (size_t) qi * C);
            if (in0) y4[d4_0] = o0[qi];
            if (in1) y4[d4_1] = o1[qi];
        }
    } else {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float sq = 0.0f;
            if (in0) sq += o0[qi].x * o0[qi].x + o0[qi].y * o0[qi].y + o0[qi].z * o0[qi].z + o0[qi].w * o0[qi].w;
            if (in1) sq += o1[qi].x * o1[qi].x + o1[qi].y * o1[qi].y + o1[qi].z * o1[qi].z + o1[qi].w * o1[qi].w;
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] = rsqrtf(s_tot / (float) C + rms_eps);
            }
        }
        __syncthreads();
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            const float rs = rs_rms[qi];
            float4* y32_4 = (float4*) outs.y32[qi];
            uint2* y16_4 = (uint2*) outs.y16[qi];
            if (in0) {
                const float4 ww = w0[qi];
                const float r0 = o0[qi].x * rs * ww.x, r1 = o0[qi].y * rs * ww.y;
                const float r2 = o0[qi].z * rs * ww.z, r3 = o0[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_0] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_0] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
            if (in1) {
                const float4 ww = w1[qi];
                const float r0 = o1[qi].x * rs * ww.x, r1 = o1[qi].y * rs * ww.y;
                const float r2 = o1[qi].z * rs * ww.z, r3 = o1[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_1] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_1] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
        }
    }
}

// == set 8: loop_gate_apply_rms_kernel over more blocks, its loads first ==
// a block of 256 threads applies CPB channels (threads tid < CPB; were 256 a block: 9 blocks a gate stood on 9 CUs,
// 108 GB/s for one gate), each active thread's 64 F + 64 G values and x, m asked for before the gate vector's sums
// and the barrier; the sums in their order; the gate's last block takes the rms with all 256 threads as before
template <int CPB>
__global__ void __launch_bounds__(256)
    loop_gate_apply_rms_wide_kernel(LoopGateArgs a, int C, int R, float eps, const float* __restrict__ sc, RmsJobs rj,
                                    float rms_eps) {
    constexpr int Rank = 64;
    __shared__ float sh_g[kLoopRankMax];
    const int j = blockIdx.y, tid = threadIdx.x;
    const int c = (int) blockIdx.x * CPB + tid;
    const bool act = tid < CPB && c < C;
    float fv[Rank], gw[Rank], xv = 0.0f, mv = 0.0f;
    if (act) {
        const float* F = (const float*) a.F[j];
        const float* G = (const float*) a.G[j];
#pragma unroll
        for (int r = 0; r < Rank; ++r) fv[r] = F[(size_t) r * C + c], gw[r] = G[(size_t) r * C + c];
        xv = a.x[j][c], mv = a.m[j][c];
    }
    const float* o = sc + (size_t) j * (2 * R + 2);
    for (int i = tid; i < R; i += blockDim.x) {
        const float rx = rsqrtf(o[2 * R] / (float) C + eps), rm = rsqrtf(o[2 * R + 1] / (float) C + eps);
        sh_g[i] = rx * o[2 * i] + rm * o[2 * i + 1];
    }
    __syncthreads();
    if (act) {
        float f = 0.0f, gn = 0.0f;
#pragma unroll
        for (int r = 0; r < Rank; ++r) {
            const float gv = sh_g[r];
            f += fv[r] * gv, gn += gw[r] * gv;
        }
        a.y[j][c] = (1.0f - tanhf(f)) * xv + tanhf(gn) * mv;
    }
    __shared__ unsigned int s_ticket;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntGate + j], 1u);
    __syncthreads();
    if (s_ticket != gridDim.x - 1) return;
    __threadfence();
    if (tid == 0) g_fuse_cnt[kFuseCntGate + j] = 0;
    // (rms_multi_kernel's vec4 path, job j - loop_gate_apply_rms_kernel's tail as is)
    __shared__ float sh[256];
    const float* xr = rj.x[j];
    const float* w = rj.w[j];
    float* y32r = rj.y[j];
    float a2 = 0.0f;
    const int n4 = C >> 2;
    const float4* x4 = (const float4*) xr;
    for (int d4 = tid; d4 < n4; d4 += 256) {
        const float4 v = x4[d4];
        a2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float rs = rsqrtf(bsum(a2, sh) / (float) C + rms_eps);
    const float4* w4 = (const float4*) w;
    float4* y32_4 = (float4*) y32r;
    for (int d4 = tid; d4 < n4; d4 += 256) {
        const float4 v = x4[d4];
        const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
        y32_4[d4] = make_float4(r0, r1, r2, r3);
    }
}

// == set 9: the FFN input's attention residual (attn_res_rms_f16, one query) and the router in ONE launch ==
// router_route_kernel's grid (a block 4 rows) at 512 threads: warps 0-3 ask for their router row first (DRAM, as
// dot_f16_pf), then every block recomputes the residual from the states (attn_res_vec4_kernel<true, 512>'s sums at its
// 512 threads: the same values; the states are a few KB in L2) while those loads are out; block 0 writes its outputs,
// every block keeps them in LDS for its rows' sums (x as the router read it); the last block routes (router_route_kernel's
// tail).  A launch less a non-loop visit, and the router's loads under the residual's arithmetic.
constexpr int kFuseCntResRouter = 4;
template <int NPL>
__global__ void __launch_bounds__(512)
    res_router_route_kernel(ResStates st, int n, const float* __restrict__ q, int C, float res_eps, float rms_eps,
                            RmsOuts outs, const uint16_t* __restrict__ Wr, float* logits,
                            const float* __restrict__ bias, int E, int K, float scale,
                            const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                            const float* __restrict__ su_t, const float* __restrict__ sd_t, int* top, float* w,
                            const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt) {
    constexpr int BLOCK_THREADS = 512, NUM_WARPS = 16;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float sc[3][kResMax];
    __shared__ float rs_rms[3];
    __shared__ __align__(16) float sx[32 * NPL];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c4 = C >> 2, d4_0 = tid, d4_1 = tid + BLOCK_THREADS;
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4;
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), one4 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const int row = (int) blockIdx.x * 4 + warp;
    const bool dot = warp < 4 && row < E;
    // == every load: the block's router rows first, then the states, q, the rms weights ==
    uint16_t wv[NPL];
    if (dot) {
        const uint16_t* wr = Wr + (size_t) row * (32 * NPL);
#pragma unroll
        for (int j = 0; j < NPL; ++j) wv[j] = wr[lane + 32 * j];
    }
    float4 xa[kResPf], xb[kResPf];
#pragma unroll
    for (int j = 0; j < kResPf; ++j) {
        const float4* x4 = (const float4*) st.st[j < n ? j : 0];
        xa[j] = in0 && j < n ? x4[d4_0] : zero4;
        xb[j] = in1 && j < n ? x4[d4_1] : zero4;
    }
    const float4 q0 = in0 ? ((const float4*) q)[d4_0] : zero4, q1 = in1 ? ((const float4*) q)[d4_1] : zero4;
    const float4* w4 = (const float4*) outs.w[0];
    const float4 w0 = w4 && in0 ? w4[d4_0] : one4, w1 = w4 && in1 ? w4[d4_1] : one4;
    // == attn_res_vec4_kernel<true, 512>, T 1, nq 1 ==
#pragma unroll
    for (int i = 0; i < kResPf; ++i) {
        if (i >= n) break;
        float sum_sq = 0.0f, qd0 = 0.0f;
        if (in0) {
            const float4 xv0 = xa[i];
            sum_sq += xv0.x * xv0.x + xv0.y * xv0.y + xv0.z * xv0.z + xv0.w * xv0.w;
            qd0 += xv0.x * q0.x + xv0.y * q0.y + xv0.z * q0.z + xv0.w * q0.w;
        }
        if (in1) {
            const float4 xv1 = xb[i];
            sum_sq += xv1.x * xv1.x + xv1.y * xv1.y + xv1.z * xv1.z + xv1.w * xv1.w;
            qd0 += xv1.x * q1.x + xv1.y * q1.y + xv1.z * q1.z + xv1.w * q1.w;
        }
        sum_sq = wsum(sum_sq);
        qd0 = wsum(qd0);
        if (lane == 0) {
            sh_red[warp][i * 4 + 0] = sum_sq;
            sh_red[warp][i * 4 + 1] = qd0;
            sh_red[warp][i * 4 + 2] = 0.0f;
            sh_red[warp][i * 4 + 3] = 0.0f;
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, 1, C, res_eps, lane);
    __syncthreads();
    float4 o0 = zero4, o1 = zero4;
#pragma unroll
    for (int i = 0; i < kResPf; ++i) {
        if (i >= n) break;
        const float w_qi = sc[0][i];
        if (in0) {
            o0.x += w_qi * xa[i].x;
            o0.y += w_qi * xa[i].y;
            o0.z += w_qi * xa[i].z;
            o0.w += w_qi * xa[i].w;
        }
        if (in1) {
            o1.x += w_qi * xb[i].x;
            o1.y += w_qi * xb[i].y;
            o1.z += w_qi * xb[i].z;
            o1.w += w_qi * xb[i].w;
        }
    }
    {
        float sq = 0.0f;
        if (in0) sq += o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w;
        if (in1) sq += o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w;
        sq = wsum(sq);
        if (lane == 0) sh_red[warp][0] = sq;
    }
    __syncthreads();
    if (warp == 0) {
        float s_tot = (lane < NUM_WARPS) ? sh_red[lane][0] : 0.0f;
        s_tot = wsum(s_tot);
        if (lane == 0) rs_rms[0] = rsqrtf(s_tot / (float) C + rms_eps);
    }
    __syncthreads();
    {
        const float rs = rs_rms[0];
        float4* y32_4 = (float4*) outs.y32[0];
        uint2* y16_4 = (uint2*) outs.y16[0];
        if (in0) {
            const float r0 = o0.x * rs * w0.x, r1 = o0.y * rs * w0.y, r2 = o0.z * rs * w0.z, r3 = o0.w * rs * w0.w;
            if (blockIdx.x == 0) {
                y32_4[d4_0] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_0] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
            ((float4*) sx)[d4_0] = make_float4(r0, r1, r2, r3);
        }
        if (in1) {
            const float r0 = o1.x * rs * w1.x, r1 = o1.y * rs * w1.y, r2 = o1.z * rs * w1.z, r3 = o1.w * rs * w1.w;
            if (blockIdx.x == 0) {
                y32_4[d4_1] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_1] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
            ((float4*) sx)[d4_1] = make_float4(r0, r1, r2, r3);
        }
    }
    __syncthreads();
    // == the router's sums: warps 0-3 a row each (dot_f16_pf's order), x from LDS ==
    if (dot) {
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < NPL; ++j) a += h2f(wv[j]) * sx[lane + 32 * j];
        a = wsum(a);
        if (lane == 0) logits[row] = a;
    }
    // == router_route_kernel's tail: the last block routes (threads < E) ==
    __shared__ unsigned int s_ticket;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntResRouter], 1u);
    __syncthreads();
    if (s_ticket != gridDim.x - 1) return;
    __threadfence();
    __shared__ float val[128];
    __shared__ uint32_t picked[NUM_WARPS];
    __shared__ float sig[17];
    const int e = tid;
    const float lg = e < E ? ((volatile const float*) logits)[e] : 0.0f;
    const float sl = e < E ? 1.0f / (1.0f + expf(-lg)) : 0.0f;
    const float v = e < E ? sl + __ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) picked[warp] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(picked[ww]);
    if (pick) sig[pos] = sl;
    __syncthreads();
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
        g_fuse_cnt[kFuseCntResRouter] = 0;
    }
    __syncthreads();
    if (pick) {
        const float wk = sl / (sig[K] + 1e-20f) * scale;
        top[pos] = e;
        w[pos] = wk;
        blob[pos] = tab[e];
        tok[pos] = 0;
        inv[pos] = pos;
        sg[pos] = sg_t ? sg_t[e] : 1.0f;
        su[pos] = su_t ? su_t[e] : 1.0f;
        wt[pos] = wk * (sd_t ? sd_t[e] : 1.0f);
    }
}

// == set 10: a loop visit's FFN input residual (attn_res, one query) and its gate's dots in ONE launch ==
// loop_gate_dots_pf_kernel's grid (a block a rank row r, one gate) at 512 threads: threads 0-127 ask for their D row's
// values and m's first (DRAM), every block recomputes the residual (attn_res_vec4_kernel<false, 512>'s sums, nq 1)
// from the states in L2 while those are out, block 0 writes it (the gate's x), each block keeps it in LDS as the x of
// its row's sums (loop_gate_dots_pf_kernel's, threads 0-127)
template <int NPT>
__global__ void __launch_bounds__(512)
    res_gate_dots_kernel(ResStates st, int n, const float* __restrict__ q, int C, float res_eps, float* res_out,
                         const float* __restrict__ m, const float* __restrict__ D, int R, float* sc) {
    constexpr int BLOCK_THREADS = 512, NUM_WARPS = 16;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float scr[3][kResMax];
    __shared__ __align__(16) float sxl[128 * NPT];
    __shared__ float sh[4][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, r = blockIdx.x;
    const int c4 = C >> 2, d4_0 = tid, d4_1 = tid + BLOCK_THREADS;
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4, dots = tid < 128;
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // == every load: the D row's halves and m (threads 0-127), the states, q ==
    float mv[NPT], dx[NPT], dm[NPT];
    if (dots) {
        const float* Dr = D + (size_t) r * 2 * C;
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            const int c = tid + 128 * i;
            mv[i] = m[c], dx[i] = Dr[c], dm[i] = Dr[C + c];
        }
    }
    float4 xa[kResPf], xb[kResPf];
#pragma unroll
    for (int j = 0; j < kResPf; ++j) {
        const float4* x4 = (const float4*) st.st[j < n ? j : 0];
        xa[j] = in0 && j < n ? x4[d4_0] : zero4;
        xb[j] = in1 && j < n ? x4[d4_1] : zero4;
    }
    const float4 q0 = in0 ? ((const float4*) q)[d4_0] : zero4, q1 = in1 ? ((const float4*) q)[d4_1] : zero4;
    // == attn_res_vec4_kernel<false, 512>, T 1, nq 1 ==
#pragma unroll
    for (int i = 0; i < kResPf; ++i) {
        if (i >= n) break;
        float sum_sq = 0.0f, qd0 = 0.0f;
        if (in0) {
            const float4 xv0 = xa[i];
            sum_sq += xv0.x * xv0.x + xv0.y * xv0.y + xv0.z * xv0.z + xv0.w * xv0.w;
            qd0 += xv0.x * q0.x + xv0.y * q0.y + xv0.z * q0.z + xv0.w * q0.w;
        }
        if (in1) {
            const float4 xv1 = xb[i];
            sum_sq += xv1.x * xv1.x + xv1.y * xv1.y + xv1.z * xv1.z + xv1.w * xv1.w;
            qd0 += xv1.x * q1.x + xv1.y * q1.y + xv1.z * q1.z + xv1.w * q1.w;
        }
        sum_sq = wsum(sum_sq);
        qd0 = wsum(qd0);
        if (lane == 0) {
            sh_red[warp][i * 4 + 0] = sum_sq;
            sh_red[warp][i * 4 + 1] = qd0;
            sh_red[warp][i * 4 + 2] = 0.0f;
            sh_red[warp][i * 4 + 3] = 0.0f;
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, scr, n, 1, C, res_eps, lane);
    __syncthreads();
    float4 o0 = zero4, o1 = zero4;
#pragma unroll
    for (int i = 0; i < kResPf; ++i) {
        if (i >= n) break;
        const float w_qi = scr[0][i];
        if (in0) {
            o0.x += w_qi * xa[i].x;
            o0.y += w_qi * xa[i].y;
            o0.z += w_qi * xa[i].z;
            o0.w += w_qi * xa[i].w;
        }
        if (in1) {
            o1.x += w_qi * xb[i].x;
            o1.y += w_qi * xb[i].y;
            o1.z += w_qi * xb[i].z;
            o1.w += w_qi * xb[i].w;
        }
    }
    if (in0) {
        if (r == 0) ((float4*) res_out)[d4_0] = o0;
        ((float4*) sxl)[d4_0] = o0;
    }
    if (in1) {
        if (r == 0) ((float4*) res_out)[d4_1] = o1;
        ((float4*) sxl)[d4_1] = o1;
    }
    __syncthreads();
    // == loop_gate_dots_pf_kernel<NPT>, one gate, x from LDS ==
    if (dots) {
        float xv[NPT];
#pragma unroll
        for (int i = 0; i < NPT; ++i) xv[i] = sxl[tid + 128 * i];
        float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            ax += dx[i] * xv[i], am += dm[i] * mv[i];
            sx += xv[i] * xv[i], sm += mv[i] * mv[i];
        }
        ax = wsum(ax), am = wsum(am), sx = wsum(sx), sm = wsum(sm);
        if (lane == 0) sh[warp][0] = ax, sh[warp][1] = am, sh[warp][2] = sx, sh[warp][3] = sm;
    }
    __syncthreads();
    if (tid == 0) {
        sc[2 * r] = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
        sc[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
        if (r == 0) {
            sc[2 * R] = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
            sc[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
        }
    }
}

// == set 11: a loop visit's FFN gate rms in the router's launch ==
// loop_gate_apply_rms_kernel's apply alone (its rms tail - a fence, a ticket, one block's rms - goes to the router)
__global__ void __launch_bounds__(256)
    loop_gate_apply1_kernel(LoopGateArgs a, int C, int R, float eps, const float* __restrict__ sc) {
    constexpr int Rank = 64;
    __shared__ float sh_g[kLoopRankMax];
    const int j = blockIdx.y, tid = threadIdx.x;
    const float* o = sc + (size_t) j * (2 * R + 2);
    for (int i = tid; i < R; i += blockDim.x) {
        const float rx = rsqrtf(o[2 * R] / (float) C + eps), rm = rsqrtf(o[2 * R + 1] / (float) C + eps);
        sh_g[i] = rx * o[2 * i] + rm * o[2 * i + 1];
    }
    __syncthreads();
    const int c = (int) blockIdx.x * blockDim.x + tid;
    if (c < C) {
        const float* F = (const float*) a.F[j];
        const float* G = (const float*) a.G[j];
        float f = 0.0f, gn = 0.0f;
#pragma unroll
        for (int r = 0; r < Rank; ++r) {
            const float gv = sh_g[r];
            f += F[(size_t) r * C + c] * gv, gn += G[(size_t) r * C + c] * gv;
        }
        a.y[j][c] = (1.0f - tanhf(f)) * a.x[j][c] + tanhf(gn) * a.m[j][c];
    }
}

// router_route_kernel's grid at 256 threads: warps 0-3 ask for their router row first, then every block takes the
// rms of the gate's y (rms_multi_kernel's vec4 path at its 256 threads: the same tree, the same values), block 0
// writes it to x_out, each block keeps it in LDS for its rows' sums; the last block routes
template <int NPL>
__global__ void __launch_bounds__(256)
    rms_router_route_kernel(const float* __restrict__ y, const float* __restrict__ nw, int C, float rms_eps,
                            float* x_out, const uint16_t* __restrict__ Wr, float* logits,
                            const float* __restrict__ bias, int E, int K, float scale,
                            const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                            const float* __restrict__ su_t, const float* __restrict__ sd_t, int* top, float* w,
                            const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt) {
    constexpr int NUM_WARPS = 8;
    __shared__ float sh[256];
    __shared__ __align__(16) float sxl[32 * NPL];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row = (int) blockIdx.x * 4 + warp;
    const bool dot = warp < 4 && row < E;
    uint16_t wv[NPL];
    if (dot) {
        const uint16_t* wr = Wr + (size_t) row * (32 * NPL);
#pragma unroll
        for (int j = 0; j < NPL; ++j) wv[j] = wr[lane + 32 * j];
    }
    // == rms_multi_kernel's vec4 path (x = y, w = nw), 256 threads ==
    {
        const int n4 = C >> 2;
        const float4* x4 = (const float4*) y;
        float a2 = 0.0f;
        for (int d4 = tid; d4 < n4; d4 += 256) {
            const float4 v = x4[d4];
            a2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float rs = rsqrtf(bsum(a2, sh) / (float) C + rms_eps);
        const float4* w4 = (const float4*) nw;
        for (int d4 = tid; d4 < n4; d4 += 256) {
            const float4 v = x4[d4];
            const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
            if (blockIdx.x == 0) ((float4*) x_out)[d4] = make_float4(r0, r1, r2, r3);
            ((float4*) sxl)[d4] = make_float4(r0, r1, r2, r3);
        }
    }
    __syncthreads();
    if (dot) {
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < NPL; ++j) a += h2f(wv[j]) * sxl[lane + 32 * j];
        a = wsum(a);
        if (lane == 0) logits[row] = a;
    }
    // == router_route_kernel's tail ==
    __shared__ unsigned int s_ticket;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntResRouter + 1], 1u);
    __syncthreads();
    if (s_ticket != gridDim.x - 1) return;
    __threadfence();
    __shared__ float val[128];
    __shared__ uint32_t picked[NUM_WARPS];
    __shared__ float sig[17];
    const int e = tid;
    const float lg = e < E ? ((volatile const float*) logits)[e] : 0.0f;
    const float sl = e < E ? 1.0f / (1.0f + expf(-lg)) : 0.0f;
    const float v = e < E ? sl + __ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) picked[warp] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(picked[ww]);
    if (pick) sig[pos] = sl;
    __syncthreads();
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
        g_fuse_cnt[kFuseCntResRouter + 1] = 0;
    }
    __syncthreads();
    if (pick) {
        const float wk = sl / (sig[K] + 1e-20f) * scale;
        top[pos] = e;
        w[pos] = wk;
        blob[pos] = tab[e];
        tok[pos] = 0;
        inv[pos] = pos;
        sg[pos] = sg_t ? sg_t[e] : 1.0f;
        su[pos] = su_t ? su_t[e] : 1.0f;
        wt[pos] = wk * (sd_t ? sd_t[e] : 1.0f);
    }
}

// == set 12: a loop visit's FFN gate apply (+ its rms) and the router in ONE launch ==
// blocks [0, na) are loop_gate_apply_rms_kernel's (one gate, j 0; its last block's rms writes rj.y[0] and then raises
// a flag), blocks [na, na + nr) are router_route_kernel's: they ask for their rows first, wait for the flag (the grid is
// small - every block resident at once - and the apply blocks come first in dispatch order), then read rj.y[0] (after
// an acquire fence, glc loads) and sum as dot_f16_pf; the last router block routes and lowers the flag
constexpr int kFuseCntGateRouter = 6;
__device__ unsigned int g_fuse_flag[4] = {};
template <int NPL>
__global__ void __launch_bounds__(256)
    gate_router_kernel(LoopGateArgs a, int C, int R, float eps, const float* __restrict__ sc, RmsJobs rj,
                       float rms_eps, int na, const uint16_t* __restrict__ Wr, float* logits,
                       const float* __restrict__ bias, int E, int K, float scale,
                       const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                       const float* __restrict__ su_t, const float* __restrict__ sd_t, int* top, float* w,
                       const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt) {
    const int tid = threadIdx.x;
    __shared__ unsigned int s_ticket;
    if ((int) blockIdx.x < na) {
        // == loop_gate_apply_rms_kernel, j 0, its grid the first na blocks ==
        constexpr int Rank = 64;
        __shared__ float sh_g[kLoopRankMax];
        const float* o = sc;
        for (int i = tid; i < R; i += blockDim.x) {
            const float rx = rsqrtf(o[2 * R] / (float) C + eps), rm = rsqrtf(o[2 * R + 1] / (float) C + eps);
            sh_g[i] = rx * o[2 * i] + rm * o[2 * i + 1];
        }
        __syncthreads();
        const int c = (int) blockIdx.x * blockDim.x + tid;
        if (c < C) {
            const float* F = (const float*) a.F[0];
            const float* G = (const float*) a.G[0];
            float f = 0.0f, gn = 0.0f;
#pragma unroll
            for (int r = 0; r < Rank; ++r) {
                const float gv = sh_g[r];
                f += F[(size_t) r * C + c] * gv, gn += G[(size_t) r * C + c] * gv;
            }
            a.y[0][c] = (1.0f - tanhf(f)) * a.x[0][c] + tanhf(gn) * a.m[0][c];
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntGate], 1u);
        __syncthreads();
        if (s_ticket != (unsigned) na - 1) return;
        __threadfence();
        if (tid == 0) g_fuse_cnt[kFuseCntGate] = 0;
        __shared__ float sh[256];
        const float* xr = rj.x[0];
        const float* wn = rj.w[0];
        float* y32r = rj.y[0];
        float a2 = 0.0f;
        const int n4 = C >> 2;
        const float4* x4 = (const float4*) xr;
        for (int d4 = tid; d4 < n4; d4 += 256) {
            const float4 v = x4[d4];
            a2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float rs = rsqrtf(bsum(a2, sh) / (float) C + rms_eps);
        const float4* w4 = (const float4*) wn;
        float4* y32_4 = (float4*) y32r;
        for (int d4 = tid; d4 < n4; d4 += 256) {
            const float4 v = x4[d4];
            const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
            y32_4[d4] = make_float4(r0, r1, r2, r3);
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicExch(&g_fuse_flag[0], 1u);
        return;
    }
    // == a router block: its rows first, then the flag, then dot_f16_pf's sums on rj.y[0] ==
    constexpr int NUM_WARPS = 8;
    const int lane = tid & 31, warp = tid >> 5;
    const int row = ((int) blockIdx.x - na) * 4 + warp;
    const bool dot = warp < 4 && row < E;
    uint16_t wv[NPL];
    if (dot) {
        const uint16_t* wr = Wr + (size_t) row * (32 * NPL);
#pragma unroll
        for (int j = 0; j < NPL; ++j) wv[j] = wr[lane + 32 * j];
    }
    if (tid == 0)
        while (atomicAdd(&g_fuse_flag[0], 0u) == 0u) __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __threadfence();
    if (dot) {
        const volatile float* x = rj.y[0];
        float xv[NPL];
#pragma unroll
        for (int j = 0; j < NPL; ++j) xv[j] = x[lane + 32 * j];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < NPL; ++j) acc += h2f(wv[j]) * xv[j];
        acc = wsum(acc);
        if (lane == 0) logits[row] = acc;
    }
    // == router_route_kernel's tail over the router blocks ==
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntGateRouter], 1u);
    __syncthreads();
    if (s_ticket != gridDim.x - (unsigned) na - 1) return;
    __threadfence();
    __shared__ float val[128];
    __shared__ uint32_t picked[NUM_WARPS];
    __shared__ float sig[17];
    const int e = tid;
    const float lg = e < E ? ((volatile const float*) logits)[e] : 0.0f;
    const float sl = e < E ? 1.0f / (1.0f + expf(-lg)) : 0.0f;
    const float v = e < E ? sl + __ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) picked[warp] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(picked[ww]);
    if (pick) sig[pos] = sl;
    __syncthreads();
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
        g_fuse_cnt[kFuseCntGateRouter] = 0;
        g_fuse_flag[0] = 0u;
    }
    __syncthreads();
    if (pick) {
        const float wk = sl / (sig[K] + 1e-20f) * scale;
        top[pos] = e;
        w[pos] = wk;
        blob[pos] = tab[e];
        tok[pos] = 0;
        inv[pos] = pos;
        sg[pos] = sg_t ? sg_t[e] : 1.0f;
        su[pos] = su_t ? su_t[e] : 1.0f;
        wt[pos] = wk * (sd_t ? sd_t[e] : 1.0f);
    }
}

}  // namespace

bool fuse_mv_f16_pair(const uint16_t* W0, const float* x0, float* y0, int N0, const uint16_t* W1, const float* x1,
                      float* y1, int N1, int K, cudaStream_t s) {
    if (K != 32 * kFuseNpl || N0 <= 0 || N1 < 0) return false;
    const int nb0 = (N0 + 3) / 4, nb1 = (N1 + 3) / 4;
    mv_f16_pf_kernel<kFuseNpl><<<(unsigned) (nb0 + nb1), 128, 0, s>>>(MvF16Job{W0, x0, y0, N0},
                                                                      MvF16Job{W1, x1, y1, N1}, nb0);
    check("fuse_mv_f16_pair");
    return true;
}

bool fuse_router_route(const uint16_t* Wr, const float* x, int C, float* logits, const float* bias, int E, int K,
                       float scale, const uint8_t* const* tab, const float* sg_t, const float* su_t, const float* sd_t,
                       int* top, float* w, const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt,
                       cudaStream_t s, bool lds) {
    if (C != 32 * kFuseNpl || E < 1 || E > 128 || K < 1 || K > 16 || K > E) return false;
    if (lds && !((uintptr_t) Wr & 15) && !((uintptr_t) x & 15))
        router_route_kernel<kFuseNpl, true><<<(unsigned) ((E + 3) / 4), 128, 0, s>>>(
            Wr, x, logits, bias, E, K, scale, tab, sg_t, su_t, sd_t, top, w, blob, tok, inv, sg, su, wt);
    else
        router_route_kernel<kFuseNpl, false><<<(unsigned) ((E + 3) / 4), 128, 0, s>>>(
            Wr, x, logits, bias, E, K, scale, tab, sg_t, su_t, sd_t, top, w, blob, tok, inv, sg, su, wt);
    check("fuse_router_route");
    return true;
}

bool fuse_prep_qkv_r_f16_at(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                            uint16_t* vc, float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip,
                            float rel_clip, float eps, const int* pos, bool log_scale, float alpha, float n_floor,
                            int kv_cap, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w, const float* sink_x,
                            int sink_k, float* sink_y) {
    if (sink_w && sink_k != 32 * kFuseNpl) return false;
    prep_qkv_r_pf_kernel<uint16_t, kFuseNpl><<<(unsigned) (2 * H + 2 * KV + (sink_w ? H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, H, KV, D, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, kv_cap, 1.0f, 1.0f, pos, q16, sink_w, sink_x, sink_y);
    check("fuse_prep_qkv_r_f16_at");
    return true;
}

bool fuse_attn_decode_f16_at_gate(const float* q, const float* r, const float* sink, const uint16_t* kc,
                                  const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int P,
                                  float qk_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                                  const float* gate, float* out, uint16_t* out16, cudaStream_t s, bool skip) {
    if (P != 16 && P != 128) return false;
    attn_split_kernel<uint16_t><<<dim3(H, P), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj,
                                                            d_rel, extent, window, scratch, pos);
    check("fuse_attn_decode_f16_at_gate (split)");
    if (P == 16)
        attn_combine_gate_kernel<16><<<H, 128, 0, s>>>(sink, scratch, 1.0f, gate, out, out16);
    else if (skip)
        attn_combine_gate_kernel<128, true><<<H, 128, 0, s>>>(sink, scratch, 1.0f, gate, out, out16, pos, window);
    else
        attn_combine_gate_kernel<128><<<H, 128, 0, s>>>(sink, scratch, 1.0f, gate, out, out16);
    check("fuse_attn_decode_f16_at_gate (combine)");
    return true;
}

// fuse_loop_gates_rms: loop_gates (one token, F32 weights, rank 64) then the rms of each gate's y - rj's job j, x = gate
// j's y (rms_multi; or rms for one gate) - in two launches (were four)
bool fuse_loop_gates_rms(const LoopGate* gt, int n, int C, int R, float eps, float* sc, const RmsJobs& rj,
                         float rms_eps, cudaStream_t s, bool wide) {
    if (n < 1 || n > 3 || R != 64 || C != 2304) return false;
    LoopGateArgs a{};
    for (int j = 0; j < n; ++j) {
        if (gt[j].compact[0] || gt[j].compact[1] || gt[j].compact[2]) return false;
        if (rj.x[j] != gt[j].y || ((uintptr_t) rj.x[j] & 15) || ((uintptr_t) rj.y[j] & 15) ||
            ((uintptr_t) rj.w[j] & 15))
            return false;
        a.x[j] = gt[j].x, a.m[j] = gt[j].m, a.y[j] = gt[j].y, a.D[j] = gt[j].D, a.F[j] = gt[j].F, a.G[j] = gt[j].G;
    }
    loop_gate_dots_pf_kernel<18><<<dim3(R, n), 128, 0, s>>>(a, C, R, sc);
    check("fuse_loop_gates_rms (dots)");
    if (wide)
        loop_gate_apply_rms_wide_kernel<64><<<dim3((C + 63) / 64, n), 256, 0, s>>>(a, C, R, eps, sc, rj, rms_eps);
    else
        loop_gate_apply_rms_kernel<<<dim3((C + 255) / 256, n), 256, 0, s>>>(a, C, R, eps, sc, rj, rms_eps);
    check("fuse_loop_gates_rms (apply, rms)");
    return true;
}

// fuse_attn_one: attn_decode_f16_at + gate_mul_f16 in ONE launch (the split's last block of a head combines)
bool fuse_attn_one_f16_at_gate(const float* q, const float* r, const float* sink, const uint16_t* kc,
                               const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale,
                               const float* proj, int d_rel, int extent, int window, float* scratch, const float* gate,
                               float* out, uint16_t* out16, cudaStream_t s) {
    // (P 128 - a full layer - stays two launches: the last block's 128 partials in registers cost the 3072 split
    //  blocks their occupancy - 34 vs 29 us at position 300)
    if (P != 16 || H > 56) return false;
    if (P == 16)
        attn_split_combine_gate_kernel<uint16_t, 16><<<dim3(H, 16), 128, 0, s>>>(
            q, r, kc, vc, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos, sink, 1.0f, gate, out,
            out16);
    else
        attn_split_combine_gate_kernel<uint16_t, 128><<<dim3(H, 128), 128, 0, s>>>(
            q, r, kc, vc, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos, sink, 1.0f, gate, out,
            out16);
    check("fuse_attn_one_f16_at_gate");
    return true;
}

// fuse_moe_attn_res: moe_combine_conv (K 8; out = the stream) and then attn_res (outs null: res_out) or attn_res_rms_f16
// (outs given) of one token over st, whose last state must be out - one launch instead of two
bool fuse_moe_attn_res(const float* y, const int* inv, const float* wt, int K, int C, float* hist, const float* w,
                       const float* res, float* out, const ResStates& st, int n, const float* q, int nq, float res_eps,
                       float rms_eps, float* res_out, const RmsOuts* outs, cudaStream_t s, bool pf) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    if (K != 8 || C != 2304 || n < 1 || n > kResMax || nq < 1 || nq > 3 || st.st[n - 1] != out) return false;
    if (!al(y) || !al(hist) || !al(w) || (res && !al(res)) || !al(out) || !al(q)) return false;
    for (int i = 0; i < n; ++i)
        if (!al(st.st[i])) return false;
    if (outs) {
        for (int qi = 0; qi < nq; ++qi)
            if ((outs->w[qi] && !al(outs->w[qi])) || (outs->y32[qi] && !al(outs->y32[qi])) ||
                (outs->y16[qi] && ((uintptr_t) outs->y16[qi] & 7)))
                return false;
        if (pf && n <= kResPf)
            attn_res_pf_kernel<true, true><<<1, 512, 0, s>>>(MoeCombine{y, inv, wt, hist, w, res, out}, st, n, q, nq, C,
                                                            res_eps, rms_eps, nullptr, *outs);
        else
            moe_attn_res_kernel<true><<<1, 512, 0, s>>>(MoeCombine{y, inv, wt, hist, w, res, out}, st, n, q, nq, C,
                                                        res_eps, rms_eps, nullptr, *outs);
    } else {
        if (!al(res_out)) return false;
        if (pf && n <= kResPf)
            attn_res_pf_kernel<false, true><<<1, 512, 0, s>>>(MoeCombine{y, inv, wt, hist, w, res, out}, st, n, q, nq,
                                                             C, res_eps, rms_eps, res_out, RmsOuts{});
        else
            moe_attn_res_kernel<false><<<1, 512, 0, s>>>(MoeCombine{y, inv, wt, hist, w, res, out}, st, n, q, nq, C,
                                                         res_eps, rms_eps, res_out, RmsOuts{});
    }
    check("fuse_moe_attn_res");
    return true;
}

// fuse_prep_attn_f16_at_gate: prep_qkv_r_f16_at (with sinks) + attn_decode_f16_at + gate_mul_f16 of one token in ONE
// launch, for P 16 (a window layer); q and r are not written back clipped (nothing reads them after the attention)
bool fuse_prep_attn_f16_at_gate(const float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                                uint16_t* vc, const float* r, const float* r_clip, int H, int KV, int D, int d_rel,
                                float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                                float n_floor, int kv_cap, const uint16_t* sink_w, const float* sink_x, int sink_k,
                                float* sink_y, float qk_scale, const float* proj, int extent, int window, int P,
                                float* scratch, const float* gate, float* out, uint16_t* out16, cudaStream_t s) {
    auto al = [](const void* p, int b) { return ((uintptr_t) p & (b - 1)) == 0; };
    if (P != 16 || D != 128 || d_rel > 32 || H > 56 || H % KV || !sink_w || sink_k != 32 * kFuseNpl) return false;
    if (!al(q, 16) || (q_clip && !al(q_clip, 16)) || !al(k_in, 16) || !al(v_in, 16) || !al(kc, 8) || !al(vc, 8))
        return false;
    attn_prep_one_kernel<16><<<dim3(H, 17), 128, 0, s>>>(q, q_clip, k_in, kc, v_in, vc, r, r_clip, H, KV, d_rel,
                                                          qk_clip, rel_clip, eps, pos, log_scale, alpha, n_floor,
                                                          kv_cap, sink_w, sink_x, sink_y, qk_scale, proj, extent,
                                                          window, scratch, 1.0f, gate, out, out16);
    check("fuse_prep_attn_f16_at_gate");
    return true;
}

// fuse_attn_res_pf (set 5): attn_res (outs null: into res_out) or attn_res_rms_f16 (outs given) of one token over st
// (n <= 6 states), every load at once - one launch, the same values
bool fuse_attn_res_pf(const ResStates& st, int n, const float* q, int nq, int C, float res_eps, float rms_eps,
                      float* res_out, const RmsOuts* outs, cudaStream_t s) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    if (C != 2304 || n < 1 || n > kResPf || nq < 1 || nq > 3 || !al(q)) return false;
    for (int i = 0; i < n; ++i)
        if (!al(st.st[i])) return false;
    if (outs) {
        for (int qi = 0; qi < nq; ++qi)
            if ((outs->w[qi] && !al(outs->w[qi])) || (outs->y32[qi] && !al(outs->y32[qi])) ||
                (outs->y16[qi] && ((uintptr_t) outs->y16[qi] & 7)))
                return false;
        attn_res_pf_kernel<true, false><<<1, 512, 0, s>>>(MoeCombine{}, st, n, q, nq, C, res_eps, rms_eps, nullptr,
                                                         *outs);
    } else {
        if (!al(res_out)) return false;
        attn_res_pf_kernel<false, false><<<1, 512, 0, s>>>(MoeCombine{}, st, n, q, nq, C, res_eps, rms_eps, res_out,
                                                          RmsOuts{});
    }
    check("fuse_attn_res_pf");
    return true;
}

// fuse_res_router_route (set 9): attn_res_rms_f16 (one token, one query; outs' y32 given) and fuse_router_route on its
// y32 in one launch
bool fuse_res_router_route(const ResStates& st, int n, const float* q, int C, float res_eps, float rms_eps,
                           const RmsOuts& outs, const uint16_t* Wr, float* logits, const float* bias, int E, int K,
                           float scale, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                           const float* sd_t, int* top, float* w, const uint8_t** blob, int* tok, int* inv, float* sg,
                           float* su, float* wt, cudaStream_t s) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    if (C != 32 * kFuseNpl || n < 1 || n > kResPf || E < 1 || E > 128 || K < 1 || K > 16 || K > E || !al(q)) return false;
    for (int i = 0; i < n; ++i)
        if (!al(st.st[i])) return false;
    if ((outs.w[0] && !al(outs.w[0])) || !outs.y32[0] || !al(outs.y32[0]) || ((uintptr_t) outs.y16[0] & 7)) return false;
    res_router_route_kernel<kFuseNpl><<<(unsigned) ((E + 3) / 4), 512, 0, s>>>(st, n, q, C, res_eps, rms_eps, outs, Wr,
                                                                              logits, bias, E, K, scale, tab, sg_t,
                                                                              su_t, sd_t, top, w, blob, tok, inv, sg,
                                                                              su, wt);
    check("fuse_res_router_route");
    return true;
}

// fuse_res_loop_gate_rms (set 10): attn_res (one token, one query) into res_out = the gate's x, then
// fuse_loop_gates_rms of that one gate - the residual rides the dots' launch (two launches, were three)
bool fuse_res_loop_gate_rms(const ResStates& st, int n, const float* q, float res_eps, float* res_out,
                            const LoopGate& gt, int C, int R, float eps, float* sc, const RmsJobs& rj, float rms_eps,
                            cudaStream_t s, bool wide) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    if (C != 32 * kFuseNpl || R != 64 || n < 1 || n > kResPf || !al(q) || !al(res_out) || gt.x != res_out) return false;
    if (gt.compact[0] || gt.compact[1] || gt.compact[2]) return false;
    if (rj.x[0] != gt.y || !al(rj.x[0]) || !al(rj.y[0]) || !al(rj.w[0])) return false;
    for (int i = 0; i < n; ++i)
        if (!al(st.st[i])) return false;
    LoopGateArgs a{};
    a.x[0] = gt.x, a.m[0] = gt.m, a.y[0] = gt.y, a.D[0] = gt.D, a.F[0] = gt.F, a.G[0] = gt.G;
    res_gate_dots_kernel<18><<<(unsigned) R, 512, 0, s>>>(st, n, q, C, res_eps, res_out, gt.m, gt.D, R, sc);
    check("fuse_res_loop_gate_rms (residual, dots)");
    if (wide)
        loop_gate_apply_rms_wide_kernel<64><<<dim3((C + 63) / 64, 1), 256, 0, s>>>(a, C, R, eps, sc, rj, rms_eps);
    else
        loop_gate_apply_rms_kernel<<<dim3((C + 255) / 256, 1), 256, 0, s>>>(a, C, R, eps, sc, rj, rms_eps);
    check("fuse_res_loop_gate_rms (apply, rms)");
    return true;
}

// fuse_loop_gate_rms_router (set 11): one loop gate (dots; apply into gt.y, a scratch apart from gt.x and x_out), then
// the rms of gt.y (weights nw) into x_out and fuse_router_route on it in one launch - fuse_loop_gates_rms (rj
// {gt.y, nw, x_out}) + fuse_router_route's values; three launches as before, the rms tail off the gate's critical path
bool fuse_loop_gate_rms_router(const LoopGate& gt, int C, int R, float eps, float* sc, const float* nw, float rms_eps,
                               float* x_out, const uint16_t* Wr, float* logits, const float* bias, int E, int K,
                               float scale, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                               const float* sd_t, int* top, float* w, const uint8_t** blob, int* tok, int* inv,
                               float* sg, float* su, float* wt, cudaStream_t s) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    if (C != 32 * kFuseNpl || R != 64 || E < 1 || E > 128 || K < 1 || K > 16 || K > E) return false;
    if (gt.compact[0] || gt.compact[1] || gt.compact[2] || gt.y == gt.x || gt.y == x_out) return false;
    if (!al(gt.y) || !al(x_out) || (nw && !al(nw))) return false;
    LoopGateArgs a{};
    a.x[0] = gt.x, a.m[0] = gt.m, a.y[0] = gt.y, a.D[0] = gt.D, a.F[0] = gt.F, a.G[0] = gt.G;
    loop_gate_dots_pf_kernel<18><<<dim3(R, 1), 128, 0, s>>>(a, C, R, sc);
    check("fuse_loop_gate_rms_router (dots)");
    loop_gate_apply1_kernel<<<dim3((C + 255) / 256, 1), 256, 0, s>>>(a, C, R, eps, sc);
    check("fuse_loop_gate_rms_router (apply)");
    rms_router_route_kernel<kFuseNpl><<<(unsigned) ((E + 3) / 4), 256, 0, s>>>(gt.y, nw, C, rms_eps, x_out, Wr, logits,
                                                                              bias, E, K, scale, tab, sg_t, su_t, sd_t,
                                                                              top, w, blob, tok, inv, sg, su, wt);
    check("fuse_loop_gate_rms_router (rms, router)");
    return true;
}

// fuse_loop_gate_router (set 12): fuse_loop_gates_rms of one gate (rj's job 0) and fuse_router_route on rj.y[0] in two
// launches (were three): the dots, then the apply + rms and the router in one grid
bool fuse_loop_gate_router(const LoopGate& gt, int C, int R, float eps, float* sc, const RmsJobs& rj, float rms_eps,
                           const uint16_t* Wr, float* logits, const float* bias, int E, int K, float scale,
                           const uint8_t* const* tab, const float* sg_t, const float* su_t, const float* sd_t, int* top,
                           float* w, const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt,
                           cudaStream_t s) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    if (C != 32 * kFuseNpl || R != 64 || E < 1 || E > 128 || K < 1 || K > 16 || K > E) return false;
    if (gt.compact[0] || gt.compact[1] || gt.compact[2] || rj.x[0] != gt.y) return false;
    if (!al(rj.x[0]) || !al(rj.y[0]) || (rj.w[0] && !al(rj.w[0]))) return false;
    LoopGateArgs a{};
    a.x[0] = gt.x, a.m[0] = gt.m, a.y[0] = gt.y, a.D[0] = gt.D, a.F[0] = gt.F, a.G[0] = gt.G;
    loop_gate_dots_pf_kernel<18><<<dim3(R, 1), 128, 0, s>>>(a, C, R, sc);
    check("fuse_loop_gate_router (dots)");
    const int na = (C + 255) / 256, nr = (E + 3) / 4;
    gate_router_kernel<kFuseNpl><<<(unsigned) (na + nr), 256, 0, s>>>(a, C, R, eps, sc, rj, rms_eps, na, Wr, logits, bias,
                                                                     E, K, scale, tab, sg_t, su_t, sd_t, top, w, blob,
                                                                     tok, inv, sg, su, wt);
    check("fuse_loop_gate_router (apply, rms, router)");
    return true;
}
