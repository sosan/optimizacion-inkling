// src/ik2/ik2_kernels.cuh - maya-ik2's kernels: the ops of Inkling 2 (thinkingmachines/Inkling-2-Small) that Maya's
// other engines do not have - the short convolutions, the attention residuals, the attention's clips, relative bias and
// sink, the softcapped SwiGLU with the experts' scales - and the decodes of the weights an `inkling2` GGUF holds
// (NVFP4, Q5_K, Q6_K, Q8_0).  The arithmetic follows ref/inkling2.py (transformers' modeling_inkling2.py), in F32; the
// experts' GEMMs of the prompt path take FP16 rows (an NVFP4 weight is exact in FP16).  docs/INKLING2.md has the model.
//
// Layouts: rows of tokens; a token's heads back to back ([T][H][D]); the K / V caches by position, [ctx][Hkv][D] F32.
// CUDA only for now (no HIP mapping yet).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ik2::k {

/// Paged global KV pool constants: 256 positions per page.
constexpr int kKvPageShift = 8;
constexpr int kKvPageSize = 1 << kKvPageShift;
constexpr int kKvPageMask = kKvPageSize - 1;

/// y = x * rsqrt(mean(x^2) + eps) * w per row of D (w null: no weight).  x and y may alias.
void rms(const float* x, const float* w, int rows, int D, float eps, float* y, cudaStream_t s);
/// rms of n <= 3 vectors [D] in one launch, each with its own weights: y[i] = rms(x[i]) * w[i] (y may be x).
struct RmsJobs {
    const float* x[3];
    const float* w[3];
    float* y[3];
};
void rms_multi(const RmsJobs& jobs, int n, int D, float eps, cudaStream_t s);
/// Fused RMSNorm + FP16 output (and optional F32 output y32 if non-null; y32 may alias x).
void rms_f16(const float* x, const float* w, int rows, int D, float eps, uint16_t* y16, float* y32, cudaStream_t s);

/// The scaled L2 clip, a group at a time: x [rows][G][D]; v = x * scale[g] (scale [G][D], null: 1);
/// y = v * min(1, th / (|v| + 1e-6)) * tau, tau = 1 + alpha ln max(1, (pos[row] + 1) / n_floor) with pos given (a full
/// layer's log scaling of q and r), else 1.  x and y may alias.
void l2clip(const float* x, const float* scale, int rows, int G, int D, float th, const int* pos, float alpha,
            float n_floor, float* y, cudaStream_t s);

/// The folded short conv (docs/INKLING2.md): y[t][c] = sum_j w[c][j] xp[t + j][c], xp = hist's 3 rows then x's T
/// (hist: the conv's last 3 inputs, oldest first; zeros at a sequence's start), taps = 4; with dyn [T][heads * 4]
/// (a full layer's k / v) + sum_lag dyn[t][h(c) * 4 + lag] xp[t + 3 - lag][c], h(c) = c / (C / heads).
/// Then hist = the last 3 rows of xp.  y must not alias x.
void conv4(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C, float* y,
           cudaStream_t s);
/// Residual-fused short conv: same as conv4, plus residual[t][c] added to y[t][c] when residual != nullptr.
void conv4_res(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C,
               const float* residual, float* y, cudaStream_t s);

/// The attention residual: the n states st[i] ([T][C] each, n <= kResMax; the block states then the stream) each
/// RMS-normed without a weight (eps), p = softmax_i(q . s_i / sqrt(C)) for each of the nq queries (q [nq][C]),
/// out [nq][T][C] = sum_i p_i s_i.
constexpr int kResMax = 12;
struct ResStates {
    const float* st[kResMax];
};
void attn_res(const ResStates& states, int n, const float* q, int nq, int T, int C, float eps, float* out,
              cudaStream_t s);
/// One token's attn_res for jobs x nq queries in one launch (a block a job): job j's queries are q + j nq C, its
/// output out + j nq C.
void attn_res_jobs(const ResStates& states, int n, const float* q, int nq, int jobs, int C, float eps, float* out,
                   cudaStream_t s);
struct RmsOuts {
    const float* w[3] = {};
    uint16_t* y16[3] = {};
    float* y32[3] = {};
};
/// Fused attn_res + per-query RMSNorm (res_eps for attn_res, rms_eps for output RMSNorm); writes FP16 outs.y16[qi]
/// and optional F32 outs.y32[qi] (outs.w[qi] may be null).  nq <= 3.
void attn_res_rms_f16(const ResStates& states, int n, const float* q, int nq, int T, int C, float res_eps,
                      float rms_eps, const RmsOuts& outs, cudaStream_t s);

/// One head's attention weights in place over lg [Tq][nk] = q . k (raw dots): query i at position qpos0 + i, key j at
/// kpos0 + j, dist = their difference;  logit = lg * qk_scale + (0 <= dist < extent ? r[i] . proj[dist] : 0), masked
/// unless 0 <= dist < window;  p = softmax over the keys and a sink logit sink[i] (whose value is zero: the weights sum
/// to less than 1).  r: this head's relative states, row i at r + i * r_ld (d_rel values); proj [extent][d_rel];
/// sink: row i at sink + i * sink_ld.
void attn_probs(float* lg, int Tq, int nk, int qpos0, int kpos0, float qk_scale, const float* r, int r_ld, int d_rel,
                const float* proj, int extent, int window, const float* sink, int sink_ld, cudaStream_t s);
/// Batched FP16-output variant across all H heads: lg [H][Tq][nk] F32 (read-only), r [Tq][H][d_rel], sink [Tq][H],
/// out16 [H][Tq][nk] FP16 in one launch.
/// attn_probs_f16 with the relative bias precomputed: bias [Tq * H][extent], row i * H + h, column the distance.
void attn_probs_bias_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0,
                         float qk_scale, const float* bias, int extent, int window, const float* sink, cudaStream_t s);
void attn_probs_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0, float qk_scale,
                    const float* r, int d_rel, const float* proj, int extent, int window, const float* sink,
                    cudaStream_t s);
/// Prefill attention fused (FP16 KV, D = 128): q16 [Tq] rows of q_ld (head h at h * D) at positions qpos0 ..;
/// kc / vc the cache [pos or pos % kv_cap][KV * D]; keys [max(0, qp - window + 1), qp] of each query; the relative
/// bias from the table bias [Tq * H][extent] (as attn_probs_bias_f16), sink [Tq][H] -> out [Tq] rows of o_ld (F32).
/// The keys split n_split ways (scratch: n_split * Tq * H * (D + 2) floats; fewer splits if it holds fewer).
/// rq given (not on the Hopper kernel): no table - the bias made in the kernel from the queries' 16 relative states
/// rq [Tq][H][16] and relp, rel_proj [extent][16] (on the tensor cores, split FP16: the F32 product to ~2^-22).
void attn_flash_f16(const uint16_t* q16, int q_ld, const uint16_t* kc, const uint16_t* vc, int kv_cap, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, const float* bias, int extent, int window,
                    const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats, int n_split,
                    cudaStream_t s, const int* page_table = nullptr, const float* rq = nullptr,
                    const float* relp = nullptr);
/// attn_flash_f16 over an FP8 (E4M3) cache: its tiles turned into FP16 as they are loaded; qk_scale with K's scale
/// folded in, the output times v_inv_scale (as attn_decode_fp8).
void attn_flash_fp8(const uint16_t* q16, int q_ld, const uint8_t* kc, const uint8_t* vc, int kv_cap, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, float v_inv_scale, const float* bias, int extent,
                    int window, const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats,
                    int n_split, cudaStream_t s, const int* page_table = nullptr, const float* rq = nullptr,
                    const float* relp = nullptr);
/// the Hopper kernel (wgmma) is the one attn_flash_f16 runs here (it does not make the bias itself)
bool fa_wgmma();
/// a split count for attn_flash_f16: Tq queries x H heads against up to nk keys on n_sm SMs
int attn_flash_splits(int Tq, int H, int nk, int n_sm);

/// o *= sigmoid(g), n values.
void gate_mul(float* o, const float* g, size_t n, cudaStream_t s);
/// o16[i] = f2h(o[i] * sigmoid(g[i])), and writes o[i] when write_f32 is true.
void gate_mul_f16(float* o, const float* g, size_t n, uint16_t* o16, bool write_f32, cudaStream_t s);

/// The router over logits [T][E]: sc = sigmoid; the K largest of sc + bias (ties: the lower expert) -> top [T][K] in
/// ascending expert order, w [T][K] = sc / (sum of the chosen + 1e-20) * scale.
void route(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
           cudaStream_t s);

/// FP16 rows gathered from F32 rows: y [n][C] = x[idx[i]][C].
void gather_f16(const float* x, const int* idx, int n, int C, uint16_t* y, cudaStream_t s);

/// The experts' activation over gate | up rows: gu [n][2 F] FP16 (gate first), row i's scales sg[i], su[i] (the
/// expert's NVFP4 scales; 1 without) -> h [n][F] FP16 = gcap tanh(g / gcap) sigmoid(g) ucap tanh(u / ucap),
/// g = gate * sg, u = up * su.
void swiglu_cap(const uint16_t* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                uint16_t* h, cudaStream_t s);

/// out [T][C] (+)= sum over the K slots i of wt[t * K + i] * y[t * K + i ... the row inv[t * K + i]] (y [rows][C] FP16;
/// wt already holds the route weight times the expert's down scale).  add: onto out, else out is overwritten.
void moe_combine(const uint16_t* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                 cudaStream_t s);

/// The same two with F32 inputs: gu [n][2 F] F32 (the experts' raw dot products can pass FP16's range: an NVFP4
/// expert's values are its true ones over its scale) -> h FP16; y [rows][C] F32.
void swiglu_cap_f32(const float* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                    uint16_t* h, cudaStream_t s);
void moe_combine_f32(const float* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                     cudaStream_t s);
/// One token's moe_combine_f32 and the conv4_res after it in one launch: out [C] = conv4(sum_k wt[k] y[inv[k]])
/// + res (hist [3][C], w [C][4]; res null: nothing added).
void moe_combine_conv(const float* y, const int* inv, const float* wt, int K, int C, float* hist, const float* w,
                      const float* res, float* out, cudaStream_t s);

/// y = a + b, n values (y may alias either).
void add(const float* a, const float* b, size_t n, float* y, cudaStream_t s);
/// FP16 copies of n values.
void to_f16(const float* x, size_t n, uint16_t* y, cudaStream_t s);

/// x = cap tanh(x / cap), n values (the head's logits).
void softcap(float* x, size_t n, float cap, cudaStream_t s);

/// The message gates of a looped model (Inkling-2-Mini: a core layer's inputs in the core's later passes), a row a
/// token: g = D [rms(x); rms(m)] (D [R][2 C]: x's columns first; the norms without a weight, eps),
/// y = (1 - tanh(F g)) x + tanh(G g) m (F, G by rank: [R][C]).  x: the input, m: the message from the pass before,
/// [T][C]; R <= 128; y may be x.  n <= 3 gates (a layer's q, k and v inputs) in two launches.
/// sc: scratch, [n][T][2 R + 2] floats.
struct LoopGate {
    const float *x, *m, *D, *F, *G;
    float* y;
    const uint16_t* compact[3] = {}; // D, F, G if each F32 is exactly its upper 16 bits
};
void loop_gates(const LoopGate* gates, int n, int T, int C, int R, float eps, float* sc, cudaStream_t s);
void loop_gate(const float* x, const float* m, const float* D, const float* F, const float* G, int T, int C, int R,
               float eps, float* sc, float* y, cudaStream_t s);
/// Sampling's candidates found on the device: of x [n] (its values within [lo, hi]: softcapped logits) every value
/// of the highest bins that together hold K values at least (1024 bins over [lo, hi]) - K and a few more - gathered
/// in no order, so that the host sorts a thousand or two values and not the vocabulary.  out [2 + 2 cap] ints:
/// [0] their count (past cap: only cap of them are there), [1] the first bin taken, then idx [cap] and val [cap]
/// (F32).  scratch: cand_scratch(n) ints.
size_t cand_scratch(int n);
void cand_topk(const float* x, int n, float lo, float hi, int K, int cap, int* scratch, int* out, cudaStream_t s);
/// The same gate over a chunk of tokens with its products taken as matrix products by the caller (loop_gates'
/// kernels are a token's): loop_gate_mix gives g [T][R] = a1 / rms(x) + a2 / rms(m) of a1 = x D_x^T, a2 = m D_m^T
/// ([T][R]); loop_gate_blend y = (1 - tanh f) x + tanh(gn) m of f = g F, gn = g G ([T][C]; n = T C; y may be x).
void loop_gate_mix(const float* x, const float* m, const float* a1, const float* a2, int T, int C, int R, float eps,
                   float* g, cudaStream_t s);
void loop_gate_blend(const float* x, const float* m, const float* f, const float* gn, size_t n, float* y,
                     cudaStream_t s);

// ---- the token path: the experts straight from their blocks, the attention of one query

/// Two signed INT8 limbs approximate each activation for DP4A with the serialized NVFP4 or Q8 weights.
/// Scratch uses nine float-sized words per 16 inputs, one row for gate/up or S rows for down.
/// A fused router can provide the gate input already packed.
void expert_limb_mv(bool gate, const uint8_t* const* blob, size_t offset, int C, int F, int S, const float* x,
                    uint16_t* h, float* y, const float* sg, const float* su, float gcap, float ucap, void* scratch,
                    cudaStream_t stream, bool q8 = false, bool packed = false);

/// The routed experts of a few tokens as matrix-vector products over the blocks as the GGUF holds them (NVFP4, Q8_0,
/// Q5_K, Q6_K), one launch a phase.  Slot s (S of them): its expert's blob at blob[s] (gate | up | down: eb0, eb1 bytes
/// in), its token's row tok[s] of x.
///   gate_up: gu [S][2 F] = (gate, up) . x[tok[s]] (x [*][C] F32), raw sums (the scales applied by swiglu_cap_f32)
///   down:    y [S][C] = down . h[s] (h [S][F] FP16), raw sums
void experts_gate_up(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                     const float* x, int C, int F, float* gu, cudaStream_t s);
/// One token's NVFP4 experts through the tensor cores (slot s: blob[s]; gate | up | down as above): the input in
/// FP16 (gate / up: x as two limbs, x16 [2][C] = hi, lo with x = hi + lo - to_f16_limbs, or route_token's x_limbs;
/// down: h [S][F]), a code times its scale exact in FP16, the sum FP32.  C and F multiples of 64.
void to_f16_limbs(const float* x, size_t n, uint16_t* y, cudaStream_t s);
bool experts_mma_fits(int ty_gate, int ty_up, int ty_down, int C, int F);
void experts_gate_up_swiglu_mma(const uint8_t* const* blob, int S, size_t eb0, const uint16_t* x16, int C, int F,
                                const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s);
void experts_down_mma(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                      cudaStream_t s);
void experts_down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C, int F,
                  float* y, cudaStream_t s);
#if defined(STRATA_USE_HIP)
/// Upstream issue #36: gfx906 native NVFP4 expert kernels (MI50 / MI60 / Radeon VII). IK2_G906=0 off.
bool ik2_g906_ok();
bool ik2_g906_experts(const uint8_t* const* blob, const int* tok, int S, size_t eb0, size_t off_down, const float* x,
                      int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                      cudaStream_t s);
/// IK2_FUSE_MASK 0x1000 / 0x2000 (ik2_experts2.inc): experts_gate_up_swiglu / experts_down for NVFP4 experts at
/// C 2304, F 1152 (experts_nv_fits), x in registers - the same values to the bit (tests/ik2_experts_test.cu)
bool experts_nv_fits(int ty_gate, int ty_up, int ty_down, int C, int F);
void experts_gate_up_swiglu_nv(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C,
                               int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                               cudaStream_t s);
void experts_down_nv(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                     cudaStream_t s);
/// IK2_FUSE_MASK 0x8000 (set 16): a visit's NVFP4 gate_up + down (sets 13 + 14's tiles) in one launch, down tiles
/// waiting on per-expert device counters (each expert's last down tile resets them: graph-safe); S <= 8
void experts_nv_visit(const uint8_t* const* blob, const int* tok, int S, size_t eb0, size_t off_down, const float* x,
                      int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                      cudaStream_t s);
int experts_nv_visit_pending();
/// IK2_FUSE_MASK 0x4000: the same for Q8_0 experts (row_dot's 4-aligned Q8_0 path; experts_q8_fits)
bool experts_q8_fits(int ty, int C, int F);
void experts_gate_up_swiglu_q8(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C,
                               int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                               cudaStream_t s);
void experts_down_q8(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                     cudaStream_t s);
#endif  // STRATA_USE_HIP

/// y [N] = W [N][K] . x [K] for a dense matrix as the GGUF holds it (rows of `type` blocks: Q8_0, NVFP4, Q5_K, Q6_K),
/// a warp a row in a fixed order - one token's products, half the bytes of an FP16 copy for Q8_0.  K <= 8192.
void mv_blocks(int type, const uint8_t* W, int N, int K, const float* x, float* y, cudaStream_t s);

/// The same grouped: group g's expert at blob[g], picked by n[g] <= 8 tokens - x rows tok[g * 8 + j], output (and, for
/// down, input h) rows row[g * 8 + j] - its blocks read once for all of them.
constexpr int kGroupMax = 8;
void experts_gate_up_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int ty_gate,
                       int ty_up, size_t eb0, const float* x, int C, int F, float* gu, cudaStream_t s);
void enable_grouped_tensor_products(bool enabled);
void experts_down_g(const uint8_t* const* blob, const int* n, const int* row, int G, int ty_down, size_t off_down,
                    const uint16_t* h, int C, int F, float* y, cudaStream_t s);
/// Fused gate+up+softcapped-SwiGLU straight to h [S][F] FP16 (or h[row[g*8+t]][F] in _g, where sg/su are also
/// indexed by row[g*8+t]).  sg, su may be null (scale 1).
void experts_gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                            const float* x, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                            uint16_t* h, cudaStream_t s);
/// experts_gate_up_swiglu with slot s's F outputs written at h + orow[s] * h_ld (orow null: s; h_ld 0: F) - a tensor-
/// parallel GPU's half of every row of a wider h
void experts_gate_up_swiglu_rows(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up,
                                 size_t eb0, const float* x, int C, int F, const float* sg, const float* su, float gcap,
                                 float ucap, uint16_t* h, const int* orow, int h_ld, cudaStream_t s);
void experts_gate_up_swiglu_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G,
                              int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F, const float* sg,
                              const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                              const uint16_t* input16 = nullptr);

/// Issue #19's fused path: llama.cpp's IQ q8_1 dot kernels through an FP32-activation adapter,
/// grouped by expert like experts_gate_up_swiglu_g. Gate and up may have different IQ types.
bool experts_gate_up_swiglu_iq_supported(int ty_gate, int ty_up, int C);
void experts_gate_up_swiglu_iq(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G,
                               int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F, const float* sg,
                               const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s);
/// The decode visit path: S slots, each with its own expert blob and token (c.K <= 8).
void experts_gate_up_swiglu_iq_visit(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up,
                                     size_t eb0, const float* x, int C, int F, const float* sg, const float* su,
                                     float gcap, float ucap, uint16_t* h, cudaStream_t s);

/// Fused attention-input prep: l2clip(q) in place, l2clip(k_in) -> FP16 kc[slot], rms(v_in) -> FP16 vc[slot],
/// l2clip(r) in place, where slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t).  pos may be null (sliding layers).
void prep_qkv_r_f16(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                    cudaStream_t s, const int* page_table = nullptr);
/// attn_decode for nq queries at positions qpos0 .. qpos0 + nq - 1 (q [nq][H][128], r [nq][H][d_rel], sink [nq][H],
/// out [nq][H][128]), each over the keys k0 .. k0 + nk - 1 it may see.  scratch: nq x H x (nk + 128 x 130) floats.
void attn_multi(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int nq, int H,
                int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel, int extent, int window,
                float* scratch, float* out, cudaStream_t s, const int* page_table = nullptr);

/// One query's attention (a token being generated), every head in one launch: q [H][D] and r [H][d_rel] as the
/// clips left them, sink [H]; the keys at positions k0 .. k0 + nk - 1 of the caches kc / vc ([ctx][Hkv][D] F32), the
/// query at qpos; logits q . k * qk_scale + the relative bias (0 <= dist < extent), masked past the window, softmax
/// with the sink; out [H][D].  scratch: H x nk + H x 128 x 130 floats (nk >= 256: the keys in runs a block each,
/// combined after).  D = 128.
void attn_decode(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int H, int Hkv,
                 int D, int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                 float* scratch, float* out, cudaStream_t s, const int* page_table = nullptr);

/// FP16 K / V cache variants of attn_multi / attn_decode with optional ring-buffer wraparound (when kv_cap > 0, key
/// k0 + j sits at slot (k0 + j) % kv_cap; when kv_cap <= 0, linear slot k0 + j).  D = 128.
void attn_multi_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                    int kv_cap, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                    int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s,
                    const int* page_table = nullptr);
void attn_decode_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale, const float* proj,
                     int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s,
                     const int* page_table = nullptr);

/// FP8 (E4M3FN) K / V cache variants: scale K by k_scale and V by v_scale on write; fold 1/k_scale into qk_scale
/// and pass v_inv_scale = 1/v_scale on read.  slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t).  D = 128.
void prep_qkv_r_fp8(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                    float k_scale, float v_scale, cudaStream_t s, const int* page_table = nullptr);
void attn_multi_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                    int kv_cap, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
                    float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                    float* out, cudaStream_t s, const int* page_table = nullptr);
void attn_decode_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale,
                     float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                     float* out, cudaStream_t s, const int* page_table = nullptr);
/// Unwrap nk KV rows starting at k0 from FP8 ring buffer into contiguous FP16 [nk, KD] scaled by k_inv_scale / v_inv_scale.
void unpack_kv_fp8_to_f16(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk, int KD,
                          float k_inv_scale, float v_inv_scale, uint16_t* k_out16, uint16_t* v_out16,
                          cudaStream_t s, const int* page_table = nullptr);

// ---- the resident token path: every expert in VRAM (found on the device), the position read on the device, so a
// token needs nothing from the host between its embedding and its logits and its launches replay as one CUDA graph

/// y [N] = W [N][K] . x [K] for a Q8_0 matrix kept planar: q [N][K] its int8 values, d [N][K / 32] its blocks' FP16
/// scales (the GGUF's blocks taken apart, the values unchanged) - a warp reads 128 values an instruction.  K a
/// multiple of 128; q aligned to 4 bytes, x to 16.  decode_q8p_f16: the same matrix as FP16 (n values).
void mv_q8p(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, cudaStream_t s);
/// ... with a short conv (conv4's, one token) as the product's epilogue: the rows [r0, r0 + n) go through it - hist
/// [3][n], w [n][4], dyn [heads * 4] or null, res [n] added or null - to y [n] instead of to the product's y, and
/// hist takes the row.  A launch less a conv: every launch of a token costs 2 us and more, whatever it does.
struct MvConv {
    int r0 = 0, n = 0;
    float* hist = nullptr;
    const float* w = nullptr;
    const float* dyn = nullptr;
    int heads = 0;
    const float* res = nullptr;
    float* y = nullptr;
};
/// ... and with rows that take another input: the rows [r0, r0 + n) of a0 and of a1 are products with a0.x and
/// a1.x (a layer's fused q | k | v | r | gate matrix whose q, k and v each have an input of their own: one launch).
struct MvAlt {
    const float* x = nullptr;
    int r0 = 0, n = 0;
};
void q8_tile(const int8_t* q, const uint16_t* d, int N, int K, const uint16_t* x, int T, float* y, int ldy,
             cudaStream_t s);
void mv_q8p_conv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                 const MvConv& c1, cudaStream_t s, const MvAlt& a0 = MvAlt{}, const MvAlt& a1 = MvAlt{});
void decode_q8p_f16(const int8_t* q, const uint16_t* d, size_t n, uint16_t* out, cudaStream_t s);
/// y [N] = W [N][K] . x [K] for a small dense matrix kept in FP16 (the sinks, a full layer's dynamic taps, the
/// router): F32 sums, a warp a row.
void mv_f16(const uint16_t* W, int N, int K, const float* x, float* y, cudaStream_t s);
/// The routed slots' tables from the router's picks, all on the device.  top, w [S]: route()'s; tab [E]: where each
/// expert's blocks are (gate | up | down); sg_t, su_t, sd_t [E]: the experts' scales (null: 1).  Slot i (token
/// i / K): blob[i] = tab[top[i]], tok[i], inv[i] = i, sg[i], su[i], wt[i] = w[i] * down's scale - what
/// experts_gate_up_swiglu, experts_down and moe_combine_f32 take.
void route_gather(const int* top, const float* w, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                  const float* sd_t, int S, int K, const uint8_t** blob, int* tok, int* inv, float* sg, float* su,
                  float* wt, cudaStream_t s);
/// One token's route() and route_gather() in one launch (the same values to the bit): top, w [K] and the K slots'
/// tables.
void route_token(const float* logits, const float* bias, int E, int K, float scale, const uint8_t* const* tab,
                 const float* sg_t, const float* su_t, const float* sd_t, int* top, float* w, const uint8_t** blob,
                 int* tok, int* inv, float* sg, float* su, float* wt, cudaStream_t s, const float* x = nullptr,
                 int width = 0, void* packed = nullptr, bool q8 = false, uint16_t* x_limbs = nullptr);
#if defined(STRATA_USE_HIP)
/// IK2_FUSE=1 (ik2_fuse.inc): the resident path's small kernels, fewer and shorter, the same values to the bit
/// (tests/ik2_fuse_test.cu).  Each returns false - nothing launched - where its shapes are not the ones it was built
/// for (rows of 2304, the router's E <= 128, P 16 or 128); the caller then takes the kernels it replaces.
/// fuse_mv_f16_pair: two mv_f16 products (y0 = W0 x0, y1 = W1 x1, rows of K) in one launch.
bool fuse_mv_f16_pair(const uint16_t* W0, const float* x0, float* y0, int N0, const uint16_t* W1, const float* x1,
                      float* y1, int N1, int K, cudaStream_t s);
/// fuse_router_route: mv_f16 (logits = Wr x, Wr [E][C]) and route_token (no x limbs / packed input) in one launch;
/// lds (set 6): the rows and x through LDS in 16-byte loads (the same values).
bool fuse_router_route(const uint16_t* Wr, const float* x, int C, float* logits, const float* bias, int E, int K,
                       float scale, const uint8_t* const* tab, const float* sg_t, const float* su_t, const float* sd_t,
                       int* top, float* w, const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt,
                       cudaStream_t s, bool lds = false);
/// fuse_prep_qkv_r_f16_at: prep_qkv_r_f16_at, its sinks' loads asked for at once.
bool fuse_prep_qkv_r_f16_at(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                            uint16_t* vc, float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip,
                            float rel_clip, float eps, const int* pos, bool log_scale, float alpha, float n_floor,
                            int kv_cap, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w, const float* sink_x,
                            int sink_k, float* sink_y);
/// fuse_attn_decode_f16_at_gate: attn_decode_f16_at then gate_mul_f16 (out = o sigmoid(gate), out16 its FP16):
/// the split as is, the combine and the gate in one launch.
bool fuse_attn_decode_f16_at_gate(const float* q, const float* r, const float* sink, const uint16_t* kc,
                                  const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int P,
                                  float qk_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                                  const float* gate, float* out, uint16_t* out16, cudaStream_t s, bool skip = false);
/// (skip, set 7: P 128's combine reads only the splits that hold keys and takes each split's factor once.)
/// fuse_loop_gates_rms: loop_gates of one token (F32 weights, rank 64, C 2304) and then rms of each gate's y (rj's
/// job j: x = gate j's y, its weights, its output; rms_multi's or rms's values) in two launches instead of four.
bool fuse_loop_gates_rms(const LoopGate* gates, int n, int C, int R, float eps, float* sc, const RmsJobs& rj,
                         float rms_eps, cudaStream_t s, bool wide = false);
/// (wide, set 8: the apply over 64 channels a block - 36 blocks a gate - each thread's loads before the barrier.)
/// fuse_attn_one_f16_at_gate: fuse_attn_decode_f16_at_gate's values in one launch (a head's last split block combines).
bool fuse_attn_one_f16_at_gate(const float* q, const float* r, const float* sink, const uint16_t* kc,
                               const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale,
                               const float* proj, int d_rel, int extent, int window, float* scratch, const float* gate,
                               float* out, uint16_t* out16, cudaStream_t s);
/// fuse_moe_attn_res: moe_combine_conv (K 8, C 2304; out = the stream, hist / w / res as moe_combine_conv's) and then
/// attn_res (outs null: into res_out) or attn_res_rms_f16 (outs given) of one token over st, whose last state is out:
/// one launch, attn_res_vec4's values.
bool fuse_moe_attn_res(const float* y, const int* inv, const float* wt, int K, int C, float* hist, const float* w,
                       const float* res, float* out, const ResStates& st, int n, const float* q, int nq, float res_eps,
                       float rms_eps, float* res_out, const RmsOuts* outs, cudaStream_t s, bool pf = false);
/// fuse_attn_res_pf (set 5): attn_res (outs null: into res_out) or attn_res_rms_f16 (outs given) of one token over st
/// (n <= 6, C 2304), every load of a thread at once; fuse_moe_attn_res's pf: its kernel the same way (n <= 6).
bool fuse_attn_res_pf(const ResStates& st, int n, const float* q, int nq, int C, float res_eps, float rms_eps,
                      float* res_out, const RmsOuts* outs, cudaStream_t s);
/// fuse_res_router_route (set 9): attn_res_rms_f16 of one token (one query, n <= 6; outs.y32 given) and then
/// fuse_router_route on outs.y32, in one launch - the same values.
bool fuse_res_router_route(const ResStates& st, int n, const float* q, int C, float res_eps, float rms_eps,
                           const RmsOuts& outs, const uint16_t* Wr, float* logits, const float* bias, int E, int K,
                           float scale, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                           const float* sd_t, int* top, float* w, const uint8_t** blob, int* tok, int* inv, float* sg,
                           float* su, float* wt, cudaStream_t s);
/// fuse_res_loop_gate_rms (set 10): attn_res of one token (one query, n <= 6) into res_out, which must be gt.x, and
/// then fuse_loop_gates_rms of that one gate (rj's job 0) - the residual in the dots' launch; the same values.
bool fuse_res_loop_gate_rms(const ResStates& st, int n, const float* q, float res_eps, float* res_out,
                            const LoopGate& gt, int C, int R, float eps, float* sc, const RmsJobs& rj, float rms_eps,
                            cudaStream_t s, bool wide = false);
/// fuse_loop_gate_rms_router (set 11): one loop gate (gt.y a scratch apart from gt.x and x_out) whose rms (weights nw)
/// goes to x_out in the router's launch, then fuse_router_route on x_out - the same values as fuse_loop_gates_rms (rj
/// {gt.y, nw, x_out}) and fuse_router_route.
bool fuse_loop_gate_rms_router(const LoopGate& gt, int C, int R, float eps, float* sc, const float* nw, float rms_eps,
                               float* x_out, const uint16_t* Wr, float* logits, const float* bias, int E, int K,
                               float scale, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                               const float* sd_t, int* top, float* w, const uint8_t** blob, int* tok, int* inv,
                               float* sg, float* su, float* wt, cudaStream_t s);
/// fuse_loop_gate_router (set 12): fuse_loop_gates_rms of one gate (rj's job 0) and then fuse_router_route on rj.y[0]
/// - the apply, its rms and the router in one grid (the router blocks wait for the rms on a flag); the same values.
bool fuse_loop_gate_router(const LoopGate& gt, int C, int R, float eps, float* sc, const RmsJobs& rj, float rms_eps,
                           const uint16_t* Wr, float* logits, const float* bias, int E, int K, float scale,
                           const uint8_t* const* tab, const float* sg_t, const float* su_t, const float* sd_t, int* top,
                           float* w, const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt,
                           cudaStream_t s);
/// fuse_prep_attn_f16_at_gate: prep_qkv_r_f16_at (sinks given) + attn_decode_f16_at + gate_mul_f16 of one token in one
/// launch (P 16: a window layer); the cache, the sinks and out / out16 as theirs; q and r are left unclipped.
bool fuse_prep_attn_f16_at_gate(const float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                                uint16_t* vc, const float* r, const float* r_clip, int H, int KV, int D, int d_rel,
                                float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                                float n_floor, int kv_cap, const uint16_t* sink_w, const float* sink_x, int sink_k,
                                float* sink_y, float qk_scale, const float* proj, int extent, int window, int P,
                                float* scratch, const float* gate, float* out, uint16_t* out16, cudaStream_t s);
#endif  // STRATA_USE_HIP
/// prep_qkv_r_f16 / _fp8 of one token whose position is at pos (device): the cache slot from it, and with log_scale
/// (a full layer) q's and r's tau.  q16 (given): an FP16 copy of the clipped q, [H][D] (attn_decode_mma's).
/// sink_w (given): the token's sinks in the same launch, sink_y [H] = sink_w [H][sink_k] (FP16) . sink_x (mv_f16's).
void prep_qkv_r_f16_at(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, int kv_cap,
                       uint16_t* q16, cudaStream_t s, const uint16_t* sink_w = nullptr, const float* sink_x = nullptr,
                       int sink_k = 0, float* sink_y = nullptr, const int* page_table = nullptr);
void prep_qkv_r_fp8_at(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, int kv_cap,
                       float k_scale, float v_scale, uint16_t* q16, cudaStream_t s,
                       const uint16_t* sink_w = nullptr, const float* sink_x = nullptr, int sink_k = 0,
                       float* sink_y = nullptr, const int* page_table = nullptr);
/// attn_decode_f16 / _fp8 of the token at position *pos (device): the keys it may see follow from the position and
/// the window (1 << 30: every one), split P ways whatever their number (P fixed, so the launches do not change from
/// a token to the next).  scratch: H x P x 130 floats.
void attn_decode_f16_at(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                        int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, const float* proj,
                        int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s,
                        const int* page_table = nullptr);
void attn_decode_fp8_at(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                        int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, float v_inv_scale,
                        const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                        cudaStream_t s, const int* page_table = nullptr);

/// One token's attention on the tensor cores - for long contexts, where the scalar attn_decode is bound by its
/// arithmetic; this one reads at the memory's speed, an FP8 cache too.  q16 [H][128]: the clipped q in FP16;
/// r [H][d_rel], sink [H] as attn_decode's.  The keys the token may see follow from its position (*pos on the device
/// if given, else qpos) and the window (1 << 30: every one); kv_cap > 0: the cache a ring of that many rows (>= 16).
/// A key's K and V rows are read once for the H / Hkv <= 8 heads that share them.  n_split: blocks a K / V head
/// (fixed, so the launches do not change with the position) - attn_decode_mma_blocks(Hkv) of them fill the device
/// once, a window of w keys needs no more than w / 64.  scratch: attn_decode_mma_scratch(H, n_split) floats.
/// gate [H][128] (or null): the output times its sigmoid (gate_mul's); out16 (or null): an FP16 copy of the output.
int attn_decode_mma_blocks(int Hkv);
size_t attn_decode_mma_scratch(int H, int n_split);
void attn_decode_mma_f16(const uint16_t* q16, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                         int kv_cap, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                         const float* gate, uint16_t* out16, cudaStream_t s, const int* page_table = nullptr);
void attn_decode_mma_fp8(const uint16_t* q16, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                         int kv_cap, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                         float* out, const float* gate, uint16_t* out16, cudaStream_t s,
                         const int* page_table = nullptr);

// ---- speculative decoding: the draft model's (Inkling 2's DSpark drafter, mtp.safetensors) and the rollback's

/// hist = the last 3 rows of [old (3 rows of C); x (T rows of C)] - a short conv's history after T of the inputs it
/// was given (the accepted part of a verify window).  hist may alias old.
void conv_hist_after(const float* old, const float* x, int T, int C, float* hist, cudaStream_t s);
/// BF16 copies of n values (nearest even).
void to_bf16(const float* x, size_t n, uint16_t* y, cudaStream_t s);
/// out [rows][C] = rms(sum_k alpha[k] x[k][row]) * w: x's taps n_tap apart (tap_stride floats), alpha (device) [n_tap].
void mix_rms(const float* x, int n_tap, size_t tap_stride, int rows, int C, const float* alpha, const float* w,
             float eps, float* out, cudaStream_t s);
/// NeoX rotary in place on x [rows][heads][128] at positions pos[row] (device), base theta (inverse frequencies in F32
/// as torch computes them).
void rope_neox(float* x, int rows, int heads, const int* pos, float theta, cudaStream_t s);
/// The drafter's attention: nq queries q [nq][H][128] over the context keys at positions k_lo .. k_hi - 1 of kc / vc
/// ([ctx][Hkv][128]) and the block's own nq keys kb / vb ([nq][Hkv][128], all visible to every query); softmax(q . k *
/// scale) . v -> out [nq][H][128].  scratch: H x nq x (k_hi - k_lo + nq) floats.
void draft_attn(const float* q, const float* kc, const float* vc, int k_lo, int k_hi, const float* kb, const float* vb,
                int nq, int H, int Hkv, float scale, float* scratch, float* out, cudaStream_t s, int cap = 0);
/// ... over a context of FP16 rows.
void draft_attn_f16(const float* q, const uint16_t* kc, const uint16_t* vc, int k_lo, int k_hi, const float* kb,
                    const float* vb, int nq, int H, int Hkv, float scale, float* scratch, float* out, cudaStream_t s,
                    int cap);
/// out [T][F] BF16 = silu(gu[t][f]) * gu[t][F + f] (gu [T][2 F] F32: gate's then up's).
void silu_mul_bf16(const float* gu, int T, int F, uint16_t* out, cudaStream_t s);
/// The Markov head's input [state (R); emb[prev] (R, BF16 table); hidden (C); f (C)] as BF16 into out [2 R + 2 C].
void rnn_inputs(const float* state, const uint16_t* emb, const int* prev, const float* hidden, const float* f, int R,
                int C, uint16_t* out, cudaStream_t s);
/// The GRU-like step over proj [3 R] (+ bias [3 R]): gate = sigmoid, cand = tanh, out = tanh; state = gate state +
/// (1 - gate) cand (in place), latent = out.
void rnn_gates(const float* proj, const float* bias, int R, float* state, float* latent, cudaStream_t s);
/// *out = argmax over v < n of a[v] + b[v] + c[v] (b, c may be null; ties: the lower index), *out written on the device.
void argmax_sum(const float* a, const float* b, const float* c, int n, int* out, cudaStream_t s);
/// out[r] = argmax over c < n_cols of x[r * ld + c] for 0 <= r < n_rows (ties: the lower index), written on the device.
void argmax_rows(const float* x, int ld, int n_rows, int n_cols, int* out, cudaStream_t s);
/// *out = argmax over v < n of (base[v] + bias_f16[v] + sum_{r<R} W[r * ld + v] * x[r]), where W is [R][ld] FP16,
/// x is [R] F32, base (optional) is [n] F32, bias_f16 (optional) is [n] FP16.  Ties: the lower index.
void argmax_gemv_f16(const uint16_t* W, int ld, const float* x, const float* base, const uint16_t* bias_f16,
                     int R, int n, int* out, cudaStream_t s);

// ---- the weights' decodes (ggml's block types; values exactly ggml's dequantize_row_*, then rounded to the output)

/// ggml type ids of the blocks these decode. The IQ2_* / IQ1_S / Q4_K / MXFP4 / Q2_0 ids are the small-quant
/// layouts from issue #19: the down experts of those files are Q2_0 (the old control quant used NVFP4).
constexpr int kQ8_0 = 8, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18,
              kIQ1_S = 19, kIQ2_S = 22, kIQ4_XS = 23, kMXFP4 = 39, kNVFP4 = 40, kQ2_0 = 42;
/// values a block / bytes a block (0: not one of the above).
int block_values(int type);
size_t block_bytes(int type);
/// n values (a multiple of the block's) of `type` blocks -> FP16 (nearest; NVFP4: exact) or F32.
void decode_f16(int type, const uint8_t* blocks, size_t n, uint16_t* out, cudaStream_t s);
void decode_f32(int type, const uint8_t* blocks, size_t n, float* out, cudaStream_t s);
/// The prompt's NVFP4 experts as one grouped GEMM (tensor cores, the weights decoded in shared memory, F32 sums): for
/// every tile, Y [row0 + r][N] = X [row0 + r][K] . W [N][K]^T over its rows r < rows (at most bm: 64 or 128), W = w[which] as
/// NVFP4 rows of K values.  K a multiple of 64 and N of 128 (experts_pf_nvfp4_fits).  tiles on the device.
struct PfTile {
    const uint8_t* w[2];
    int row0, rows;
};
bool experts_pf_nvfp4_fits(int K, int N);
/// The rows a tile may hold for the kernel experts_pf_nvfp4 runs (64 or 128), and that kernel (IK2_PF_KERNEL: 21 the
/// default - raw weights decoded in registers, 128 x 128, 2 stages; 1 the first version; tests and benches set it).
int experts_pf_nvfp4_bm();
void experts_pf_nvfp4_variant(int v);
void experts_pf_nvfp4(const PfTile* tiles, int n_tiles, int bm, int which, const uint16_t* X, int K, int N, float* Y,
                      cudaStream_t s);
/// ... on the host (the reference of the tests, and the embedding's rows).
void decode_host(int type, const uint8_t* blocks, size_t n, float* out);

// ---- the block path: a few tokens at once (2 .. 8, a speculative step's verify window) on the resident path, every
// position read from the device (pos [T]: the tokens' positions, consecutive) so that a step's launches replay

/// attn_decode_mma for the T tokens of a window (T * H / Hkv <= 24): q16 [T][H][D]; r: a token's [H][d_rel] r_ld
/// floats apart; sink [T][H]; a token's keys its own (causal, its window); gate rows gate_ld apart; out, out16
/// [T][H][D].  scratch: attn_block_scratch floats.
void attn_block_f16(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint16_t* kc,
                    const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int n_split, float qk_scale,
                    const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                    const float* gate, int gate_ld, uint16_t* out16, cudaStream_t s,
                    const int* page_table = nullptr);
void attn_block_fp8(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint8_t* kc,
                    const uint8_t* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int n_split, float qk_scale,
                    float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                    float* out, const float* gate, int gate_ld, uint16_t* out16, cudaStream_t s,
                    const int* page_table = nullptr);
size_t attn_block_scratch(int H, int T, int n_split);
/// conv4_res for T tokens with row strides (x, res, y); with rec_old [3][C] and rec_x [T][C] given, the history
/// before and the inputs kept for rollback_block.
void conv_block(const float* x, int x_ld, float* hist, const float* w, const float* dyn, int d_ld, int heads, int T,
                int C, const float* res, int res_ld, float* y, int y_ld, float* rec_old, float* rec_x,
                cudaStream_t s);
/// A conv's record of a window (conv_block's rec_old, rec_x) and its history.
struct ConvRec {
    const float* old;
    const float* x;
    float* hist;
    int C;
};
/// Every recorded conv's history as after the first n of the window's tokens (recs: device [n_recs]).
void rollback_block(const ConvRec* recs, int n_recs, int n, cudaStream_t s);
/// y [T][N] (rows y_ld apart) = W [N][K] (FP16) . x [T][K] (rows x_ld apart).
void mv_f16_rows(const uint16_t* W, int N, int K, const float* x, int x_ld, int T, float* y, int y_ld,
                 cudaStream_t s);
/// x [T][C] as two FP16 limbs a token: out [T][2][C] (hi, lo).
void f16_limbs_rows(const float* x, int T, int C, uint16_t* out, cudaStream_t s);
/// route()'s picks of a window (S = T K <= 64 slots) grouped by expert on the device: blob [S], n [S] (0 past the
/// last group), tok / row [S][kGroupMax], and a slot's sg, su, wt (down's scale folded) and inv (the slot).
void group_route(const int* top, const float* w, int S, int K, const uint8_t* const* tab, const float* sg_t,
                 const float* su_t, const float* sd_t, const uint8_t** blob, int* n, int* tok, int* row, float* sg,
                 float* su, float* wt, int* inv, cudaStream_t s);
/// group_route's groups of NVFP4 experts on the tensor cores: gate | up (the input as f16_limbs_rows' limbs) to
/// h [slot][F] through the capped SwiGLU; down from h to y [slot][C].  G: the launch's groups (those past the last
/// return at once); T: the window's tokens (a group's members).
void experts_block_gu(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int T,
                      size_t off_up, int C, int F, const uint16_t* limbs, const float* sg, const float* su, float gcap,
                      float ucap, uint16_t* h, cudaStream_t s);
void experts_block_down(const uint8_t* const* blob, const int* n, const int* row, int G, size_t off_down, int C,
                        int F, const uint16_t* h, float* y, cudaStream_t s);
/// prep_qkv_r_*_at for the T tokens of a window: q and r rows q_ld and r_ld floats apart (q16 [T][H][D]), the
/// positions pos [T] on the device; sinks for each token (sink_x [T][sink_k], sink_y [T][H]).
void prep_qkv_r_fp8_block(float* q, int q_ld, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in,
                          uint8_t* vc, float* r, int r_ld, const float* r_clip, int T, int H, int KV, int D, int d_rel,
                          float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                          float n_floor, int kv_cap, float k_scale, float v_scale, uint16_t* q16, cudaStream_t s,
                          const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y,
                          const int* page_table = nullptr);
void prep_qkv_r_f16_block(float* q, int q_ld, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                          uint16_t* vc, float* r, int r_ld, const float* r_clip, int T, int H, int KV, int D, int d_rel,
                          float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                          float n_floor, int kv_cap, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w,
                          const float* sink_x, int sink_k, float* sink_y, const int* page_table = nullptr);
/// route() for a window, a block a token (the same picks and weights to the bit).
void route_rows(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
                cudaStream_t s);
/// experts_block_gu / _down over Q8_0 experts (a block's sum scaled in FP32: the int8 values exact).
void experts_block_gu_q8(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int T,
                         size_t off_up, int C, int F, const uint16_t* limbs, const float* sg, const float* su,
                         float gcap, float ucap, uint16_t* h, cudaStream_t s);
void experts_block_down_q8(const uint8_t* const* blob, const int* n, const int* row, int G, size_t off_down, int C,
                           int F, const uint16_t* h, float* y, cudaStream_t s);
/// attn_res_jobs for T tokens: job j's queries q + j nq C, its output out + j nq T C ([nq][T][C]).
void attn_res_jobs_block(const ResStates& states, int n, const float* q, int nq, int jobs, int T, int C, float eps,
                         float* out, cudaStream_t s);

// ---- multi-slot batched decode path (B concurrent independent requests, each on its own slot)

/// Short conv for B independent requests: batch item b reads/updates hist_base + slot_ids[b] * 3 * C.
void conv_slots(const float* x, int x_ld, float* hist_base, const int* slot_ids, const float* w, const float* dyn,
                int d_ld, int heads, int B, int C, const float* res, int res_ld, float* y, int y_ld, cudaStream_t s);
/// Attention-input prep for B independent requests at positions pos_d[b] on slots slot_ids[b].
/// If page_tables != nullptr, writes into shared paged pool using page_tables + slot_ids[b] * pt_stride;
/// otherwise writes into per-slot sliding ring kc_base + slot_ids[b] * kv_cap * KV * D.
void prep_qkv_r_fp8_slots(float* q, int q_ld, const float* q_clip, const float* k_in, uint8_t* kc_base,
                          const float* v_in, uint8_t* vc_base, float* r, int r_ld, const float* r_clip, int B, int H,
                          int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos_d,
                          const int* slot_ids, const int* page_tables, int pt_stride, bool log_scale, float alpha,
                          float n_floor, int kv_cap, float k_scale, float v_scale, uint16_t* q16, cudaStream_t s,
                          const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y);
void prep_qkv_r_f16_slots(float* q, int q_ld, const float* q_clip, const float* k_in, uint16_t* kc_base,
                          const float* v_in, uint16_t* vc_base, float* r, int r_ld, const float* r_clip, int B, int H,
                          int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos_d,
                          const int* slot_ids, const int* page_tables, int pt_stride, bool log_scale, float alpha,
                          float n_floor, int kv_cap, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w,
                          const float* sink_x, int sink_k, float* sink_y);
/// Batched decode attention over B independent slots on the tensor cores.
size_t attn_slots_scratch(int H, int B, int n_split);
void attn_slots_fp8(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint8_t* kc_base,
                    const uint8_t* vc_base, int kv_cap, int H, int Hkv, const int* pos_d, const int* slot_ids,
                    const int* page_tables, int pt_stride, int B, int n_split, float qk_scale, float v_inv_scale,
                    const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                    const float* gate, int gate_ld, uint16_t* out16, cudaStream_t s);
void attn_slots_f16(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint16_t* kc_base,
                    const uint16_t* vc_base, int kv_cap, int H, int Hkv, const int* pos_d, const int* slot_ids,
                    const int* page_tables, int pt_stride, int B, int n_split, float qk_scale, const float* proj,
                    int d_rel, int extent, int window, float* scratch, float* out, const float* gate, int gate_ld,
                    uint16_t* out16, cudaStream_t s);

#if defined(STRATA_USE_HIP)
/// IK2_HBPF (ik2_hbpf.inc: prefill kernels ported from headbouyJB/inkling2-gufo, MIT): f16 WMMA tiles over the blocks
/// bit 0: the experts of a chunk grouped on the GPU, gate_up + activation and down as two tiled GEMMs, slots summed
/// in a fixed order (C 2304, F 1152, E 128, K 8; NVFP4 or Q8_0 experts, all three roles one type: hb_moe_fits)
struct HbMoeWs {
    int *cnt, *goff, *fill, *nitems, *item_e, *item_start, *item_n, *postok, *pair;
    float* posw;
};
bool hb_moe_fits(int ty_gate, int ty_up, int ty_down, int C, int F, int E, int K);
void hb_moe_route(const int* ids, const float* w, int S, int tile, const HbMoeWs& ws, cudaStream_t s);
/// tab: the experts' blobs (gate rows, up rows, then down at off_down); h16 [T K][F] and pairs [T K][C] scratch; y [T][C];
/// pipe (bit 6): the GEMMs' late-dequant K step (k_gemm4 PIPE), the same result bit for bit;
/// probe (with pipe, NVFP4, tile 64): 1 / 2 also launch k_gemm4 without fragment reads / without dequant first (a price
/// in a kernel trace; their output is overwritten); rg (bit 7): the GEMMs as upstream
/// gufo's routed kernel shape (k_rg), tile 192 or 128 (the items hb_moe_route made), the same sums as k_gemm4
void hb_moe_prefill(int type, int tile, const uint8_t* const* tab, size_t off_down, const float* sg, const float* su,
                    const float* sd, float gcap, float ucap, const uint16_t* x16, uint16_t* h16, float* pairs, float* y,
                    int T, const HbMoeWs& ws, bool pipe, bool rg, int probe, cudaStream_t s);
/// bit 1: Y [T][ldy] = X16 [T][K] . W^T over a planar Q8_0 matrix (IK2_DENSE q8: int8 [N][K], F16 d [N][K / 32])
bool hb_dense_fits(int N, int K);
void hb_dense(const int8_t* q8, const uint16_t* q8d, int N, int K, const uint16_t* x16, int T, float* Y, int ldy,
              cudaStream_t s);
/// bit 2: prefill attention over the FP16 cache (slot = pos % cap) after prep_qkv_r_f16: q16 = to_f16(q) [T][H][D],
/// r [T][H][d_rel], sink [T][H], rel F32 [extent][d_rel]; out [T][H][D] (the gate not applied)
bool hb_attn_fits(int H, int KV, int D, int d_rel);
void hb_attn_prefill(const uint16_t* q16, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                     int cap, int T, int pos0, int window, int extent, const float* rel, float qk_scale, float* out,
                     cudaStream_t s);
/// bit 8 (with bit 2): the same over a turbo6 cache (kc6 / vc6, ring kv_cap as unpack_kv_t6_to_f16;
/// window (1 << 30) = a full visit): its keys decoded once to FP16 rows k16 / v16 [rows][KV * 128] (unpack_kv_t6_to_f16's
/// values), then hb_attn_prefill over them - or, a full visit of more than `pass` keys (a multiple of 64, <= rows), in
/// passes with the softmax state in st (st_floats >= T x 24 x 132), the one call's bits.  hb_attn_t6_plan: 1 one call,
/// n passes, 0 cannot (the caller's old path; hb_attn_prefill_t6 aborts on it)
int hb_attn_t6_plan(int T, int pos0, int window, int rows, int pass, size_t st_floats);
void hb_attn_prefill_t6(const uint16_t* q16, const float* r, const float* sink, const uint8_t* kc6, const uint8_t* vc6,
                        int kv_cap, int T, int pos0, int window, int extent, const float* rel, float qk_scale,
                        float* out, uint16_t* k16, uint16_t* v16, int rows, int pass, float* st, size_t st_floats,
                        cudaStream_t s);
/// IK2_KV=turbo6 (ik2_turbo6.h / .inc; tubo's 98-byte block, one a 128-value head): caches [kv_cap][KV][98] bytes.
/// kv_t6_encode: F32 rows k32 / v32 [T][KV][128] -> the blocks at slot (pos + t) % kv_cap (kv_cap 0: pos + t), pos =
/// *pos_d if given else pos0 - the bytes of ik2::t6::QuantizeBlock.  prep_qkv_r_t6: prep_qkv_r_f16's work (q, r clips,
/// sinks) with k and v staged in F32 (k_stage, v_stage [T][KV][128]), then kv_t6_encode.  attn_decode_t6: attn_multi /
/// attn_decode_f16_at over turbo6 caches (pos_d: the replayed graph's position; P splits; scratch nq H P 130 floats).
/// unpack_kv_t6_to_f16: positions k0 .. k0 + nk - 1 -> FP16 [nk][KV * 128] (the prompt path's GEMMs).
void kv_t6_encode(const float* k32, const float* v32, uint8_t* kc, uint8_t* vc, int T, int KV, int kv_cap, int pos0,
                  const int* pos_d, cudaStream_t s);
void prep_qkv_r_t6(float* q, const float* q_clip, const float* k_in, const float* v_in, float* r, const float* r_clip,
                   int T, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos,
                   float alpha, float n_floor, float* k_stage, float* v_stage, uint8_t* kc, uint8_t* vc, int kv_cap,
                   int pos0, const int* pos0_d, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                   int sink_k, float* sink_y, cudaStream_t s);
void attn_decode_t6(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc, int kv_cap,
                    int nq, int H, int Hkv, int k0, int nk, int qpos0, const int* pos_d, int P, float qk_scale,
                    const float* proj, int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s);
void unpack_kv_t6_to_f16(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk, int Hkv, uint16_t* k16,
                         uint16_t* v16, cudaStream_t s);
/// attn_tok_t6: one token's prep_qkv_r_t6 + attn_decode_t6 + gate_mul_f16 (the position at pos; out = o sigmoid(gate),
/// out16 its FP16), the same bytes and values, a group a block: one_launch (P 16, a window layer) all of it in one
/// launch, else prep + split + combine (skip: set 7's combine).  q is left rotated (prep), r clipped; scratch H P 130
/// floats.  false (nothing launched): a shape it does not take (3 q heads a group, D 128, d_rel <= 32, sinks of
/// C = 2304, P 16 / 128, 16-byte rows) - the caller takes the two calls above.
bool attn_tok_t6(float* q, const float* q_clip, const float* k_in, const float* v_in, float* r, const float* r_clip,
                 int H, int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos,
                 bool log_scale, float alpha, float n_floor, uint8_t* kc, uint8_t* vc, int kv_cap, bool sliding,
                 const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y, float qk_scale,
                 const float* proj, int extent, int window, int P, bool one_launch, bool skip, float* scratch,
                 const float* gate, float* out, uint16_t* out16, cudaStream_t s);
/// bit 4 (ours): up to 3 loop gates over a chunk, each one kernel: y = (1 - tanh(F g)) x + tanh(G g) m with
/// g = D [rms(x); rms(m)] (eps_gate), then y16 = rms(y) * w (eps_norm) F16, y32 the same F32 when not null; x, m
/// [T][C] F32 (y32 may be x: a block reads its own rows before it writes them; n 1 then), D [R][2C], F (forget),
/// G (gain) [R][C]; C 2304, R 64 (hb_ctl_fits)
struct CtlJob {
    const float *x, *m, *D, *F, *G, *w;
    uint16_t* y16;
    float* y32;
};
bool hb_ctl_fits(int C, int R);
void hb_ctl_prefill(const CtlJob* jobs, int n, int T, float eps_gate, float eps_norm, cudaStream_t s);
// ---- the exact verify window (ik2_dsc.inc, IK2_DSC=1): every row the single-token decode's float bits

/// mv_q8p for T tokens (1 .. 8): y [t] rows ldy apart = W . x [t] (rows x_ld apart, x_ld % 4 == 0), each token's sums
/// mv_q8p_kernel's (lane, order, expression, butterfly); a0 / a1 as mv_q8p_conv's (their bounds multiples of 4).
void mv_q8p_rows(const int8_t* q, const uint16_t* d, int N, int K, const float* x, int x_ld, int T, float* y, int ldy,
                 cudaStream_t s, const MvAlt& a0 = MvAlt{}, const MvAlt& a1 = MvAlt{});
/// attn_decode_f16_at + gate_mul_f16 for T queries at pos [T] (device), each split P ways as the decode splits one
/// token; q / r rows q_ld / r_ld apart, sink [T][H], gate rows gate_ld apart -> out [t] rows o_ld apart (F32, gated).
/// scratch: T x H x P x 130 floats.
void attn_rows_f16(const float* q, int q_ld, const float* r, int r_ld, const float* sink, const uint16_t* kc,
                   const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int P, float qk_scale,
                   const float* proj, int d_rel, int extent, int window, float* scratch, const float* gate,
                   int gate_ld, float* out, int o_ld, cudaStream_t s);
/// route()'s S = T K picks (S <= 64) as slots sorted by (expert, pick): slot tables blob / tok / sg / su by slot,
/// wt (w times down's scale) and inv (the pick's slot) by pick - moe_combine_f32's.
void dsc_slots(const int* top, const float* w, int S, int K, const uint8_t* const* tab, const float* sg_t,
               const float* su_t, const float* sd_t, const uint8_t** blob, int* tok, int* inv, float* sg, float* su,
               float* wt, cudaStream_t s);
/// sets 13 + 14 (set 16's tiles) over S slots, the slot the fast grid index: h [S][F], y [S][C] (NVFP4, 2304 / 1152).
void experts_rows_nv(const uint8_t* const* blob, const int* tok, int S, size_t eb0, size_t off_down, const float* x,
                     int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                     cudaStream_t s);

#endif  // STRATA_USE_HIP
}  // namespace ik2::k
