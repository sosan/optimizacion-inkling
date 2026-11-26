// tests/ik2_dsc_test.cu: the exact window's kernels (ik2_dsc.inc) against the decode's,
// bit for bit, on synthetic data at Inkling-2-Mini's shapes - every row / slot of a window kernel vs the decode kernel
// run on that token alone:
//   1a mv_q8p_rows (T 1 .. 8, K 2304 / 3072, a part with its own inputs, a ragged N) vs mv_q8p_conv a token at a time
//   1c attn_rows_f16 (a window layer: ring cache, P 16; a full layer: P 128) vs attn_decode_f16_at + gate_mul_f16
//   1b dsc_slots + experts_rows_nv vs experts_nv_visit (set 16) a token's 8 picks at a time; then moe_combine_f32 +
//      conv_block (the window's) vs moe_combine_conv a token at a time (the decode's, history carried)
//   (1a twice: x from cache and x through LDS - IK2_DSC bit 4's variant)
// Prints each check's mismatches; "dsc test: ALL OK" and exit 0, else exit 1.
// Arg "time": then each kernel's speed in-stream (events around repeated calls; weights from pools past the 32 MB
// Infinity Cache), the window's one launch against the decode's T: the dense head, one layer's experts, attention.  GPU: seconds, < 1 GiB (the head-sized product: 0.46 GB of int8).
// build: the AMD CMake build (ik2_dsc_test), or hipcc -x hip --offload-arch=gfx1151 -O3 -std=c++20
//        -ffp-contract=on -Isrc/ik2 -Iinclude -c tests/ik2_dsc_test.cu, linked with build/ik2_kernels.o
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <vector>
#include "ik2_kernels.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s -> %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
namespace ik2::k { extern int g_dsc_lds; }
namespace k = ik2::k;

static uint32_t hsh(uint32_t v) { v *= 2654435761u; v ^= v >> 15; v *= 2246822519u; v ^= v >> 13; return v; }
static float rnd(uint32_t i, float a) { return ((int) (hsh(i) % 20001) - 10000) * 1e-4f * a; }
template <class T> static T* up(const std::vector<T>& h) {
    T* d; CK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template <class T> static T* zeros(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
static uint16_t f16(float f) { const __half h = __float2half(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
static int fails = 0;
static void report(const char* what, size_t bad, size_t n) {
    std::printf("  %-58s %s (%zu of %zu differ)\n", what, bad ? "DIFFER" : "ok", bad, n);
    fails += bad != 0;
}
template <class T> static size_t ndiff(const T* a, const T* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += std::memcmp(a + i, b + i, sizeof(T)) != 0;
    return d;
}

// 1a
static void test_mv(int K, int N, int T) {
    std::vector<int8_t> q((size_t) N * K);
    std::vector<uint16_t> d((size_t) N * (K / 32));
    for (size_t i = 0; i < q.size(); ++i) q[i] = (int8_t) hsh((uint32_t) i * 7 + K);
    for (size_t i = 0; i < d.size(); ++i) d[i] = (uint16_t) (0x2000 + (hsh((uint32_t) i + 3) & 0x0FFF));
    const int x_ld = K + 128;   // (rows apart, a multiple of 4)
    std::vector<float> x(3 * (size_t) 8 * x_ld);   // three inputs: x, a0's, a1's
    for (size_t i = 0; i < x.size(); ++i) x[i] = rnd((uint32_t) i + 11, 1.0f);
    int8_t* dq = up(q); uint16_t* dd = up(d); float* dx = up(x);
    const size_t X = (size_t) 8 * x_ld;
    const k::MvAlt a0{dx + X, 256, 128}, a1{dx + 2 * X, 512, 64};
    const int ldy = N + 8;
    float* yw = zeros<float>((size_t) T * ldy);
    k::mv_q8p_rows(dq, dd, N, K, dx, x_ld, T, yw, ldy, nullptr, a0, a1);
    std::vector<float> ref((size_t) T * ldy);
    float* yr = zeros<float>(N);
    for (int t = 0; t < T; ++t) {
        k::mv_q8p_conv(dq, dd, N, K, dx + (size_t) t * x_ld, yr, k::MvConv{}, k::MvConv{}, nullptr,
                       k::MvAlt{a0.x + (size_t) t * x_ld, a0.r0, a0.n}, k::MvAlt{a1.x + (size_t) t * x_ld, a1.r0, a1.n});
        const auto h = down(yr, N);
        std::memcpy(ref.data() + (size_t) t * ldy, h.data(), (size_t) N * 4);
    }
    const auto w = down(yw, (size_t) T * ldy);
    size_t bad = 0;
    for (int t = 0; t < T; ++t) bad += ndiff(w.data() + (size_t) t * ldy, ref.data() + (size_t) t * ldy, N);
    char s[96];
    std::snprintf(s, sizeof s, "1a mv_q8p_rows%s K %d N %d T %d", k::g_dsc_lds ? " (LDS x)" : "", K, N, T);
    report(s, bad, (size_t) T * N);
    cudaFree(dq), cudaFree(dd), cudaFree(dx), cudaFree(yw), cudaFree(yr);
}

// 1c
static void test_attn(bool sliding, int T) {
    const int H = 24, KV = 8, D = 128, d_rel = 16, extent = 256, HD = H * D, KD = KV * D, HR = H * d_rel;
    const int window = sliding ? 1024 : (1 << 30), cap = sliding ? 1024 + 2048 : 4096, ring = sliding ? cap : 0;
    const int P = sliding ? std::min(128, std::max(1, (1024 + 63) / 64)) : 128;
    const int pos0 = sliding ? 5000 : 700;   // (the ring wrapped; a full layer's keys from 0)
    const int N = HD + 2 * KD + HR + HD;     // a window layer's fused q | k | v | r | gate rows
    std::vector<uint16_t> kc((size_t) cap * KD), vc((size_t) cap * KD);
    for (size_t i = 0; i < kc.size(); ++i) kc[i] = f16(rnd((uint32_t) i + 101, 1.0f)), vc[i] = f16(rnd((uint32_t) i + 202, 1.0f));
    std::vector<float> qkv((size_t) T * N), sink((size_t) T * H), proj((size_t) extent * d_rel);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = rnd((uint32_t) i + 303, 0.3f);
    for (size_t i = 0; i < sink.size(); ++i) sink[i] = rnd((uint32_t) i + 404, 2.0f);
    for (size_t i = 0; i < proj.size(); ++i) proj[i] = rnd((uint32_t) i + 505, 0.5f);
    std::vector<int> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = pos0 + t;
    uint16_t *dk = up(kc), *dv = up(vc);
    float *dq = up(qkv), *ds = up(sink), *dp = up(proj);
    int* dpos = up(pos);
    const float* q = dq;
    const float *r = dq + HD + 2 * KD, *g = r + HR;
    const float qk_scale = 1.0f / std::sqrt((float) D);
    float* scratch = zeros<float>((size_t) 8 * H * 128 * 130);
    float* ow = zeros<float>((size_t) T * HD);
    k::attn_rows_f16(q, N, r, N, ds, dk, dv, ring, H, KV, dpos, T, P, qk_scale, dp, d_rel, extent, window, scratch, g, N,
                     ow, HD, nullptr);
    const auto w = down(ow, (size_t) T * HD);
    float* orf = zeros<float>(HD);
    uint16_t* o16 = zeros<uint16_t>(HD);
    size_t bad = 0;
    for (int t = 0; t < T; ++t) {
        k::attn_decode_f16_at(q + (size_t) t * N, r + (size_t) t * N, ds + (size_t) t * H, dk, dv, ring, H, KV, dpos + t,
                              P, qk_scale, dp, d_rel, extent, window, scratch, orf, nullptr);
        k::gate_mul_f16(orf, g + (size_t) t * N, HD, o16, true, nullptr);
        const auto h = down(orf, HD);
        bad += ndiff(w.data() + (size_t) t * HD, h.data(), HD);
    }
    char s[96];
    std::snprintf(s, sizeof s, "1c attn_rows_f16 %s P %d T %d at %d", sliding ? "window layer" : "full layer", P, T, pos0);
    report(s, bad, (size_t) T * HD);
    cudaFree(dk), cudaFree(dv), cudaFree(dq), cudaFree(ds), cudaFree(dp), cudaFree(dpos), cudaFree(scratch), cudaFree(ow);
    cudaFree(orf), cudaFree(o16);
}

// 1b
static void test_experts(int T) {
    const int C = 2304, F = 1152, K = 8, E = 12, S = T * K;
    const size_t RG = (size_t) C / 64 * 36, RD = (size_t) F / 64 * 36;
    const size_t EB0 = F * RG, EB1 = F * RG, EB2 = C * RD, EX = (EB0 + EB1 + EB2 + 255) & ~(size_t) 255;
    std::vector<uint8_t> hb(EX * E);
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = (uint8_t) hsh((uint32_t) i ^ 0x5bd1e995u);
    for (int e = 0; e < E; ++e) {   // NVFP4 scale bytes E4M3 in 0x28 .. 0x47 (ik2_experts_test's)
        auto fix = [&](size_t off, size_t rows, size_t rb) {
            for (size_t rr = 0; rr < rows; ++rr)
                for (size_t b = 0; b < rb / 36; ++b)
                    for (int qq = 0; qq < 4; ++qq) {
                        uint8_t& v = hb[e * EX + off + rr * rb + b * 36 + qq];
                        v = (uint8_t) (0x28 + (v & 0x1F));
                    }
        };
        fix(0, F, RG), fix(EB0, F, RG), fix(EB0 + EB1, C, RD);
    }
    uint8_t* dex = up(hb);
    std::vector<const uint8_t*> tab(E);
    for (int e = 0; e < E; ++e) tab[e] = dex + (size_t) e * EX;
    std::vector<float> sg_t(E), su_t(E), sd_t(E);
    for (int e = 0; e < E; ++e) sg_t[e] = 0.5f + e * 0.07f, su_t[e] = 0.8f + e * 0.03f, sd_t[e] = 1.1f - e * 0.02f;
    std::vector<int> top(S);
    std::vector<float> w(S), x((size_t) T * C);
    for (int t = 0; t < T; ++t) {   // K distinct experts a token, ascending (route's order), overlapping across tokens
        int n = 0;
        for (int e = 0; e < E && n < K; ++e)
            if ((hsh((uint32_t) (t * 31 + e)) % 3) != 0 || E - e <= K - n) top[t * K + n++] = e;
        for (int j = 0; j < K; ++j) w[t * K + j] = 0.05f + (hsh((uint32_t) (t * 8 + j) + 9) % 100) * 1e-3f;
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] = rnd((uint32_t) i + 606, 1.0f);
    const uint8_t** dtab = up(tab);
    float *dsg = up(sg_t), *dsu = up(su_t), *dsd = up(sd_t), *dw = up(w), *dx = up(x);
    int* dtop = up(top);
    const uint8_t** blob = zeros<const uint8_t*>(64);
    int *tok = zeros<int>(64), *inv = zeros<int>(64);
    float *sg = zeros<float>(64), *su = zeros<float>(64), *wt = zeros<float>(64);
    uint16_t* h = zeros<uint16_t>((size_t) 64 * F);
    float* y = zeros<float>((size_t) 64 * C);
    const float gcap = 10.0f, ucap = 12.0f;
    k::dsc_slots(dtop, dw, S, K, dtab, dsg, dsu, dsd, blob, tok, inv, sg, su, wt, nullptr);
    k::experts_rows_nv(blob, tok, S, EB0, EB0 + EB1, dx, C, F, sg, su, gcap, ucap, h, y, nullptr);
    const auto hinv = down(inv, S);
    const auto hblob = down(blob, S);
    const auto hh = down(h, (size_t) S * F);
    const auto hy = down(y, (size_t) S * C);
    {   // the slots: a permutation, sorted by expert
        std::vector<int> seen(S, 0);
        size_t bad = 0;
        for (int i = 0; i < S; ++i) bad += hinv[i] < 0 || hinv[i] >= S || seen[hinv[i]]++;
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                if (top[i] < top[j] && hinv[i] > hinv[j]) ++bad;
        for (int i = 0; i < S; ++i) bad += hblob[hinv[i]] != tab[top[i]];
        report("1b dsc_slots: a permutation sorted by expert, blob right", bad, S);
    }
    // the decode: a token's K picks, set 16 (gate_up + down in one launch)
    const uint8_t** blob1 = zeros<const uint8_t*>(K);
    int* tok1 = zeros<int>(K);
    float *sg1 = zeros<float>(K), *su1 = zeros<float>(K);
    uint16_t* h1 = zeros<uint16_t>((size_t) K * F);
    float* y1 = zeros<float>((size_t) K * C);
    size_t bh = 0, by = 0;
    std::vector<std::vector<float>> yref(T);
    for (int t = 0; t < T; ++t) {
        std::vector<const uint8_t*> b(K);
        std::vector<float> a(K), u(K);
        for (int j = 0; j < K; ++j) b[j] = tab[top[t * K + j]], a[j] = sg_t[top[t * K + j]], u[j] = su_t[top[t * K + j]];
        CK(cudaMemcpy(blob1, b.data(), K * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sg1, a.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(su1, u.data(), K * 4, cudaMemcpyHostToDevice));
        k::experts_nv_visit(blob1, tok1, K, EB0, EB0 + EB1, dx + (size_t) t * C, C, F, sg1, su1, gcap, ucap, h1, y1, nullptr);
        const auto rh = down(h1, (size_t) K * F);
        yref[t] = down(y1, (size_t) K * C);
        for (int j = 0; j < K; ++j) {
            const int sl = hinv[t * K + j];
            bh += ndiff(hh.data() + (size_t) sl * F, rh.data() + (size_t) j * F, F);
            by += ndiff(hy.data() + (size_t) sl * C, yref[t].data() + (size_t) j * C, C);
        }
    }
    char s[96];
    std::snprintf(s, sizeof s, "1b experts_rows_nv h (FP16) vs set 16, T %d", T);
    report(s, bh, (size_t) S * F);
    std::snprintf(s, sizeof s, "1b experts_rows_nv y (F32) vs set 16, T %d", T);
    report(s, by, (size_t) S * C);
    // the combine and the MoE's short conv: moe_combine_f32 + conv_block (window) vs moe_combine_conv (decode)
    std::vector<float> hist((size_t) 3 * C), cw((size_t) C * 4), res((size_t) T * C);
    for (size_t i = 0; i < hist.size(); ++i) hist[i] = rnd((uint32_t) i + 707, 1.0f);
    for (size_t i = 0; i < cw.size(); ++i) cw[i] = rnd((uint32_t) i + 808, 0.6f);
    for (size_t i = 0; i < res.size(); ++i) res[i] = rnd((uint32_t) i + 909, 1.0f);
    float *dh_w = up(hist), *dh_r = up(hist), *dcw = up(cw), *dres = up(res);
    float *tmp = zeros<float>((size_t) T * C), *out_w = zeros<float>((size_t) T * C), *out_r = zeros<float>(C);
    float *rec_old = zeros<float>((size_t) 3 * C), *rec_x = zeros<float>((size_t) T * C);
    k::moe_combine_f32(y, inv, wt, T, K, C, tmp, false, nullptr);
    k::conv_block(tmp, C, dh_w, dcw, nullptr, 0, 0, T, C, dres, C, out_w, C, rec_old, rec_x, nullptr);
    const auto ow = down(out_w, (size_t) T * C);
    const auto hwt = down(wt, S);
    std::vector<float> wref(S);   // the decode's wt, computed here: w * down's scale (route_gather's one product)
    for (int i = 0; i < S; ++i) wref[i] = w[i] * sd_t[top[i]];
    std::snprintf(s, sizeof s, "1b dsc_slots wt vs w * sd (host), T %d", T);
    report(s, ndiff(hwt.data(), wref.data(), S), S);
    std::vector<int> id(K);
    for (int j = 0; j < K; ++j) id[j] = j;
    int* did = up(id);
    float* wt1 = zeros<float>(K);
    size_t bc = 0;
    for (int t = 0; t < T; ++t) {   // (the decode's wt: w * down's scale, the same product - pair 9)
        CK(cudaMemcpy(y1, yref[t].data(), (size_t) K * C * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(wt1, wref.data() + (size_t) t * K, K * 4, cudaMemcpyHostToDevice));
        k::moe_combine_conv(y1, did, wt1, K, C, dh_r, dcw, dres + (size_t) t * C, out_r, nullptr);
        const auto o = down(out_r, C);
        bc += ndiff(ow.data() + (size_t) t * C, o.data(), C);
    }
    const auto hw = down(dh_w, (size_t) 3 * C), hr = down(dh_r, (size_t) 3 * C);
    std::snprintf(s, sizeof s, "1b moe_combine_f32 + conv_block vs moe_combine_conv, T %d", T);
    report(s, bc, (size_t) T * C);
    report("   ... the conv history after the window", ndiff(hw.data(), hr.data(), (size_t) 3 * C), (size_t) 3 * C);
    cudaFree(dex), cudaFree(dtab), cudaFree(dsg), cudaFree(dsu), cudaFree(dsd), cudaFree(dw), cudaFree(dx), cudaFree(dtop);
    cudaFree(blob), cudaFree(tok), cudaFree(inv), cudaFree(sg), cudaFree(su), cudaFree(wt), cudaFree(h), cudaFree(y);
    cudaFree(blob1), cudaFree(tok1), cudaFree(sg1), cudaFree(su1), cudaFree(h1), cudaFree(y1), cudaFree(dh_w), cudaFree(dh_r);
    cudaFree(dcw), cudaFree(dres), cudaFree(tmp), cudaFree(out_w), cudaFree(out_r), cudaFree(rec_old), cudaFree(rec_x);
    cudaFree(did), cudaFree(wt1);
}

// ---- "time"
static float time_ms(int reps, const std::function<void(int)>& f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(0);   // (warm: code, page tables)
    CK(cudaEventRecord(a, nullptr));
    for (int i = 0; i < reps; ++i) f(i);
    CK(cudaEventRecord(b, nullptr));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a), cudaEventDestroy(b);
    return ms / reps;
}
static void time_dense() {   // the head: 201,024 x 2304 planar Q8_0 (0.49 GB a read: DRAM every call)
    const int N = 201024, K = 2304, x_ld = K;
    int8_t* q; uint16_t* d;
    CK(cudaMalloc(&q, (size_t) N * K)); CK(cudaMalloc(&d, (size_t) N * (K / 32) * 2));
    CK(cudaMemset(q, 3, (size_t) N * K)); CK(cudaMemset(d, 0x20, (size_t) N * (K / 32) * 2));
    std::vector<float> hx((size_t) 8 * x_ld);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = rnd((uint32_t) i, 1.0f);
    float* x = up(hx);
    float* y = zeros<float>((size_t) 8 * N);
    const double gb = ((double) N * K + (double) N * (K / 32) * 2) / 1e9;
    std::printf("time, dense head %d x %d (%.2f GB):  T  decode T calls (us)  window x cache (us, GB/s)  window x LDS\n", N, K, gb);
    for (int T = 1; T <= 8; ++T) {
        const float td = time_ms(5, [&](int) {
            for (int t = 0; t < T; ++t) k::mv_q8p_conv(q, d, N, K, x + (size_t) t * x_ld, y + (size_t) t * N, k::MvConv{}, k::MvConv{}, nullptr);
        });
        float tw[2];
        for (int v = 0; v < 2; ++v) {
            k::g_dsc_lds = v;
            tw[v] = time_ms(5, [&](int) { k::mv_q8p_rows(q, d, N, K, x, x_ld, T, y, N, nullptr); });
        }
        std::printf("  %d  %8.0f  %8.0f (%4.0f)  %8.0f (%4.0f)\n", T, td * 1e3, tw[0] * 1e3, gb / tw[0] * 1e3, tw[1] * 1e3,
                    gb / tw[1] * 1e3);
    }
    cudaFree(q), cudaFree(d), cudaFree(x), cudaFree(y);
}
static void time_experts() {   // one layer's 128 NVFP4 experts (0.57 GB); picks: 5 of a hot 12 + 3 others a token
    const int C = 2304, F = 1152, K = 8, E = 128, NR = 8;
    const size_t RG = (size_t) C / 64 * 36, RD = (size_t) F / 64 * 36;
    const size_t EB0 = F * RG, EB1 = F * RG, EX = (EB0 + EB1 + C * RD + 255) & ~(size_t) 255;
    uint8_t* dex;
    CK(cudaMalloc(&dex, EX * E));
    CK(cudaMemset(dex, 0x33, EX * E));   // (scale bytes 0x33: E4M3 ~0.6; nibbles 3)
    std::vector<const uint8_t*> tab(E);
    for (int e = 0; e < E; ++e) tab[e] = dex + (size_t) e * EX;
    const uint8_t** dtab = up(tab);
    float* ones = up(std::vector<float>(E, 1.0f));
    std::vector<float> hx((size_t) 8 * C);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = rnd((uint32_t) i + 1, 0.1f);
    float* x = up(hx);
    const uint8_t** blob = zeros<const uint8_t*>(64);
    int *tok = zeros<int>(64), *inv = zeros<int>(64), *tok0 = zeros<int>(K);
    float *sg = zeros<float>(64), *su = zeros<float>(64), *wt = zeros<float>(64);
    uint16_t* h = zeros<uint16_t>((size_t) 64 * F);
    float* y = zeros<float>((size_t) 64 * C);
    std::printf("time, one layer's experts (NVFP4, %.1f MB each):  T  distinct  decode T x set 16 (us)  window lazy (us)  union bound at 233 GB/s (us)\n",
                EX / 1e6);
    for (int T = 1; T <= 8; ++T) {
        const int S = T * K;
        std::vector<int> top((size_t) NR * S);
        std::vector<const uint8_t*> b1((size_t) NR * S);
        double dist = 0;
        for (int r = 0; r < NR; ++r) {
            const int hot0 = (r * 37) % (E - 12);   // (each routing its own experts: no cache help across calls)
            std::vector<int> used(E, 0);
            for (int t = 0; t < T; ++t) {
                std::vector<int> pick;
                for (int j = 0; pick.size() < 5; ++j) {
                    const int e = hot0 + (int) (hsh((uint32_t) (r * 1000 + t * 50 + j)) % 12);
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                for (int j = 0; pick.size() < (size_t) K; ++j) {
                    const int e = (int) (hsh((uint32_t) (r * 7777 + t * 333 + j) + 99) % E);
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                std::sort(pick.begin(), pick.end());
                for (int j = 0; j < K; ++j) top[(size_t) r * S + t * K + j] = pick[j], b1[(size_t) r * S + t * K + j] = tab[pick[j]], used[pick[j]] = 1;
            }
            for (int e = 0; e < E; ++e) dist += used[e];
        }
        dist /= NR;
        int* dtop = up(top);
        const uint8_t** db1 = up(b1);
        float* w = up(std::vector<float>((size_t) NR * S, 0.1f));
        const float td = time_ms(NR, [&](int r) {
            for (int t = 0; t < T; ++t)
                k::experts_nv_visit(db1 + (size_t) r * S + t * K, tok0, K, EB0, EB0 + EB1, x + (size_t) t * C, C, F, ones, ones,
                                    10.0f, 12.0f, h, y, nullptr);
        });
        const float tw = time_ms(NR, [&](int r) {
            k::dsc_slots(dtop + (size_t) r * S, w + (size_t) r * S, S, K, dtab, ones, ones, ones, blob, tok, inv, sg, su, wt, nullptr);
            k::experts_rows_nv(blob, tok, S, EB0, EB0 + EB1, x, C, F, sg, su, 10.0f, 12.0f, h, y, nullptr);
        });
        std::printf("  %d  %5.1f  %8.0f  %8.0f  %8.0f\n", T, dist, td * 1e3, tw * 1e3, dist * EX / 233e9 * 1e6);
        cudaFree(dtop), cudaFree(db1), cudaFree(w);
    }
    cudaFree(dex), cudaFree(dtab), cudaFree(ones), cudaFree(x), cudaFree(blob), cudaFree(tok), cudaFree(inv), cudaFree(tok0);
    cudaFree(sg), cudaFree(su), cudaFree(wt), cudaFree(h), cudaFree(y);
}
static void time_attn() {   // a full layer at 4K keys (P 128) and a window layer (1024, P 16): T x decode vs one window
    const int H = 24, KV = 8, D = 128, d_rel = 16, extent = 256, HD = H * D, KD = KV * D, cap = 4096 + 16;
    std::vector<uint16_t> kc((size_t) cap * KD);
    for (size_t i = 0; i < kc.size(); ++i) kc[i] = f16(rnd((uint32_t) i, 1.0f));
    uint16_t *dk = up(kc), *dv = up(kc);
    std::vector<float> qkv((size_t) 8 * 2 * HD), sink((size_t) 8 * H, 0.0f), proj((size_t) extent * d_rel);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = rnd((uint32_t) i + 5, 0.3f);
    for (size_t i = 0; i < proj.size(); ++i) proj[i] = rnd((uint32_t) i + 9, 0.5f);
    float *dq = up(qkv), *ds = up(sink), *dp = up(proj);
    std::vector<int> pos(8);
    for (int t = 0; t < 8; ++t) pos[t] = 4096 + t - 8;
    int* dpos = up(pos);
    float* scratch = zeros<float>((size_t) 8 * H * 128 * 130);
    float* o = zeros<float>((size_t) 8 * HD);
    uint16_t* o16 = zeros<uint16_t>((size_t) 8 * HD);
    const int ld = 2 * HD;   // (q then the gate, a row)
    std::printf("time, attention at 4K:  T  full: decode T x (split + combine + gate) (us)  window  |  window layer: decode  window\n");
    for (int T = 1; T <= 8; ++T) {
        float tt[2][2];
        for (int full = 0; full < 2; ++full) {
            const int window = full ? (1 << 30) : 1024, P = full ? 128 : 16, ring = full ? 0 : cap;
            tt[full][0] = time_ms(20, [&](int) {
                for (int t = 0; t < T; ++t) {
                    k::attn_decode_f16_at(dq + (size_t) t * ld, dq, ds + (size_t) t * H, dk, dv, ring, H, KV, dpos + t, P,
                                          0.088f, dp, d_rel, extent, window, scratch, o, nullptr);
                    k::gate_mul_f16(o, dq + (size_t) t * ld + HD, HD, o16, true, nullptr);
                }
            });
            tt[full][1] = time_ms(20, [&](int) {
                k::attn_rows_f16(dq, ld, dq, 0, ds, dk, dv, ring, H, KV, dpos, T, P, 0.088f, dp, d_rel, extent, window,
                                 scratch, dq + HD, ld, o, HD, nullptr);
            });
        }
        std::printf("  %d  %8.0f  %8.0f  |  %8.0f  %8.0f\n", T, tt[1][0] * 1e3, tt[1][1] * 1e3, tt[0][0] * 1e3, tt[0][1] * 1e3);
    }
    cudaFree(dk), cudaFree(dv), cudaFree(dq), cudaFree(ds), cudaFree(dp), cudaFree(dpos), cudaFree(scratch), cudaFree(o), cudaFree(o16);
}

int main(int argc, char** argv) {
    std::printf("dsc test (ik2_dsc.inc vs the decode's kernels, bit for bit):\n");
    for (int v = 0; v < 2; ++v) {
        k::g_dsc_lds = v;
        for (int T : {1, 2, 5, 8}) test_mv(2304, 1030, T);
        test_mv(3072, 2304, 7);
        test_mv(2304, 201024, 5);   // (the head's rows)
    }
    for (int T : {2, 5, 8}) test_attn(true, T), test_attn(false, T);
    for (int T : {2, 5, 8}) test_experts(T);
    std::printf(fails ? "dsc test: %d checks DIFFER\n" : "dsc test: ALL OK\n", fails);
    if (argc > 1 && !std::strcmp(argv[1], "time")) time_dense(), time_experts(), time_attn();
    return fails ? 1 : 0;
}
