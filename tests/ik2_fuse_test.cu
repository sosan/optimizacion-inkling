// tests/ik2_fuse_test.cu - IK2_FUSE's kernels (src/ik2/ik2_fuse.inc) against the kernels they replace, on the GPU,
// random data at Inkling-2-Mini's shapes (C 2304, H 24, KV 8, D 128, d_rel 16, E 128, K 8): every output compared
// bit for bit (memcmp); a difference prints its count and largest |a - b|.
#include "hip/hip_runtime.h"
#include "../src/ik2/ik2_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = ik2::k;

namespace {

int g_fail = 0;
std::mt19937 rng(20261122);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            std::printf("HIP %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);       \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }
std::vector<float> fvec(size_t n, float a = 1.0f) {
    std::vector<float> v(n);
    for (auto& x : v) x = frand(a);
    return v;
}
std::vector<float> pvec(size_t n, float lo, float hi) {   // positive
    std::vector<float> v(n);
    for (auto& x : v) x = std::uniform_real_distribution<float>(lo, hi)(rng);
    return v;
}
// FP16 bits of moderate random values: sign, exponent 2^-9 .. 2^0, any mantissa
std::vector<uint16_t> hvec(size_t n) {
    std::vector<uint16_t> v(n);
    for (auto& x : v)
        x = (uint16_t) (((rng() & 1u) << 15) | ((6u + rng() % 10u) << 10) | (rng() & 1023u));
    return v;
}

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
T* dev_n(size_t n) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    CU(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    CU(cudaDeviceSynchronize());
    CU(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template <class T>
void put(T* p, const std::vector<T>& h) {
    CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

double as_double(float v) { return v; }
double as_double(uint16_t v) {   // FP16 bits
    const int e = (v >> 10) & 31, m = v & 1023;
    const double s = (v >> 15) ? -1.0 : 1.0;
    if (e == 0) return s * std::ldexp((double) m, -24);
    if (e == 31) return m ? NAN : s * INFINITY;
    return s * std::ldexp(1024.0 + m, e - 25);
}
double as_double(int v) { return v; }
double as_double(const uint8_t* v) { return (double) (uintptr_t) v; }

// a (fused) against b (the kernels it replaces), bit for bit
template <class T>
bool same(const char* what, const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T))) {
        std::printf("  ok   %-44s %zu values bit for bit\n", what, a.size());
        return true;
    }
    size_t bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T))) {
            ++bad;
            const double d = std::fabs(as_double(a[i]) - as_double(b[i]));
            worst = std::isnan(d) ? INFINITY : std::max(worst, d);
        }
    std::printf("  FAIL %-44s %zu of %zu differ, max |a - b| %.3g\n", what, bad, a.size(), worst);
    ++g_fail;
    return false;
}

constexpr int C = 2304, H = 24, KV = 8, D = 128, DREL = 16, E = 128, KTOP = 8;

// us an iteration of f, 200 back to back on stream 0 (a stream's launches as the decode's: each after the last)
template <class F>
double us_per(F f) {
    for (int i = 0; i < 20; ++i) f();
    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    CU(cudaEventRecord(a, 0));
    for (int i = 0; i < 200; ++i) f();
    CU(cudaEventRecord(b, 0));
    CU(cudaEventSynchronize(b));
    float ms = 0.0f;
    CU(cudaEventElapsedTime(&ms, a, b));
    CU(cudaEventDestroy(a));
    CU(cudaEventDestroy(b));
    return ms * 1000.0 / 200.0;
}
template <class F0, class F1>
void timing(const char* what, F0 unfused, F1 fused) {
    const double u = us_per(unfused), f = us_per(fused);
    std::printf("  time %-44s unfused %.2f us, fused %.2f us\n", what, u, f);
}

void test_mv_f16_pair() {
    std::printf("fuse_mv_f16_pair (a full layer's dynamic taps) vs 2 x mv_f16\n");
    const int N = KV * 4;
    uint16_t *W0 = dev(hvec((size_t) N * C)), *W1 = dev(hvec((size_t) N * C));
    float *x0 = dev(fvec(C)), *x1 = dev(fvec(C, 3.0f));
    float *a = dev_n<float>(2 * N), *b = dev_n<float>(2 * N);
    k::mv_f16(W0, N, C, x0, b, 0);
    k::mv_f16(W1, N, C, x1, b + N, 0);
    if (!k::fuse_mv_f16_pair(W0, x0, a, N, W1, x1, a + N, N, C, 0)) {
        std::printf("  FAIL not launched\n"), ++g_fail;
        return;
    }
    same("y0 | y1", host(a, 2 * N), host(b, 2 * N));
    timing("dynamic taps", [&] { k::mv_f16(W0, N, C, x0, b, 0); k::mv_f16(W1, N, C, x1, b + N, 0); },
           [&] { k::fuse_mv_f16_pair(W0, x0, a, N, W1, x1, a + N, N, C, 0); });
    for (void* p : {(void*) W0, (void*) W1, (void*) x0, (void*) x1, (void*) a, (void*) b}) CU(cudaFree(p));
}

void test_router_route(bool lds = false) {
    std::printf("fuse_router_route%s vs mv_f16 + route_token (3 launches each: the last-block counter goes back to 0)\n",
                lds ? " (LDS, set 6)" : "");
    uint16_t* Wr = dev(hvec((size_t) E * C));
    float* bias = dev(fvec(E, 0.05f));
    std::vector<const uint8_t*> tab_h(E);
    for (int e = 0; e < E; ++e) tab_h[e] = (const uint8_t*) (uintptr_t) (0x100000u + 4096u * e);
    const uint8_t** tab = dev(tab_h);
    float *sg_t = dev(pvec(E, 0.5f, 2.0f)), *su_t = dev(pvec(E, 0.5f, 2.0f)), *sd_t = dev(pvec(E, 0.5f, 2.0f));
    float* x = dev_n<float>(C);
    struct Out {
        float* logits = dev_n<float>(E);
        int* top = dev_n<int>(KTOP);
        float* w = dev_n<float>(KTOP);
        const uint8_t** blob = dev_n<const uint8_t*>(KTOP);
        int* tok = dev_n<int>(KTOP);
        int* inv = dev_n<int>(KTOP);
        float* sg = dev_n<float>(KTOP);
        float* su = dev_n<float>(KTOP);
        float* wt = dev_n<float>(KTOP);
    } A, B;
    for (int rep = 0; rep < 3; ++rep) {
        put(x, fvec(C, 1.0f + rep));
        k::mv_f16(Wr, E, C, x, B.logits, 0);
        k::route_token(B.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, B.top, B.w, B.blob, B.tok, B.inv, B.sg,
                       B.su, B.wt, 0);
        if (!k::fuse_router_route(Wr, x, C, A.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, A.top, A.w, A.blob,
                                  A.tok, A.inv, A.sg, A.su, A.wt, 0, lds)) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        std::printf(" launch %d:\n", rep + 1);
        same("logits", host(A.logits, E), host(B.logits, E));
        same("top", host(A.top, KTOP), host(B.top, KTOP));
        same("w", host(A.w, KTOP), host(B.w, KTOP));
        same("blob", host(A.blob, KTOP), host(B.blob, KTOP));
        same("tok", host(A.tok, KTOP), host(B.tok, KTOP));
        same("inv", host(A.inv, KTOP), host(B.inv, KTOP));
        same("sg", host(A.sg, KTOP), host(B.sg, KTOP));
        same("su", host(A.su, KTOP), host(B.su, KTOP));
        same("wt", host(A.wt, KTOP), host(B.wt, KTOP));
    }
    timing("router + route",
           [&] {
               k::mv_f16(Wr, E, C, x, B.logits, 0);
               k::route_token(B.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, B.top, B.w, B.blob, B.tok, B.inv,
                              B.sg, B.su, B.wt, 0);
           },
           [&] {
               k::fuse_router_route(Wr, x, C, A.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, A.top, A.w, A.blob,
                                    A.tok, A.inv, A.sg, A.su, A.wt, 0, lds);
           });
    timing("router mv_f16 alone", [&] { k::mv_f16(Wr, E, C, x, B.logits, 0); },
           [&] { k::mv_f16(Wr, E, C, x, B.logits, 0); });
}

void test_prep(bool sliding, int pos_v) {
    std::printf("fuse_prep_qkv_r_f16_at vs prep_qkv_r_f16_at (%s, position %d)\n", sliding ? "window layer" : "full layer",
                pos_v);
    const int kv_cap = sliding ? 1024 : 0, slots = sliding ? 1024 : 8192;
    const auto q0 = fvec((size_t) H * D, 4.0f), r0 = fvec((size_t) H * DREL, 4.0f);
    const auto k_in = fvec((size_t) KV * D, 4.0f), v_in = fvec((size_t) KV * D, 4.0f);
    float *q_clip = dev(pvec((size_t) H * D, 0.5f, 1.5f)), *r_clip = dev(pvec((size_t) H * DREL, 0.5f, 1.5f));
    float *kd = dev(k_in), *vd = dev(v_in);
    uint16_t* sink_w = dev(hvec((size_t) H * C));
    float* sink_x = dev(fvec(C));
    int* pos = dev(std::vector<int>{pos_v});
    const auto cache0 = hvec((size_t) slots * KV * D);
    struct Out {
        float *q, *r, *sink;
        uint16_t *kc, *vc;
    } A, B;
    for (Out* o : {&A, &B}) {
        o->q = dev(q0), o->r = dev(r0), o->sink = dev_n<float>(H);
        o->kc = dev(cache0), o->vc = dev(cache0);
    }
    k::prep_qkv_r_f16_at(B.q, q_clip, kd, B.kc, vd, B.vc, B.r, r_clip, H, KV, D, DREL, 22.627417f, 8.0f, 1e-6f, pos,
                         !sliding, 0.1f, 8192.0f, kv_cap, nullptr, 0, sink_w, sink_x, C, B.sink);
    if (!k::fuse_prep_qkv_r_f16_at(A.q, q_clip, kd, A.kc, vd, A.vc, A.r, r_clip, H, KV, D, DREL, 22.627417f, 8.0f,
                                   1e-6f, pos, !sliding, 0.1f, 8192.0f, kv_cap, nullptr, 0, sink_w, sink_x, C,
                                   A.sink)) {
        std::printf("  FAIL not launched\n"), ++g_fail;
        return;
    }
    same("q (clipped in place)", host(A.q, (size_t) H * D), host(B.q, (size_t) H * D));
    same("r (clipped in place)", host(A.r, (size_t) H * DREL), host(B.r, (size_t) H * DREL));
    same("sinks", host(A.sink, H), host(B.sink, H));
    same("K cache", host(A.kc, cache0.size()), host(B.kc, cache0.size()));
    same("V cache", host(A.vc, cache0.size()), host(B.vc, cache0.size()));
    timing("prep_qkv_r", [&] {
        k::prep_qkv_r_f16_at(B.q, q_clip, kd, B.kc, vd, B.vc, B.r, r_clip, H, KV, D, DREL, 22.627417f, 8.0f, 1e-6f,
                             pos, !sliding, 0.1f, 8192.0f, kv_cap, nullptr, 0, sink_w, sink_x, C, B.sink);
    }, [&] {
        k::fuse_prep_qkv_r_f16_at(A.q, q_clip, kd, A.kc, vd, A.vc, A.r, r_clip, H, KV, D, DREL, 22.627417f, 8.0f,
                                  1e-6f, pos, !sliding, 0.1f, 8192.0f, kv_cap, nullptr, 0, sink_w, sink_x, C, A.sink);
    });
}

void test_attn_gate(bool sliding, int pos_v, bool skip = false) {
    const int P = sliding ? 16 : 128;
    std::printf("fuse_attn_decode_f16_at_gate%s vs attn_decode_f16_at + gate_mul_f16 (%s, P %d, position %d)\n",
                skip ? " (skip, set 7)" : "", sliding ? "window layer" : "full layer", P, pos_v);
    const int kv_cap = sliding ? 1024 : 0, slots = sliding ? 1024 : 8192, window = sliding ? 1024 : (1 << 30);
    float *q = dev(fvec((size_t) H * D, 0.3f)), *r = dev(fvec((size_t) H * DREL, 1.0f));
    float* sink = dev(fvec(H, 2.0f));
    uint16_t *kc = dev(hvec((size_t) slots * KV * D)), *vc = dev(hvec((size_t) slots * KV * D));
    float* proj = dev(fvec((size_t) 1024 * DREL, 0.5f));
    int* pos = dev(std::vector<int>{pos_v});
    float* gate = dev(fvec((size_t) H * D, 4.0f));
    float *sa = dev_n<float>((size_t) H * P * 130), *sb = dev_n<float>((size_t) H * P * 130);
    float *oa = dev_n<float>((size_t) H * D), *ob = dev_n<float>((size_t) H * D);
    uint16_t *ha = dev_n<uint16_t>((size_t) H * D), *hb = dev_n<uint16_t>((size_t) H * D);
    const float qk_scale = 1.0f / std::sqrt((float) D);
    k::attn_decode_f16_at(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window, sb, ob, 0);
    k::gate_mul_f16(ob, gate, (size_t) H * D, hb, true, 0);
    if (!k::fuse_attn_decode_f16_at_gate(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window,
                                         sa, gate, oa, ha, 0, skip)) {
        std::printf("  FAIL not launched\n"), ++g_fail;
        return;
    }
    same("out F32 (o sigmoid(gate))", host(oa, (size_t) H * D), host(ob, (size_t) H * D));
    same("out FP16", host(ha, (size_t) H * D), host(hb, (size_t) H * D));
    timing("attention + gate", [&] {
        k::attn_decode_f16_at(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window, sb, ob, 0);
        k::gate_mul_f16(ob, gate, (size_t) H * D, hb, true, 0);
    }, [&] {
        k::fuse_attn_decode_f16_at_gate(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window,
                                        sa, gate, oa, ha, 0, skip);
    });
    if (skip)
        timing("set 1 (no skip) vs set 7 (skip)", [&] {
            k::fuse_attn_decode_f16_at_gate(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024,
                                            window, sa, gate, oa, ha, 0, false);
        }, [&] {
            k::fuse_attn_decode_f16_at_gate(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024,
                                            window, sa, gate, oa, ha, 0, true);
        });
}

void test_attn_one(bool sliding, int pos_v) {
    const int P = sliding ? 16 : 128;
    std::printf("fuse_attn_one_f16_at_gate vs attn_decode_f16_at + gate_mul_f16 (%s, P %d, position %d, 2 launches)\n",
                sliding ? "window layer" : "full layer", P, pos_v);
    const int kv_cap = sliding ? 1024 : 0, slots = sliding ? 1024 : 8192, window = sliding ? 1024 : (1 << 30);
    float *q = dev(fvec((size_t) H * D, 0.3f)), *r = dev(fvec((size_t) H * DREL, 1.0f));
    float* sink = dev(fvec(H, 2.0f));
    uint16_t *kc = dev(hvec((size_t) slots * KV * D)), *vc = dev(hvec((size_t) slots * KV * D));
    float* proj = dev(fvec((size_t) 1024 * DREL, 0.5f));
    int* pos = dev(std::vector<int>{pos_v});
    float* gate = dev(fvec((size_t) H * D, 4.0f));
    float *sa = dev_n<float>((size_t) H * P * 130), *sb = dev_n<float>((size_t) H * P * 130);
    float *oa = dev_n<float>((size_t) H * D), *ob = dev_n<float>((size_t) H * D);
    uint16_t *ha = dev_n<uint16_t>((size_t) H * D), *hb = dev_n<uint16_t>((size_t) H * D);
    const float qk_scale = 1.0f / std::sqrt((float) D);
    k::attn_decode_f16_at(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window, sb, ob, 0);
    k::gate_mul_f16(ob, gate, (size_t) H * D, hb, true, 0);
    for (int rep = 0; rep < 2; ++rep) {
        CU(cudaMemset(oa, 0, (size_t) H * D * 4));
        if (!k::fuse_attn_one_f16_at_gate(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window,
                                          sa, gate, oa, ha, 0)) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        same("out F32 (o sigmoid(gate))", host(oa, (size_t) H * D), host(ob, (size_t) H * D));
        same("out FP16", host(ha, (size_t) H * D), host(hb, (size_t) H * D));
    }
    timing("attention + gate (one launch)", [&] {
        k::attn_decode_f16_at(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window, sb, ob, 0);
        k::gate_mul_f16(ob, gate, (size_t) H * D, hb, true, 0);
    }, [&] {
        k::fuse_attn_one_f16_at_gate(q, r, sink, kc, vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window,
                                     sa, gate, oa, ha, 0);
    });
}

void test_loop_gates(int n, bool wide = false) {
    std::printf("fuse_loop_gates_rms%s vs loop_gates + %s (%d gate%s, 2 launches)\n", wide ? " (wide, set 8)" : "",
                n == 3 ? "rms_multi" : "rms", n, n > 1 ? "s" : "");
    const int R = 64;
    std::vector<float*> D_(n), F_(n), G_(n), m_(n), nw(n);
    for (int j = 0; j < n; ++j) {
        D_[j] = dev(fvec((size_t) R * 2 * C, 0.05f)), F_[j] = dev(fvec((size_t) R * C, 0.05f));
        G_[j] = dev(fvec((size_t) R * C, 0.05f)), m_[j] = dev(fvec(C, 2.0f)), nw[j] = dev(pvec(C, 0.5f, 1.5f));
    }
    std::vector<std::vector<float>> x0(n);
    for (int j = 0; j < n; ++j) x0[j] = fvec(C, 2.0f);
    // n 3: x (lp_in) and y (x3) apart; n 1: y = x (the FFN gate's W.xq), its rms in place
    struct Arm {
        std::vector<float*> x, y;
        float* sc;
    } A, B;
    for (Arm* a : {&A, &B}) {
        a->sc = dev_n<float>((size_t) n * (2 * R + 2));
        for (int j = 0; j < n; ++j) {
            a->x.push_back(dev(x0[j]));
            a->y.push_back(n == 1 ? a->x[j] : dev_n<float>(C));
        }
    }
    auto gates = [&](Arm& a) {
        std::vector<k::LoopGate> gt(n);
        for (int j = 0; j < n; ++j) gt[j] = {a.x[j], m_[j], D_[j], F_[j], G_[j], a.y[j]};
        return gt;
    };
    auto jobs = [&](Arm& a) {
        k::RmsJobs rj{};
        for (int j = 0; j < n; ++j) rj.x[j] = a.y[j], rj.w[j] = nw[j], rj.y[j] = a.y[j];
        return rj;
    };
    for (int rep = 0; rep < 2; ++rep) {
        for (int j = 0; j < n; ++j) put(A.x[j], x0[j]), put(B.x[j], x0[j]);
        const auto gb = gates(B);
        k::loop_gates(gb.data(), n, 1, C, R, 1e-6f, B.sc, 0);
        if (n == 3) k::rms_multi(jobs(B), 3, C, 1e-6f, 0);
        else k::rms(B.y[0], nw[0], 1, C, 1e-6f, B.y[0], 0);
        const auto ga = gates(A);
        if (!k::fuse_loop_gates_rms(ga.data(), n, C, R, 1e-6f, A.sc, jobs(A), 1e-6f, 0, wide)) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        same("dots (scratch)", host(A.sc, (size_t) n * (2 * R + 2)), host(B.sc, (size_t) n * (2 * R + 2)));
        for (int j = 0; j < n; ++j) same("y = rms(gate)", host(A.y[j], C), host(B.y[j], C));
    }
    const auto ga = gates(A), gb = gates(B);
    timing(n == 3 ? "3 gates + rms_multi" : "1 gate + rms", [&] {
        k::loop_gates(gb.data(), n, 1, C, R, 1e-6f, B.sc, 0);
        if (n == 3) k::rms_multi(jobs(B), 3, C, 1e-6f, 0);
        else k::rms(B.y[0], nw[0], 1, C, 1e-6f, B.y[0], 0);
    }, [&] { k::fuse_loop_gates_rms(ga.data(), n, C, R, 1e-6f, A.sc, jobs(A), 1e-6f, 0, wide); });
    if (wide)
        timing("set 2 vs set 8 (wide)", [&] { k::fuse_loop_gates_rms(ga.data(), n, C, R, 1e-6f, A.sc, jobs(A), 1e-6f, 0); },
               [&] { k::fuse_loop_gates_rms(ga.data(), n, C, R, 1e-6f, A.sc, jobs(A), 1e-6f, 0, true); });
}

void test_moe_attn_res(int n, int nq, bool rms, bool pf = false) {
    std::printf("fuse_moe_attn_res%s vs moe_combine_conv + %s (%d states, %d quer%s)\n", pf ? " (pf, set 5)" : "",
                rms ? "attn_res_rms_f16" : "attn_res", n, nq, nq > 1 ? "ies" : "y");
    const int K = 8;
    float* y = dev(fvec((size_t) K * C, 2.0f));
    int* inv = dev(std::vector<int>{3, 0, 7, 1, 5, 2, 6, 4});
    float *wt = dev(fvec(K, 1.0f)), *w = dev(fvec((size_t) C * 4, 0.7f)), *res = dev(fvec(C, 3.0f));
    float* q = dev(fvec((size_t) nq * C, 0.5f));
    std::vector<float*> blocks(n - 1);
    for (auto& b : blocks) b = dev(fvec(C, 3.0f));
    std::vector<float*> nw(nq);
    for (auto& p : nw) p = dev(pvec(C, 0.5f, 1.5f));
    const auto hist0 = fvec((size_t) 3 * C, 2.0f);
    struct Arm {
        float *hist, *h, *res_out;
        float* y32[3];
        uint16_t* y16[3];
    } A, B;
    for (Arm* a : {&A, &B}) {
        a->hist = dev(hist0), a->h = dev_n<float>(C), a->res_out = dev_n<float>((size_t) nq * C);
        for (int i = 0; i < 3; ++i) a->y32[i] = dev_n<float>(C), a->y16[i] = dev_n<uint16_t>(C);
    }
    auto states = [&](Arm& a) {
        k::ResStates st{};
        for (int i = 0; i < n - 1; ++i) st.st[i] = blocks[i];
        st.st[n - 1] = a.h;
        return st;
    };
    auto outs_of = [&](Arm& a) {
        k::RmsOuts o{};
        for (int i = 0; i < nq; ++i) o.w[i] = nw[i], o.y16[i] = a.y16[i], o.y32[i] = a.y32[i];
        return o;
    };
    for (int rep = 0; rep < 2; ++rep) {   // (2: the conv's history carried)
        k::moe_combine_conv(y, inv, wt, K, C, B.hist, w, res, B.h, 0);
        const auto ob = outs_of(B), oa = outs_of(A);
        if (rms) k::attn_res_rms_f16(states(B), n, q, nq, 1, C, 1e-6f, 1e-6f, ob, 0);
        else k::attn_res(states(B), n, q, nq, 1, C, 1e-6f, B.res_out, 0);
        if (!k::fuse_moe_attn_res(y, inv, wt, K, C, A.hist, w, res, A.h, states(A), n, q, nq, 1e-6f, 1e-6f,
                                  A.res_out, rms ? &oa : nullptr, 0, pf)) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        same("stream (combine out)", host(A.h, C), host(B.h, C));
        same("conv history", host(A.hist, (size_t) 3 * C), host(B.hist, (size_t) 3 * C));
        if (rms) {
            for (int i = 0; i < nq; ++i) {
                same("rms out F32", host(A.y32[i], C), host(B.y32[i], C));
                same("rms out FP16", host(A.y16[i], C), host(B.y16[i], C));
            }
        } else {
            same("attn_res out", host(A.res_out, (size_t) nq * C), host(B.res_out, (size_t) nq * C));
        }
    }
    const auto ob = outs_of(B), oa = outs_of(A);
    timing("combine + attention residual", [&] {
        k::moe_combine_conv(y, inv, wt, K, C, B.hist, w, res, B.h, 0);
        if (rms) k::attn_res_rms_f16(states(B), n, q, nq, 1, C, 1e-6f, 1e-6f, ob, 0);
        else k::attn_res(states(B), n, q, nq, 1, C, 1e-6f, B.res_out, 0);
    }, [&] {
        k::fuse_moe_attn_res(y, inv, wt, K, C, A.hist, w, res, A.h, states(A), n, q, nq, 1e-6f, 1e-6f, A.res_out,
                             rms ? &oa : nullptr, 0, pf);
    });
}

// set 5: fuse_attn_res_pf against attn_res / attn_res_rms_f16 (one token)
void test_attn_res_pf(int n, int nq, bool rms) {
    std::printf("fuse_attn_res_pf (set 5) vs %s (%d states, %d quer%s)\n", rms ? "attn_res_rms_f16" : "attn_res", n, nq,
                nq > 1 ? "ies" : "y");
    k::ResStates st{};
    for (int i = 0; i < n; ++i) st.st[i] = dev(fvec(C, 3.0f));
    float* q = dev(fvec((size_t) nq * C, 0.5f));
    float *ra = dev_n<float>((size_t) nq * C), *rb = dev_n<float>((size_t) nq * C);
    k::RmsOuts oa{}, ob{};
    for (int i = 0; i < nq; ++i) {
        oa.w[i] = ob.w[i] = dev(pvec(C, 0.5f, 1.5f));
        oa.y32[i] = dev_n<float>(C), ob.y32[i] = dev_n<float>(C);
        oa.y16[i] = dev_n<uint16_t>(C), ob.y16[i] = dev_n<uint16_t>(C);
    }
    auto unfused = [&] {
        if (rms) k::attn_res_rms_f16(st, n, q, nq, 1, C, 1e-6f, 1e-6f, ob, 0);
        else k::attn_res(st, n, q, nq, 1, C, 1e-6f, rb, 0);
    };
    auto fused = [&] {
        return k::fuse_attn_res_pf(st, n, q, nq, C, 1e-6f, 1e-6f, rms ? nullptr : ra, rms ? &oa : nullptr, 0);
    };
    unfused();
    if (!fused()) {
        std::printf("  FAIL not launched\n"), ++g_fail;
        return;
    }
    if (rms) {
        for (int i = 0; i < nq; ++i) {
            same("rms out F32", host(oa.y32[i], C), host(ob.y32[i], C));
            same("rms out FP16", host(oa.y16[i], C), host(ob.y16[i], C));
        }
    } else {
        same("attn_res out", host(ra, (size_t) nq * C), host(rb, (size_t) nq * C));
    }
    timing("attention residual", unfused, [&] { fused(); });
}

void test_prep_attn(int pos_v) {
    std::printf("fuse_prep_attn_f16_at_gate vs prep_qkv_r_f16_at + attn_decode_f16_at + gate_mul_f16 (window layer, "
                "P 16, position %d, 2 launches)\n", pos_v);
    const int P = 16, kv_cap = 1024, slots = 1024, window = 1024;
    const auto q0 = fvec((size_t) H * D, 4.0f), r0 = fvec((size_t) H * DREL, 4.0f);
    float *q_clip = dev(pvec((size_t) H * D, 0.5f, 1.5f)), *r_clip = dev(pvec((size_t) H * DREL, 0.5f, 1.5f));
    float *kd = dev(fvec((size_t) KV * D, 4.0f)), *vd = dev(fvec((size_t) KV * D, 4.0f));
    uint16_t* sink_w = dev(hvec((size_t) H * C));
    float* sink_x = dev(fvec(C));
    int* pos = dev(std::vector<int>{pos_v});
    float* proj = dev(fvec((size_t) 1024 * DREL, 0.5f));
    float* gate = dev(fvec((size_t) H * D, 4.0f));
    const auto cache0 = hvec((size_t) slots * KV * D);
    const float qk_scale = 1.0f / std::sqrt((float) D);
    struct Arm {
        float *q, *r, *sink, *scr, *o;
        uint16_t *kc, *vc, *o16;
    } A, B;
    for (Arm* a : {&A, &B}) {
        a->q = dev(q0), a->r = dev(r0), a->sink = dev_n<float>(H), a->scr = dev_n<float>((size_t) H * P * 130);
        a->o = dev_n<float>((size_t) H * D), a->kc = dev(cache0), a->vc = dev(cache0), a->o16 = dev_n<uint16_t>((size_t) H * D);
    }
    auto unfused = [&] {
        k::prep_qkv_r_f16_at(B.q, q_clip, kd, B.kc, vd, B.vc, B.r, r_clip, H, KV, D, DREL, 22.627417f, 8.0f, 1e-6f, pos,
                             false, 0.1f, 8192.0f, kv_cap, nullptr, 0, sink_w, sink_x, C, B.sink);
        k::attn_decode_f16_at(B.q, B.r, B.sink, B.kc, B.vc, kv_cap, H, KV, pos, P, qk_scale, proj, DREL, 1024, window,
                              B.scr, B.o, 0);
        k::gate_mul_f16(B.o, gate, (size_t) H * D, B.o16, true, 0);
    };
    auto fused = [&] {
        return k::fuse_prep_attn_f16_at_gate(A.q, q_clip, kd, A.kc, vd, A.vc, A.r, r_clip, H, KV, D, DREL, 22.627417f,
                                             8.0f, 1e-6f, pos, false, 0.1f, 8192.0f, kv_cap, sink_w, sink_x, C, A.sink,
                                             qk_scale, proj, 1024, window, P, A.scr, gate, A.o, A.o16, 0);
    };
    unfused();
    for (int rep = 0; rep < 2; ++rep) {
        if (!fused()) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        same("out F32", host(A.o, (size_t) H * D), host(B.o, (size_t) H * D));
        same("out FP16", host(A.o16, (size_t) H * D), host(B.o16, (size_t) H * D));
        same("sinks", host(A.sink, H), host(B.sink, H));
        same("K cache", host(A.kc, cache0.size()), host(B.kc, cache0.size()));
        same("V cache", host(A.vc, cache0.size()), host(B.vc, cache0.size()));
    }
    timing("prep + attention + gate (one launch)", [&] {
        put(B.q, q0), put(B.r, r0);
        unfused();
    }, [&] {
        put(A.q, q0), put(A.r, r0);
        fused();
    });
}

}  // namespace

// set 9: fuse_res_router_route against attn_res_rms_f16 + mv_f16 + route_token (3 launches: the counter goes back)
void test_res_router(int n) {
    std::printf("fuse_res_router_route (set 9) vs attn_res_rms_f16 + mv_f16 + route_token (%d states)\n", n);
    k::ResStates st{};
    for (int i = 0; i < n; ++i) st.st[i] = dev(fvec(C, 3.0f));
    float* q = dev(fvec(C, 0.5f));
    float* nw = dev(pvec(C, 0.5f, 1.5f));
    uint16_t* Wr = dev(hvec((size_t) E * C));
    float* bias = dev(fvec(E, 0.05f));
    std::vector<const uint8_t*> tab_h(E);
    for (int e = 0; e < E; ++e) tab_h[e] = (const uint8_t*) (uintptr_t) (0x100000u + 4096u * e);
    const uint8_t** tab = dev(tab_h);
    float *sg_t = dev(pvec(E, 0.5f, 2.0f)), *su_t = dev(pvec(E, 0.5f, 2.0f)), *sd_t = dev(pvec(E, 0.5f, 2.0f));
    struct Out {
        float* y32 = dev_n<float>(C);
        uint16_t* y16 = dev_n<uint16_t>(C);
        float* logits = dev_n<float>(E);
        int* top = dev_n<int>(KTOP);
        float* w = dev_n<float>(KTOP);
        const uint8_t** blob = dev_n<const uint8_t*>(KTOP);
        int* tok = dev_n<int>(KTOP);
        int* inv = dev_n<int>(KTOP);
        float* sg = dev_n<float>(KTOP);
        float* su = dev_n<float>(KTOP);
        float* wt = dev_n<float>(KTOP);
    } A, B;
    k::RmsOuts oa{}, ob{};
    oa.w[0] = ob.w[0] = nw, oa.y32[0] = A.y32, oa.y16[0] = A.y16, ob.y32[0] = B.y32, ob.y16[0] = B.y16;
    auto unfused = [&] {
        k::attn_res_rms_f16(st, n, q, 1, 1, C, 1e-6f, 1e-6f, ob, 0);
        k::mv_f16(Wr, E, C, B.y32, B.logits, 0);
        k::route_token(B.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, B.top, B.w, B.blob, B.tok, B.inv, B.sg,
                       B.su, B.wt, 0);
    };
    auto fused = [&] {
        return k::fuse_res_router_route(st, n, q, C, 1e-6f, 1e-6f, oa, Wr, A.logits, bias, E, KTOP, 8.0f, tab, sg_t,
                                        su_t, sd_t, A.top, A.w, A.blob, A.tok, A.inv, A.sg, A.su, A.wt, 0);
    };
    for (int rep = 0; rep < 3; ++rep) {
        put(st.st[0] == nullptr ? (float*) nullptr : (float*) st.st[0], fvec(C, 3.0f + rep));
        unfused();
        if (!fused()) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        std::printf(" launch %d:\n", rep + 1);
        same("rms out F32", host(A.y32, C), host(B.y32, C));
        same("rms out FP16", host(A.y16, C), host(B.y16, C));
        same("logits", host(A.logits, E), host(B.logits, E));
        same("top", host(A.top, KTOP), host(B.top, KTOP));
        same("w", host(A.w, KTOP), host(B.w, KTOP));
        same("blob", host(A.blob, KTOP), host(B.blob, KTOP));
        same("tok", host(A.tok, KTOP), host(B.tok, KTOP));
        same("inv", host(A.inv, KTOP), host(B.inv, KTOP));
        same("sg", host(A.sg, KTOP), host(B.sg, KTOP));
        same("su", host(A.su, KTOP), host(B.su, KTOP));
        same("wt", host(A.wt, KTOP), host(B.wt, KTOP));
    }
    timing("residual + router (set 1: 2 launches) vs set 9", [&] {
        k::attn_res_rms_f16(st, n, q, 1, 1, C, 1e-6f, 1e-6f, ob, 0);
        k::fuse_router_route(Wr, B.y32, C, B.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, B.top, B.w, B.blob,
                             B.tok, B.inv, B.sg, B.su, B.wt, 0);
    }, [&] { fused(); });
}

// set 10: fuse_res_loop_gate_rms against attn_res + loop_gates + rms (the FFN gate: y = x, in place)
void test_res_loop_gate(int n) {
    std::printf("fuse_res_loop_gate_rms (set 10) vs attn_res + loop_gates + rms (%d states, 1 gate in place)\n", n);
    const int R = 64;
    k::ResStates st{};
    for (int i = 0; i < n; ++i) st.st[i] = dev(fvec(C, 3.0f));
    float* q = dev(fvec(C, 0.5f));
    float *D = dev(fvec((size_t) R * 2 * C, 0.05f)), *F = dev(fvec((size_t) R * C, 0.05f));
    float *G = dev(fvec((size_t) R * C, 0.05f)), *mm = dev(fvec(C, 2.0f)), *nw = dev(pvec(C, 0.5f, 1.5f));
    float *xa = dev_n<float>(C), *xb = dev_n<float>(C), *sa = dev_n<float>(2 * R + 2), *sb = dev_n<float>(2 * R + 2);
    const k::LoopGate ga{xa, mm, D, F, G, xa}, gb{xb, mm, D, F, G, xb};
    const k::RmsJobs ra{{xa}, {nw}, {xa}}, rb{{xb}, {nw}, {xb}};
    for (int rep = 0; rep < 2; ++rep) {
        put((float*) st.st[0], fvec(C, 3.0f + rep));
        k::attn_res(st, n, q, 1, 1, C, 1e-6f, xb, 0);
        k::loop_gates(&gb, 1, 1, C, R, 1e-6f, sb, 0);
        k::rms(xb, nw, 1, C, 1e-6f, xb, 0);
        if (!k::fuse_res_loop_gate_rms(st, n, q, 1e-6f, xa, ga, C, R, 1e-6f, sa, ra, 1e-6f, 0)) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        same("dots (scratch)", host(sa, 2 * R + 2), host(sb, 2 * R + 2));
        same("y = rms(gate(residual))", host(xa, C), host(xb, C));
    }
    timing("residual + gate (set 2: 3 launches) vs set 10", [&] {
        k::attn_res(st, n, q, 1, 1, C, 1e-6f, xb, 0);
        k::fuse_loop_gates_rms(&gb, 1, C, R, 1e-6f, sb, rb, 1e-6f, 0);
    }, [&] { k::fuse_res_loop_gate_rms(st, n, q, 1e-6f, xa, ga, C, R, 1e-6f, sa, ra, 1e-6f, 0); });
}

// set 11: fuse_loop_gate_rms_router against loop_gates + rms (in place) + mv_f16 + route_token (3 launches each)
void test_loop_gate_rms_router(bool s12 = false) {
    std::printf("fuse_loop_gate_%s vs loop_gates + rms + mv_f16 + route_token (1 gate)\n",
                s12 ? "router (set 12, in place)" : "rms_router (set 11)");
    const int R = 64;
    float *D = dev(fvec((size_t) R * 2 * C, 0.05f)), *F = dev(fvec((size_t) R * C, 0.05f));
    float *G = dev(fvec((size_t) R * C, 0.05f)), *mm = dev(fvec(C, 2.0f)), *nw = dev(pvec(C, 0.5f, 1.5f));
    float *xa = dev_n<float>(C), *xb = dev_n<float>(C), *ya = dev_n<float>(C);
    float *sa = dev_n<float>(2 * R + 2), *sb = dev_n<float>(2 * R + 2);
    uint16_t* Wr = dev(hvec((size_t) E * C));
    float* bias = dev(fvec(E, 0.05f));
    std::vector<const uint8_t*> tab_h(E);
    for (int e = 0; e < E; ++e) tab_h[e] = (const uint8_t*) (uintptr_t) (0x100000u + 4096u * e);
    const uint8_t** tab = dev(tab_h);
    float *sg_t = dev(pvec(E, 0.5f, 2.0f)), *su_t = dev(pvec(E, 0.5f, 2.0f)), *sd_t = dev(pvec(E, 0.5f, 2.0f));
    struct Out {
        float* logits = dev_n<float>(E);
        int* top = dev_n<int>(KTOP);
        float* w = dev_n<float>(KTOP);
        const uint8_t** blob = dev_n<const uint8_t*>(KTOP);
        int* tok = dev_n<int>(KTOP);
        int* inv = dev_n<int>(KTOP);
        float* sg = dev_n<float>(KTOP);
        float* su = dev_n<float>(KTOP);
        float* wt = dev_n<float>(KTOP);
    } A, B;
    const k::LoopGate ga{xa, mm, D, F, G, s12 ? xa : ya}, gb{xb, mm, D, F, G, xb};
    const k::RmsJobs ra{{xa}, {nw}, {xa}};
    auto fused = [&] {
        return s12 ? k::fuse_loop_gate_router(ga, C, R, 1e-6f, sa, ra, 1e-6f, Wr, A.logits, bias, E, KTOP, 8.0f, tab,
                                              sg_t, su_t, sd_t, A.top, A.w, A.blob, A.tok, A.inv, A.sg, A.su, A.wt, 0)
                   : k::fuse_loop_gate_rms_router(ga, C, R, 1e-6f, sa, nw, 1e-6f, xa, Wr, A.logits, bias, E, KTOP, 8.0f,
                                                  tab, sg_t, su_t, sd_t, A.top, A.w, A.blob, A.tok, A.inv, A.sg, A.su,
                                                  A.wt, 0);
    };
    for (int rep = 0; rep < 3; ++rep) {
        const auto x0 = fvec(C, 2.0f + rep);
        put(xa, x0), put(xb, x0);
        k::loop_gates(&gb, 1, 1, C, R, 1e-6f, sb, 0);
        k::rms(xb, nw, 1, C, 1e-6f, xb, 0);
        k::mv_f16(Wr, E, C, xb, B.logits, 0);
        k::route_token(B.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, B.top, B.w, B.blob, B.tok, B.inv, B.sg,
                       B.su, B.wt, 0);
        if (!fused()) {
            std::printf("  FAIL not launched\n"), ++g_fail;
            return;
        }
        std::printf(" launch %d:\n", rep + 1);
        same("dots (scratch)", host(sa, 2 * R + 2), host(sb, 2 * R + 2));
        same("x = rms(gate)", host(xa, C), host(xb, C));
        same("logits", host(A.logits, E), host(B.logits, E));
        same("top", host(A.top, KTOP), host(B.top, KTOP));
        same("w", host(A.w, KTOP), host(B.w, KTOP));
        same("blob", host(A.blob, KTOP), host(B.blob, KTOP));
        same("tok", host(A.tok, KTOP), host(B.tok, KTOP));
        same("inv", host(A.inv, KTOP), host(B.inv, KTOP));
        same("sg", host(A.sg, KTOP), host(B.sg, KTOP));
        same("su", host(A.su, KTOP), host(B.su, KTOP));
        same("wt", host(A.wt, KTOP), host(B.wt, KTOP));
    }
    const k::RmsJobs rb{{xb}, {nw}, {xb}};
    timing(s12 ? "gate + rms + router (sets 1, 2) vs set 12" : "gate + rms + router (sets 1, 2) vs set 11", [&] {
        k::fuse_loop_gates_rms(&gb, 1, C, R, 1e-6f, sb, rb, 1e-6f, 0);
        k::fuse_router_route(Wr, xb, C, B.logits, bias, E, KTOP, 8.0f, tab, sg_t, su_t, sd_t, B.top, B.w, B.blob,
                             B.tok, B.inv, B.sg, B.su, B.wt, 0);
    }, [&] { fused(); });
}

int main() {
    test_mv_f16_pair();
    test_router_route();
    test_router_route(true);
    test_prep(true, 300);
    test_prep(true, 2000);
    test_prep(false, 300);
    test_prep(false, 5000);
    test_attn_gate(true, 300);
    test_attn_gate(true, 2000);
    test_attn_gate(false, 300);
    test_attn_gate(false, 5000);
    test_attn_gate(false, 0, true);
    test_attn_gate(false, 50, true);
    test_attn_gate(false, 300, true);
    test_attn_gate(false, 1000, true);
    test_attn_gate(false, 5000, true);
    test_attn_one(true, 300);
    test_attn_one(true, 2000);
    test_loop_gates(3);
    test_loop_gates(1);
    test_loop_gates(3, true);
    test_loop_gates(1, true);
    test_moe_attn_res(1, 1, true);
    test_moe_attn_res(5, 3, true);
    test_moe_attn_res(6, 1, true);
    test_moe_attn_res(3, 1, false);
    test_moe_attn_res(5, 3, false);
    test_attn_res_pf(1, 1, true);
    test_attn_res_pf(2, 3, true);
    test_attn_res_pf(5, 3, true);
    test_attn_res_pf(5, 1, true);
    test_attn_res_pf(3, 3, false);
    test_attn_res_pf(6, 1, false);
    test_moe_attn_res(1, 1, true, true);
    test_moe_attn_res(2, 3, true, true);
    test_moe_attn_res(5, 3, true, true);
    test_moe_attn_res(6, 1, true, true);
    test_moe_attn_res(3, 1, false, true);
    test_moe_attn_res(5, 3, false, true);
    test_prep_attn(0);
    test_prep_attn(5);
    test_prep_attn(300);
    test_prep_attn(2000);
    test_res_router(1);
    test_res_router(2);
    test_res_router(4);
    test_res_router(5);
    test_res_loop_gate(1);
    test_res_loop_gate(2);
    test_res_loop_gate(4);
    test_res_loop_gate(5);
    test_loop_gate_rms_router();
    test_loop_gate_rms_router(true);
    std::printf(g_fail ? "FUSE TESTS: %d FAILED\n" : "FUSE TESTS: all bit for bit\n", g_fail);
    return g_fail ? 1 : 0;
}
