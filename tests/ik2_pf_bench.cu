// tests/ik2_pf_bench.cu - the prompt's NVFP4 experts at the model's shapes, timed: each expert decoded to FP16 and a
// cuBLAS GEMM (run_prefill_moe's batch path) against k::experts_pf_nvfp4's kernels (IK2_PF_KERNEL's variants).
// 128 experts with random NVFP4 weights, S rows (a chunk's tokens x 8) dealt to them unevenly; gate|up (K = 2304,
// N = 2304) and down (K = 1152, N = 2304) timed apart; each variant's output against the first kernel's.
//   build/ik2_pf_bench [S=32768] [reps=5]        (~1.5 GiB of VRAM)
#include "../src/ik2/ik2_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = ik2::k;

#define CU(x)                                                                               \
    do {                                                                                    \
        cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                            \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);   \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

int main(int argc, char** argv) {
    const int S = argc > 1 ? std::atoi(argv[1]) : 32768, reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const int E = 128, C = 2304, F = 1152;
    std::mt19937 rng(7);
    // uneven counts: weights from a log-normal, as a router's are
    std::vector<double> wt(E);
    std::lognormal_distribution<double> ln(0.0, 0.8);
    double tot = 0;
    for (auto& w : wt) tot += (w = ln(rng));
    std::vector<int> cnt(E), start(E + 1, 0);
    int given = 0;
    for (int e = 0; e < E; ++e) given += (cnt[e] = (int) (S * wt[e] / tot));
    for (int i = 0; given < S; ++i, ++given) ++cnt[i % E];
    for (int e = 0; e < E; ++e) start[e + 1] = start[e] + cnt[e];
    std::printf("S %d rows over %d experts: %d .. %d a expert\n", S, E, *std::min_element(cnt.begin(), cnt.end()),
                *std::max_element(cnt.begin(), cnt.end()));

    const size_t gu_b = (size_t) 2 * F * C / 64 * 36, dn_b = (size_t) C * F / 64 * 36, eb = gu_b + dn_b;
    std::vector<uint8_t> wh(eb * E);
    for (size_t b = 0; b < wh.size(); b += 36) {
        for (int i = 0; i < 4; ++i) wh[b + i] = (uint8_t) (((4 + rng() % 4) << 3) | (rng() & 7));
        for (int i = 4; i < 36; ++i) wh[b + i] = (uint8_t) rng();
    }
    uint8_t* w_d;
    CU(cudaMalloc(&w_d, wh.size()));
    CU(cudaMemcpy(w_d, wh.data(), wh.size(), cudaMemcpyHostToDevice));
    std::vector<uint16_t> xh((size_t) S * C);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : xh) v = __half_as_ushort(__float2half_rn(nd(rng)));
    uint16_t *x_d, *h_d, *w16_d;
    float *gu_d, *dn_d, *ref_gu, *ref_dn;
    CU(cudaMalloc(&x_d, xh.size() * 2));
    CU(cudaMemcpy(x_d, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&h_d, (size_t) S * F * 2));
    CU(cudaMemcpy(h_d, x_d, (size_t) S * F * 2, cudaMemcpyDeviceToDevice));   // (down's input: any FP16 rows)
    CU(cudaMalloc(&w16_d, (size_t) 2 * F * C * 2));
    CU(cudaMalloc(&gu_d, (size_t) S * 2 * F * 4));
    CU(cudaMalloc(&dn_d, (size_t) S * C * 4));
    CU(cudaMalloc(&ref_gu, (size_t) S * 2 * F * 4));
    CU(cudaMalloc(&ref_dn, (size_t) S * C * 4));

    cublasHandle_t blas;
    cublasCreate(&blas);
    auto gemm16 = [&](const uint16_t* W, int N, int Kd, const uint16_t* X, int T, float* Y) {
        const float one = 1.0f, zero = 0.0f;
        cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, Kd, &one, W, CUDA_R_16F, Kd, X, CUDA_R_16F, Kd, &zero, Y,
                     CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    auto base_gu = [&]() {
        for (int e = 0; e < E; ++e) {
            if (!cnt[e]) continue;
            k::decode_f16(k::kNVFP4, w_d + e * eb, (size_t) 2 * F * C, w16_d, 0);
            gemm16(w16_d, 2 * F, C, x_d + (size_t) start[e] * C, cnt[e], gu_d + (size_t) start[e] * 2 * F);
        }
    };
    auto base_dn = [&]() {
        for (int e = 0; e < E; ++e) {
            if (!cnt[e]) continue;
            k::decode_f16(k::kNVFP4, w_d + e * eb + gu_b, (size_t) C * F, w16_d, 0);
            gemm16(w16_d, C, F, h_d + (size_t) start[e] * F, cnt[e], dn_d + (size_t) start[e] * C);
        }
    };
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto timed = [&](auto&& f) {
        f();
        CU(cudaDeviceSynchronize());
        std::vector<float> ts;
        for (int r = 0; r < reps; ++r) {
            cudaEventRecord(e0);
            f();
            cudaEventRecord(e1);
            CU(cudaEventSynchronize(e1));
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            ts.push_back(ms);
        }
        std::sort(ts.begin(), ts.end());
        return ts[ts.size() / 2];
    };
    const double fl_gu = 2.0 * S * 2 * F * C, fl_dn = 2.0 * S * C * F;
    const float tb_gu = timed(base_gu), tb_dn = timed(base_dn);
    std::printf("%-26s gate|up %7.3f ms (%5.1f TFLOPS)  down %7.3f ms (%5.1f)  both %7.3f ms\n",
                "decode + cuBLAS", tb_gu, fl_gu / tb_gu * 1e-9, tb_dn, fl_dn / tb_dn * 1e-9, tb_gu + tb_dn);
    // the bound: the same cuBLAS GEMMs over weights already in FP16 (one expert's, decoded once, for all of them)
    k::decode_f16(k::kNVFP4, w_d, (size_t) 2 * F * C, w16_d, 0);
    const float tc_gu = timed([&] {
        for (int e = 0; e < E; ++e)
            if (cnt[e]) gemm16(w16_d, 2 * F, C, x_d + (size_t) start[e] * C, cnt[e], gu_d + (size_t) start[e] * 2 * F);
    });
    const float tc_dn = timed([&] {
        for (int e = 0; e < E; ++e)
            if (cnt[e]) gemm16(w16_d, C, F, h_d + (size_t) start[e] * F, cnt[e], dn_d + (size_t) start[e] * C);
    });
    std::printf("%-26s gate|up %7.3f ms (%5.1f TFLOPS)  down %7.3f ms (%5.1f)  both %7.3f ms\n",
                "cuBLAS only (no decode)", tc_gu, fl_gu / tc_gu * 1e-9, tc_dn, fl_dn / tc_dn * 1e-9, tc_gu + tc_dn);

    k::PfTile* t_d;
    CU(cudaMalloc(&t_d, (size_t) (E + S / 64 + 1) * sizeof(k::PfTile)));
    const int variants[] = {1, 21, 22, 23, 24, 25};
    bool have_ref = false;
    for (int v : variants) {
        k::experts_pf_nvfp4_variant(v);
        for (int bmv : (v == 1 ? std::vector<int>{64, 128} : std::vector<int>{0})) {
            if (v == 1) setenv("IK2_PF_BM", bmv == 64 ? "64" : "128", 1);
            const int bm = v == 1 ? bmv : k::experts_pf_nvfp4_bm();
            std::vector<k::PfTile> th;
            for (int e = 0; e < E; ++e)
                for (int m = 0; m < cnt[e]; m += bm)
                    th.push_back({{w_d + e * eb, w_d + e * eb + gu_b}, start[e] + m, std::min(bm, cnt[e] - m)});
            CU(cudaMemcpy(t_d, th.data(), th.size() * sizeof(k::PfTile), cudaMemcpyHostToDevice));
            const int nt = (int) th.size();
            const float tg = timed([&] { k::experts_pf_nvfp4(t_d, nt, bm, 0, x_d, C, 2 * F, gu_d, 0); });
            const float td = timed([&] { k::experts_pf_nvfp4(t_d, nt, bm, 1, h_d, F, C, dn_d, 0); });
            CU(cudaGetLastError());
            if (!have_ref) {
                CU(cudaMemcpy(ref_gu, gu_d, (size_t) S * 2 * F * 4, cudaMemcpyDeviceToDevice));
                CU(cudaMemcpy(ref_dn, dn_d, (size_t) S * C * 4, cudaMemcpyDeviceToDevice));
                have_ref = true;
            }
            // against the first kernel's output (the same products: equal unless a variant is wrong)
            auto cmp = [&](const float* a_d, const float* b_d, size_t n) {
                std::vector<float> a(n), b(n);
                CU(cudaMemcpy(a.data(), a_d, n * 4, cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(b.data(), b_d, n * 4, cudaMemcpyDeviceToHost));
                double worst = 0, scale = 0;
                for (size_t i = 0; i < n; ++i) {
                    worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
                    scale = std::max(scale, (double) std::fabs(b[i]));
                }
                return worst / std::max(scale, 1e-30);
            };
            const double d1 = cmp(gu_d, ref_gu, (size_t) S * 2 * F), d2 = cmp(dn_d, ref_dn, (size_t) S * C);
            char name[64];
            std::snprintf(name, sizeof name, "kernel %d (tiles of %d)", v, bm);
            std::printf("%-26s gate|up %7.3f ms (%5.1f TFLOPS)  down %7.3f ms (%5.1f)  both %7.3f ms  %+5.1f %%  diff %.1e\n",
                        name, tg, fl_gu / tg * 1e-9, td, fl_dn / td * 1e-9, tg + td,
                        ((tb_gu + tb_dn) / (tg + td) - 1) * 100, std::max(d1, d2));
        }
    }
    return 0;
}
