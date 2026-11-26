// tests/ik2_expert_bench.cu - P3-4's speed gate: the grouped NVFP4 expert GEMM (maya-ik2's grouped path) timed at
// the model's shapes - C = 2304 input / F = 1152 expert FFN width, 8 tokens a group - and its output dumped, so the
// emulated path (IK2_GROUP_MMA=3 IK2_EXPERT_WMMA=0) and the gfx12 WMMA path (IK2_GROUP_MMA=3 IK2_EXPERT_WMMA=1)
// can be compared: the same shapes, the same weights, the same activations, both from one binary.
//
//   ./ik2_expert_bench gate 40 /tmp/gate.bin            (NVFP4, the old default)
//   ./ik2_expert_bench gate 40 - iq2_xxs                (the scalar IQ path from issue #19)
//   ./ik2_expert_bench gatefused 40 - iq2_xxs           (llama.cpp IQ kernel, FP16 activations)
//   ./ik2_expert_bench downfused 40 - q2_0              (native_mmvq, FP16 activations)
//   ./ik2_expert_bench down 40 - mxfp4
//
// Last argument is the quant: nvfp4, q8_0, iq2_xxs, iq2_xs, iq1_s, iq2_s, mxfp4, q2_0, q4_k, q5_k, q6_k.
// Use `-` to skip the output dump. Compare GFLOP/s across quants to size a dedicated kernel.
#include "../src/ik2/ik2_kernels.cuh"
#include "strata/kernels/iq_f16.hpp"
#include "strata/kernels/native_mmvq.hpp"

// HIP pulls these in through its runtime headers; nvcc needs them explicitly for
// __float2half_rn / __half_as_ushort in the host-side activation generator.
#if !defined(STRATA_USE_HIP)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

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
std::mt19937 rng(20261121);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// The quantizations the expert path can run at the model's shapes. `bv` = values per block,
// `bb` = bytes per block (both from ik2::k::block_values / block_bytes). IQ2_* / IQ1_S have
// 256-value blocks: they fit C = 2304 (gate/up rows) but not F = 1152 (down rows), which is why
// a small quant uses MXFP4 / Q2_0 on down and IQ2_* on gate/up.
struct Quant {
    const char* name;
    int id, bv;
    size_t bb;
};
const Quant QUANTS[] = {
    {"nvfp4", k::kNVFP4, 64, 36},   {"q8_0", k::kQ8_0, 32, 34},     {"iq2_xxs", k::kIQ2_XXS, 256, 66},
    {"iq2_xs", k::kIQ2_XS, 256, 74}, {"iq1_s", k::kIQ1_S, 256, 50},  {"iq2_s", k::kIQ2_S, 256, 82},
    {"mxfp4", k::kMXFP4, 32, 17},   {"q2_0", k::kQ2_0, 64, 18},     {"q4_k", k::kQ4_K, 256, 144},
    {"q5_k", k::kQ5_K, 256, 176},
    {"q6_k", k::kQ6_K, 256, 210},
};

// Random but numerically sane blocks: the half scale is 1.0 (or a sane e8m0 exponent for MXFP4),
// every other byte is random. The bit patterns are what the decoder actually reads, so this
// exercises the same arithmetic path as a real GGUF - it is not a fidelity test against ggml.
std::vector<uint8_t> rand_blocks(const Quant& q, size_t n_values) {
    const size_t blocks = n_values / q.bv;
    std::vector<uint8_t> v(blocks * q.bb);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = v.data() + b * q.bb;
        for (size_t i = 0; i < q.bb; ++i) p[i] = (uint8_t) rng();
        if (q.id == k::kNVFP4) {
            for (int i = 0; i < 4; ++i) p[i] = (uint8_t) (0x38 | (rng() & 7));   // e = 7
        } else if (q.id == k::kMXFP4) {
            p[0] = 127;                                                          // e8m0 -> 1.0
        } else {
            p[0] = 0, p[1] = 0x3c;                                               // fp16 1.0 at the block start
        }
        if (q.id == k::kQ6_K) p[q.bb - 2] = 0, p[q.bb - 1] = 0x3c;                // its `d` is LAST
    }
    return v;
}
}   // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "gate";
    const bool fused_gate = std::strcmp(mode, "gatefused") == 0, fused_down = std::strcmp(mode, "downfused") == 0;
    const bool fused = fused_gate || fused_down;
    const bool gate = std::strcmp(mode, "gate") == 0 || std::strcmp(mode, "gate1") == 0 || fused_gate;
    const bool single = std::strcmp(mode, "gate1") == 0 || std::strcmp(mode, "down1") == 0;   // (the one-token path)
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    const char* out_path = argc > 3 ? argv[3] : nullptr;
    const Quant* q = &QUANTS[0];
    if (argc > 4) {
        bool found = false;
        for (const auto& cand : QUANTS)
            if (std::strcmp(cand.name, argv[4]) == 0) q = &cand, found = true;
        if (!found) {
            std::printf("unknown quant %s; choose one of: ", argv[4]);
            for (const auto& cand : QUANTS) std::printf("%s ", cand.name);
            std::printf("\n");
            return 2;
        }
    }
    const int C = 2304, F = 1152, G = 128, K = k::kGroupMax;   // (the model's shapes: 128 experts, 8 tokens a group)
    const int row_values = gate ? C : F;
    if (row_values % q->bv) {
        std::printf("quant %s has %d-value blocks, %s rows are %d wide (not a whole block)\n", q->name, q->bv,
                    gate ? "gate/up" : "down", row_values);
        return 2;
    }
    if (single && q->id != k::kNVFP4) {
        std::printf("gate1/down1 only drives the NVFP4 MMA path; use gate/down for %s\n", q->name);
        return 2;
    }
    const int S = G * K;
    const size_t stride_g = (size_t) (C / q->bv) * q->bb, stride_d = (size_t) (F / q->bv) * q->bb;
    const size_t eb0 = (size_t) F * stride_g, eb1 = eb0, eb2 = (size_t) C * stride_d;
    const size_t slot = (eb0 + eb1 + eb2 + 255) / 256 * 256, E = 8;   // (8 distinct expert blobs, read round-robin)
    std::vector<uint8_t> pool(slot * E);
    for (size_t e = 0; e < E; ++e) {
        auto g = rand_blocks(*q, (size_t) F * C), u = rand_blocks(*q, (size_t) F * C);
        // `down` has C rows of F values, i.e. eb2 bytes. Generate exactly that many (F is a whole
        // number of blocks only for the quants accepted for `down` above).
        auto d = rand_blocks(*q, (size_t) C * (F / q->bv) * q->bv);
        std::memcpy(pool.data() + e * slot, g.data(), g.size());
        std::memcpy(pool.data() + e * slot + eb0, u.data(), u.size());
        std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data(), d.size());
    }
    uint8_t* dpool = dev(pool);
    std::vector<const uint8_t*> blob(G);
    for (int g = 0; g < G; ++g) blob[g] = dpool + (g % E) * slot;
    std::vector<int> n(G), tok(S), row(S);
    for (int g = 0; g < G; ++g) {
        n[g] = K;
        for (int i = 0; i < K; ++i) tok[(size_t) g * K + i] = g * K + i, row[(size_t) g * K + i] = g * K + i;
    }
    std::vector<float> sg(S, 1.0f), su(S, 1.0f);
    std::vector<uint16_t> input(S * (size_t) C), h(S * (size_t) F), input2(2 * (size_t) C);   // (input2: the gate limb pair)
    std::vector<float> input32(S * (size_t) C);   // the grouped path reads FP16 activations staged as F32
    for (size_t i = 0; i < input.size(); ++i) {
        const float v = (float) (rng() % 2001) / 1000.0f - 1.0f;
        input[i] = (uint16_t) __half_as_ushort(__float2half_rn(v));
        input32[i] = v;
    }
    for (int i = 0; i < C; ++i) input2[i] = input[i], input2[C + i] = input[i];   // (hi limb, lo limb)
    std::vector<float> y((size_t) S * C, 0.0f);

    uint8_t* dblob_pool = dpool;
    const uint8_t** dblob = nullptr;
    const int slots = std::max(G, S);   // (the one-token path has a slot a token, the grouped path a group)
    CU(cudaMalloc(&dblob, slots * sizeof(uint8_t*)));
    std::vector<const uint8_t*> hblob(slots);
    for (int i = 0; i < slots; ++i) hblob[i] = blob[i % G];
    CU(cudaMemcpy(dblob, hblob.data(), slots * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    (void) dblob_pool;
    int* dn = dev(n);
    int* dtok = dev(tok);
    int* drow = dev(row);
    float* dsg = dev(sg);
    float* dsu = dev(su);
    uint16_t* dinput = dev(input);
    uint16_t* dinput2 = dev(input2);
    float* dinput32 = dev(input32);
    uint16_t* dh = dev(h);
    float* dy = dev(y);
    float* df32 = nullptr;
    void* dq81 = nullptr;
    if (fused) {
        CU(cudaMalloc(&df32, (size_t) K * C * sizeof(float)));
        CU(cudaMalloc(&dq81, strata::kernels::native_q8_1_bytes(C, K)));
    }

    cudaStream_t s = nullptr;
    // native_mmvq() requires an explicit non-null stream; the scalar ik2 kernels accept nullptr.
    if (fused) CU(cudaStreamCreate(&s));
    auto once = [&] {
        if (single) {
            if (gate)
                ik2::k::experts_gate_up_swiglu_mma(dblob, S, eb0, dinput2, C, F, dsg, dsu, 20.0f, 20.0f, dh, s);
            else
                ik2::k::experts_down_mma(dblob, S, eb0 + eb1, dh, C, F, dy, s);
        } else if (fused) {
            // llama.cpp's IQ / native dot kernels through the FP16 activation adapter:
            // one call per expert group, 8 token columns at a time, same weights and shapes.
            for (int g = 0; g < G; ++g) {
                // `blob` is the HOST array of device pointers. `dblob` is the device copy of that
                // array: dereferencing it here would fault in main, before any kernel runs.
                const uint8_t* bl = blob[g];
                if (fused_gate) {
                    const uint16_t* x = dinput + (size_t) g * K * C;
                    float* out = dy + (size_t) g * K * F;
                    strata::kernels::iq_mmvq_f16(q->id, bl, x, out, C, F, K, df32, dq81, s);
                    strata::kernels::iq_mmvq_f16(q->id, bl + eb0, x, out, C, F, K, df32, dq81, s);
                } else {
                    const uint16_t* h = dh + (size_t) g * K * F;
                    float* out = dy + (size_t) g * K * C;
                    strata::kernels::native_mmvq_f16(q->id, bl + eb0 + eb1, h, out, F, C, K, df32, dq81, s);
                }
            }
        } else if (gate) {
            // The grouped kernel reads FP32 activations from `x`. `input16` is only for the sm_86 MMA
            // path, which this A/B deliberately does not use (pass nullptr so it always takes the
            // generic row_dot path where the scalar IQ decoders live).
            ik2::k::experts_gate_up_swiglu_g(dblob, dn, dtok, drow, G, q->id, q->id, eb0, dinput32, C, F, dsg,
                                             dsu, 20.0f, 20.0f, dh, s, nullptr);
        } else {
            ik2::k::experts_down_g(dblob, dn, drow, G, q->id, eb0 + eb1, dh, C, F, dy, s);
        }
    };
    once();   // (warm up: JIT, the first page-ins)
    CU(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < iters; ++i) {
        cudaEvent_t a, b;
        CU(cudaEventCreate(&a));
        CU(cudaEventCreate(&b));
        CU(cudaEventRecord(a, s));
        once();
        CU(cudaEventRecord(b, s));
        CU(cudaEventSynchronize(b));
        float t = 0;
        CU(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    std::sort(ms.begin(), ms.end());
    const double best = ms.front(), med = ms[ms.size() / 2];
    const double dots = single ? (double) S : (double) G * K;   // (dots: gate + up counted as two)
    const double flops = gate ? 2.0 * dots * (double) F * 2 * C : 2.0 * slots * (double) C * F;
    std::printf("ik2_expert_bench %-7s %s C=%d F=%d %s: best %.1f us, median %.1f us, %.0f GFLOP/s\n", q->name, mode,
                C, F, single ? "one token a slot" : "8 tokens a group", best * 1e3, med * 1e3,
                flops / (med * 1e-3) / 1e9);
    if (!fused && out_path && std::strcmp(out_path, "-") != 0) {
        std::vector<uint8_t> raw;
        if (gate) {
            std::vector<uint16_t> out(h.size());
            CU(cudaMemcpy(out.data(), dh, out.size() * 2, cudaMemcpyDeviceToHost));
            raw.resize(out.size() * 2);
            std::memcpy(raw.data(), out.data(), raw.size());
        } else {
            std::vector<float> out(y.size());
            CU(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost));
            raw.resize(out.size() * 4);
            std::memcpy(raw.data(), out.data(), raw.size());
        }
        FILE* f = std::fopen(out_path, "wb");
        if (!f) return 2;
        std::fwrite(raw.data(), 1, raw.size(), f);
        std::fclose(f);
        std::printf("wrote %s (%zu bytes)\n", out_path, raw.size());
    }
    return g_fail ? 1 : 0;
}
