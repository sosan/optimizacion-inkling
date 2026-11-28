// tests/ik2_nvfp4_s8_gate_up_test.cu - correctness of the Basalt-derived NVFP4
// gate/up mma.sync.s8 kernel against an independent scalar reference that reads
// the same GGUF weights and the same q8_1 activations. Run on a GPU machine:
//   ./build/ik2_nvfp4_s8_gate_up_test
// CI only builds it (ubuntu-24.04 has no GPU); it prints "SKIP" there.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/nvfp4_s8_gate_up.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = ik2::k;

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

namespace {

std::mt19937 rng(20261128);

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

float ue4m3(uint8_t x) {
    const uint32_t v = x & 0x7F, e = v >> 3, m = v & 7;
    if (v == 0x7F) return 0.0f;
    if (e == 0) return float(m) * 0x1.0p-10f;
    uint32_t bits = ((e + 119u) << 23) | (m << 20);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int8_t fp4_code(uint8_t n) {
    static const int8_t t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return t[n & 15];
}

float q8_1_scale(const uint8_t* b) {
    uint16_t h = uint16_t(b[0]) | (uint16_t(b[1]) << 8);
    return __half2float(__ushort_as_half(h));
}

// Independent scalar dot of one NVFP4 weight row against one q8_1 token.
float row_dot(const uint8_t* w, const uint8_t* x, int C) {
    const int nb = C / 64;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = w + (size_t) b * 36;
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3(blk[s]);
            const uint8_t* q = blk + 4 + (size_t) s * 8;
            for (int j = 0; j < 8; ++j) {
                const int v0 = b * 64 + 16 * s + j, v1 = v0 + 8;
                const uint8_t* xb = x + (size_t) (v0 / 32) * 36;
                acc += d * float(fp4_code(q[j] & 15)) * q8_1_scale(xb) * float((int8_t) xb[4 + v0 % 32]);
                const uint8_t* xb1 = x + (size_t) (v1 / 32) * 36;
                acc += d * float(fp4_code(q[j] >> 4)) * q8_1_scale(xb1) * float((int8_t) xb1[4 + v1 % 32]);
            }
        }
    }
    return acc;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("ik2_nvfp4_s8_gate_up_test: SKIP (no CUDA device)\n");
        return 0;
    }

    // Small shapes that still satisfy both paths' alignment rules.
    const int G = 4, T = 8, S = G * T, C = 256, F = 64;
    const size_t stride = (size_t) (C / 64) * 36;       // NVFP4 row bytes
    const size_t eb0 = (size_t) F * stride;
    std::vector<uint8_t> pool(eb0 * 2 * G);
    for (int g = 0; g < G; ++g) {
        uint8_t* p = pool.data() + (size_t) g * 2 * eb0;
        for (size_t i = 0; i < 2 * eb0; ++i) p[i] = uint8_t(rng());
        for (size_t r = 0; r < 2 * F; ++r) {
            uint8_t* row = p + r * stride;
            for (int b = 0; b < C / 64; ++b)
                for (int s = 0; s < 4; ++s) row[(size_t) b * 36 + s] = uint8_t(0x38 | (rng() & 7));
        }
    }
    std::vector<float> x((size_t) S * C);
    for (auto& v : x) v = float(rng() % 2001) / 1000.0f - 1.0f;
    std::vector<int> tok(S), row(S);
    std::vector<float> sg(S, 1.0f), su(S, 1.0f);
    for (int s = 0; s < S; ++s) tok[s] = row[s] = s;

    std::vector<const uint8_t*> hblob(G);
    uint8_t* dpool = dev(pool);
    for (int g = 0; g < G; ++g) hblob[g] = dpool + (size_t) g * 2 * eb0;
    const uint8_t** dblob = nullptr;
    CU(cudaMalloc(&dblob, (size_t) G * sizeof(const uint8_t*)));
    CU(cudaMemcpy(dblob, hblob.data(), (size_t) G * sizeof(const uint8_t*), cudaMemcpyHostToDevice));
    int* dtok = dev(tok);
    int* drow = dev(row);
    float* dsg = dev(sg);
    float* dsu = dev(su);
    float* dx = dev(x);
    void* dq81 = nullptr;
    CU(cudaMalloc(&dq81, (size_t) S * (C / 32) * 36));
    CU(cudaMemset(dq81, 0, (size_t) S * (C / 32) * 36));
    strata::kernels::quantize_q8_1_rows(dx, S, C, dq81, nullptr);

    uint16_t* dh = nullptr;
    CU(cudaMalloc(&dh, (size_t) S * F * sizeof(uint16_t)));
    CU(cudaMemset(dh, 0, (size_t) S * F * sizeof(uint16_t)));
    if (!strata::kernels::nvfp4s8::gate_up_s8(dblob, dtok, drow, G, eb0, C, F, dq81, dsg, dsu, 20.0f, 20.0f, dh, nullptr)) {
        std::printf("FAIL: gate_up_s8 rejected the launch\n");
        return 1;
    }
    std::vector<uint16_t> h_s8((size_t) S * F);
    CU(cudaMemcpy(h_s8.data(), dh, h_s8.size() * 2, cudaMemcpyDeviceToHost));

    // Independent reference on the host, from the same weight and q8_1 bytes.
    std::vector<uint8_t> hq81((size_t) S * (C / 32) * 36);
    CU(cudaMemcpy(hq81.data(), dq81, hq81.size(), cudaMemcpyDeviceToHost));
    const float gcap = 20.0f, ucap = 20.0f;
    double max_diff = 0.0;
    int bad = 0;
    for (int s = 0; s < S; ++s) {
        const int e = s / T;
        const uint8_t* wblob = pool.data() + (size_t) e * 2 * eb0;
        const uint8_t* xtok = hq81.data() + (size_t) tok[s] * (C / 32) * 36;
        for (int r = 0; r < F; ++r) {
            const float g = row_dot(wblob + (size_t) r * stride, xtok, C);
            const float u = row_dot(wblob + eb0 + (size_t) r * stride, xtok, C);
            const float gv = gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g)));
            const float uv = ucap * tanhf(u / ucap);
            const float want = gv * uv;
            const float got = __half2float(__ushort_as_half(h_s8[(size_t) row[s] * F + r]));
            const float diff = std::fabs(got - want);
            max_diff = std::max(max_diff, (double) diff);
            if (diff > 0.02f * std::max(1.0f, std::fabs(want))) ++bad;
        }
    }
    std::printf("ik2_nvfp4_s8_gate_up_test: G=%d C=%d F=%d, max |diff| %.5f, %d/%d over tolerance\n", G, C, F,
                max_diff, bad, S * F);
    return bad == 0 ? 0 : 1;
}
