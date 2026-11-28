// tests/ik2_nvfp4_s8_test.cu - bit-exactness of Basalt's NVFP4 -> int8 fragment
// against a host-side __byte_perm reference, on the engine's GGUF block layout.
// Run: ./build/ik2_nvfp4_s8_test   (CUDA only; works on sm_89 and sm_120)
#include "strata/kernels/nvfp4_s8.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::kernels::nvfp4s8;

static uint32_t byte_perm_ref(uint32_t x, uint32_t y, uint32_t s) {
    const uint8_t b[8] = {uint8_t(x), uint8_t(x >> 8), uint8_t(x >> 16), uint8_t(x >> 24),
                          uint8_t(y), uint8_t(y >> 8), uint8_t(y >> 16), uint8_t(y >> 24)};
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (s >> (4 * i)) & 7u;
        out |= uint32_t(b[n]) << (8 * i);
    }
    return out;
}

static uint32_t word_ref(const uint8_t* p, int i) {
    return uint32_t(p[4 * i]) | (uint32_t(p[4 * i + 1]) << 8) |
           (uint32_t(p[4 * i + 2]) << 16) | (uint32_t(p[4 * i + 3]) << 24);
}

static uint32_t frag_ref(uint32_t q4, int t) {
    const uint32_t nibs = (t & 2) ? ((q4 >> 4) & 0x0F0F0F0Fu) : (q4 & 0x0F0F0F0Fu);
    const uint32_t sel = (nibs & 0x0Fu) | ((nibs >> 4) & 0xF0u) |
                         ((nibs >> 8) & 0xF00u) | ((nibs >> 12) & 0xF000u);
    const uint32_t lo = byte_perm_ref(0x03020100u, 0x0c080604u, sel);
    const uint32_t hi = byte_perm_ref(0xfdfeff00u, 0xf4f8fafcu, sel);
    return byte_perm_ref(lo, hi, 0x3210u | ((sel & 0x8888u) >> 1));
}

__global__ void frag_test(const Block* __restrict__ blocks, const uint32_t* __restrict__ want, int* __restrict__ bad,
                          int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i & 3, k = i >> 2;
    const uint32_t got = frag(blocks + k, 0, t);
    if (got != want[i]) atomicAdd(bad, 1);
}

int main() {
    const int n = 4096;
    std::mt19937 rng(1234);
    std::vector<Block> h(n / 4);
    for (auto& b : h)
        for (int i = 0; i < 32; ++i) b.qs[i] = uint8_t(rng());
    std::vector<uint32_t> want(n);
    for (int k = 0; k < n / 4; ++k)
        for (int t = 0; t < 4; ++t) want[k * 4 + t] = frag_ref(word_ref(h[k].qs, 0), t);

    Block *db = nullptr;
    uint32_t *dw = nullptr;
    int *dbad = nullptr;
    cudaMalloc(&db, h.size() * sizeof(Block));
    cudaMalloc(&dw, want.size() * sizeof(uint32_t));
    cudaMalloc(&dbad, sizeof(int));
    cudaMemcpy(db, h.data(), h.size() * sizeof(Block), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, want.data(), want.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(dbad, 0, sizeof(int));
    frag_test<<<(n + 255) / 256, 256>>>(db, dw, dbad, n);
    int bad = 0;
    cudaMemcpy(&bad, dbad, sizeof(int), cudaMemcpyDeviceToHost);
    std::printf("ik2_nvfp4_s8_test: %d fragments, %d mismatches\n", n, bad);
    if (bad) return 1;
    return 0;
}
