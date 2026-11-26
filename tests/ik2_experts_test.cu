// tests/ik2_experts_test.cu: IK2_FUSE_MASK sets 13 / 14 (ik2_experts2.inc)
// against the kernels they replace, on synthetic NVFP4 experts at the model's shape (C 2304, F 1152, 8 experts):
//   parity: experts_gate_up_swiglu_nv vs experts_gate_up_swiglu (h, FP16 bits) and experts_down_nv vs experts_down
//           (y, F32 bits), every row of every expert, each launch variant; prints mismatches + max abs diff
//   speed (arg "time"): each kernel R times on fresh 1 GiB-pool windows (weights from DRAM), for a rocprofv3 trace
//           (labels printed in launch order)
// build: hipcc -x hip --offload-arch=gfx1151 -O3 -std=c++20 -ffp-contract=on -Isrc/ik2 -Iinclude -c tests/ik2_experts_test.cu
//        then link with build/ik2_kernels.o
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include "ik2_kernels.cuh"
namespace ik2::k { extern int g_exp_gu, g_exp_dn; }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s -> %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
namespace k = ik2::k;
static const int C = 2304, F = 1152, S = 8;
static int TY = k::kNVFP4;   // the expert type under test (main runs NVFP4, then Q8_0)
static size_t RG, RD, EB0, EB1, EB2, EX;
static void set_type(int ty) {
    TY = ty;
    RG = ty == k::kNVFP4 ? (size_t) C / 64 * 36 : (size_t) C / 32 * 34;
    RD = ty == k::kNVFP4 ? (size_t) F / 64 * 36 : (size_t) F / 32 * 34;
    EB0 = F * RG, EB1 = F * RG, EB2 = C * RD, EX = (EB0 + EB1 + EB2 + 255) & ~(size_t) 255;
}

static int* d_tok_g;
static float* d_x_g;
static uint32_t hsh(uint32_t v) { v *= 2654435761u; v ^= v >> 15; v *= 2246822519u; v ^= v >> 13; return v; }
// a valid NVFP4 expert: scale bytes E4M3 in 0x28..0x47 (about 2^-2 .. 2^1), nibbles random
static void make_expert(uint8_t* e, uint32_t seed) {
    for (size_t i = 0; i < EX; ++i) e[i] = (uint8_t) hsh((uint32_t) i ^ (seed * 0x9E3779B9u));
    auto fix = [&](size_t off, size_t rows, size_t rb) {
        for (size_t r = 0; r < rows; ++r) {
            if (TY == k::kQ8_0) {   // FP16 scale 2^-7 .. 2^-4 (0x2000 .. 0x2FFF), int8 codes random
                for (size_t b = 0; b < rb / 34; ++b) e[off + r * rb + b * 34 + 1] = (uint8_t) (0x20 + (e[off + r * rb + b * 34 + 1] & 0x0F));
                continue;
            }
            for (size_t b = 0; b < rb / 36; ++b)
                for (int q = 0; q < 4; ++q) e[off + r * rb + b * 36 + q] = (uint8_t) (0x28 + (e[off + r * rb + b * 36 + q] & 0x1F));
        }
    };
    fix(0, F, RG), fix(EB0, F, RG), fix(EB0 + EB1, C, RD);
}

static int run(int ty, bool timing) {
    set_type(ty);
    const bool nv = ty == k::kNVFP4;
    const int nvar = nv ? 4 : 0;
    const char* tn = nv ? "NVFP4" : "Q8_0";
    auto gu_new = [&](const uint8_t* const* b, const float* sg, const float* su, uint16_t* h, cudaStream_t st) {
        if (nv) k::experts_gate_up_swiglu_nv(b, d_tok_g, S, EB0, d_x_g, C, F, sg, su, 10.0f, 12.0f, h, st);
        else k::experts_gate_up_swiglu_q8(b, d_tok_g, S, EB0, d_x_g, C, F, sg, su, 10.0f, 12.0f, h, st);
    };
    auto dn_new = [&](const uint8_t* const* b, const uint16_t* h, float* y, cudaStream_t st) {
        if (nv) k::experts_down_nv(b, S, EB0 + EB1, h, C, F, y, st);
        else k::experts_down_q8(b, S, EB0 + EB1, h, C, F, y, st);
    };
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    // ---- parity
    std::vector<uint8_t> hb(EX * S);
    for (int e = 0; e < S; ++e) make_expert(hb.data() + e * EX, 1000 + e);
    uint8_t* d_ex; CK(cudaMalloc(&d_ex, hb.size())); CK(cudaMemcpy(d_ex, hb.data(), hb.size(), cudaMemcpyHostToDevice));
    std::vector<const uint8_t*> ptr(S);
    for (int e = 0; e < S; ++e) ptr[e] = d_ex + (size_t) ((e * 5) % S) * EX;   // slots in a scrambled expert order
    const uint8_t** d_blob; CK(cudaMalloc(&d_blob, S * sizeof(void*))); CK(cudaMemcpy(d_blob, ptr.data(), S * sizeof(void*), cudaMemcpyHostToDevice));
    std::vector<float> hx(2 * C), hs(2 * S);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = ((int) (hsh((uint32_t) i + 77) % 20001) - 10000) * 1e-4f;
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = 0.5f + (hsh((uint32_t) i + 5) % 1000) * 1e-3f;
    std::vector<int> htok(S);
    for (int s = 0; s < S; ++s) htok[s] = s & 1;   // two x rows
    float *d_x, *d_sg, *d_y0, *d_y1; int* d_tok; uint16_t *d_h0, *d_h1;
    CK(cudaMalloc(&d_x, hx.size() * 4)); CK(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_sg, hs.size() * 4)); CK(cudaMemcpy(d_sg, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_tok, S * 4)); CK(cudaMemcpy(d_tok, htok.data(), S * 4, cudaMemcpyHostToDevice));
    d_tok_g = d_tok, d_x_g = d_x;
    CK(cudaMalloc(&d_h0, S * F * 2)); CK(cudaMalloc(&d_h1, S * F * 2));
    CK(cudaMalloc(&d_y0, S * C * 4)); CK(cudaMalloc(&d_y1, S * C * 4));
    const float gcap = 10.0f, ucap = 12.0f;
    std::vector<uint16_t> h0(S * F), h1(S * F);
    std::vector<float> y0(S * C), y1(S * C);
    int fails = 0;
    for (int sgnull = 0; sgnull < 2; ++sgnull) {
        const float* sg = sgnull ? nullptr : d_sg;
        const float* su = sgnull ? nullptr : d_sg + S;
        CK(cudaMemsetAsync(d_h0, 0xFF, S * F * 2, st));   // (on st: a null-stream fill does not order with the non-blocking st and could land after the kernel)
        k::experts_gate_up_swiglu(d_blob, d_tok, S, ty, ty, EB0, d_x, C, F, sg, su, gcap, ucap, d_h0, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(h0.data(), d_h0, S * F * 2, cudaMemcpyDeviceToHost));
        for (int v = 0; v <= nvar; ++v) {
            k::g_exp_gu = v;
            CK(cudaMemsetAsync(d_h1, 0xFF, S * F * 2, st));   // (on st: a null-stream fill does not order with the non-blocking st and could land after the kernel)
            gu_new(d_blob, sg, su, d_h1, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(h1.data(), d_h1, S * F * 2, cudaMemcpyDeviceToHost));
            int bad = 0, nz = 0, nn = 0, first = -1; float mx = 0;   // (nz: +0 vs -0; nn: a NaN, e.g. the 0xFFFF fill = unwritten)
            for (size_t i = 0; i < h0.size(); ++i)
                if (h0[i] != h1[i]) {
                    ++bad;
                    if (first < 0) first = (int) i;
                    __half a, b; std::memcpy(&a, &h0[i], 2); std::memcpy(&b, &h1[i], 2);
                    const float fa = __half2float(a), fb = __half2float(b);
                    nz += fa == 0.0f && fb == 0.0f;
                    nn += std::isnan(fa) || std::isnan(fb);
                    mx = std::fmax(mx, std::fabs(fa - fb));
                }
            std::printf("parity %s gate_up variant %d sg %s: %d / %zu rows differ (max abs %.3g)%s\n", tn, v, sgnull ? "null" : "set", bad,
                        h0.size(), mx, bad ? "  FAIL" : "  bit for bit");
            if (bad)
                std::printf("  of them %d +0/-0, %d with a NaN (0xFFFF = not written); first at %d (expert %d, row %d): %04x vs %04x\n",
                            nz, nn, first, first / F, first % F, h0[first], h1[first]);
            fails += bad != 0;
        }
        // down on h0
        CK(cudaMemsetAsync(d_y0, 0xFF, S * C * 4, st));   // (on st: a null-stream fill does not order with the non-blocking st and could land after the kernel)
        k::experts_down(d_blob, S, ty, EB0 + EB1, d_h0, C, F, d_y0, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(y0.data(), d_y0, S * C * 4, cudaMemcpyDeviceToHost));
        for (int v = 0; v <= nvar; ++v) {
            k::g_exp_dn = v;
            CK(cudaMemsetAsync(d_y1, 0xFF, S * C * 4, st));   // (on st: a null-stream fill does not order with the non-blocking st and could land after the kernel)
            dn_new(d_blob, d_h0, d_y1, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(y1.data(), d_y1, S * C * 4, cudaMemcpyDeviceToHost));
            int bad = 0; float mx = 0;
            for (size_t i = 0; i < y0.size(); ++i)
                if (std::memcmp(&y0[i], &y1[i], 4)) { ++bad; mx = std::fmax(mx, std::fabs(y0[i] - y1[i])); }
            std::printf("parity %s down variant %d (h of sg %s): %d / %zu rows differ (max abs %.3g)%s\n", tn, v, sgnull ? "null" : "set", bad,
                        y0.size(), mx, bad ? "  FAIL" : "  bit for bit");
            fails += bad != 0;
        }
    }
    if (nv) {   // set 16: gate_up + down in one launch, 3 visits in a row (its counters only grow), vs the old pair
        for (int rep = 0; rep < 3; ++rep) {
            const float* sg = rep == 1 ? nullptr : d_sg;
            const float* su = rep == 1 ? nullptr : d_sg + S;
            k::experts_gate_up_swiglu(d_blob, d_tok, S, ty, ty, EB0, d_x, C, F, sg, su, gcap, ucap, d_h0, st);
            k::experts_down(d_blob, S, ty, EB0 + EB1, d_h0, C, F, d_y0, st);
            CK(cudaMemsetAsync(d_h1, 0xFF, S * F * 2, st));
            CK(cudaMemsetAsync(d_y1, 0xFF, S * C * 4, st));
            k::experts_nv_visit(d_blob, d_tok, S, EB0, EB0 + EB1, d_x, C, F, sg, su, gcap, ucap, d_h1, d_y1, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(h0.data(), d_h0, S * F * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h1.data(), d_h1, S * F * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y0.data(), d_y0, S * C * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y1.data(), d_y1, S * C * 4, cudaMemcpyDeviceToHost));
            int bh = 0, by = 0;
            for (size_t i = 0; i < h0.size(); ++i) bh += h0[i] != h1[i];
            for (size_t i = 0; i < y0.size(); ++i) by += std::memcmp(&y0[i], &y1[i], 4) != 0;
            const int pend = k::experts_nv_visit_pending();
            std::printf("parity NVFP4 one-launch visit %d (sg %s): h %d / %zu, y %d / %zu rows differ, counters after %d%s\n", rep,
                        sg ? "set" : "null", bh, h0.size(), by, y0.size(), pend, bh + by || pend ? "  FAIL" : "  bit for bit");
            fails += bh + by != 0 || pend != 0;
        }
    }
    std::printf("sample h0[0..3] %04x %04x %04x %04x  y0[0..1] %g %g\n", h0[0], h0[1], h0[2], h0[3], y0[0], y0[1]);
    std::printf("PARITY %s %s (%d failing checks)\n", tn, fails ? "FAIL" : "OK", fails);
    if (!timing) return fails;
    // ---- speed: fresh pool windows of 8 experts (the pool is uninitialised-safe: hashed bytes)
    const size_t POOL = (size_t) (getenv("EXP_POOL_GIB") ? atoi(getenv("EXP_POOL_GIB")) : 1) << 30;   // (footprint: TLB reach)
    uint8_t* pool; CK(cudaMalloc(&pool, POOL));
    {
        std::vector<uint8_t> chunk(EX);
        make_expert(chunk.data(), 7);
        for (size_t o = 0; o + EX <= POOL; o += EX) CK(cudaMemcpy(pool + o, chunk.data(), EX, cudaMemcpyHostToDevice));
    }
    const int nwin = (int) (POOL / (8 * EX)), R = 24;
    if (getenv("EXP_TOK0")) {   // every slot reads x row 0, as a decode token does (W.tok all 0)
        CK(cudaMemset(d_tok, 0, S * 4));
        d_tok_g = d_tok;
    }
    std::vector<const uint8_t*> wp;
    for (int wi = 0; wi < nwin; ++wi) for (int e = 0; e < 8; ++e) wp.push_back(pool + ((size_t) wi * 8 + e) * EX);
    const uint8_t** d_wb; CK(cudaMalloc(&d_wb, wp.size() * sizeof(void*)));
    CK(cudaMemcpy(d_wb, wp.data(), wp.size() * sizeof(void*), cudaMemcpyHostToDevice));
    std::vector<std::string> labels;
    int it = 0;
    if (nv && getenv("EXP_STREAM")) {   // visits back to back in one stream (no serialization): ms a visit by events
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const int NV = 200;
        for (int rep = 0; rep < 3; ++rep) {
            for (int arm = 0; arm < 3; ++arm) {
                auto visit = [&](const uint8_t* const* b) {
                    if (arm == 0) {
                        k::experts_gate_up_swiglu(b, d_tok, S, ty, ty, EB0, d_x, C, F, d_sg, d_sg + S, gcap, ucap, d_h0, st);
                        k::experts_down(b, S, ty, EB0 + EB1, d_h0, C, F, d_y0, st);
                    } else if (arm == 1) {
                        k::g_exp_gu = 0, k::g_exp_dn = 0;
                        k::experts_gate_up_swiglu_nv(b, d_tok, S, EB0, d_x, C, F, d_sg, d_sg + S, gcap, ucap, d_h0, st);
                        k::experts_down_nv(b, S, EB0 + EB1, d_h0, C, F, d_y0, st);
                    } else {
                        k::experts_nv_visit(b, d_tok, S, EB0, EB0 + EB1, d_x, C, F, d_sg, d_sg + S, gcap, ucap, d_h0, d_y0, st);
                    }
                };
                for (int w = 0; w < 4; ++w) visit(d_wb + (size_t) ((it++) % nwin) * 8);
                CK(cudaEventRecord(e0, st));
                for (int v = 0; v < NV; ++v) visit(d_wb + (size_t) ((it++) % nwin) * 8);
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                std::printf("STREAM rep %d %s: %.2f us a visit (%d visits, 35.8 MB each = %.1f GB/s)\n", rep,
                            arm == 0 ? "old gu + old dn  " : arm == 1 ? "set 13 + set 14  " : "set 16 one launch",
                            ms * 1e3 / NV, NV, 35831808.0 * NV / (ms * 1e-3) / 1e9);
            }
        }
        CK(cudaFree(pool));
        return fails;
    }
    auto win = [&] { return d_wb + (size_t) ((it++) % nwin) * 8; };
    const char* order = nv ? (getenv("EXP_ORDER") ? getenv("EXP_ORDER") : "o0o0") : "o0o0";
    for (int rep = 0; rep < 2; ++rep) {
        for (const char* c = order; *c; ++c) {
            if (*c == 'o') {
                labels.push_back(std::string("old experts_gate_up_swiglu ") + tn + " B=" + std::to_string(8 * (EB0 + EB1)));
                for (int r = 0; r < R; ++r) k::experts_gate_up_swiglu(win(), d_tok, S, ty, ty, EB0, d_x, C, F, d_sg, d_sg + S, gcap, ucap, d_h0, st);
                CK(cudaStreamSynchronize(st));
                labels.push_back(std::string("old experts_down ") + tn + " B=" + std::to_string(8 * EB2));
                for (int r = 0; r < R; ++r) k::experts_down(win(), S, ty, EB0 + EB1, d_h0, C, F, d_y0, st);
                CK(cudaStreamSynchronize(st));
            } else {
                const int v = *c - '0';
                if (v > nvar) continue;
                k::g_exp_gu = v, k::g_exp_dn = v;
                labels.push_back(std::string("new gate_up ") + tn + " v" + std::to_string(v) + " B=" + std::to_string(8 * (EB0 + EB1)));
                for (int r = 0; r < R; ++r) gu_new(win(), d_sg, d_sg + S, d_h1, st);
                CK(cudaStreamSynchronize(st));
                labels.push_back(std::string("new down ") + tn + " v" + std::to_string(v) + " B=" + std::to_string(8 * EB2));
                for (int r = 0; r < R; ++r) dn_new(win(), d_h0, d_y1, st);
                CK(cudaStreamSynchronize(st));
            }
        }
    }
    for (size_t i = 0; i < labels.size(); ++i) std::printf("LABEL - %s\n", labels[i].c_str());
    CK(cudaFree(pool));
    return fails;
}

int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::strcmp(argv[1], "time") == 0;
    int fails = run(k::kNVFP4, timing);
    fails += run(k::kQ8_0, timing);
    std::printf("ALL %s\n", fails ? "FAIL" : "OK (bit for bit)");
    return fails ? 1 : 0;
}
