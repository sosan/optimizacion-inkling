// src/ik2/ik2_cpu.cpp - maya-ik2's CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

// the AVX-512 kernels: GCC / Clang through target attributes, MSVC (whose intrinsics need no /arch) as they are
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define IK2_X86_AVX512 1
#define IK2_TARGET(t) __attribute__((target(t)))
#elif defined(_MSC_VER) && defined(_M_X64)
#define IK2_X86_AVX512 1
#define IK2_TARGET(t)
#else
#define IK2_X86_AVX512 0
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ik2::cpu {

namespace {
#if IK2_X86_AVX512
bool avx512_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    // MSVC: the CPU's AVX-512 F / BW / VL bits (leaf 7) and the OS saving the ZMM state (XCR0 bits 1, 2, 5, 6, 7)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    if (!(r[2] & (1 << 27)) || !(r[2] & (1 << 12)) || !(r[2] & (1 << 29))) return false;   // OSXSAVE, FMA, F16C
    if ((_xgetbv(0) & 0xE6) != 0xE6) return false;
    __cpuidex(r, 7, 0);
    const unsigned ebx = (unsigned) r[1], f = 1u << 16, bw = 1u << 30, vl = 1u << 31;
    return (ebx & f) && (ebx & bw) && (ebx & vl);
#endif
}
// AVX2 + FMA + F16C (client cores since Alder Lake have no AVX-512): CPUID leaf 1 and 7, XCR0 YMM state.
bool avx2_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    if (!(r[2] & (1 << 27)) || !(r[2] & (1 << 12)) || !(r[2] & (1 << 29))) return false;   // OSXSAVE, FMA, F16C
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;   // AVX2
#endif
}
#else
bool avx512_ok() { return false; }
bool avx2_ok() { return false; }
#endif
}  // namespace

bool available() { return avx512_ok() || avx2_ok(); }

Pool::Pool(int n) {
    for (int i = 0; i < n; ++i)
        th_.emplace_back([this]() {
            for (;;) {
                std::function<void()> f;
                {
                    // spin a while before sleeping: an expert's chunks come ~1 ms apart, a wake-up costs tens of us
                    std::unique_lock<std::mutex> lk(m_, std::defer_lock);
                    for (int spin = 0;; ++spin) {
                        lk.lock();
                        if (stop_ || !q_.empty()) break;
                        if (spin > 20000) {
                            cv_.wait(lk, [this]() { return stop_ || !q_.empty(); });
                            break;
                        }
                        lk.unlock();
                        _mm_pause();
                    }
                    if (stop_ && q_.empty()) return;
                    f = std::move(q_.front());
                    q_.pop_front();
                }
                f();
                {
                    std::lock_guard<std::mutex> lk(m_);
                    if (--pending_ == 0) done_.notify_all();
                }
            }
        });
}
Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}
void Pool::post(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back(std::move(f));
        ++pending_;
    }
    cv_.notify_one();
}
void Pool::wait() {
    std::unique_lock<std::mutex> lk(m_, std::defer_lock);
    for (int spin = 0; spin < 200000; ++spin) {   // (the caller spins too: the work is short)
        lk.lock();
        if (pending_ == 0) return;
        lk.unlock();
        _mm_pause();
    }
    lk.lock();
    done_.wait(lk, [this]() { return pending_ == 0; });
}

namespace {

// ggml_ue4m3_to_fp32 of every byte
struct ScaleTable {
    float v[256];
    ScaleTable() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) {
                v[x] = 0.0f;
                continue;
            }
            const int e = (x >> 3) & 0xF, m = x & 7;
            v[x] = (e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7)) * 0.5f;
        }
    }
};
const ScaleTable g_scale;

#if IK2_X86_AVX512
// a 36-byte block: four UE4M3 scales, then 8 bytes a 16-value sub-block (value j < 8 the low nibble of byte j)
IK2_TARGET("avx512f,avx512bw,avx512vl,fma")
void nvfp4_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
#define IK2_SUB(sb, acc)                                                                       \
    {                                                                                          \
        const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + (sb) * 8));             \
        const __m128i lo = _mm_and_si128(raw, m4);                                             \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);                          \
        const __m128i val = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(lo, hi));                 \
        const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(val));                        \
        const __m512 xs = _mm512_mul_ps(_mm512_loadu_ps(xp + (sb) * 16), _mm512_set1_ps(g_scale.v[b[sb]])); \
        acc = _mm512_fmadd_ps(v, xs, acc);                                                     \
    }
            IK2_SUB(0, acc0)
            IK2_SUB(1, acc1)
            IK2_SUB(2, acc0)
            IK2_SUB(3, acc1)
#undef IK2_SUB
        }
        out[r - r0] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}
#endif

#if IK2_X86_AVX512
IK2_TARGET("avx2,fma")
static inline float hsum256(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

// the same 36-byte NVFP4 blocks as above, 8 values a step (upstream PR #20)
IK2_TARGET("avx2,fma")
void nvfp4_rows_avx2(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
            for (int sb = 0; sb < 4; ++sb) {
                const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + sb * 8));
                const __m128i lo = _mm_and_si128(raw, m4);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
                const __m128i val = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(lo, hi));
                const __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(val));
                const __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(val, 8)));
                const __m256 sc = _mm256_set1_ps(g_scale.v[b[sb]]);
                acc0 = _mm256_fmadd_ps(v0, _mm256_mul_ps(_mm256_loadu_ps(xp + sb * 16), sc), acc0);
                acc1 = _mm256_fmadd_ps(v1, _mm256_mul_ps(_mm256_loadu_ps(xp + sb * 16 + 8), sc), acc1);
            }
        }
        out[r - r0] = hsum256(_mm256_add_ps(acc0, acc1));
    }
}

IK2_TARGET("avx2,fma,f16c")
void q8_0_rows_avx2(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 34, xp += 32) {
            uint16_t h;
            std::memcpy(&h, b, 2);
            const __m256 d = _mm256_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))));
            for (int q = 0; q < 4; ++q) {
                const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (b + 2 + q * 8))));
                __m256& acc = (q & 1) ? acc1 : acc0;
                acc = _mm256_fmadd_ps(v, _mm256_mul_ps(_mm256_loadu_ps(xp + q * 8), d), acc);
            }
        }
        out[r - r0] = hsum256(_mm256_add_ps(acc0, acc1));
    }
}
#endif

void nvfp4_rows_scalar(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 36)
            for (int sb = 0; sb < 4; ++sb) {
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t by = b[4 + sb * 8 + j];
                    a += kv[by & 0xF] * x[i * 64 + sb * 16 + j] + kv[by >> 4] * x[i * 64 + sb * 16 + j + 8];
                }
                acc += (double) g_scale.v[b[sb]] * a;
            }
        out[r - r0] = (float) acc;
    }
}

#if IK2_X86_AVX512
IK2_TARGET("avx512f,avx512bw,avx512vl,fma,f16c")
void q8_0_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 34, xp += 32) {
            uint16_t h;
            std::memcpy(&h, b, 2);
            const __m512 d = _mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))));   // (MSVC has no _cvtsh_ss)
            const __m512 v0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (b + 2))));
            const __m512 v1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (b + 18))));
            acc0 = _mm512_fmadd_ps(v0, _mm512_mul_ps(_mm512_loadu_ps(xp), d), acc0);
            acc1 = _mm512_fmadd_ps(v1, _mm512_mul_ps(_mm512_loadu_ps(xp + 16), d), acc1);
        }
        out[r - r0] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}
#endif

float fp16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) return (s ? -1.0f : 1.0f) * std::ldexp((float) m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    const uint32_t bits = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void q8_0_rows_scalar(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const int nb = K / 32;
    const size_t rb = (size_t) nb * 34;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 34) {
            uint16_t h;
            std::memcpy(&h, b, 2);
            float a = 0.0f;
            for (int j = 0; j < 32; ++j) a += (float) (int8_t) b[2 + j] * x[i * 32 + j];
            acc += (double) fp16_to_f32(h) * a;
        }
        out[r - r0] = (float) acc;
    }
}

}  // namespace

void q8_0_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if IK2_X86_AVX512
    static const int fast = avx512_ok() ? 2 : avx2_ok() ? 1 : 0;
    if (fast == 2) { q8_0_rows_avx512(W, K, r0, r1, x, out); return; }
    if (fast == 1) { q8_0_rows_avx2(W, K, r0, r1, x, out); return; }
#endif
    q8_0_rows_scalar(W, K, r0, r1, x, out);
}

void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if IK2_X86_AVX512
    static const int fast = avx512_ok() ? 2 : avx2_ok() ? 1 : 0;
    if (fast == 2) { nvfp4_rows_avx512(W, K, r0, r1, x, out); return; }
    if (fast == 1) { nvfp4_rows_avx2(W, K, r0, r1, x, out); return; }
#endif
    nvfp4_rows_scalar(W, K, r0, r1, x, out);
}

void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap) {
    job->gu.resize((size_t) 2 * F);
    job->h.resize(F);
    const int n = std::max(1, pool.size());
    auto down = [&pool, job, C, F, n]() {
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) C * c / n), r1 = (int) ((long) C * (c + 1) / n);
            pool.post([job, F, r0, r1]() {
                (job->q8 ? q8_0_rows : nvfp4_rows)(job->down, F, r0, r1, job->h.data(), job->y + r0);
            });
        }
    };
    job->left = n;   // the chunks of gate and up rows, each with its rows' activation
    for (int c = 0; c < n; ++c) {
        const int r0 = (int) ((long) F * c / n), r1 = (int) ((long) F * (c + 1) / n);
        pool.post([job, C, F, gcap, ucap, r0, r1, down]() {
            auto rows = job->q8 ? q8_0_rows : nvfp4_rows;
            float* g = job->gu.data();
            float* u = job->gu.data() + F;
            rows(job->gate, C, r0, r1, job->x, g + r0);
            rows(job->up, C, r0, r1, job->x, u + r0);
            for (int i = r0; i < r1; ++i) {
                const float gs = g[i] * job->sg, us = u[i] * job->su;
                job->h[i] = gcap * std::tanh(gs / gcap) * (1.0f / (1.0f + std::exp(-gs))) * ucap * std::tanh(us / ucap);
            }
            if (--job->left == 0) down();   // the last chunk: down's rows, which need all of h
        });
    }
}

}  // namespace ik2::cpu
