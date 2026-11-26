// src/ik2/ik2_turbo6.h - the turbo6 KV block (IK2_KV=turbo6), host + device: 128 values in 98 bytes - the FP16 norm,
// the codes' low nibbles (qs [64]: value j in byte j / 2 at shift (j % 2) * 4) and their top 2 bits (qh [32]: value j
// in byte j / 4 at shift (j % 4) * 2); a code indexes 64 Lloyd-Max centroids of N(0, 1/128), the values being the
// block divided by its L2 norm and rotated (signs1, the normalized 128-point Walsh-Hadamard transform, signs2); the
// norm stored is the L2 norm over the centroids picked (a decoded block keeps the input's norm).
// From tubo's turbo4_codec.hpp (MIT; a gufo-based engine), its k6 path: the tables and FloatToHalfBits ..
// RotateInverse copied (the line numbers below are that file's), the functions that take its Bits reduced to the k6
// branch.  Its turbo6 block and tables come from TheTom/llama-cpp-turboquant PR #381 (MIT), an implementation of
// TurboQuant (arXiv 2504.19874).  Inkling's head_dim 128 is one block a head.  The device code (ik2_turbo6.inc) writes
// these bytes exactly: tests/ik2_kernels_test.cu test_turbo6 compares them with QuantizeBlock below.
// AMD build only (included under STRATA_USE_HIP).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#define T6_HD __host__ __device__
#else
#define T6_HD
#endif

namespace ik2::t6 {

inline constexpr std::size_t kBlockValues = 128;
inline constexpr std::size_t kNormBytes = 2;
inline constexpr std::size_t kQhOffset = 66;     // kNormBytes + 64 qs bytes
inline constexpr std::size_t kBlockBytes = 98;   // kQhOffset + 32 qh bytes
inline constexpr float kInvSqrt128 = 0.08838834764831845f;   // turbo4_codec.hpp:77
inline constexpr float kNormEpsilon = 1e-10f;                // turbo4_codec.hpp:79

// turbo4_codec.hpp:162
inline constexpr float TURBO_CENTROIDS_6BIT[64] = {
    -0.330935f, -0.286417f, -0.257865f, -0.236198f, -0.218435f, -0.203203f,
    -0.189753f, -0.177626f, -0.166522f, -0.156230f, -0.146600f, -0.137517f,
    -0.128895f, -0.120663f, -0.112765f, -0.105157f, -0.097801f, -0.090663f,
    -0.083717f, -0.076940f, -0.070310f, -0.063809f, -0.057422f, -0.051133f,
    -0.044929f, -0.038798f, -0.032729f, -0.026710f, -0.020733f, -0.014787f,
    -0.008863f, -0.002953f, 0.002953f,  0.008863f,  0.014787f,  0.020733f,
    0.026710f,  0.032729f,  0.038798f,  0.044929f,  0.051133f,  0.057422f,
    0.063809f,  0.070310f,  0.076940f,  0.083717f,  0.090663f,  0.097801f,
    0.105157f,  0.112765f,  0.120663f,  0.128895f,  0.137517f,  0.146600f,
    0.156230f,  0.166522f,  0.177626f,  0.189753f,  0.203203f,  0.218435f,
    0.236198f,  0.257865f,  0.286417f,  0.330935f,
};

// turbo4_codec.hpp:176
inline constexpr float TURBO_MID_6BIT[63] = {
    -0.308676f, -0.272141f, -0.247031f, -0.227316f, -0.210819f, -0.196478f,
    -0.183690f, -0.172074f, -0.161376f, -0.151415f, -0.142059f, -0.133206f,
    -0.124779f, -0.116714f, -0.108961f, -0.101479f, -0.094232f, -0.087190f,
    -0.080329f, -0.073625f, -0.067060f, -0.060616f, -0.054277f, -0.048031f,
    -0.041864f, -0.035763f, -0.029719f, -0.023722f, -0.017760f, -0.011825f,
    -0.005908f, 0.000000f,  0.005908f,  0.011825f,  0.017760f,  0.023722f,
    0.029719f,  0.035763f,  0.041864f,  0.048031f,  0.054277f,  0.060616f,
    0.067060f,  0.073625f,  0.080329f,  0.087190f,  0.094232f,  0.101479f,
    0.108961f,  0.116714f,  0.124779f,  0.133206f,  0.142059f,  0.151415f,
    0.161376f,  0.172074f,  0.183690f,  0.196478f,  0.210819f,  0.227316f,
    0.247031f,  0.272141f,  0.308676f,
};

// turbo4_codec.hpp:190
inline constexpr float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,
    1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f,
    -1.0f, 1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
    1.0f,  1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f,
    -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, -1.0f,
    1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f,
    -1.0f, 1.0f,  1.0f,  1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,  1.0f,  -1.0f, -1.0f,
    1.0f,  1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,
    -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,
    -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    1.0f,  1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,
};

// turbo4_codec.hpp:205
inline constexpr float TURBO_WHT_SIGNS2[128] = {
    1.0f,  1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, -1.0f,
    -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, 1.0f,  -1.0f,
    1.0f,  -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  1.0f,  1.0f,
    1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  1.0f,  -1.0f, -1.0f,
    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  1.0f,  -1.0f,
    1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f,
    -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f,
    -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,
    1.0f,  1.0f,  -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f,
    -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f,
};

// turbo4_codec.hpp:223
T6_HD inline std::uint16_t FloatToHalfBits(float value) noexcept {
  std::uint32_t bits;
#if defined(__clang__) || defined(__GNUC__)
  __builtin_memcpy(&bits, &value, sizeof(bits));
#else
  std::memcpy(&bits, &value, sizeof(bits));
#endif
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::int32_t exponent =
      static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  const std::uint32_t mantissa = bits & 0x7FFFFFu;

  if (((bits >> 23) & 0xFFu) == 0xFFu) {  // inf / nan
    return static_cast<std::uint16_t>(sign | 0x7C00u |
                                      (mantissa ? 0x200u : 0u));
  }
  if (exponent >= 0x1F)
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<std::uint16_t>(sign);
    const std::uint32_t sub = mantissa | 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t rounded = sub >> shift;
    const std::uint32_t remainder = sub & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  // Round to nearest, ties to even.
  std::uint32_t half_mantissa = mantissa >> 13;
  const std::uint32_t rest = mantissa & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (half_mantissa & 1u)))
    ++half_mantissa;
  std::uint32_t out =
      (static_cast<std::uint32_t>(exponent) << 10) + half_mantissa;
  if (out >= 0x7C00u)
    out = 0x7C00u;
  return static_cast<std::uint16_t>(sign | out);
}

// turbo4_codec.hpp:262
T6_HD inline float HalfBitsToFloat(std::uint16_t half) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
  std::uint32_t exponent = (half >> 10) & 0x1Fu;
  std::uint32_t mantissa = half & 0x3FFu;

  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // Subnormal half -> normal float.
      exponent = 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFu;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// turbo4_codec.hpp:293
T6_HD inline void FastWalshHadamard128(float* x) noexcept {
  for (std::size_t h = 1; h < kBlockValues; h *= 2) {
    for (std::size_t i = 0; i < kBlockValues; i += h * 2) {
      for (std::size_t j = i; j < i + h; ++j) {
        const float a = x[j];
        const float b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
  for (std::size_t i = 0; i < kBlockValues; ++i)
    x[i] *= kInvSqrt128;
}

// turbo4_codec.hpp:309
T6_HD inline void RotateForward(float* x) noexcept {
  for (std::size_t i = 0; i < kBlockValues; ++i)
    x[i] *= TURBO_WHT_SIGNS1[i];
  FastWalshHadamard128(x);
  for (std::size_t i = 0; i < kBlockValues; ++i)
    x[i] *= TURBO_WHT_SIGNS2[i];
}

// turbo4_codec.hpp:326
T6_HD inline void RotateInverse(float* x) noexcept {
  for (std::size_t i = 0; i < kBlockValues; ++i)
    x[i] *= TURBO_WHT_SIGNS2[i];
  FastWalshHadamard128(x);
  for (std::size_t i = 0; i < kBlockValues; ++i)
    x[i] *= TURBO_WHT_SIGNS1[i];
}

// NearestCode's k6 branch (turbo4_codec.hpp:412): the first i with value < MID[i], a binary search (strict '<': a value
// on a midpoint takes the higher code)
[[nodiscard]] T6_HD inline std::uint8_t NearestCode(float value) noexcept {
  int lo = 0;
  int hi = 63;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (value < TURBO_MID_6BIT[mid])
      hi = mid;
    else
      lo = mid + 1;
  }
  return static_cast<std::uint8_t>(lo);
}

// CodeAt's k6 branch (turbo4_codec.hpp:462; IndexAt :362): element j's code in a stored block (`block` at its norm)
[[nodiscard]] T6_HD inline std::uint32_t CodeAt(const std::uint8_t* block, std::size_t j) noexcept {
  const std::uint32_t nib = static_cast<std::uint8_t>((block[kNormBytes + j / 2] >> ((j % 2) * 4)) & 0xF);
  const std::uint32_t hi = (block[kQhOffset + j / 4U] >> (2U * (j % 4U))) & 0x3U;
  return nib | (hi << 4U);
}

// PackRotatedBlock's k6 path (turbo4_codec.hpp:480): 128 rotated values -> qs and qh (past the norm)
T6_HD inline void PackRotatedBlock(const float* rotated, std::uint8_t* block) noexcept {
  std::memset(block + kNormBytes, 0, kBlockBytes - kNormBytes);
  for (std::size_t j = 0; j < kBlockValues; ++j) {
    const std::uint32_t code = NearestCode(rotated[j]);
    block[kNormBytes + j / 2] |= static_cast<std::uint8_t>((code & 0xFU) << ((j % 2) * 4));
    block[kQhOffset + j / 4U] |= static_cast<std::uint8_t>(((code >> 4U) & 0x3U) << (2U * (j % 4U)));
  }
}

// ReconstructionNormBlock's k6 path (turbo4_codec.hpp:498): the picked centroids' L2 norm, j = 0..127 in order
[[nodiscard]] T6_HD inline float ReconstructionNormBlock(const std::uint8_t* block) noexcept {
  float acc = 0.0f;
  for (std::size_t j = 0; j < kBlockValues; ++j) {
    const float c = TURBO_CENTROIDS_6BIT[CodeAt(block, j) & 63U];
    acc += c * c;
  }
  return std::sqrt(acc);
}

// QuantizeBlock with Bits::k6 (turbo4_codec.hpp:538): 98 bytes; returns the group norm
T6_HD inline float QuantizeBlock(const float* src, std::uint8_t* dst) noexcept {
  float sum = 0.0f;
  for (std::size_t j = 0; j < kBlockValues; ++j)
    sum += src[j] * src[j];

  const float group_norm = std::sqrt(sum);

  const float inv_norm = group_norm > kNormEpsilon ? 1.0f / group_norm : 0.0f;
  float work[kBlockValues];
  for (std::size_t j = 0; j < kBlockValues; ++j)
    work[j] = src[j] * inv_norm;
  RotateForward(work);
  PackRotatedBlock(work, dst);

  const float recon = ReconstructionNormBlock(dst);
  const float corrected =
      recon > kNormEpsilon ? group_norm / recon : group_norm;
  const std::uint16_t half = FloatToHalfBits(corrected);
  std::memcpy(dst, &half, sizeof(half));
  return group_norm;
}

// DequantizeBlock with Bits::k6 (turbo4_codec.hpp:597): centroid x norm, in the ROTATED domain (RotateInverse after)
T6_HD inline void DequantizeBlock(const std::uint8_t* src, float* dst) noexcept {
  std::uint16_t half = 0;
  std::memcpy(&half, src, sizeof(half));
  const float norm = HalfBitsToFloat(half);
  for (std::size_t j = 0; j < kBlockValues; ++j)
    dst[j] = TURBO_CENTROIDS_6BIT[CodeAt(src, j) & 63U] * norm;
}

}  // namespace ik2::t6
