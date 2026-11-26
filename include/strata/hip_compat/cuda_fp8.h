#pragma once
// CUDA's cuda_fp8.h for HIP builds of CUDA-shaped sources.
// HIP's amd_hip_fp8.h provides the same operations under __hip_* names; this header
// maps the __nv_* spelling onto them so a source written for CUDA compiles unchanged.
#include <hip/hip_fp8.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)

using __nv_fp8_storage_t = __hip_fp8_storage_t;
using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;
using __nv_fp8_interpretation_t = __hip_fp8_interpretation_t;
using __nv_saturation_t = __hip_saturation_t;

static constexpr __nv_saturation_t __NV_SATFINITE = __HIP_SATFINITE;
static constexpr __nv_saturation_t __NV_NOSAT = __HIP_NOSAT;
static constexpr __nv_fp8_interpretation_t __NV_E4M3 = __HIP_E4M3;
static constexpr __nv_fp8_interpretation_t __NV_E5M2 = __HIP_E5M2;

#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_fp8x2_to_halfraw2 __hip_cvt_fp8x2_to_halfraw2
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2
#define __nv_cvt_halfraw_to_fp8 __hip_cvt_halfraw_to_fp8
#define __nv_cvt_halfraw2_to_fp8x2 __hip_cvt_halfraw2_to_fp8x2

#endif  // HIP
