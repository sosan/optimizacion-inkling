cmake_minimum_required(VERSION 3.24)

# Self-contained Inkling 2 Mini engine. CUDA by default; AMD HIP with
# -DSTRATA_ENABLE_HIP=ON -DSTRATA_HIP_IK2_SPIKE=ON -DAMDGPU_TARGETS=gfx1201.
# For HIP we use hipcc as the C++ compiler instead of CMake's HIP language:
# the latter passes -x hip to the host-only .cpp sources of a mixed-language target.
if(STRATA_ENABLE_HIP)
  set(CMAKE_CXX_COMPILER hipcc CACHE FILEPATH "")
endif()
project(inkling_custom_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STRATA_ENABLE_HIP "Build the ik2 kernels with AMD HIP instead of CUDA" OFF)
option(STRATA_HIP_IK2_SPIKE "Also build the RDNA4 expert GEMM benchmark" OFF)

find_package(Threads REQUIRED)

set(IK2_GPU_SOURCES src/ik2/ik2_kernels.cu src/ik2/ik2_main.cu src/ik2/ik2_draft.cu)
# llama.cpp's IQ / native GGUF dot kernels (issue #19: the fused path for small quants).
set(IQ_GPU_SOURCES src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu)

if(STRATA_ENABLE_HIP)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  # Upstream PR #26 guards the gfx1151/turbo6 paths with STRATA_USE_HIP (set by the full Maya CMake).
  add_compile_definitions(STRATA_USE_HIP=1)
  set(IK2_GPU_BLAS roc::hipblas)
  set(IK2_GPU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat")
  # No CUDA/HIP CMake language is enabled, so .cu must be declared as CXX explicitly; hipcc
  # compiles them as device code thanks to -x hip. The .cpp files stay plain host C++.
  set_source_files_properties(${IK2_GPU_SOURCES} ${IQ_GPU_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;hip")
  set(IK2_GPU_TESTS)
  # ik2_fuse_test / ik2_experts_test / ik2_dsc_test exercise AMD-only paths (PR #25/#26).
  foreach(t ik2_expert_bench ik2_kernels_test)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cu")
      list(APPEND IK2_GPU_TESTS ${t})
      set_source_files_properties(tests/${t}.cu PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;hip")
    endif()
  endforeach()
  message(STATUS "Inkling: HIP (${AMDGPU_TARGETS}) with hip_compat headers")
else()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Basalt-derived NVFP4 -> int8 fragment test (sm_80+; sm_89 and sm_120 both covered).
  add_executable(ik2_nvfp4_s8_test tests/ik2_nvfp4_s8_test.cu)
  target_include_directories(ik2_nvfp4_s8_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set(IK2_GPU_BLAS CUDA::cublas)
  set(IK2_GPU_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

add_library(inkling_custom_kernels STATIC src/ik2/ik2_kernels.cu ${IQ_GPU_SOURCES})
set_target_properties(inkling_custom_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(inkling_custom_kernels PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ik2"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml"
  ${IK2_GPU_INCLUDE}
)

add_executable(inkling-custom-engine
  src/ik2/ik2_main.cu
  src/ik2/ik2_draft.cu
  src/ik2/ik2_cpu.cpp
)
target_include_directories(inkling-custom-engine PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ik2"
  ${IK2_GPU_INCLUDE}
)
target_link_libraries(inkling-custom-engine PRIVATE
  inkling_custom_kernels
  ${IK2_GPU_BLAS}
  Threads::Threads
)

# Kernel A/B and parity harnesses (upstream PRs #8, #25, #26)
if(STRATA_ENABLE_HIP)
  foreach(t ${IK2_GPU_TESTS})
    add_executable(${t} tests/${t}.cu)
    set_target_properties(${t} PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(${t} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src/ik2" ${IK2_GPU_INCLUDE})
    target_link_libraries(${t} PRIVATE inkling_custom_kernels ${IK2_GPU_BLAS} Threads::Threads)
  endforeach()
else()
  # The expert A/B benchmark is useful on CUDA too: issue #19's scalar IQ decoders have to be
  # measured against NVFP4 before deciding whether a dedicated matrix kernel is worth it.
  add_executable(ik2_expert_bench tests/ik2_expert_bench.cu)
  target_include_directories(ik2_expert_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src/ik2" ${IK2_GPU_INCLUDE})
  target_link_libraries(ik2_expert_bench PRIVATE inkling_custom_kernels ${IK2_GPU_BLAS} Threads::Threads)
  # Upstream PR #31: prompt NVFP4 grouped GEMMs, decode + cuBLAS against the kernel variants.
  add_executable(ik2_pf_bench tests/ik2_pf_bench.cu)
  target_include_directories(ik2_pf_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src/ik2" ${IK2_GPU_INCLUDE})
  target_link_libraries(ik2_pf_bench PRIVATE inkling_custom_kernels ${IK2_GPU_BLAS} Threads::Threads)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|HIP")
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_sources(inkling-custom-engine PRIVATE src/ik2/ik2_cpu_deep.cpp)
    set_source_files_properties(src/ik2/ik2_cpu_deep.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
    target_compile_definitions(inkling-custom-engine PRIVATE IK2_HOST_AVX2=1)
    target_link_libraries(inkling-custom-engine PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
