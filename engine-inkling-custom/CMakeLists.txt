cmake_minimum_required(VERSION 3.24)

project(inkling_custom_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(INKLING_CUSTOM_ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(inkling_custom_kernels STATIC
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2/ik2_kernels.cu"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/kernels/cuda/iq_kernels.cu"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/kernels/cuda/native_mmvq.cu"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/kernels/cuda/nvfp4_s8_gate_up.cu"
)

target_include_directories(inkling_custom_kernels PUBLIC
  "${INKLING_CUSTOM_ENGINE_ROOT}/include"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/kernels/cuda"
  "${INKLING_CUSTOM_ENGINE_ROOT}/third_party/ggml"
)
target_link_libraries(inkling_custom_kernels PUBLIC CUDA::cudart)

add_executable(inkling-custom-engine
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2/ik2_main.cu"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2/ik2_draft.cu"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2/ik2_cpu.cpp"
)

set_target_properties(inkling-custom-engine PROPERTIES
  OUTPUT_NAME "inkling-custom-engine"
)

target_include_directories(inkling-custom-engine PRIVATE
  "${INKLING_CUSTOM_ENGINE_ROOT}/include"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2"
)

target_link_libraries(inkling-custom-engine PRIVATE
  inkling_custom_kernels
  CUDA::cublas
  CUDA::cudart
  Threads::Threads
)

# Expert-kernel A/B benchmark (issue #19: scalar IQ decoders vs NVFP4). Built by CI so the
# binary can be downloaded and run on a GPU machine; the engine itself does not depend on it.
add_executable(ik2_expert_bench
  "${INKLING_CUSTOM_ENGINE_ROOT}/tests/ik2_expert_bench.cu"
)
target_include_directories(ik2_expert_bench PRIVATE
  "${INKLING_CUSTOM_ENGINE_ROOT}/include"
  "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2"
)
target_link_libraries(ik2_expert_bench PRIVATE
  inkling_custom_kernels
  CUDA::cublas
  CUDA::cudart
  Threads::Threads
)

# Basalt-derived NVFP4 -> int8 fragment test (CUDA CI, sm_89 + sm_120).
add_executable(ik2_nvfp4_s8_test
  "${INKLING_CUSTOM_ENGINE_ROOT}/tests/ik2_nvfp4_s8_test.cu"
)
target_include_directories(ik2_nvfp4_s8_test PRIVATE
  "${INKLING_CUSTOM_ENGINE_ROOT}/include"
)

# Correctness of the s8 MMA gate/up kernel against an independent scalar
# reference. CI only builds it; it needs a GPU, so the user runs it locally.
add_executable(ik2_nvfp4_s8_gate_up_test
  "${INKLING_CUSTOM_ENGINE_ROOT}/tests/ik2_nvfp4_s8_gate_up_test.cu"
)
target_include_directories(ik2_nvfp4_s8_gate_up_test PRIVATE
  "${INKLING_CUSTOM_ENGINE_ROOT}/include"
)
target_link_libraries(ik2_nvfp4_s8_gate_up_test PRIVATE
  inkling_custom_kernels
  CUDA::cudart
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_sources(inkling-custom-engine PRIVATE
      "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2/ik2_cpu_deep.cpp"
    )
    set_source_files_properties(
      "${INKLING_CUSTOM_ENGINE_ROOT}/src/ik2/ik2_cpu_deep.cpp"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off"
    )
    target_compile_definitions(inkling-custom-engine PRIVATE IK2_HOST_AVX2=1)
    target_link_libraries(inkling-custom-engine PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
