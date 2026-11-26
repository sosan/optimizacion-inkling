// include/strata/kernels/iq_f16.hpp - FP16-activation adapter for llama.cpp's IQ dot kernels.
//
// strata::kernels::iq_mmvq() uses q8_1 activations (llama.cpp's CUDA contract), while the ik2
// engine stages its activations as FP16. This adapter converts a column-major batch of FP16
// activations to q8_1 on the device and then calls the existing IQ mmvq kernel. The caller owns
// both scratch buffers (no allocation or host sync inside the call), so it can be used from a
// decode loop. The conversion is two cheap elementwise passes over the activation batch, not
// over the weights.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// y[c][r] = W[r] . x[c], W in GGUF IQ blocks, x[c] FP16 (column c at x + c * n_in),
/// y column c at y + c * n_out. `f32` must hold ncols*n_in floats and `q8_1` must hold
/// native_q8_1_bytes(n_in, ncols) bytes. Both are device pointers.
void iq_mmvq_f16(int ggml_type, const void* w, const uint16_t* x_f16, float* y,
                 int n_in, int n_out, int ncols, float* f32, void* q8_1, void* stream);

}
