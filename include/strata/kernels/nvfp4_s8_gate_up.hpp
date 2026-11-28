// Host entry for Basalt's NVFP4 gate/up dot via mma.sync.s8 (sm_80+).
// Weights are GGUF NVFP4 blocks (type 40), activations are q8_1 (36 bytes per
// 32 values), so the arithmetic matches ik2's scalar decode but the 16-value
// sub-block products are summed in int8 MMA lanes. Eight tokens per expert;
// h[orow[slot] * F + r] = SwiGLU(gate, up) in FP16, like
// experts_gate_up_swiglu_g. Returns false if the shape cannot be handled (or
// on HIP, where mma.sync.s8 is unavailable) so the caller keeps the old path.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::nvfp4s8 {

bool gate_up_s8(const uint8_t* const* blob, const int* tok, const int* orow, int G, size_t eb0, int C, int F,
                const void* x_q8_1, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                void* stream);

}  // namespace strata::kernels::nvfp4s8
