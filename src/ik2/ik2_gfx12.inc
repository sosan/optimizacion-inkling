// src/ik2/ik2_gfx12.inc - RDNA4 (gfx1200/gfx1201) matrix-core paths for ik2's expert GEMMs.
// Ported from upstream PR #8 (alesha-pro/maya-inkling-mini): the m16n8k16 emulation and the
// gfx12 v_wmma_f32_16x16x16_f16_w32_gfx12 implementation, plus the grouped/one-token expert
// kernels. Included only by the HIP branch of ik2_kernels.cu. CUDA is untouched.
#if defined(STRATA_USE_HIP)

// mma.sync m16n8k16's fragment layout is kept exactly, so every kernel that builds fragments is unchanged: a
// lane's group g = lane >> 2 and tig t = lane & 3 hold rows g and g + 8, columns 2 t and 2 t + 1 of the 16 x 8
// result; A's a[0..3] are A[g][2t, 2t+1], A[g+8][2t, 2t+1], A[g][2t+8, 2t+9], A[g+8][2t+8, 2t+9]; B's b0, b1
// are B[2t, 2t+1][g] and B[2t+8, 2t+9][g].
// P3-4: on gfx12 (RDNA4) the arithmetic is the matrix core: v_wmma_f32_16x16x16_f16_w32_gfx12, llvm's
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 (wave32; the plain ..._w32 builtin is gfx11's, needs
// wmma-256b-insts and does not exist on gfx1201).  The instruction is 16 x 16 x 16 - N = 16, not CUDA's 8 - so
// its two 16 x 8 halves are the two m16n8k16 n8 blocks: one instruction takes the B fragments of both, and its
// accumulator splits back into the caller's two c[4] sets.  An operand is 8 halves a lane (4 dwords), the
// accumulator 8 F32 a lane, and the hardware's distribution is (verified on gfx1201, and the layout of the
// repo's other gfx12 WMMA kernels - src/ds4/ds4_kernels.cu, src/kernels/cuda/exl3.cu):
//     operand A (M x K = 16 x 16): lane l holds A[l % 16][8 (l >> 4) .. + 7]      (one row, 8 of the 16 k)
//     operand B (K x N = 16 x 16): lane l holds B[8 (l >> 4) .. + 7][l % 16]      (one column, 8 of the 16 k)
//     accumulator D:                lane l holds D[8 (l >> 4) + i][l % 16], i = 0..7
// i.e. a lane owns one column of the result, 8 rows of it.  m16n8k16's lane owns 2 rows x 2 columns per n8
// block instead, so the two distributions are not a permutation of each other: the accumulator has to be
// re-laid.  Doing that on the product of two n8 blocks costs 8 shuffles per lane (below); doing it per n8 block
// costs the same 8 (a shuffle carries one register to every lane at once), which is why the callers that have
// two neighbouring n8 blocks of one A - mma16816x2 - should use it.
//
// The re-lay is cheapest with the product TRANSPOSED, which is what this does: with C = A (16 queries x 16 k) x
// B (16 k x 8 keys), feed the hardware X = B^T (keys x k, the caller's b0, b1) as its A and Y = A^T (k x
// queries, the caller's a[0..3]) as its B.  Then C' = X Y = (A B)^T = C^T, so the accumulator holds
//     lane l, register i  =  C'[8 (l >> 4) + i][l % 16]  =  C[query l % 16][key 8 (l >> 4) + i]
// - one query a lane, 8 keys of it in registers - and the caller's lane 4 g + t needs queries {g, g + 8} at
// keys {2t, 2t + 1} (and {2t + 8, 2t + 9} for the second n8 block), which are 4 shuffles from lanes g and g + 8
// (16 + g, 24 + g) of registers 2t, 2t + 1: the lane keeps only the shuffles whose register is its own key.
// The dot products are the same 16 products either way round (A^T B^T = (B A)^T, and a product of FP16 values
// is exact in FP32), so the arithmetic is the CUDA path's up to the summation order.
//
// The accumulation is done in the caller's layout (+= below, the instruction run with a zero accumulator):
// carrying the accumulator across calls in the hardware's layout would need the inverse re-lay on entry, 4 more
// shuffles a call for the same sums.
//
// IK2_FA_WMMA: the flash-attention kernels pick their product at run time (the matrix core by default, the
// emulation below with 0), so the emulation is defined on every AMD arch - gfx12 included, where the rest of the
// engine's MMA stays on the core - and mma16816 / mma16816x2 are the core's on gfx12, the emulation's elsewhere.
__device__ __forceinline__ void mma16816_emu(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3;
    const int ra = (int) (g * 4), cb0 = (int) (2 * t) * 4, cb1 = cb0 + 4;
    float d0 = c[0], d1 = c[1], d2 = c[2], d3 = c[3];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint32_t ar0 = __shfl_sync(0xffffffffu, a[0], ra + u);   // A[g][2u], A[g][2u + 1]
        const uint32_t ar1 = __shfl_sync(0xffffffffu, a[1], ra + u);   // A[g + 8][2u], [2u + 1]
        const uint32_t ar2 = __shfl_sync(0xffffffffu, a[2], ra + u);   // A[g][2u + 8], [2u + 9]
        const uint32_t ar3 = __shfl_sync(0xffffffffu, a[3], ra + u);   // A[g + 8][2u + 8], [2u + 9]
        const uint32_t b00 = __shfl_sync(0xffffffffu, b0, cb0 + u);    // B[2u, 2u + 1][2t]
        const uint32_t b01 = __shfl_sync(0xffffffffu, b1, cb0 + u);    // B[2u + 8, 2u + 9][2t]
        const uint32_t b10 = __shfl_sync(0xffffffffu, b0, cb1 + u);    // ... the column 2t + 1
        const uint32_t b11 = __shfl_sync(0xffffffffu, b1, cb1 + u);
        d0 = fmaf(h2f((uint16_t) ar0), h2f((uint16_t) b00), d0);
        d0 = fmaf(h2f((uint16_t) (ar0 >> 16)), h2f((uint16_t) (b00 >> 16)), d0);
        d0 = fmaf(h2f((uint16_t) ar2), h2f((uint16_t) b01), d0);
        d0 = fmaf(h2f((uint16_t) (ar2 >> 16)), h2f((uint16_t) (b01 >> 16)), d0);
        d1 = fmaf(h2f((uint16_t) ar0), h2f((uint16_t) b10), d1);
        d1 = fmaf(h2f((uint16_t) (ar0 >> 16)), h2f((uint16_t) (b10 >> 16)), d1);
        d1 = fmaf(h2f((uint16_t) ar2), h2f((uint16_t) b11), d1);
        d1 = fmaf(h2f((uint16_t) (ar2 >> 16)), h2f((uint16_t) (b11 >> 16)), d1);
        d2 = fmaf(h2f((uint16_t) ar1), h2f((uint16_t) b00), d2);
        d2 = fmaf(h2f((uint16_t) (ar1 >> 16)), h2f((uint16_t) (b00 >> 16)), d2);
        d2 = fmaf(h2f((uint16_t) ar3), h2f((uint16_t) b01), d2);
        d2 = fmaf(h2f((uint16_t) (ar3 >> 16)), h2f((uint16_t) (b01 >> 16)), d2);
        d3 = fmaf(h2f((uint16_t) ar1), h2f((uint16_t) b10), d3);
        d3 = fmaf(h2f((uint16_t) (ar1 >> 16)), h2f((uint16_t) (b10 >> 16)), d3);
        d3 = fmaf(h2f((uint16_t) ar3), h2f((uint16_t) b11), d3);
        d3 = fmaf(h2f((uint16_t) (ar3 >> 16)), h2f((uint16_t) (b11 >> 16)), d3);
    }
    c[0] = d0, c[1] = d1, c[2] = d2, c[3] = d3;
}
__device__ __forceinline__ void mma16816x2_emu(float* c0, float* c1, const uint32_t* a, uint32_t b0, uint32_t b1,
                                               uint32_t b2, uint32_t b3) {
    mma16816_emu(c0, a, b0, b1);
    mma16816_emu(c1, a, b2, b3);
}
#if defined(__gfx1200__) || defined(__gfx1201__)
typedef _Float16 ik2_wmma_h8 __attribute__((ext_vector_type(8)));
typedef float ik2_wmma_f8 __attribute__((ext_vector_type(8)));
struct ik2_wmma_frag {   // (a 16 x 16 operand: 8 halves a lane)
    uint32_t v[4];
};
// A's m16n8k16 fragment -> the hardware's B operand of the transposed product: lane l holds
// A[query l % 16][k 8 (l >> 4) .. + 7], whose four dwords are a[0], a[1] (k 0..7) or a[2], a[3] (k 8..15) of the
// four lanes 4 (l % 8) + j (rows 8 and up live in a[1], a[3] of the lane's group).
__device__ __forceinline__ ik2_wmma_frag wmma_y_from_a(const uint32_t* a) {
    const unsigned lane = threadIdx.x & 31u, r = lane & 15u, h = lane >> 4, m = r & 7u, lo = r < 8 ? 1u : 0u;
    ik2_wmma_frag q;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int s = (int) (4 * m + (unsigned) j);
        const uint32_t v0 = __shfl_sync(0xffffffffu, a[0], s);   // A[r][2j, 2j + 1]
        const uint32_t v1 = __shfl_sync(0xffffffffu, a[1], s);   // A[r + 8][2j, 2j + 1]
        const uint32_t v2 = __shfl_sync(0xffffffffu, a[2], s);   // A[r][2j + 8, 2j + 9]
        const uint32_t v3 = __shfl_sync(0xffffffffu, a[3], s);
        q.v[j] = h ? (lo ? v2 : v3) : (lo ? v0 : v1);
    }
    return q;
}
// B's two m16n8k16 fragments (the n8 blocks at keys 0..7 and 8..15) -> the hardware's A operand: lane l holds
// B[k 8 (l >> 4) .. + 7][key l % 16], from b0, b1 (keys < 8) or b2, b3 (keys 8 and up) of the four lanes
// 4 (l % 8) + j - a shuffle's source is one lane for every reader, so the key's half is chosen per reader.
__device__ __forceinline__ ik2_wmma_frag wmma_x_from_b(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
    const unsigned lane = threadIdx.x & 31u, c = lane & 15u, h = lane >> 4, lo = c < 8 ? 1u : 0u;
    ik2_wmma_frag q;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int s = (int) (4 * (c & 7u) + (unsigned) j);
        const uint32_t v0 = __shfl_sync(0xffffffffu, b0, s);   // B[2j, 2j + 1][key]
        const uint32_t v1 = __shfl_sync(0xffffffffu, b1, s);   // B[2j + 8, 2j + 9][key]
        const uint32_t v2 = __shfl_sync(0xffffffffu, b2, s);   // ... [key + 8]
        const uint32_t v3 = __shfl_sync(0xffffffffu, b3, s);
        q.v[j] = h ? (lo ? v1 : v3) : (lo ? v0 : v2);
    }
    return q;
}
// D (the transposed product) -> the caller's n8 c fragments: c0 = {C[g][2t], C[g][2t+1], C[g+8][2t],
// C[g+8][2t+1]}, c1 the same at keys 2t + 8, 2t + 9 (lanes g + 16, g + 24).  A shuffle carries one register,
// so the loop is over the key (the register) and each lane keeps the two that are its own t's.
template <bool BOTH>
__device__ __forceinline__ void wmma_c_into(float* c0, float* c1, const ik2_wmma_f8& d, bool add) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    const int j0 = 2 * (int) t, s0 = (int) g, s1 = (int) g + 8, s2 = (int) g + 16, s3 = (int) g + 24;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v0 = __shfl_sync(0xffffffffu, d[j], s0), v1 = __shfl_sync(0xffffffffu, d[j], s1);
        if (j == j0) c0[0] = add ? c0[0] + v0 : v0, c0[2] = add ? c0[2] + v1 : v1;
        if (j == j0 + 1) c0[1] = add ? c0[1] + v0 : v0, c0[3] = add ? c0[3] + v1 : v1;
        if constexpr (BOTH) {
            const float v2 = __shfl_sync(0xffffffffu, d[j], s2), v3 = __shfl_sync(0xffffffffu, d[j], s3);
            if (j == j0) c1[0] = add ? c1[0] + v2 : v2, c1[2] = add ? c1[2] + v3 : v3;
            if (j == j0 + 1) c1[1] = add ? c1[1] + v2 : v2, c1[3] = add ? c1[3] + v3 : v3;
        }
    }
}
__device__ __forceinline__ ik2_wmma_f8 wmma_run(const ik2_wmma_frag& x, const ik2_wmma_frag& y,
                                                const ik2_wmma_f8& acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(ik2_wmma_h8, x),
                                                            __builtin_bit_cast(ik2_wmma_h8, y), acc);
}
__device__ __forceinline__ ik2_wmma_f8 wmma_run(const ik2_wmma_frag& x, const ik2_wmma_frag& y) {
    ik2_wmma_f8 z;
#pragma unroll
    for (int i = 0; i < 8; ++i) z[i] = 0.0f;
    return wmma_run(x, y, z);
}
// D[16 x 8] (+)= A[16 x 16] x B[16 x 8], the m16n8k16 of the CUDA branch, through the matrix core
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const ik2_wmma_f8 d = wmma_run(wmma_x_from_b(b0, b1, 0u, 0u), wmma_y_from_a(a));
    wmma_c_into<false>(c, c, d, true);
}
// the same for both n8 blocks of one A in one instruction: D[16 x 16] (+)= A x B (b2, b3: keys 8..15)
__device__ __forceinline__ void mma16816x2(float* c0, float* c1, const uint32_t* a, uint32_t b0, uint32_t b1,
                                           uint32_t b2, uint32_t b3) {
    wmma_c_into<true>(c0, c1, wmma_run(wmma_x_from_b(b0, b1, b2, b3), wmma_y_from_a(a)), true);
}
#else   // (not gfx12: the emulation, defined above so the gate's off arm is the same code on every arch)
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    mma16816_emu(c, a, b0, b1);
}
__device__ __forceinline__ void mma16816x2(float* c0, float* c1, const uint32_t* a, uint32_t b0, uint32_t b1,
                                           uint32_t b2, uint32_t b3) {
    mma16816x2_emu(c0, c1, a, b0, b1, b2, b3);
}
#endif   // (__gfx1200__ / __gfx1201__: the matrix core; anything else: the emulation)
// IK2_FA_WMMA's product for the flash-attention kernels: W = the arch's own (the gfx12 matrix core, or the CUDA
// mma.sync), else the emulated m16n8k16.  Both are the same sums (FP16 products exact in FP32), so the two arms
// differ in summation order only - the parity harness is what says so.
template <bool W>
__device__ __forceinline__ void fa_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    if constexpr (W) mma16816(c, a, b0, b1);
    else mma16816_emu(c, a, b0, b1);
}
template <bool W>
__device__ __forceinline__ void fa_mma2(float* c0, float* c1, const uint32_t* a, uint32_t b0, uint32_t b1,
                                        uint32_t b2, uint32_t b3) {
    if constexpr (W) mma16816x2(c0, c1, a, b0, b1, b2, b3);
    else mma16816x2_emu(c0, c1, a, b0, b1, b2, b3);
}
// ldmatrix's addressing, read as shared memory is: a lane's address is the start of one row of an 8 x 8 block
// (lanes 0-7: block 0's rows 0-7, 8-15: block 1's, ...), and the fragment a lane holds is block row lane >> 2,
// columns 2 (lane & 3) and 2 (lane & 3) + 1.  The row's address is the one its lane passed, gathered by a
// shuffle - no shared-memory descriptor, so no __cvta_generic_to_shared and no swizzle to undo.
__device__ __forceinline__ const uint16_t* ldsm_row(const void* p, int src) {
    const uint32_t lo = (uint32_t) (uintptr_t) p, hi = (uint32_t) ((uint64_t) (uintptr_t) p >> 32);
    const uint32_t l = __shfl_sync(0xffffffffu, lo, src), h = __shfl_sync(0xffffffffu, hi, src);
    return (const uint16_t*) (uintptr_t) (((uint64_t) h << 32) | l);
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned lane = threadIdx.x & 31u, row = lane >> 2, col = (lane & 3u) * 2u;
    r0 = *(const uint32_t*) (ldsm_row(p, (int) row) + col);
    r1 = *(const uint32_t*) (ldsm_row(p, 8 + (int) row) + col);
    r2 = *(const uint32_t*) (ldsm_row(p, 16 + (int) row) + col);
    r3 = *(const uint32_t*) (ldsm_row(p, 24 + (int) row) + col);
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned lane = threadIdx.x & 31u, row = lane >> 2, col = (lane & 3u) * 2u;
    // transposed: a lane holds block rows 2 (lane & 3) and 2 (lane & 3) + 1 at the column lane >> 2
    const auto frag = [&](int m) {
        const uint16_t* a0 = ldsm_row(p, m * 8 + (int) col) + row;
        const uint16_t* a1 = ldsm_row(p, m * 8 + (int) col + 1) + row;
        return (uint32_t) a0[0] | ((uint32_t) a1[0] << 16);
    };
    r0 = frag(0), r1 = frag(1), r2 = frag(2), r3 = frag(3);
}
// cp.async: the copy is made as the instruction is reached - no group is in flight and nothing is waited for;
// the kernel's own __syncthreads() (and __syncwarp(), for the bias) order it, and commit / wait stay as the
// compiler's ordering points (valid == false: the bytes zero-filled, as the instruction's 0 source size does).
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    if (valid) *(uint4*) dst = *(const uint4*) src;
    else *(uint4*) dst = make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
    if (valid) *(uint32_t*) dst = *(const uint32_t*) src;
    else *(uint32_t*) dst = 0u;
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    __threadfence_block();   // (the copies' shared-memory writes, ordered before what follows)
}

// ---- P3-4: the grouped expert GEMM on gfx12's matrix cores (v_mmac_f32_16x16x16_f16, wave32).  One warp owns
// a 16-row x 16-column tile - 16 rows of weights and the group's <= 8 token columns (columns 8..15 stay zero and
// their writes are guarded) - over its stride of the K blocks: one WMMA a 16-value k step in place of each
// emulated m16n8k16.  The gfx12 register layout (checked on gfx1201 against src/prefill/moe_fused_rdna4.cu's iu8
// kernels and by a basis probe): lane l holds A[l % 16][8 (l / 16) + j] and B[8 (l / 16) + j][l % 16], and
// D[8 (l / 16) + i][l % 16].  So a lane reads 8 consecutive k of ONE weight row (its A fragment) and 8 consecutive
// k of one activation row (its B column, one token) - no shuffles, no shared-memory staging of the operands.
// Arithmetic identical to the emulated path: an NVFP4 code times its sub-block scale is exact in FP16 and the
// products are exactly summed in FP32, so only the summation order inside the 16-value dot differs.
#if defined(__HIP_PLATFORM_AMD__)
typedef _Float16 ik2_mh8 __attribute__((ext_vector_type(8)));
typedef float ik2_mf8 __attribute__((ext_vector_type(8)));
union ik2_mfrag {
    uint32_t u[4];
    ik2_mh8 h;
};
__device__ __forceinline__ ik2_mf8 ik2_vmmac(ik2_mh8 a, ik2_mh8 b, ik2_mf8 c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    (void) a, (void) b;
    return c;   // (never launched: the host dispatch checks the device's arch first)
#endif
}
// One 64-value NVFP4 block: 4 bytes of 4 ue4m3 sub-block scales, then 32 bytes of codes.  Sub-block t16 (its
// scale byte t16) is the 16 values k0 + 16 t16 .. + 15, and its 8 bytes carry value i in byte i's low nibble and
// value i + 8 in that byte's high nibble.  This lane's 8 consecutive values (k0 + 16 t16 + 8 n .. + 7) are
// therefore the low nibbles (n = 0) or the high nibbles (n = 1) of that sub-block's 8 bytes, in order.
__device__ __forceinline__ void ik2_nv_a_frag(ik2_mfrag& A, uint2 q, int n, uint32_t scale) {
    const uint32_t w[4] = {q.x & 0xFFFFu, q.x >> 16, q.y & 0xFFFFu, q.y >> 16};
#pragma unroll
    for (int p = 0; p < 4; ++p) A.u[p] = ik2_fp4pair(n ? (w[p] >> 4) : w[p], scale);
}
template <bool Gate>
__global__ void __launch_bounds__(128)
grouped_expert_wmma_kernel(const uint8_t* const* __restrict__ blobs, const int* __restrict__ tok, size_t offset,
                           int C, int F, const uint16_t* __restrict__ h, uint16_t* hidden, float* out,
                           const float* sg, const float* su, float gcap, float ucap, const int* counts,
                           const int* rows_out, const uint16_t* input16) {
    __shared__ float part[4][16 * 16];
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5;
    const int n = lane >> 4, c = lane & 15, rr = lane & 15;
    const int s = blockIdx.y, count = counts[s];
    if (count <= 0) return;
    const int K = Gate ? C : F;
    const int r0 = Gate ? blockIdx.x * 8 : blockIdx.x * 16;
    const int stride = K / 64 * 36;
    // this lane's A row: rows 0..7 are the (gate | down) rows r0.. and 8..15 the up rows (gate) or down r0 + 8..
    const int arow = Gate ? r0 + (rr & 7) : r0 + rr;
    const uint8_t* ap = blobs[s] + (Gate ? (rr < 8 ? 0 : offset) : offset) + (size_t) arow * stride;
    // this lane's B column: the group's token c, zero past the group's count
    const bool live = c < count;
    const int xrow = live ? (Gate ? tok[s * kGroupMax + c] : rows_out[s * kGroupMax + c]) : 0;
    const uint16_t* bp = (Gate ? input16 : h) + (size_t) xrow * K;
    ik2_mfrag A, B;
    ik2_mf8 d;
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] = 0.0f;
    const int nk = K / 64;
    for (int k = split; k < nk; k += 4) {
        const uint32_t* sp = (const uint32_t*) ap + (size_t) k * 9;   // (36 bytes = 9 words)
        const uint32_t s0 = __ldg(sp);
#pragma unroll
        for (int t16 = 0; t16 < 4; ++t16) {
            const uint2 q = *(const uint2*) (ap + (size_t) k * 36 + 4 + 8 * t16);
            ik2_nv_a_frag(A, q, n, ue4m3_h2((s0 >> (8 * t16)) & 0xFFu));
            if (live) {
                const uint4 v = *(const uint4*) (bp + (size_t) k * 64 + 16 * t16 + 8 * n);
                B.u[0] = v.x, B.u[1] = v.y, B.u[2] = v.z, B.u[3] = v.w;
            } else {
                B.u[0] = B.u[1] = B.u[2] = B.u[3] = 0u;
            }
            d = ik2_vmmac(A.h, B.h, d);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) part[split][(8 * n + i) * 16 + c] = d[i];
    __syncthreads();
    float* tot = part[0];
#pragma unroll
    for (int i = 0; i < 2; ++i) {   // (128 threads, 256 values)
        const int j = threadIdx.x + 128 * i;
#pragma unroll
        for (int w = 1; w < 4; ++w) part[0][j] += part[w][j];
    }
    __syncthreads();
    if constexpr (Gate) {
        const int r = threadIdx.x >> 4;   // (8 feature rows x 16 token columns)
        const int col = threadIdx.x & 15;
        if (r < 8 && col < count) {
            const int dst = rows_out[s * kGroupMax + col];
            const float g = tot[r * 16 + col] * (sg ? sg[dst] : 1.f);
            const float u = tot[(r + 8) * 16 + col] * (su ? su[dst] : 1.f);
            hidden[(size_t) dst * F + r0 + r] =
                f2h(gcap * tanhf(g / gcap) * (1.f / (1.f + expf(-g))) * ucap * tanhf(u / ucap));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 2; ++i) {   // (128 threads, 16 rows x 16 columns)
            const int j = threadIdx.x + 128 * i;
            const int r = j >> 4, col = j & 15;
            if (col < count) out[(size_t) rows_out[s * kGroupMax + col] * C + r0 + r] = tot[j];
        }
    }
}
template <bool Gate>
void launch_grouped_expert_wmma(const uint8_t* const* blob, const int* tok, int G, size_t off, int C, int F,
                                const uint16_t* h, uint16_t* hidden, float* y, const float* sg, const float* su,
                                float gcap, float ucap, const int* n, const int* row, const uint16_t* input16,
                                cudaStream_t s) {
    const dim3 g((unsigned) (Gate ? F / 8 : C / 16), (unsigned) G);
    grouped_expert_wmma_kernel<Gate><<<g, 128, 0, s>>>(blob, tok, off, C, F, h, hidden, y, sg, su, gcap, ucap, n,
                                                       row, input16);
}
// The one-token expert tile on the same instruction: a warp is 16 rows (gate: the 8 gate rows and the same 8 of up,
// so a row's SwiGLU is both its halves; down: 16 rows) over its stride of the K blocks.  The two columns the dot
// needs are the input's FP16 limbs for gate (hi, then lo: a row's two columns add up to the product with the FP32
// input) and the hidden row itself for down; the other 14 are zero and their outputs unused.
template <bool Gate, int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_wmma_kernel(const uint8_t* const* __restrict__ blob, size_t off, int C, int F,
                        const uint16_t* __restrict__ x16, int x_ld, uint16_t* hidden, float* y, const float* sg,
                        const float* su, float gcap, float ucap) {
    __shared__ float part[SPLIT][16 * 16];
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5;
    const int n = lane >> 4, c = lane & 15, rr = lane & 15, slot = blockIdx.y;
    const int K = Gate ? C : F;
    const int r0 = blockIdx.x * (Gate ? 8 : 16);
    const int stride = K / 64 * 36;
    const int arow = Gate ? r0 + (rr & 7) : r0 + rr;
    const uint8_t* ap = blob[slot] + (Gate ? (rr < 8 ? 0 : off) : off) + (size_t) arow * stride;
    const bool live = Gate ? c < 2 : c == 0;
    const uint16_t* bp = Gate ? x16 + (c == 1 ? (size_t) C : 0) : x16 + (size_t) slot * x_ld;
    ik2_mfrag A, B;
    ik2_mf8 d;
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] = 0.0f;
    const int nk = K / 64;
    for (int k = split; k < nk; k += SPLIT) {
        const uint32_t* sp = (const uint32_t*) ap + (size_t) k * 9;
        const uint32_t s0 = __ldg(sp);
#pragma unroll
        for (int t16 = 0; t16 < 4; ++t16) {
            const uint2 q = *(const uint2*) (ap + (size_t) k * 36 + 4 + 8 * t16);
            ik2_nv_a_frag(A, q, n, ue4m3_h2((s0 >> (8 * t16)) & 0xFFu));
            if (live) {
                const uint4 v = *(const uint4*) (bp + (size_t) k * 64 + 16 * t16 + 8 * n);
                B.u[0] = v.x, B.u[1] = v.y, B.u[2] = v.z, B.u[3] = v.w;
            } else {
                B.u[0] = B.u[1] = B.u[2] = B.u[3] = 0u;
            }
            d = ik2_vmmac(A.h, B.h, d);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) part[split][(8 * n + i) * 16 + c] = d[i];
    __syncthreads();
    const int rows = Gate ? 8 : 16;
    if (threadIdx.x < rows) {
        const int r = threadIdx.x;
        float tot = 0.0f, totu = 0.0f;
#pragma unroll
        for (int w = 0; w < SPLIT; ++w) {
            if constexpr (Gate) {
                tot += part[w][r * 16] + part[w][r * 16 + 1];            // (hi + lo limbs)
                totu += part[w][(r + 8) * 16] + part[w][(r + 8) * 16 + 1];
            } else {
                tot += part[w][r * 16];
            }
        }
        if constexpr (Gate) {
            const float g = tot * (sg ? sg[slot] : 1.0f), u = totu * (su ? su[slot] : 1.0f);
            hidden[(size_t) slot * F + r0 + r] =
                f2h(gcap * tanhf(g / gcap) * (1.f / (1.f + expf(-g))) * ucap * tanhf(u / ucap));
        } else {
            y[(size_t) slot * C + r0 + r] = tot;
        }
    }
}
template <bool Gate>
void launch_experts_mma_wmma(const uint8_t* const* blob, int S, size_t off, const uint16_t* x16, int x_ld,
                                    int C, int F, const float* sg, const float* su, float gcap, float ucap,
                                    uint16_t* h, float* y, cudaStream_t s) {
    const dim3 g((unsigned) (Gate ? F / 8 : C / 16), (unsigned) S);
    if (Gate)
        experts_mma_wmma_kernel<true, 8><<<g, 8 * 32, 0, s>>>(blob, off, C, F, x16, x_ld, h, y, sg, su, gcap, ucap);
    else
        experts_mma_wmma_kernel<false, 4><<<g, 4 * 32, 0, s>>>(blob, off, C, F, x16, x_ld, h, y, sg, su, gcap, ucap);
}
// (a grouped expert tile runs on the matrix cores only on gfx12 / RDNA4: anywhere else the emulated path stays)
inline bool ik2_expert_wmma_ok() {
    static const bool ok = [] {
        if (getenv("IK2_EXPERT_WMMA") && atoi(getenv("IK2_EXPERT_WMMA")) == 0) return false;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return std::strncmp(p.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(p.gcnArchName, "gfx1201", 7) == 0;
    }();
    return ok;
}
#endif   // __HIP_PLATFORM_AMD__

#endif   // STRATA_USE_HIP
