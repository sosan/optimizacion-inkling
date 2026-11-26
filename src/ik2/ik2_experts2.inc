// ik2_experts2.inc - batch-1 routed-expert GEMVs for NVFP4 experts, behind
// IK2_FUSE_MASK bits 0x1000 (gate_up, set 13) and 0x2000 (down, set 14).  Included at the end of ik2_kernels.cu (it
// uses that file's helpers); nothing here is on the default path.
// Why (measured, serialized): a plain read at the experts' launch shape (1152 blocks x 128) reaches
// 238-247 GB/s, so the shape is not the limit.  experts_gate_up_swiglu_kernel copies x (10.4 KB padded) into LDS a
// block before its 20.7 KB of weights (that prologue alone: plain read 96.9 -> 111.0 us), and both kernels walk a
// runtime-K loop that asks for one sub-block's 3 dwords a row, then waits.  Here K is a template constant, a lane keeps
// the x values of its own sub-blocks in registers (row_dot's lane -> sub-block map: sb = lane, lane + 32, ...; no LDS,
// no barrier) and asks for every load of its rows before the first sum.  Same sums in the same order: the same values
// to the bit (tests/ik2_experts_test.cu).

namespace {

template <int K> struct NvRow {
    static constexpr int NSUB = K / 16;              // 16-value sub-blocks a row
    static constexpr int NMAX = (NSUB + 31) / 32;   // a lane's most (lanes < NSUB % 32 take one more)
    static constexpr size_t RB = (size_t) K / 64 * 36;
};

template <bool NT> __device__ __forceinline__ uint32_t ld32(const uint8_t* p) {
    if constexpr (NT) return __builtin_nontemporal_load((const uint32_t*) p);
    else return __ldg((const uint32_t*) p);
}

// a lane's x of its sub-blocks (F32), in registers
template <int K> __device__ __forceinline__ void nv_x_f32(const float* x, float4 (&xv)[NvRow<K>::NMAX][4]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < NvRow<K>::NMAX; ++i) {
        const int sb = lane + 32 * i;
        if (sb < NvRow<K>::NSUB) {
            const float4* p = (const float4*) (x + sb * 16);
            xv[i][0] = p[0], xv[i][1] = p[1], xv[i][2] = p[2], xv[i][3] = p[3];
        } else {
            xv[i][0] = xv[i][1] = xv[i][2] = xv[i][3] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}
// the same from FP16 (h): load_xs_f16's values (an FP16 -> F32 conversion is exact)
template <int K> __device__ __forceinline__ void nv_x_f16(const uint16_t* h, float4 (&xv)[NvRow<K>::NMAX][4]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < NvRow<K>::NMAX; ++i) {
        const int sb = lane + 32 * i;
        if (sb < NvRow<K>::NSUB) {
            const uint4* p = (const uint4*) (h + sb * 16);
            const uint4 a = p[0], b = p[1];
            const float2 f0 = u32_to_f2(a.x), f1 = u32_to_f2(a.y), f2 = u32_to_f2(a.z), f3 = u32_to_f2(a.w);
            const float2 f4 = u32_to_f2(b.x), f5 = u32_to_f2(b.y), f6 = u32_to_f2(b.z), f7 = u32_to_f2(b.w);
            xv[i][0] = make_float4(f0.x, f0.y, f1.x, f1.y), xv[i][1] = make_float4(f2.x, f2.y, f3.x, f3.y);
            xv[i][2] = make_float4(f4.x, f4.y, f5.x, f5.y), xv[i][3] = make_float4(f6.x, f6.y, f7.x, f7.y);
        } else {
            xv[i][0] = xv[i][1] = xv[i][2] = xv[i][3] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

// a lane's 3 dwords a sub-block of one row (scale word, two nibble words), every one asked for before any is used
template <int K, bool NT> struct NvLoads {
    uint32_t sw[NvRow<K>::NMAX], q0[NvRow<K>::NMAX], q1[NvRow<K>::NMAX];
    __device__ __forceinline__ void load(const uint8_t* w) {
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int i = 0; i < NvRow<K>::NMAX; ++i) {
            const int sb = lane + 32 * i;
            if (sb < NvRow<K>::NSUB) {
                const uint8_t* b = w + (size_t) (sb >> 2) * 36;
                const int sub = sb & 3;
                sw[i] = ld32<NT>(b), q0[i] = ld32<NT>(b + 4 + sub * 8), q1[i] = ld32<NT>(b + 8 + sub * 8);
            }
        }
    }
    // row_dot's lane sum (kNVFP4 branch): acc += d * nvfp4_sub_dot(..) over sb = lane, lane + 32, .. in that order
    __device__ __forceinline__ float lane_sum(const float4 (&xv)[NvRow<K>::NMAX][4]) const {
        const int lane = threadIdx.x & 31;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < NvRow<K>::NMAX; ++i) {
            const int sb = lane + 32 * i;
            if (sb < NvRow<K>::NSUB) {
                const float d = ue4m3((uint8_t) (sw[i] >> (8 * (sb & 3))));
                acc += d * nvfp4_sub_dot(q0[i], q1[i], xv[i][0], xv[i][1], xv[i][2], xv[i][3]);
            }
        }
        return acc;
    }
};

// experts_gate_up_swiglu_kernel for NVFP4 gate and up, K = C: a warp RPW rows (gate's and up's), block WARPS warps
template <int K, int WARPS, int RPW, bool NT>
__device__ __forceinline__ void gu_nv_tile(int bx, int s, const uint8_t* const* blob, const int* tok, size_t eb0,
                                           const float* x, int F, const float* sg, const float* su, float gcap,
                                           float ucap, uint16_t* h) {
    const int warp = threadIdx.x >> 5;
    float4 xv[NvRow<K>::NMAX][4];
    nv_x_f32<K>(x + (size_t) tok[s] * K, xv);
    const uint8_t* bl = blob[s];
    const float sgs = sg ? sg[s] : 1.0f;
    const float sus = su ? su[s] : 1.0f;
#pragma unroll 1
    for (int j = 0; j < RPW; ++j) {
        const int r = (bx * WARPS + warp) * RPW + j;
        if (r >= F) break;
        NvLoads<K, NT> g, u;
        g.load(bl + (size_t) r * NvRow<K>::RB);
        u.load(bl + eb0 + (size_t) r * NvRow<K>::RB);
        float vg = g.lane_sum(xv), vu = u.lane_sum(xv);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            vg += __shfl_xor_sync(0xffffffffu, vg, o);
            vu += __shfl_xor_sync(0xffffffffu, vu, o);
        }
        if ((threadIdx.x & 31) == 0) {
            const float gg = vg * sgs, uu = vu * sus;
            h[(size_t) s * F + r] = f2h(gcap * tanhf(gg / gcap) * (1.0f / (1.0f + expf(-gg))) * ucap * tanhf(uu / ucap));
        }
    }
}
template <int K, int WARPS, int RPW, bool NT>
__global__ void __launch_bounds__(WARPS * 32) experts_gu_nv_kernel(const uint8_t* const* blob, const int* tok,
                                                                  size_t eb0, const float* x, int F, const float* sg,
                                                                  const float* su, float gcap, float ucap, uint16_t* h) {
    gu_nv_tile<K, WARPS, RPW, NT>(blockIdx.x, blockIdx.y, blob, tok, eb0, x, F, sg, su, gcap, ucap, h);
}

// experts_down_kernel for NVFP4, K = F: a warp RPW rows (two at a time, as row_dot_2), block WARPS warps
template <int K, int WARPS, int RPW, bool NT>
__device__ __forceinline__ void dn_nv_tile(int bx, int s, const uint8_t* const* blob, size_t off_down,
                                           const uint16_t* h, int C, float* y) {
    const int warp = threadIdx.x >> 5;
    float4 xv[NvRow<K>::NMAX][4];
    nv_x_f16<K>(h + (size_t) s * K, xv);
    const uint8_t* w = blob[s] + off_down;
#pragma unroll 1
    for (int j = 0; j < RPW; j += 2) {
        const int row = (bx * WARPS + warp) * RPW + j;
        if (row >= C) break;
        const bool two = row + 1 < C;
        NvLoads<K, NT> a, b;
        a.load(w + (size_t) row * NvRow<K>::RB);
        if (two) b.load(w + (size_t) (row + 1) * NvRow<K>::RB);
        float v0 = a.lane_sum(xv), v1 = two ? b.lane_sum(xv) : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v0 += __shfl_xor_sync(0xffffffffu, v0, o);
            v1 += __shfl_xor_sync(0xffffffffu, v1, o);
        }
        if ((threadIdx.x & 31) == 0) {
            y[(size_t) s * C + row] = v0;
            if (two) y[(size_t) s * C + row + 1] = v1;
        }
    }
}
template <int K, int WARPS, int RPW, bool NT>
__global__ void __launch_bounds__(WARPS * 32) experts_dn_nv_kernel(const uint8_t* const* blob, size_t off_down,
                                                                  const uint16_t* h, int C, float* y) {
    dn_nv_tile<K, WARPS, RPW, NT>(blockIdx.x, blockIdx.y, blob, off_down, h, C, y);
}

// ---- set 16 (bit 0x8000): NVFP4 gate_up and down of a visit in ONE launch.  Blocks [0, S * NGU) are gate_up tiles
// (expert-major), then S * NDN down tiles; the hardware hands out blocks in id order, so when a down block of expert s
// runs, every gate_up block of s is running or done.  Each gate_up block raises g_nv_cnt[0][s] (release) after its h
// rows are stored; a down block's thread 0 waits (acquire) until it reaches NGU, then every wave takes an acquire
// fence before it reads h.  The expert's last down block (g_nv_cnt[1][s] reaches NDN: every down block of s is past
// its wait) sets both counters back to 0 - nothing kept on the host, so a graph replay works.  A wait that never ends
// (a bug) gives up after ~2^22 sleeps and leaves NaN rows instead of hanging the GPU.
__device__ int g_nv_cnt[2][8];
template <int KG, int KD>
__global__ void __launch_bounds__(128) experts_nv_visit_kernel(const uint8_t* const* blob, const int* tok, size_t eb0,
                                                               size_t off_down, const float* x, int C, int F,
                                                               const float* sg, const float* su, float gcap,
                                                               float ucap, uint16_t* h, float* y, int S) {
    const int ngu = (F + 7) / 8, ndn = (C + 15) / 16;
    const int id = blockIdx.x;
    if (id < S * ngu) {
        const int s = id / ngu;
        gu_nv_tile<KG, 4, 2, false>(id - s * ngu, s, blob, tok, eb0, x, F, sg, su, gcap, ucap, h);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // this wave's h stores reach L2 first
        __syncthreads();
        if (threadIdx.x == 0) __hip_atomic_fetch_add(&g_nv_cnt[0][s], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        return;
    }
    const int id2 = id - S * ngu, s = id2 / ndn;
    __shared__ int ok;
    if (threadIdx.x == 0) {
        int spins = 0;
        while (__hip_atomic_load(&g_nv_cnt[0][s], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < ngu &&
               ++spins <= (1 << 22))
            __builtin_amdgcn_s_sleep(2);
        ok = spins <= (1 << 22);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // every wave: h from L2, never a stale L0 / GL1 line
    if (ok) dn_nv_tile<KD, 4, 4, false>(id2 - s * ndn, s, blob, off_down, h, C, y);
    else if ((threadIdx.x & 31) == 0) y[(size_t) s * C + (id2 - s * ndn) * 16 + (threadIdx.x >> 5) * 4] = __int_as_float(0x7fc00000);
    __syncthreads();
    if (threadIdx.x == 0 &&
        __hip_atomic_fetch_add(&g_nv_cnt[1][s], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == ndn - 1) {
        __hip_atomic_store(&g_nv_cnt[0][s], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&g_nv_cnt[1][s], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// ---- Q8_0 experts (set 15, bit 0x4000): row_dot's 4-aligned Q8_0 path (lane l: block pairs p = l, l + 32, ..;
// q8_blk0_dot + q8_blk1_dot a pair) with the pair's 17 dwords loaded first and x in registers
template <int K> struct Q8Row {
    static constexpr int NB = K / 32;               // 34-byte blocks a row (even here)
    static constexpr int NP = NB / 2;               // block pairs a row
    static constexpr int NMAX = (NP + 31) / 32;
    static constexpr size_t RB = (size_t) NB * 34;
    static_assert(NB % 2 == 0, "row_dot's odd tail block is not here");
};
// q8_blk0_dot / q8_blk1_dot on dwords already loaded (the same expressions, so the same roundings)
__device__ __forceinline__ float q8_b0(const uint32_t (&u)[17], const float4 (&f)[16], float& d1_out) {
    const float4 f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4], f5 = f[5], f6 = f[6], f7 = f[7];
    const float d0 = half_bits_to_float((uint16_t) (u[0] & 0xFFFFu));
    float a0 = (float) (int8_t) ((u[0] >> 16) & 0xFFu) * f0.x + (float) (int8_t) (u[0] >> 24) * f0.y;
    a0 += dot_u32_f4(u[1], make_float4(f0.z, f0.w, f1.x, f1.y));
    a0 += dot_u32_f4(u[2], make_float4(f1.z, f1.w, f2.x, f2.y));
    a0 += dot_u32_f4(u[3], make_float4(f2.z, f2.w, f3.x, f3.y));
    a0 += dot_u32_f4(u[4], make_float4(f3.z, f3.w, f4.x, f4.y));
    a0 += dot_u32_f4(u[5], make_float4(f4.z, f4.w, f5.x, f5.y));
    a0 += dot_u32_f4(u[6], make_float4(f5.z, f5.w, f6.x, f6.y));
    a0 += dot_u32_f4(u[7], make_float4(f6.z, f6.w, f7.x, f7.y));
    a0 += (float) (int8_t) (u[8] & 0xFFu) * f7.z + (float) (int8_t) ((u[8] >> 8) & 0xFFu) * f7.w;
    d1_out = half_bits_to_float((uint16_t) (u[8] >> 16));
    return d0 * a0;
}
__device__ __forceinline__ float q8_b1(const uint32_t (&u)[17], float d1, const float4 (&f)[16]) {
    return d1 * (dot_u32_f4(u[9], f[8]) + dot_u32_f4(u[10], f[9]) + dot_u32_f4(u[11], f[10]) +
                 dot_u32_f4(u[12], f[11]) + dot_u32_f4(u[13], f[12]) + dot_u32_f4(u[14], f[13]) +
                 dot_u32_f4(u[15], f[14]) + dot_u32_f4(u[16], f[15]));
}
template <int K> __device__ __forceinline__ void q8_x_f32(const float* x, float4 (&xv)[Q8Row<K>::NMAX][16]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < Q8Row<K>::NMAX; ++i) {
        const int p = lane + 32 * i;
#pragma unroll
        for (int t = 0; t < 16; ++t)
            xv[i][t] = p < Q8Row<K>::NP ? ((const float4*) (x + p * 64))[t] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
}
template <int K> __device__ __forceinline__ void q8_x_f16(const uint16_t* h, float4 (&xv)[Q8Row<K>::NMAX][16]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < Q8Row<K>::NMAX; ++i) {
        const int p = lane + 32 * i;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (p < Q8Row<K>::NP) {
                const uint4 a = ((const uint4*) (h + p * 64))[t];
                const float2 f0 = u32_to_f2(a.x), f1 = u32_to_f2(a.y), f2 = u32_to_f2(a.z), f3 = u32_to_f2(a.w);
                xv[i][2 * t] = make_float4(f0.x, f0.y, f1.x, f1.y), xv[i][2 * t + 1] = make_float4(f2.x, f2.y, f3.x, f3.y);
            } else {
                xv[i][2 * t] = xv[i][2 * t + 1] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }
}
template <int K> struct Q8Loads {
    uint32_t u[Q8Row<K>::NMAX][17];
    __device__ __forceinline__ void load(const uint8_t* w) {
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int i = 0; i < Q8Row<K>::NMAX; ++i) {
            const int p = lane + 32 * i;
            if (p < Q8Row<K>::NP) {
                const uint32_t* q = (const uint32_t*) (w + (size_t) p * 68);
#pragma unroll
                for (int t = 0; t < 17; ++t) u[i][t] = __ldg(q + t);
            }
        }
    }
    __device__ __forceinline__ float lane_sum(const float4 (&xv)[Q8Row<K>::NMAX][16]) const {
        const int lane = threadIdx.x & 31;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < Q8Row<K>::NMAX; ++i) {
            if (lane + 32 * i < Q8Row<K>::NP) {
                float d1;
                acc += q8_b0(u[i], xv[i], d1);
                acc += q8_b1(u[i], d1, xv[i]);
            }
        }
        return acc;
    }
};
template <int K, int WARPS, int RPW>
__global__ void __launch_bounds__(WARPS * 32) experts_gu_q8_kernel(const uint8_t* const* blob, const int* tok,
                                                                  size_t eb0, const float* x, int F, const float* sg,
                                                                  const float* su, float gcap, float ucap, uint16_t* h) {
    const int s = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const uint8_t* bl = blob[s];
    if (((uintptr_t) bl & 3) != 0) __builtin_trap();   // (row_dot's byte path would sum in another order)
    float4 xv[Q8Row<K>::NMAX][16];
    q8_x_f32<K>(x + (size_t) tok[s] * K, xv);
    const float sgs = sg ? sg[s] : 1.0f;
    const float sus = su ? su[s] : 1.0f;
#pragma unroll 1
    for (int j = 0; j < RPW; ++j) {
        const int r = (blockIdx.x * WARPS + warp) * RPW + j;
        if (r >= F) break;
        Q8Loads<K> g, u;
        g.load(bl + (size_t) r * Q8Row<K>::RB);
        u.load(bl + eb0 + (size_t) r * Q8Row<K>::RB);
        float vg = g.lane_sum(xv), vu = u.lane_sum(xv);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            vg += __shfl_xor_sync(0xffffffffu, vg, o);
            vu += __shfl_xor_sync(0xffffffffu, vu, o);
        }
        if ((threadIdx.x & 31) == 0) {
            const float gg = vg * sgs, uu = vu * sus;
            h[(size_t) s * F + r] = f2h(gcap * tanhf(gg / gcap) * (1.0f / (1.0f + expf(-gg))) * ucap * tanhf(uu / ucap));
        }
    }
}
template <int K, int WARPS, int RPW>
__global__ void __launch_bounds__(WARPS * 32) experts_dn_q8_kernel(const uint8_t* const* blob, size_t off_down,
                                                                  const uint16_t* h, int C, float* y) {
    const int s = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const uint8_t* w = blob[s] + off_down;
    if (((uintptr_t) w & 3) != 0) __builtin_trap();
    float4 xv[Q8Row<K>::NMAX][16];
    q8_x_f16<K>(h + (size_t) s * K, xv);
#pragma unroll 1
    for (int j = 0; j < RPW; j += 2) {
        const int row = (blockIdx.x * WARPS + warp) * RPW + j;
        if (row >= C) break;
        const bool two = row + 1 < C;
        Q8Loads<K> a, b;
        a.load(w + (size_t) row * Q8Row<K>::RB);
        if (two) b.load(w + (size_t) (row + 1) * Q8Row<K>::RB);
        float v0 = a.lane_sum(xv), v1 = two ? b.lane_sum(xv) : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v0 += __shfl_xor_sync(0xffffffffu, v0, o);
            v1 += __shfl_xor_sync(0xffffffffu, v1, o);
        }
        if ((threadIdx.x & 31) == 0) {
            y[(size_t) s * C + row] = v0;
            if (two) y[(size_t) s * C + row + 1] = v1;
        }
    }
}

}  // namespace

bool experts_nv_fits(int ty_gate, int ty_up, int ty_down, int C, int F) {
    return ty_gate == kNVFP4 && ty_up == kNVFP4 && ty_down == kNVFP4 && C == 2304 && F == 1152;
}
// the launch variant (tests/ik2_experts_test.cu sets these; -1: IK2_EXP_GU / IK2_EXP_DN, else 0 = the default)
int g_exp_gu = -1, g_exp_dn = -1;
static int exp_variant(int& g, const char* e) {
    if (g < 0) g = getenv(e) ? atoi(getenv(e)) : 0;
    return g;
}

void experts_gate_up_swiglu_nv(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C,
                               int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                               cudaStream_t s) {
    if (S <= 0) return;
    const int v = exp_variant(g_exp_gu, "IK2_EXP_GU");
    switch (v) {
    case 1: experts_gu_nv_kernel<2304, 4, 4, false><<<dim3((F + 15) / 16, S), 128, 0, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h); break;
    case 2: experts_gu_nv_kernel<2304, 4, 2, true><<<dim3((F + 7) / 8, S), 128, 0, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h); break;
    case 3: experts_gu_nv_kernel<2304, 2, 4, false><<<dim3((F + 7) / 8, S), 64, 0, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h); break;
    case 4: experts_gu_nv_kernel<2304, 4, 8, false><<<dim3((F + 31) / 32, S), 128, 0, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h); break;
    default: experts_gu_nv_kernel<2304, 4, 2, false><<<dim3((F + 7) / 8, S), 128, 0, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h);
    }
    (void) C;
    check("experts_gate_up_swiglu_nv");
}
void experts_down_nv(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                     cudaStream_t s) {
    if (S <= 0) return;
    const int v = exp_variant(g_exp_dn, "IK2_EXP_DN");
    switch (v) {
    case 1: experts_dn_nv_kernel<1152, 4, 8, false><<<dim3((C + 31) / 32, S), 128, 0, s>>>(blob, off_down, h, C, y); break;
    case 2: experts_dn_nv_kernel<1152, 4, 4, true><<<dim3((C + 15) / 16, S), 128, 0, s>>>(blob, off_down, h, C, y); break;
    case 3: experts_dn_nv_kernel<1152, 4, 2, false><<<dim3((C + 7) / 8, S), 128, 0, s>>>(blob, off_down, h, C, y); break;
    case 4: experts_dn_nv_kernel<1152, 2, 4, false><<<dim3((C + 7) / 8, S), 64, 0, s>>>(blob, off_down, h, C, y); break;
    default: experts_dn_nv_kernel<1152, 4, 4, false><<<dim3((C + 15) / 16, S), 128, 0, s>>>(blob, off_down, h, C, y);
    }
    (void) F;
    check("experts_down_nv");
}

bool experts_q8_fits(int ty, int C, int F) { return ty == kQ8_0 && C == 2304 && F == 1152; }
void experts_gate_up_swiglu_q8(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int C,
                               int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                               cudaStream_t s) {
    if (S <= 0) return;
    experts_gu_q8_kernel<2304, 4, 2><<<dim3((F + 7) / 8, S), 128, 0, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h);
    (void) C;
    check("experts_gate_up_swiglu_q8");
}
void experts_down_q8(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                     cudaStream_t s) {
    if (S <= 0) return;
    experts_dn_q8_kernel<1152, 4, 4><<<dim3((C + 15) / 16, S), 128, 0, s>>>(blob, off_down, h, C, y);
    (void) F;
    check("experts_down_q8");
}

// set 16: the visit's NVFP4 gate_up + down in one launch (experts_nv_fits; S <= 8; graph-safe: device counters)
void experts_nv_visit(const uint8_t* const* blob, const int* tok, int S, size_t eb0, size_t off_down, const float* x,
                      int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                      cudaStream_t s) {
    if (S <= 0) return;
    const int ngu = (F + 7) / 8, ndn = (C + 15) / 16;
    experts_nv_visit_kernel<2304, 1152><<<S * (ngu + ndn), 128, 0, s>>>(blob, tok, eb0, off_down, x, C, F, sg, su, gcap,
                                                                        ucap, h, y, S);
    check("experts_nv_visit");
}
// the visit kernel's counters summed (0 between launches; tests/ik2_experts_test.cu)
int experts_nv_visit_pending() {
    int c[16] = {};
    if (hipMemcpyFromSymbol(c, HIP_SYMBOL(g_nv_cnt), sizeof c) != cudaSuccess) return -1;
    int t = 0;
    for (int v : c) t += v;
    return t;
}
