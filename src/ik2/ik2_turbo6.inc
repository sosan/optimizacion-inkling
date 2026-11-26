// src/ik2/ik2_turbo6.inc - IK2_KV=turbo6 on the device (the format and its CPU reference: ik2_turbo6.h, tubo's codec):
// the KV write (prep_qkv_r_impl_kernel<float> to an F32 staging row, then kv_t6_encode_kernel, the CPU reference's bytes),
// the token path's reader (q rotated once - q.k = (R q).(R k) - the codes decoded in the loop, V summed rotated, the
// combine rotating the head's output back) and the prompt path's unpack to FP16 for its GEMMs.  Included at the end of
// ik2_kernels.cu, inside ik2::k.

// One wave a 128-value block (KV head g of token t; blockIdx.z: 0 k, 1 v): QuantizeBlock's steps in its order - the two
// L2 sums j = 0..127 by one lane, no fused multiply-add, correctly rounded sqrt and divisions - so the same 98 bytes.
// ponytail: two serial 128-term sums a block (~0.5 us); a tree in this order's place only if the write shows in a profile
__global__ void kv_t6_encode_kernel(const float* __restrict__ k32, const float* __restrict__ v32, uint8_t* kc,
                                    uint8_t* vc, int KVh, int kv_cap, int pos0, const int* pos_d) {
    __shared__ float s[128];
    __shared__ float sh_gn, sh_inv;
    const int g = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const float* src = (blockIdx.z ? v32 : k32) + ((size_t) t * KVh + g) * 128;
    const int p = (pos_d ? *pos_d : pos0) + t, slot = kv_cap > 0 ? p % kv_cap : p;
    uint8_t* dst = (blockIdx.z ? vc : kc) + ((size_t) slot * KVh + g) * t6::kBlockBytes;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = src[4 * lane + i], s[4 * lane + i] = x[i];
    __syncthreads();
    if (lane == 0) {
        float sum = 0.0f;
        for (int j = 0; j < 128; ++j) sum = __fadd_rn(sum, __fmul_rn(s[j], s[j]));
        const float gn = __fsqrt_rn(sum);
        sh_gn = gn, sh_inv = gn > t6::kNormEpsilon ? __fdiv_rn(1.0f, gn) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) s[4 * lane + i] = __fmul_rn(x[i], sh_inv) * t6::TURBO_WHT_SIGNS1[4 * lane + i];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {   // FastWalshHadamard128's butterflies: 64 a stage, two a lane
#pragma unroll
        for (int b = lane; b < 64; b += 32) {
            const int j = (b / h) * 2 * h + (b % h);
            const float a = s[j], c = s[j + h];
            s[j] = a + c, s[j + h] = a - c;
        }
        __syncthreads();
    }
    uint32_t code[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int j = 4 * lane + i;
        code[i] = t6::NearestCode(__fmul_rn(s[j], t6::kInvSqrt128) * t6::TURBO_WHT_SIGNS2[j]);
    }
    dst[2 + 2 * lane] = (uint8_t) ((code[0] & 0xF) | ((code[1] & 0xF) << 4));
    dst[3 + 2 * lane] = (uint8_t) ((code[2] & 0xF) | ((code[3] & 0xF) << 4));
    dst[t6::kQhOffset + lane] = (uint8_t) (((code[0] >> 4) & 3) | (((code[1] >> 4) & 3) << 2) |
                                           (((code[2] >> 4) & 3) << 4) | (((code[3] >> 4) & 3) << 6));
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) s[4 * lane + i] = t6::TURBO_CENTROIDS_6BIT[code[i]];
    __syncthreads();
    if (lane == 0) {
        float acc = 0.0f;
        for (int j = 0; j < 128; ++j) acc = __fadd_rn(acc, __fmul_rn(s[j], s[j]));
        const float recon = __fsqrt_rn(acc), gn = sh_gn;
        const uint16_t half = t6::FloatToHalfBits(recon > t6::kNormEpsilon ? __fdiv_rn(gn, recon) : gn);
        dst[0] = (uint8_t) half, dst[1] = (uint8_t) (half >> 8);
    }
}

// 128 threads: x (in LDS) -> the normalized Walsh-Hadamard transform's butterflies (the caller scales by 1/sqrt(128))
__device__ __forceinline__ void t6_fwht128(float* x) {
    const int tid = threadIdx.x;
    for (int h = 1; h < 128; h <<= 1) {
        float a = 0.0f, c = 0.0f;
        int j = 0;
        if (tid < 64) j = (tid / h) * 2 * h + (tid % h), a = x[j], c = x[j + h];
        __syncthreads();
        if (tid < 64) x[j] = a + c, x[j + h] = a - c;
        __syncthreads();
    }
}
// a lane's 4 centroids (values 4 lane .. 4 lane + 3 of block b) from cent [64]; returns the block's norm
__device__ __forceinline__ float t6_load4(const uint8_t* b, int lane, const float* cent, float (&c)[4]) {
    const uint32_t q2 = __ldg((const uint16_t*) (b + 2 + 2 * lane)), hb = __ldg(b + t6::kQhOffset + lane);
    c[0] = cent[(q2 & 0xF) | ((hb & 3) << 4)];
    c[1] = cent[((q2 >> 4) & 0xF) | (((hb >> 2) & 3) << 4)];
    c[2] = cent[((q2 >> 8) & 0xF) | (((hb >> 4) & 3) << 4)];
    c[3] = cent[(q2 >> 12) | (((hb >> 6) & 3) << 4)];
    return half_bits_to_float(__ldg((const uint16_t*) b));
}

// attn_split_kernel (attn_flash_range inlined) over turbo6 blocks: the same grid, keys, window, bias and partials; the
// partials' V in the rotated domain (attn_combine_t6_kernel turns them back)
__global__ void attn_split_t6_kernel(const float* q, const float* r, const uint8_t* kc, const uint8_t* vc, int kv_cap,
                                     int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                                     int d_rel, int extent, int window, float* part, const int* pos_d) {
    if (pos_d) {
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + (int) gridDim.z - k0;
    }
    __shared__ float sm_m[4], sm_s[4], sm_c[64], sm_q[128];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, qi = blockIdx.z, g = h / (H / Hkv), tid = threadIdx.x;
    const int qpos = qpos0 + qi, len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    const int lane = tid & 31, warp = tid >> 5, o = lane * 4;
    const float* rh = r + ((size_t) qi * H + h) * d_rel;
    if (tid < 64) sm_c[tid] = t6::TURBO_CENTROIDS_6BIT[tid];
    sm_q[tid] = q[((size_t) qi * H + h) * 128 + tid] * t6::TURBO_WHT_SIGNS1[tid];
    __syncthreads();
    t6_fwht128(sm_q);
    float q4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) q4[i] = sm_q[o + i] * t6::kInvSqrt128 * t6::TURBO_WHT_SIGNS2[o + i];
    const float r_lane = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
    float m_w = -INFINITY, s_w = 0.0f, o_w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j, dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const size_t blk = ((size_t) (kv_cap > 0 ? pos % kv_cap : pos) * Hkv + g) * t6::kBlockBytes;
        float c4[4];
        const float kn = t6_load4(kc + blk, lane, sm_c, c4);
        float l = wsum(q4[0] * c4[0] + q4[1] * c4[1] + q4[2] * c4[2] + q4[3] * c4[3]) * kn * qk_scale;
        if (dist < extent) {
            const float b = (lane < d_rel) ? r_lane * __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
            l += wsum(b);
        }
        if (l > -INFINITY) {
            const float vn = t6_load4(vc + blk, lane, sm_c, c4);
            const float m_new = fmaxf(m_w, l);
            const float alpha = (m_w == -INFINITY) ? 0.0f : expf(m_w - m_new);
            const float beta = expf(l - m_new);
            m_w = m_new;
            s_w = s_w * alpha + beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) o_w[i] = o_w[i] * alpha + c4[i] * vn * beta;
        }
    }
    if (lane == 0) sm_m[warp] = m_w, sm_s[warp] = s_w;
#pragma unroll
    for (int i = 0; i < 4; ++i) sm_o[warp][o + i] = o_w[i];
    __syncthreads();
    const float M = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
    float sum = 0.0f, oo = 0.0f;
    if (M > -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (sm_m[w] == -INFINITY) continue;
            const float f = expf(sm_m[w] - M);
            sum += sm_s[w] * f;
            oo += sm_o[w][tid] * f;
        }
    }
    float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
    if (tid == 0) pp[0] = M, pp[1] = sum;
    pp[2 + tid] = oo;
}

// attn_combine_kernel, then RotateInverse (signs2, the transform, 1/sqrt(128), signs1) of the head's output
__global__ void attn_combine_t6_kernel(const float* sink, const float* part, int H, int P, float* out) {
    __shared__ float sx[128];
    const int h = blockIdx.x, qi = blockIdx.y, tid = threadIdx.x;
    const float* pp = part + ((size_t) qi * H + h) * P * 130;
    const float sk = sink[(size_t) qi * H + h];
    float M = sk;
    for (int p = 0; p < P; ++p) M = fmaxf(M, pp[(size_t) p * 130]);
    float den = expf(sk - M), o = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = pp[(size_t) p * 130];
        if (mp == -INFINITY) continue;
        const float f = expf(mp - M);
        den += pp[(size_t) p * 130 + 1] * f;
        o += pp[(size_t) p * 130 + 2 + tid] * f;
    }
    sx[tid] = (o / den) * t6::TURBO_WHT_SIGNS2[tid];
    __syncthreads();
    t6_fwht128(sx);
    out[((size_t) qi * H + h) * 128 + tid] = sx[tid] * t6::kInvSqrt128 * t6::TURBO_WHT_SIGNS1[tid];
}

// positions k0 .. k0 + nk - 1 (ring slots) -> FP16 rows [nk][Hkv * 128]: DequantizeBlock + RotateInverse, rounded
__global__ void unpack_kv_t6_to_f16_kernel(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int Hkv,
                                           uint16_t* k16, uint16_t* v16) {
    __shared__ float sx[128];
    const int j = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, pos = k0 + j;
    const uint8_t* b = (blockIdx.z ? vc : kc) + ((size_t) (kv_cap > 0 ? pos % kv_cap : pos) * Hkv + g) * t6::kBlockBytes;
    const uint32_t code = ((b[2 + tid / 2] >> ((tid % 2) * 4)) & 0xF) | (((b[t6::kQhOffset + tid / 4] >> (2 * (tid % 4))) & 3) << 4);
    sx[tid] = t6::TURBO_CENTROIDS_6BIT[code] * half_bits_to_float((uint16_t) (b[0] | (b[1] << 8))) *
              t6::TURBO_WHT_SIGNS2[tid];
    __syncthreads();
    t6_fwht128(sx);
    (blockIdx.z ? v16 : k16)[((size_t) j * Hkv + g) * 128 + tid] = f2h(sx[tid] * t6::kInvSqrt128 * t6::TURBO_WHT_SIGNS1[tid]);
}

void kv_t6_encode(const float* k32, const float* v32, uint8_t* kc, uint8_t* vc, int T, int KV, int kv_cap, int pos0,
                  const int* pos_d, cudaStream_t s) {
    if (T <= 0) return;
    kv_t6_encode_kernel<<<dim3((unsigned) KV, (unsigned) T, 2), 32, 0, s>>>(k32, v32, kc, vc, KV, kv_cap, pos0, pos_d);
    check("kv_t6_encode");
}
void prep_qkv_r_t6(float* q, const float* q_clip, const float* k_in, const float* v_in, float* r, const float* r_clip,
                   int T, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos,
                   float alpha, float n_floor, float* k_stage, float* v_stage, uint8_t* kc, uint8_t* vc, int kv_cap,
                   int pos0, const int* pos0_d, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                   int sink_k, float* sink_y, cudaStream_t s) {
    if (T <= 0) return;
    if (D != 128) {
        std::fprintf(stderr, "ik2::k::prep_qkv_r_t6: head_dim %d (turbo6 is one 128-value block a head)\n", D);
        std::abort();
    }
    prep_qkv_r_impl_kernel<float><<<(unsigned) (T * (2 * H + 2 * KV) + (sink_w ? T * H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, k_stage, v_in, v_stage, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha,
        n_floor, 0, 0, 1.0f, 1.0f, nullptr, q16, sink_w, sink_x, sink_k, sink_y);
    check("prep_qkv_r_t6");
    kv_t6_encode(k_stage, v_stage, kc, vc, T, KV, kv_cap, pos0, pos0_d, s);
}
void attn_decode_t6(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc, int kv_cap,
                    int nq, int H, int Hkv, int k0, int nk, int qpos0, const int* pos_d, int P, float qk_scale,
                    const float* proj, int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    if (d_rel > 32 || H % Hkv) {
        std::fprintf(stderr, "ik2::k::attn_decode_t6: d_rel %d, %d heads on %d\n", d_rel, H, Hkv);
        std::abort();
    }
    attn_split_t6_kernel<<<dim3((unsigned) H, (unsigned) P, (unsigned) nq), 128, 0, s>>>(
        q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos0, qk_scale, proj, d_rel, extent, window, scratch, pos_d);
    check("attn_decode_t6 (split)");
    attn_combine_t6_kernel<<<dim3((unsigned) H, (unsigned) nq), 128, 0, s>>>(sink, scratch, H, P, out);
    check("attn_decode_t6 (combine)");
}
void unpack_kv_t6_to_f16(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk, int Hkv, uint16_t* k16,
                         uint16_t* v16, cudaStream_t s) {
    if (nk <= 0) return;
    unpack_kv_t6_to_f16_kernel<<<dim3((unsigned) nk, (unsigned) Hkv, 2), 128, 0, s>>>(kc, vc, kv_cap, k0, Hkv, k16, v16);
    check("unpack_kv_t6_to_f16");
}

// ==== the token path's reader: the bytes and, head by head, the sums of prep_qkv_r_t6 +
// attn_decode_t6 + gate_mul_f16 above, in their order (tests/ik2_kernels_test.cu test_turbo6_tok: bit for bit), with
// a GROUP a block - a row's bytes read and decoded once for the group's q heads (were once a head) -, the next row's
// bytes asked for while a row is summed, q rotated once a head (was once a split block), the encode a warp in
// registers and the combine, RotateInverse and output gate in one launch.  IK2_FUSE_MASK bit 22 (0x400000): the old.

// the WHT's butterflies (t6_fwht128's pairs and sums, unscaled) over one warp's 128 values, lane L holding 4 L .. 4 L
// + 3: stages h 1 and 2 inside a lane, h = 4 m (m 1 .. 16) between lanes L and L ^ m (the lower lane keeps a + c, the
// upper a - c, a the lower's value)
__device__ __forceinline__ void t6_fwht_warp(float (&x)[4], int lane) {
    {
        const float a0 = x[0], c0 = x[1], a1 = x[2], c1 = x[3];
        x[0] = a0 + c0, x[1] = a0 - c0, x[2] = a1 + c1, x[3] = a1 - c1;
    }
    {
        const float a0 = x[0], c0 = x[2], a1 = x[1], c1 = x[3];
        x[0] = a0 + c0, x[2] = a0 - c0, x[1] = a1 + c1, x[3] = a1 - c1;
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float y = __shfl_xor_sync(0xffffffffu, x[i], m);
            x[i] = hi ? y - x[i] : x[i] + y;
        }
    }
}

// NearestCode (the first i with v < MID[i], 63 if none) as a count: MID rises strictly, so that i is the number of
// midpoints with !(v < MID[i]) - the same code for every float (NaN: 63), no table load
constexpr bool t6_mid_rises() {
    for (int i = 1; i < 63; ++i)
        if (!(t6::TURBO_MID_6BIT[i - 1] < t6::TURBO_MID_6BIT[i])) return false;
    return true;
}
static_assert(t6_mid_rises(), "turbo6: the code count needs strictly rising midpoints");
__device__ __forceinline__ uint32_t t6_code(float v) {
    uint32_t n = 0;
#pragma unroll
    for (int i = 0; i < 63; ++i) n += !(v < t6::TURBO_MID_6BIT[i]) ? 1u : 0u;
    return n;
}

// QuantizeBlock (k6) of the 128 values a warp holds (lane L: 4 L .. 4 L + 3) into the 98 bytes at dst and at dst2
// (either null: not written), kv_t6_encode_kernel's operations in its order - the two sums over j = 0 .. 127 one
// after the other (every lane runs the chain over the values read lane by lane: v_readlane, no LDS, no barrier), no
// contraction, correctly rounded sqrt and divisions: its bytes.  code: the lane's 4 codes; returns the norm's bits
__device__ __forceinline__ uint16_t t6_encode_warp(const float (&x)[4], uint8_t* dst, uint8_t* dst2,
                                                   uint32_t (&code)[4], int lane) {
    float sum = 0.0f;
#pragma unroll
    for (int L = 0; L < 32; ++L)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float v = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x[i]), L));
            sum = __fadd_rn(sum, __fmul_rn(v, v));
        }
    const float gn = __fsqrt_rn(sum), inv = gn > t6::kNormEpsilon ? __fdiv_rn(1.0f, gn) : 0.0f;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = __fmul_rn(x[i], inv) * t6::TURBO_WHT_SIGNS1[4 * lane + i];
    t6_fwht_warp(s, lane);
    float c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        code[i] = t6_code(__fmul_rn(s[i], t6::kInvSqrt128) * t6::TURBO_WHT_SIGNS2[4 * lane + i]);
        c[i] = t6::TURBO_CENTROIDS_6BIT[code[i]];
    }
    float acc = 0.0f;
#pragma unroll
    for (int L = 0; L < 32; ++L)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float v = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(c[i]), L));
            acc = __fadd_rn(acc, __fmul_rn(v, v));
        }
    const float recon = __fsqrt_rn(acc);
    const uint16_t half = t6::FloatToHalfBits(recon > t6::kNormEpsilon ? __fdiv_rn(gn, recon) : gn);
    const uint8_t b0 = (uint8_t) ((code[0] & 0xF) | ((code[1] & 0xF) << 4));
    const uint8_t b1 = (uint8_t) ((code[2] & 0xF) | ((code[3] & 0xF) << 4));
    const uint8_t bh = (uint8_t) (((code[0] >> 4) & 3) | (((code[1] >> 4) & 3) << 2) | (((code[2] >> 4) & 3) << 4) |
                                  (((code[3] >> 4) & 3) << 6));
    uint8_t* const ds[2] = {dst, dst2};
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        uint8_t* d = ds[k];
        if (!d) continue;
        d[2 + 2 * lane] = b0, d[3 + 2 * lane] = b1, d[t6::kQhOffset + lane] = bh;
        if (lane == 0) d[0] = (uint8_t) half, d[1] = (uint8_t) (half >> 8);
    }
    return half;
}

// the clipped k (store_k_warp<float>: l2clip_warp's vec4 sums, tau 1) and the normed v (store_v_warp<float>: its
// strided sums) of a group, a lane's 4 values 4 L .. 4 L + 3 (the encode's layout)
__device__ __forceinline__ void t6_k_clip(const float* kp, float qk_clip, int lane, float (&x)[4]) {
    const float4 kx = ((const float4*) kp)[lane];
    const float k0v = kx.x * 1.0f, k1v = kx.y * 1.0f, k2v = kx.z * 1.0f, k3v = kx.w * 1.0f;
    const float knrm = sqrtf(wsum(k0v * k0v + k1v * k1v + k2v * k2v + k3v * k3v));
    const float kf = fminf(1.0f, qk_clip / (knrm + 1e-6f)) * 1.0f;
    x[0] = k0v * kf, x[1] = k1v * kf, x[2] = k2v * kf, x[3] = k3v * kf;
}
__device__ __forceinline__ void t6_v_norm(const float* vp, float eps, int lane, float (&x)[4]) {
    float a = 0.0f;
    for (int d = lane; d < 128; d += 32) a += vp[d] * vp[d];
    const float rs = rsqrtf(wsum(a) / (float) 128 + eps);
    const float4 vx = ((const float4*) vp)[lane];
    x[0] = vx.x * rs, x[1] = vx.y * rs, x[2] = vx.z * rs, x[3] = vx.w * rs;
}
// q of a head: l2clip_warp's vec4 sums (scale, tau), then attn_split_t6_kernel's rotation of it: signs1, the
// butterflies, 1/sqrt(128), signs2 - a lane's 4 rotated values
__device__ __forceinline__ void t6_q_rot(const float* qp, const float* qc, float qk_clip, float tau, int lane,
                                         float (&x)[4]) {
    const float4 xv = ((const float4*) qp)[lane];
    const float4 sv = qc ? __ldg((const float4*) qc + lane) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
    const float nrm = sqrtf(wsum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3));
    const float f = fminf(1.0f, qk_clip / (nrm + 1e-6f)) * tau;
    x[0] = v0 * f, x[1] = v1 * f, x[2] = v2 * f, x[3] = v3 * f;
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = x[i] * t6::TURBO_WHT_SIGNS1[4 * lane + i];
    t6_fwht_warp(x, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = x[i] * t6::kInvSqrt128 * t6::TURBO_WHT_SIGNS2[4 * lane + i];
}
// r of a head, l2clip_warp's loop path (d_rel < 128) as attn_prep_one_kernel takes it: lane d's clipped value
__device__ __forceinline__ float t6_r_clip(const float* src, const float* sc, int d_rel, float rel_clip, float tau,
                                           int lane) {
    float a = 0.0f;
    for (int d = lane; d < d_rel; d += 32) {
        const float v = src[d] * (sc ? sc[d] : 1.0f);
        a += v * v;
    }
    const float nrm = sqrtf(wsum(a));
    const float f = fminf(1.0f, rel_clip / (nrm + 1e-6f)) * tau;
    return lane < d_rel ? src[lane] * (sc ? sc[lane] : 1.0f) * f : 0.0f;
}

// a row's words for a lane: K and V low nibbles (16 bits: qs bytes 2 L, 2 L + 1), top bits (qh byte L), norms (FP16
// bits), and the bias column's value (0 past the extent or for lane >= d_rel)
struct T6Row {
    uint32_t kq, kh, kn, vq, vh, vn;
    float pr;
};
__device__ __forceinline__ void t6_row_load(const uint8_t* kb, const uint8_t* vb, const float* prow, int lane,
                                            T6Row& w) {
    w.kq = __ldg((const uint16_t*) (kb + 2 + 2 * lane)), w.kh = __ldg(kb + t6::kQhOffset + lane);
    w.kn = __ldg((const uint16_t*) kb);
    w.vq = __ldg((const uint16_t*) (vb + 2 + 2 * lane)), w.vh = __ldg(vb + t6::kQhOffset + lane);
    w.vn = __ldg((const uint16_t*) vb);
    w.pr = prow ? __ldg(prow + lane) : 0.0f;
}
// t6_load4's centroids from the words
__device__ __forceinline__ void t6_cent4(uint32_t q2, uint32_t hb, const float* cent, float (&c)[4]) {
    c[0] = cent[(q2 & 0xF) | ((hb & 3) << 4)];
    c[1] = cent[((q2 >> 4) & 0xF) | (((hb >> 2) & 3) << 4)];
    c[2] = cent[((q2 >> 8) & 0xF) | (((hb >> 4) & 3) << 4)];
    c[3] = cent[((q2 >> 12) & 0xF) | (((hb >> 6) & 3) << 4)];
}
// one row for the GQ heads: attn_split_t6_kernel's loop body, head by head the same expressions in the same order
template <int GQ>
__device__ __forceinline__ void t6_row_math(const float (&q4)[GQ][4], const float (&r_lane)[GQ], const T6Row& w,
                                            int dist, int d_rel, int extent, float qk_scale, const float* cent,
                                            float (&m_w)[GQ], float (&s_w)[GQ], float (&o_w)[GQ][4]) {
    const int lane = threadIdx.x & 31;
    float c4[4], v4[4];
    t6_cent4(w.kq, w.kh, cent, c4);
    const float kn = half_bits_to_float((uint16_t) w.kn);
    float l[GQ];
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh)
        l[hh] = wsum(q4[hh][0] * c4[0] + q4[hh][1] * c4[1] + q4[hh][2] * c4[2] + q4[hh][3] * c4[3]) * kn * qk_scale;
    if (dist < extent) {
#pragma unroll
        for (int hh = 0; hh < GQ; ++hh) {
            const float b = (lane < d_rel) ? r_lane[hh] * w.pr : 0.0f;
            l[hh] += wsum(b);
        }
    }
    t6_cent4(w.vq, w.vh, cent, v4);
    const float vn = half_bits_to_float((uint16_t) w.vn);
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh) {
        if (l[hh] > -INFINITY) {
            const float m_new = fmaxf(m_w[hh], l[hh]);
            const float alpha = (m_w[hh] == -INFINITY) ? 0.0f : expf(m_w[hh] - m_new);
            const float beta = expf(l[hh] - m_new);
            m_w[hh] = m_new;
            s_w[hh] = s_w[hh] * alpha + beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) o_w[hh][i] = o_w[hh][i] * alpha + v4[i] * vn * beta;
        }
    }
}
// a warp's cache rows j = jb, jb + 4, .. < je (positions k0 + j; ring > 0: slot pos % ring) for group g, the next
// row's words loaded while the current one is summed
template <int GQ>
__device__ __forceinline__ void t6_rows(const float (&q4)[GQ][4], const float (&r_lane)[GQ], const uint8_t* kc,
                                        const uint8_t* vc, int ring, int Hkv, int g, int k0, int jb, int je, int qpos,
                                        float qk_scale, const float* proj, int d_rel, int extent, int window,
                                        const float* cent, float (&m_w)[GQ], float (&s_w)[GQ], float (&o_w)[GQ][4]) {
    const int lane = threadIdx.x & 31;
    auto load = [&](int j, T6Row& w) {
        const int pos = k0 + j, dist = qpos - pos;
        const size_t off = ((size_t) (ring > 0 ? pos % ring : pos) * Hkv + g) * t6::kBlockBytes;
        const bool bias = dist >= 0 && dist < extent && lane < d_rel;
        t6_row_load(kc + off, vc + off, bias ? proj + (size_t) dist * d_rel : nullptr, lane, w);
    };
    T6Row cur{}, nxt{};
    if (jb < je) load(jb, cur);
    for (int j = jb; j < je; j += 4) {
        if (j + 4 < je) load(j + 4, nxt);
        const int dist = qpos - (k0 + j);
        if (!(dist < 0 || dist >= window))
            t6_row_math<GQ>(q4, r_lane, cur, dist, d_rel, extent, qk_scale, cent, m_w, s_w, o_w);
        cur = nxt;
    }
}
// the block's 4 warps -> the GQ heads' partials (attn_split_t6_kernel's merge, [H][P][130]: max, sum, 128 values)
template <int GQ>
__device__ __forceinline__ void t6_partials(const float (&m_w)[GQ], const float (&s_w)[GQ], const float (&o_w)[GQ][4],
                                            float (*sm_m)[4], float (*sm_s)[4], float (*sm_o)[4][128], float* part,
                                            int h0, int P, int p) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh) {
        if (lane == 0) sm_m[hh][warp] = m_w[hh], sm_s[hh][warp] = s_w[hh];
#pragma unroll
        for (int i = 0; i < 4; ++i) sm_o[hh][warp][lane * 4 + i] = o_w[hh][i];
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh) {
        const float M = fmaxf(fmaxf(sm_m[hh][0], sm_m[hh][1]), fmaxf(sm_m[hh][2], sm_m[hh][3]));
        float sum = 0.0f, oo = 0.0f;
        if (M > -INFINITY) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[hh][w] == -INFINITY) continue;
                const float f = expf(sm_m[hh][w] - M);
                sum += sm_s[hh][w] * f;
                oo += sm_o[hh][w][tid] * f;
            }
        }
        float* pp = part + ((size_t) (h0 + hh) * P + p) * 130;
        if (tid == 0) pp[0] = M, pp[1] = sum;
        pp[2 + tid] = oo;
    }
}
// a lane's 4 values of one head's output: attn_combine_t6_kernel's sums over np partials (pm / ps: max, sum; ov: the
// lane's values) and the sink, RotateInverse, then gate_mul's o sigmoid(g) - to out (F32) and out16 (FP16)
template <int NP>
__device__ __forceinline__ void t6_out4(const float* pm, const float* ps, const float (*ov)[4], float sk,
                                        const float* gate, float* out, uint16_t* out16, int lane) {
    float M = sk;
    for (int p = 0; p < NP; ++p) M = fmaxf(M, pm[p]);
    float den = expf(sk - M), o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        if (pm[p] == -INFINITY) continue;
        const float f = expf(pm[p] - M);
        den += ps[p] * f;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += ov[p][i] * f;
    }
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = (o[i] / den) * t6::TURBO_WHT_SIGNS2[4 * lane + i];
    t6_fwht_warp(x, lane);
    const float4 gv = ((const float4*) gate)[lane];
    const float g4[4] = {gv.x, gv.y, gv.z, gv.w};
    float y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v = x[i] * t6::kInvSqrt128 * t6::TURBO_WHT_SIGNS1[4 * lane + i];
        y[i] = v * (1.0f / (1.0f + expf(-g4[i])));
    }
    ((float4*) out)[lane] = make_float4(y[0], y[1], y[2], y[3]);
    ((uint2*) out16)[lane] = make_uint2(f2_to_u32(y[0], y[1]), f2_to_u32(y[2], y[3]));
}

// one token's prep (the position at pos0_d): a warp a q head (clip, then rotate: q in place ROTATED - its one reader is
// attn_split_t6g_kernel), a group's k (clip) and v (norm) encoded to the cache at slot pos % kv_cap, r clipped in
// place, the sinks (dot_f16_pf: prep's sums)
template <int NPL>
__global__ void __launch_bounds__(32)
    prep_t6_tok_kernel(float* q, const float* q_clip, const float* k_in, const float* v_in, uint8_t* kc, uint8_t* vc,
                       float* r, const float* r_clip, int H, int KVh, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, float alpha, float n_floor, int kv_cap, const int* pos0_d,
                       const uint16_t* sink_w, const float* sink_x, float* sink_y) {
    const int gid = blockIdx.x, lane = threadIdx.x;
    if (gid >= 2 * H + 2 * KVh) {
        const int h = gid - 2 * H - 2 * KVh;
        const float a = dot_f16_pf<NPL>(sink_w + (size_t) h * (32 * NPL), sink_x, lane);
        if (lane == 0) sink_y[h] = a;
        return;
    }
    const float tau = pos ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[0] + 1.0f) / n_floor))) : 1.0f;
    if (gid < H) {
        float x[4];
        t6_q_rot(q + (size_t) gid * 128, q_clip ? q_clip + (size_t) gid * 128 : nullptr, qk_clip, tau, lane, x);
        ((float4*) (q + (size_t) gid * 128))[lane] = make_float4(x[0], x[1], x[2], x[3]);
    } else if (gid < H + 2 * KVh) {
        const bool is_v = gid >= H + KVh;
        const int g = gid - H - (is_v ? KVh : 0), p0 = *pos0_d, slot = kv_cap > 0 ? p0 % kv_cap : p0;
        float x[4];
        if (is_v) t6_v_norm(v_in + (size_t) g * 128, eps, lane, x);
        else t6_k_clip(k_in + (size_t) g * 128, qk_clip, lane, x);
        uint32_t code[4];
        t6_encode_warp(x, (is_v ? vc : kc) + ((size_t) slot * KVh + g) * t6::kBlockBytes, nullptr, code, lane);
    } else {
        const int h = gid - H - 2 * KVh;
        float* rp = r + (size_t) h * d_rel;
        l2clip_warp(rp, r_clip ? r_clip + (size_t) h * d_rel : nullptr, rp, nullptr, d_rel, rel_clip, tau);
    }
}

// attn_split_t6_kernel for one token (position at pos_d), a GROUP a block (grid Hkv x P): q read rotated (the prep's)
template <int GQ>
__global__ void __launch_bounds__(128)
    attn_split_t6g_kernel(const float* q, const float* r, const uint8_t* kc, const uint8_t* vc, int ring, int Hkv,
                          float qk_scale, const float* proj, int d_rel, int extent, int window, float* part,
                          const int* pos_d) {
    const int qpos = *pos_d, k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0, nk = qpos + 1 - k0;
    __shared__ float sm_c[64], sm_m[GQ][4], sm_s[GQ][4];
    __shared__ __align__(16) float sm_o[GQ][4][128];
    const int g = blockIdx.x, p = blockIdx.y, P = gridDim.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len), h0 = g * GQ;
    if (tid < 64) sm_c[tid] = t6::TURBO_CENTROIDS_6BIT[tid];
    float q4[GQ][4], r_lane[GQ];
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh) {
        const float4 qv = ((const float4*) (q + (size_t) (h0 + hh) * 128))[lane];
        q4[hh][0] = qv.x, q4[hh][1] = qv.y, q4[hh][2] = qv.z, q4[hh][3] = qv.w;
        r_lane[hh] = (lane < d_rel) ? __ldg(r + (size_t) (h0 + hh) * d_rel + lane) : 0.0f;
    }
    __syncthreads();
    float m_w[GQ], s_w[GQ], o_w[GQ][4];
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh) {
        m_w[hh] = -INFINITY, s_w[hh] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) o_w[hh][i] = 0.0f;
    }
    t6_rows<GQ>(q4, r_lane, kc, vc, ring, Hkv, g, k0, j0 + warp, j1, qpos, qk_scale, proj, d_rel, extent, window, sm_c,
                m_w, s_w, o_w);
    t6_partials<GQ>(m_w, s_w, o_w, sm_m, sm_s, sm_o, part, h0, P, p);
}

// attn_combine_t6_kernel + gate_mul_f16 of one token, a head a block (SKIP: attn_combine_gate_kernel's set 7 - the
// splits without keys taken as such, not read): the sums a thread a value, then warp 0 rotates back and gates
template <int NP, bool SKIP>
__global__ void __launch_bounds__(128)
    attn_combine_gate_t6_kernel(const float* __restrict__ sink, const float* __restrict__ part,
                                const float* __restrict__ gate, float* out, uint16_t* out16, const int* pos_d,
                                int window) {
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const float* pp = part + (size_t) h * NP * 130;
    __shared__ float sm_m[NP], sm_s[NP], sm_f[SKIP ? NP : 1];
    __shared__ __align__(16) float sx[128];
    int used = NP;
    if constexpr (SKIP) {
        const int qpos = *pos_d, k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0, nk = qpos + 1 - k0;
        const int len = (nk + NP - 1) / NP;
        used = (nk + len - 1) / len;
    }
    for (int p = tid; p < NP; p += 128) {
        sm_m[p] = p < used ? pp[(size_t) p * 130] : -INFINITY;
        sm_s[p] = p < used ? pp[(size_t) p * 130 + 1] : 0.0f;
    }
    float ov[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) ov[p] = p < used ? pp[(size_t) p * 130 + 2 + tid] : 0.0f;
    const float sk = sink[h];
    __syncthreads();
    float M = sk;
    for (int p = 0; p < NP; ++p) M = fmaxf(M, sm_m[p]);
    if constexpr (SKIP) {
        for (int p = tid; p < NP; p += 128) sm_f[p] = sm_m[p] == -INFINITY ? 0.0f : expf(sm_m[p] - M);
        __syncthreads();
    }
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const float mp = sm_m[p];
        if (mp == -INFINITY) continue;
        const float f = SKIP ? sm_f[p] : expf(mp - M);
        den += sm_s[p] * f;
        o += ov[p] * f;
    }
    sx[tid] = (o / den) * t6::TURBO_WHT_SIGNS2[tid];
    __syncthreads();
    if (tid >= 32) return;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = sx[4 * lane + i];
    t6_fwht_warp(x, lane);
    const float4 gv = ((const float4*) (gate + (size_t) h * 128))[lane];
    const float g4[4] = {gv.x, gv.y, gv.z, gv.w};
    float y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v = x[i] * t6::kInvSqrt128 * t6::TURBO_WHT_SIGNS1[4 * lane + i];
        y[i] = v * (1.0f / (1.0f + expf(-g4[i])));
    }
    ((float4*) (out + (size_t) h * 128))[lane] = make_float4(y[0], y[1], y[2], y[3]);
    ((uint2*) (out16 + (size_t) h * 128))[lane] = make_uint2(f2_to_u32(y[0], y[1]), f2_to_u32(y[2], y[3]));
}

// a window visit (set 4) in ONE launch, grid (Hkv, NP + 1), a group a block: warps 0 .. GQ - 1 clip and rotate the
// group's q heads and clip their r (into LDS); the block holding the token's position encodes its k (warp 3) and v
// (warp 0, after its head), writes them to the cache and LDS and its owner warp takes that row from LDS (the bytes
// the cache now holds); block NP the group's sinks; the group's last block of NP + 1 combines, a warp a head
template <int NP, int GQ>
__global__ void __launch_bounds__(128)
    attn_prep_one_t6_kernel(const float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in,
                            uint8_t* vc, const float* r, const float* r_clip, int Hkv, int d_rel, float qk_clip,
                            float rel_clip, float eps, const int* pos_d, bool log_scale, float alpha, float n_floor,
                            int kv_cap, int ring, const uint16_t* sink_w, const float* sink_x, float* sink,
                            float qk_scale, const float* proj, int extent, int window, float* part,
                            const float* __restrict__ gate, float* out, uint16_t* out16) {
    static_assert(GQ <= 3, "warp 3 encodes k");
    const int g = blockIdx.x, p = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, h0 = g * GQ;
    if (p == NP) {   // == the group's sinks: prep_qkv_r_pf_kernel's sink warps ==
        if (warp < GQ) {
            const float a = dot_f16_pf<kFuseNpl>(sink_w + (size_t) (h0 + warp) * (32 * kFuseNpl), sink_x, lane);
            if (lane == 0) sink[h0 + warp] = a;
        }
    } else {
        __shared__ float sm_c[64], sm_m[GQ][4], sm_s[GQ][4], sm_r[GQ][32];
        __shared__ __align__(16) float sm_q[GQ][128];
        __shared__ __align__(16) float sm_o[GQ][4][128];
        __shared__ __align__(16) uint8_t sm_row[2][128];   // the token's own K and V blocks
        const int qpos = *pos_d, k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0, nk = qpos + 1 - k0;
        const int len = (nk + NP - 1) / NP, j0 = p * len, j1 = min(nk, j0 + len);
        const bool self = j0 <= nk - 1 && nk - 1 < j1;
        const float tau = log_scale ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) qpos + 1.0f) / n_floor))) : 1.0f;
        if (tid < 64) sm_c[tid] = t6::TURBO_CENTROIDS_6BIT[tid];
        const int slot = kv_cap > 0 ? qpos % kv_cap : qpos;
        if (warp < GQ) {
            const int h = h0 + warp;
            float x[4];
            t6_q_rot(q + (size_t) h * 128, q_clip ? q_clip + (size_t) h * 128 : nullptr, qk_clip, tau, lane, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) sm_q[warp][4 * lane + i] = x[i];
            sm_r[warp][lane] = t6_r_clip(r + (size_t) h * d_rel, r_clip ? r_clip + (size_t) h * d_rel : nullptr, d_rel,
                                         rel_clip, tau, lane);
            if (self && warp == 0) {
                uint32_t code[4];
                t6_v_norm(v_in + (size_t) g * 128, eps, lane, x);
                t6_encode_warp(x, vc + ((size_t) slot * Hkv + g) * t6::kBlockBytes, sm_row[1], code, lane);
            }
        } else if (self && warp == 3) {
            float x[4];
            uint32_t code[4];
            t6_k_clip(k_in + (size_t) g * 128, qk_clip, lane, x);
            t6_encode_warp(x, kc + ((size_t) slot * Hkv + g) * t6::kBlockBytes, sm_row[0], code, lane);
        }
        __syncthreads();
        float q4[GQ][4], r_lane[GQ], m_w[GQ], s_w[GQ], o_w[GQ][4];
#pragma unroll
        for (int hh = 0; hh < GQ; ++hh) {
#pragma unroll
            for (int i = 0; i < 4; ++i) q4[hh][i] = sm_q[hh][4 * lane + i], o_w[hh][i] = 0.0f;
            r_lane[hh] = lane < d_rel ? sm_r[hh][lane] : 0.0f;
            m_w[hh] = -INFINITY, s_w[hh] = 0.0f;
        }
        // the cache rows; the token's own (j = nk - 1, the last of its warp's rows) from LDS
        const int jb = j0 + warp;
        const bool own = self && jb <= nk - 1 && ((nk - 1 - jb) & 3) == 0;
        t6_rows<GQ>(q4, r_lane, kc, vc, ring, Hkv, g, k0, jb, own ? nk - 1 : j1, qpos, qk_scale, proj, d_rel, extent,
                    window, sm_c, m_w, s_w, o_w);
        if (own && window > 0) {   // (dist 0: inside every window and extent > 0)
            T6Row w;
            w.kq = *(const uint16_t*) (sm_row[0] + 2 + 2 * lane), w.kh = sm_row[0][t6::kQhOffset + lane];
            w.kn = *(const uint16_t*) sm_row[0];
            w.vq = *(const uint16_t*) (sm_row[1] + 2 + 2 * lane), w.vh = sm_row[1][t6::kQhOffset + lane];
            w.vn = *(const uint16_t*) sm_row[1];
            w.pr = (0 < extent && lane < d_rel) ? __ldg(proj + lane) : 0.0f;
            t6_row_math<GQ>(q4, r_lane, w, 0, d_rel, extent, qk_scale, sm_c, m_w, s_w, o_w);
        }
        t6_partials<GQ>(m_w, s_w, o_w, sm_m, sm_s, sm_o, part, h0, NP, p);
    }
    // == the group's last block of NP + 1: a warp a head, t6_out4 ==
    __shared__ unsigned int s_ticket;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_ticket = atomicAdd(&g_fuse_cnt[kFuseCntAttn + g], 1u);
    __syncthreads();
    if (s_ticket != NP) return;
    __threadfence();
    if (tid == 0) g_fuse_cnt[kFuseCntAttn + g] = 0;
    if (warp >= GQ) return;
    const int h = h0 + warp;
    const float* pp = part + (size_t) h * NP * 130;
    float pm[NP], ps[NP], ov[NP][4];
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        pm[i] = pp[(size_t) i * 130], ps[i] = pp[(size_t) i * 130 + 1];
#pragma unroll
        for (int c = 0; c < 4; ++c) ov[i][c] = pp[(size_t) i * 130 + 2 + 4 * lane + c];
    }
    const float sk = ((volatile const float*) sink)[h];
    t6_out4<NP>(pm, ps, ov, sk, gate + (size_t) h * 128, out + (size_t) h * 128, out16 + (size_t) h * 128, lane);
}

bool attn_tok_t6(float* q, const float* q_clip, const float* k_in, const float* v_in, float* r, const float* r_clip,
                 int H, int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos,
                 bool log_scale, float alpha, float n_floor, uint8_t* kc, uint8_t* vc, int kv_cap, bool sliding,
                 const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y, float qk_scale,
                 const float* proj, int extent, int window, int P, bool one_launch, bool skip, float* scratch,
                 const float* gate, float* out, uint16_t* out16, cudaStream_t s) {
    auto al = [](const void* p, int b) { return ((uintptr_t) p & (b - 1)) == 0; };
    if (D != 128 || KV < 1 || KV > 56 || H != 3 * KV || d_rel > 32 || !sink_w || sink_k != 32 * kFuseNpl) return false;
    if (P != 16 && P != 128) return false;
    if (!al(q, 16) || (q_clip && !al(q_clip, 16)) || !al(k_in, 16) || !al(v_in, 16) || !al(gate, 16) ||
        !al(out, 16) || !al(out16, 8))
        return false;
    const int ring = sliding ? kv_cap : 0;
    if (one_launch && P == 16) {
        attn_prep_one_t6_kernel<16, 3><<<dim3((unsigned) KV, 17), 128, 0, s>>>(
            q, q_clip, k_in, kc, v_in, vc, r, r_clip, KV, d_rel, qk_clip, rel_clip, eps, pos, log_scale, alpha,
            n_floor, kv_cap, ring, sink_w, sink_x, sink_y, qk_scale, proj, extent, window, scratch, gate, out, out16);
        check("attn_tok_t6 (one launch)");
        return true;
    }
    prep_t6_tok_kernel<kFuseNpl><<<(unsigned) (3 * H + 2 * KV), 32, 0, s>>>(
        q, q_clip, k_in, v_in, kc, vc, r, r_clip, H, KV, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, kv_cap, pos, sink_w, sink_x, sink_y);
    check("attn_tok_t6 (prep)");
    attn_split_t6g_kernel<3><<<dim3((unsigned) KV, (unsigned) P), 128, 0, s>>>(q, r, kc, vc, ring, KV, qk_scale, proj,
                                                                              d_rel, extent, window, scratch, pos);
    check("attn_tok_t6 (split)");
    if (P == 16)
        attn_combine_gate_t6_kernel<16, false><<<(unsigned) H, 128, 0, s>>>(sink_y, scratch, gate, out, out16, pos,
                                                                            window);
    else if (skip)
        attn_combine_gate_t6_kernel<128, true><<<(unsigned) H, 128, 0, s>>>(sink_y, scratch, gate, out, out16, pos,
                                                                            window);
    else
        attn_combine_gate_t6_kernel<128, false><<<(unsigned) H, 128, 0, s>>>(sink_y, scratch, gate, out, out16, pos,
                                                                             window);
    check("attn_tok_t6 (combine, gate)");
    return true;
}
