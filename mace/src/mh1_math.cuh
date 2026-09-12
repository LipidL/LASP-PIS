#pragma once
#include <lapis/pair.cuh>
namespace mace {
using namespace lapis;
// Small model-local tangents. The runtime itself provides no differentiation.
struct D1 {
  float v, d;
  __device__ D1(float a = 0, float b = 0) : v(a), d(b) {}
};
__device__ inline D1 operator+(D1 a, D1 b) {
  return {a.v + b.v, a.d + b.d};
}
__device__ inline D1 operator-(D1 a, D1 b) {
  return {a.v - b.v, a.d - b.d};
}
__device__ inline D1 operator*(D1 a, D1 b) {
  return {a.v * b.v, a.d * b.v + a.v * b.d};
}
__device__ inline D1 operator/(D1 a, D1 b) {
  float v = a.v / b.v;
  return {v, (a.d - v * b.d) / b.v};
}
__device__ inline D1 power(D1 a, float b) {
  float v = powf(a.v, b);
  return {v, b * v / a.v * a.d};
}
__device__ inline D1 exponential(D1 a) {
  float v = expf(a.v);
  return {v, v * a.d};
}
__device__ inline D1 sine(D1 a) {
  return {sinf(a.v), cosf(a.v) * a.d};
}
__device__ inline D1 envelope(D1 r, float rc, float p) {
  if (r.v >= rc)
    return {};
  D1 x = r / D1(rc);
  return D1(1) - D1((p + 1) * (p + 2) * .5f) * power(x, p) + D1(p * (p + 2)) * power(x, p + 1) -
         D1(p * (p + 1) * .5f) * power(x, p + 2);
}
struct D3 {
  float v, x, y, z;
  __device__ D3(float a = 0, float b = 0, float c = 0, float d = 0) : v(a), x(b), y(c), z(d) {}
};
__device__ inline D3 operator+(D3 a, D3 b) {
  return {a.v + b.v, a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ inline D3 operator-(D3 a, D3 b) {
  return {a.v - b.v, a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ inline D3 operator*(D3 a, D3 b) {
  return {a.v * b.v, a.x * b.v + a.v * b.x, a.y * b.v + a.v * b.y, a.z * b.v + a.v * b.z};
}
__device__ inline void harmonics(float3 d, float r, D3 *sh) {
  float ir = 1 / r;
  D3 x(d.x * ir), y(d.y * ir), z(d.z * ir);
  x.x = (1 - x.v * x.v) * ir;
  x.y = -x.v * y.v * ir;
  x.z = -x.v * z.v * ir;
  y.x = x.y;
  y.y = (1 - y.v * y.v) * ir;
  y.z = -y.v * z.v * ir;
  z.x = x.z;
  z.y = y.z;
  z.z = (1 - z.v * z.v) * ir;
  sh[0] = D3(1);
  sh[1] = x;
  sh[2] = y;
  sh[3] = z;
  D3 yy = y * y, xxzz = x * x + z * z;
  sh[4] = D3(1.7320508075688772f) * x * z;
  sh[5] = D3(1.7320508075688772f) * x * y;
  sh[6] = yy - D3(.5f) * xxzz;
  sh[7] = D3(1.7320508075688772f) * y * z;
  sh[8] = D3(.8660254037844386f) * (z * z - x * x);
  sh[9] = D3(.9128709291752769f) * (sh[4] * z + sh[8] * x);
  sh[10] = D3(2.23606797749979f) * sh[4] * y;
  sh[11] = D3(.6123724356957945f) * (D3(4) * yy - xxzz) * x;
  sh[12] = D3(.5f) * y * (D3(2) * yy - D3(3) * xxzz);
  sh[13] = D3(.6123724356957945f) * z * (D3(4) * yy - xxzz);
  sh[14] = D3(2.23606797749979f) * sh[8] * y;
  sh[15] = D3(.9128709291752769f) * (sh[8] * z - sh[4] * x);
  for (int l = 1; l <= 3; ++l)
    for (int k = l * l; k < (l + 1) * (l + 1); ++k)
      sh[k] = D3(sqrtf(float(2 * l + 1))) * sh[k];
}
__device__ inline float sigmoid(float x) {
  return 1 / (1 + expf(-x));
}
struct RadialWeights {
  const float *source, *target, *radial, *bias;
  const float *weight[3], *affine_bias[3], *gamma[3], *beta[3];
  const float *output_weight, *output_bias;
  int depth, outputs;
};
struct RadialScratch {
  float input[10], dinput[10], a[64], da[64];
};
__device__ inline float block_sum128(float v, float *tmp) {
  int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  for (int step = 16; step; step /= 2)
    v += __shfl_down_sync(0xffffffff, v, step);
  if (lane == 0)
    tmp[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < 4 ? tmp[lane] : 0;
    for (int step = 16; step; step /= 2)
      v += __shfl_down_sync(0xffffffff, v, step);
    if (lane == 0)
      tmp[0] = v;
  }
  __syncthreads();
  return tmp[0];
}
__device__ inline float warp_sum(float x) {
  for (int step = 16; step; step /= 2)
    x += __shfl_down_sync(0xffffffff, x, step);
  return __shfl_sync(0xffffffff, x, 0);
}
template <bool Derivative>
__device__ inline void radial_hidden(const RadialWeights &w, int zs, int zr, RadialScratch &s) {
  int lane = threadIdx.x & 31;
  float a[2], da[2] = {};
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    int t = lane + 32 * q;
    a[q] = w.bias[t] + w.source[zs * 64 + t] + w.target[zr * 64 + t];
    for (int k = 0; k < 10; ++k) {
      float v = w.radial[k * 64 + t];
      a[q] += v * s.input[k];
      if constexpr (Derivative)
        da[q] += v * s.dinput[k];
    }
  }
  for (int layer = 0; layer < w.depth; ++layer) {
    if (layer) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        int t = lane + 32 * q;
        a[q] = w.affine_bias[layer][t];
        da[q] = 0;
        for (int k = 0; k < 64; ++k) {
          float v = w.weight[layer][k * 64 + t];
          a[q] += v * s.a[k];
          if constexpr (Derivative)
            da[q] += v * s.da[k];
        }
      }
    }
    float mean = warp_sum(a[0] + a[1]) / 64, dmean = 0;
    if constexpr (Derivative)
      dmean = warp_sum(da[0] + da[1]) / 64;
    a[0] -= mean;
    a[1] -= mean;
    da[0] -= dmean;
    da[1] -= dmean;
    float inv = rsqrtf(warp_sum(a[0] * a[0] + a[1] * a[1]) / 64 + 1e-5f), dinv = 0;
    if constexpr (Derivative)
      dinv = -inv * inv * inv * warp_sum(a[0] * da[0] + a[1] * da[1]) / 64;
    __syncwarp();
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      int t = lane + 32 * q;
      float u = a[q] * inv * w.gamma[layer][t] + w.beta[layer][t];
      float sig = sigmoid(u);
      s.a[t] = u * sig;
      if constexpr (Derivative)
        s.da[t] = (da[q] * inv + a[q] * dinv) * w.gamma[layer][t] * sig * (1 + u * (1 - sig));
    }
    __syncwarp();
  }
}
} // namespace mace
