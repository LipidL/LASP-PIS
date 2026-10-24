#include "mh1_math.cuh"
#include "mace/mh1.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <fstream>
#include <sstream>
#include "weights.hpp"
#include <numeric>

namespace mace {
using namespace lapis;
namespace {
void blas_check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw Error("cuBLAS failure " + std::to_string(int(s)));
}
struct Tensor {
  std::vector<char> host;
  Buffer device;
};
struct LinearPath {
  int in_offset, out_offset, in_mul, out_mul, dim, weight_offset;
};
struct Linear {
  int input, output;
  std::vector<LinearPath> paths;
  const float *weight;
};
struct TPEntry {
  int path, input, sh, output;
};
struct PolyTerm {
  int output, degree, a, b, c;
};
struct Product {
  const PolyTerm *terms;
  const float *coefficients;
  int count, outputs;
  Linear linear;
};
struct Layer {
  Linear up, res, one, two, skip;
  Product product;
  RadialWeights radial, density;
  float alpha, beta;
  const TPEntry *tp;
  const float *cg;
  int terms, paths, in_comp, out_comp;
};
// Opt-in stage timing when a system profiler cannot collect GPU kernel activity.
// Event creation is completely absent from ordinary calculator calls.
struct StageTimer {
  cudaStream_t stream;
  bool enabled = std::getenv("MACE_PROFILE_MH1") != nullptr;
  std::vector<std::pair<std::string, cudaEvent_t>> events;
  explicit StageTimer(cudaStream_t s) : stream(s) { mark("start"); }
  void mark(const std::string &name) {
    if (!enabled)
      return;
    cudaEvent_t event;
    check(cudaEventCreate(&event));
    events.emplace_back(name, event);
    check(cudaEventRecord(event, stream));
  }
  void report() {
    if (!enabled)
      return;
    check(cudaEventSynchronize(events.back().second));
    for (size_t i = 1; i < events.size(); ++i) {
      float ms;
      check(cudaEventElapsedTime(&ms, events[i - 1].second, events[i].second));
      std::fprintf(stderr, "MH1 %s: %.4f ms\n", events[i].first.c_str(), ms);
    }
  }
  ~StageTimer() {
    for (auto &event : events)
      cudaEventDestroy(event.second);
  }
};
// Matrices use [irrep component][node][channel]. This makes each component a
// column-major channel x node matrix and every equivariant map a batched GEMM.
void linear(Context &ctx, cublasHandle_t handle, const Linear &l, const float *x, float *y,
            bool backward = false, bool add = false) {
  int n = ctx.geometry.n;
  if (!n)
    return;
  if (!add)
    ctx.clear(y, size_t(n) * (backward ? l.input : l.output) * 4);
  const float one = 1;
  for (const auto &p : l.paths) {
    const float *w = l.weight + p.weight_offset;
    if (!backward)
      blas_check(cublasSgemmStridedBatched(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, p.out_mul, n, p.in_mul, &one, w, p.out_mul, 0,
          x + size_t(p.in_offset) * n, p.in_mul, size_t(n) * p.in_mul, &one,
          y + size_t(p.out_offset) * n, p.out_mul, size_t(n) * p.out_mul, p.dim));
    else
      blas_check(cublasSgemmStridedBatched(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, p.in_mul, n, p.out_mul, &one, w, p.out_mul, 0,
          x + size_t(p.out_offset) * n, p.out_mul, size_t(n) * p.out_mul, &one,
          y + size_t(p.in_offset) * n, p.in_mul, size_t(n) * p.in_mul, p.dim));
  }
}
__global__ void embed_kernel(DeviceGeometry g, const int *lookup, const float *embedding,
                             const float *e0, float shift, float *x, float *e) {
  int i = blockIdx.x, c = threadIdx.x;
  int z = lookup[g.elements[i]];
  if (z < 0) {
    if (c == 0)
      atomicCAS(g.status, 0, 2);
    z = 0;
  }
  for (int k = c; k < 512; k += blockDim.x)
    x[i * 512 + k] = embedding[z * 512 + k];
  if (c == 0)
    e[i] = shift;
}
__global__ void normalize_kernel(int n, const float *raw, const float *res, const float *density,
                                 float alpha, float beta, float *pre) {
  size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= size_t(n) * 9728)
    return;
  int i = k < size_t(n) * 2048 ? int(k / 2048) : int(((k - size_t(n) * 2048) / 512) % n);
  pre[k] = raw[k] / (alpha + beta * density[i]) + res[k];
}
__device__ size_t gate_index(int n, int i, int c, int comp) {
  return comp == 0 ? size_t(i) * 2048 + c : size_t(n) * 2048 + size_t((comp - 1) * n + i) * 512 + c;
}
template <bool Backward>
__global__ void gate_kernel(int n, const float *pre, const float *grad, float *output,
                            float scalar_scale, float gate_scale) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n * 512)
    return;
  int i = q / 512, c = q % 512;
  float u = pre[i * 2048 + c], sig = sigmoid(u);
  if constexpr (!Backward)
    output[q] = scalar_scale * u * sig;
  else
    output[i * 2048 + c] = grad[q] * scalar_scale * sig * (1 + u * (1 - sig));
  for (int l = 1; l <= 3; ++l) {
    int gi = i * 2048 + l * 512 + c;
    float g = sigmoid(pre[gi]), dg = 0;
    for (int m = l * l; m < (l + 1) * (l + 1); ++m) {
      size_t src = gate_index(n, i, c, m), dst = size_t(m) * n * 512 + q;
      if constexpr (!Backward)
        output[dst] = pre[src] * gate_scale * g;
      else {
        output[src] = grad[dst] * gate_scale * g;
        dg += grad[dst] * pre[src] * gate_scale * g * (1 - g);
      }
    }
    if constexpr (Backward)
      output[gi] = dg;
  }
}
__global__ void norm_backward(int n, const float *pre_grad, const float *raw, const float *density,
                              float alpha, float beta, float *raw_grad, float *density_grad) {
  int i = blockIdx.x, t = threadIdx.x;
  __shared__ float sum[128];
  float den = alpha + beta * density[i], s = 0;
  for (int k = t; k < 9728; k += 128) {
    size_t idx =
        k < 2048 ? size_t(i) * 2048 + k
                 : size_t(n) * 2048 + size_t(((k - 2048) / 512) * n + i) * 512 + (k - 2048) % 512;
    float g = pre_grad[idx], value = raw[idx];
    raw_grad[idx] = g / den;
    s -= g * value * beta / (den * den);
  }
  float total = block_sum128(s, sum);
  if (t == 0)
    density_grad[i] = total;
}
template <bool Backward>
__global__ void product_kernel(int n, Product p, const float *x, const float *grad, float *y) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n * 512)
    return;
  int c = q % 512;
  // Every lane owns one channel. Indexed component accesses stay in shared
  // memory instead of spilling dynamic-index local arrays to device memory.
  __shared__ float v[16][128], accum[Backward ? 16 : 4][128];
  int lane = threadIdx.x;
  for (int m = 0; m < 16; ++m)
    v[m][lane] = x[size_t(m) * n * 512 + q];
  for (int m = 0; m < (Backward ? 16 : 4); ++m)
    accum[m][lane] = 0;
  for (int k = 0; k < p.count; ++k) {
    PolyTerm t = p.terms[k];
    float w = p.coefficients[k * 512 + c];
    float a = v[t.a][lane], b = t.degree >= 2 ? v[t.b][lane] : 1,
          d = t.degree >= 3 ? v[t.c][lane] : 1;
    if constexpr (!Backward)
      accum[t.output][lane] += w * a * b * d;
    else {
      float g = w * grad[size_t(t.output) * n * 512 + q];
      accum[t.a][lane] += g * b * d;
      if (t.degree >= 2)
        accum[t.b][lane] += g * a * d;
      if (t.degree >= 3)
        accum[t.c][lane] += g * a * b;
    }
  }
  if constexpr (!Backward) {
    for (int m = 0; m < p.outputs; ++m)
      y[size_t(m) * n * 512 + q] = accum[m][lane];
  } else {
    for (int m = 0; m < 16; ++m)
      y[size_t(m) * n * 512 + q] = accum[m][lane];
  }
}
__global__ void add_kernel(size_t count, const float *a, float *b, float scale) {
  size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < count)
    b[k] += scale * a[k];
}
__global__ void fill_kernel(size_t count, float *a, float value) {
  size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < count)
    a[k] = value;
}
template <bool Backward>
__global__ void read_act(int count, const float *x, const float *grad, float *y, float scale) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < count) {
    float sig = sigmoid(x[k]);
    if constexpr (Backward)
      y[k] = grad[k] * scale * sig * (1 + x[k] * (1 - sig));
    else
      y[k] = scale * x[k] * sig;
  }
}
template <bool Backward, int Weights> struct PairScratch {
  // Radial intermediates are dead before tensor-product adjoints start.
  union {
    RadialScratch radial[4];
    float up_grad[4 * 128];
  };
  D3 sh[4][16];
  float weights[4][Weights], derivatives[Backward ? 4 * Weights : 1];
  float reduce[12], cut[4], dcut[4], density[4], ddensity[4];
};
template <int LayerIndex, bool Backward> struct MH1Pair {
  static constexpr int threads = 128, tile_edges = 4, scratch_alignment = 16;
  static const char *name() {
    if constexpr (Backward)
      return LayerIndex == 0 ? "MH1.backward0" : "MH1.backward1";
    else
      return LayerIndex == 0 ? "MH1.forward0" : "MH1.forward1";
  }
  static constexpr int NC = LayerIndex == 0 ? 16 : 40;
  static constexpr int NW = LayerIndex == 0 ? 512 : 1280;
  using Scratch = PairScratch<Backward, NW>;
  struct State {
    float density = 0, force[3] = {}, virial[9] = {};
  };
  DeviceGeometry g;
  Layer layer;
  const int *lookup;
  const float *constants, *radii, *bessel, *up, *grad, *density_grad;
  float *message, *density_out, *up_grad, *force, *virial;
  WorkspaceSpec plan(int) const {
    return {0, sizeof(Scratch) + (Backward ? 0 : NC * 128 * sizeof(float)), 16};
  }
  void setup(Context &, void *) {}
  void postprocess(Context &, void *) {}
  __device__ void prepare(int, State &, void *scratch) {
    if constexpr (!Backward) {
      float *accum = reinterpret_cast<float *>(static_cast<Scratch *>(scratch) + 1);
      for (int k = 0; k < NC; ++k)
        accum[k * 128 + threadIdx.x] = 0;
    }
  }
  template <class Tile>
  __device__ void consume(int i, const Tile &tile, State &state, void *scratch) {
    auto &s = *static_cast<Scratch *>(scratch);
    int c = threadIdx.x, worker = c / 32, lane = c % 32, zs = 0, zr = 0;
    if (worker < tile.size()) {
      int j = tile.sender(worker);
      zs = lookup[g.elements[j]];
      zr = lookup[g.elements[i]];
      if (zs < 0 || zr < 0) {
        if (lane == 0)
          atomicCAS(g.status, 0, 2);
        zs = zr = 0;
      }
      float r = sqrtf(tile.squared_distance(worker));
      float3 d = tile.displacement(worker);
      if (r <= 0) {
        if (lane == 0)
          atomicCAS(g.status, 0, 1);
        r = 1e-10f;
      }
      if (lane == 0) {
        D1 distance(r, 1),
            ratio = distance / D1(.5f * (radii[g.elements[j]] + radii[g.elements[i]]));
        D1 transformed = D1(1) / (D1(1) + D1(constants[3]) * power(ratio, constants[4]) /
                                              (D1(1) + power(ratio, constants[4] - constants[5])));
        D1 cut = envelope(distance, constants[0], constants[2]);
        s.cut[worker] = cut.v;
        s.dcut[worker] = cut.d;
        for (int k = 0; k < 10; ++k) {
          D1 b = D1(constants[1]) * sine(D1(bessel[k]) * transformed) / transformed;
          s.radial[worker].input[k] = b.v;
          s.radial[worker].dinput[k] = b.d;
        }
        // Official MACE uses the opposite angular vector orientation.
        harmonics(make_float3(-d.x, -d.y, -d.z), r, s.sh[worker]);
        for (int k = 0; k < 16; ++k) {
          s.sh[worker][k].x = -s.sh[worker][k].x;
          s.sh[worker][k].y = -s.sh[worker][k].y;
          s.sh[worker][k].z = -s.sh[worker][k].z;
        }
      }
      __syncwarp();
      radial_hidden<Backward>(layer.radial, zs, zr, s.radial[worker]);
    } else {
      for (int k = lane; k < 64; k += 32) {
        s.radial[worker].a[k] = 0;
        s.radial[worker].da[k] = 0;
      }
      if (lane == 0) {
        s.cut[worker] = 0;
        s.dcut[worker] = 0;
      }
    }
    __syncthreads();
    // One projection-weight load serves all four transient edges.
    for (int p = 0; p < layer.paths; ++p) {
      int k = p * 128 + c;
      float w[4], dw[4] = {};
#pragma unroll
      for (int e = 0; e < 4; ++e)
        w[e] = layer.radial.output_bias[k];
      for (int a = 0; a < 64; ++a) {
        float v = layer.radial.output_weight[a * layer.radial.outputs + k];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w[e] += v * s.radial[e].a[a];
          if constexpr (Backward)
            dw[e] += v * s.radial[e].da[a];
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        s.weights[e][k] = w[e] * s.cut[e];
        if constexpr (Backward)
          s.derivatives[e * NW + k] = dw[e] * s.cut[e] + w[e] * s.dcut[e];
      }
    }
    __syncthreads();
    if (worker < tile.size()) {
      radial_hidden<Backward>(layer.density, zs, zr, s.radial[worker]);
      if (lane == 0) {
        float a = layer.density.output_bias[0], da = 0;
        for (int k = 0; k < 64; ++k) {
          a += layer.density.output_weight[k] * s.radial[worker].a[k];
          if constexpr (Backward)
            da += layer.density.output_weight[k] * s.radial[worker].da[k];
        }
        float den = tanhf(a * a);
        s.density[worker] = den * s.cut[worker];
        s.ddensity[worker] = 2 * a * da * (1 - den * den) * s.cut[worker] + den * s.dcut[worker];
      }
    }
    __syncthreads();
    for (int edge = 0; edge < tile.size(); ++edge) {
      int j = tile.sender(edge);
      float r = fmaxf(sqrtf(tile.squared_distance(edge)), 1e-10f);
      float3 d = tile.displacement(edge);
      float u0 = up[size_t(j) * 128 + c], u1 = 0, u2 = 0, u3 = 0;
      if constexpr (LayerIndex > 0) {
        u1 = up[(size_t(g.n) + j) * 128 + c];
        u2 = up[(size_t(2) * g.n + j) * 128 + c];
        u3 = up[(size_t(3) * g.n + j) * 128 + c];
      }
      if constexpr (!Backward) {
        float *accum = reinterpret_cast<float *>(&s + 1);
        for (int k = 0; k < layer.terms; ++k) {
          auto t = layer.tp[k];
          float v = t.input == 0 ? u0 : t.input == 1 ? u1 : t.input == 2 ? u2 : u3;
          accum[t.output * 128 + c] +=
              layer.cg[k] * v * s.sh[edge][t.sh].v * s.weights[edge][t.path * 128 + c];
        }
        if (c == 0)
          state.density += s.density[edge];
      } else {
        float gx = 0, gy = 0, gz = 0, dr = 0;
        if constexpr (LayerIndex > 0)
          for (int k = 0; k < 4; ++k)
            s.up_grad[k * 128 + c] = 0;
        for (int k = 0; k < layer.terms; ++k) {
          auto t = layer.tp[k];
          float a = grad[(size_t(t.output) * g.n + i) * 128 + c] * layer.cg[k];
          float v = t.input == 0   ? u0
                    : t.input == 1 ? u1
                    : t.input == 2 ? u2
                                   : u3,
                w = s.weights[edge][t.path * 128 + c];
          D3 sh = s.sh[edge][t.sh];
          if constexpr (LayerIndex > 0)
            s.up_grad[t.input * 128 + c] += a * sh.v * w;
          dr += a * v * sh.v * s.derivatives[edge * NW + t.path * 128 + c];
          gx += a * v * w * sh.x;
          gy += a * v * w * sh.y;
          gz += a * v * w * sh.z;
        }
        if constexpr (LayerIndex > 0)
          for (int k = 0; k < 4; ++k)
            atomicAdd(up_grad + (size_t(k) * g.n + j) * 128 + c, s.up_grad[k * 128 + c]);
        if (c == 0)
          dr += density_grad[i] * s.ddensity[edge];
        gx += dr * d.x / r;
        gy += dr * d.y / r;
        gz += dr * d.z / r;
        // Reduce all three Cartesian components with one block rendezvous.
        for (int step = 16; step; step /= 2) {
          gx += __shfl_down_sync(0xffffffff, gx, step);
          gy += __shfl_down_sync(0xffffffff, gy, step);
          gz += __shfl_down_sync(0xffffffff, gz, step);
        }
        if (lane == 0) {
          s.reduce[worker] = gx;
          s.reduce[worker + 4] = gy;
          s.reduce[worker + 8] = gz;
        }
        __syncthreads();
        if (worker == 0) {
          gx = lane < 4 ? s.reduce[lane] : 0;
          gy = lane < 4 ? s.reduce[lane + 4] : 0;
          gz = lane < 4 ? s.reduce[lane + 8] : 0;
          for (int step = 16; step; step /= 2) {
            gx += __shfl_down_sync(0xffffffff, gx, step);
            gy += __shfl_down_sync(0xffffffff, gy, step);
            gz += __shfl_down_sync(0xffffffff, gz, step);
          }
        }
        if (c == 0) {
          float f[3] = {gx, gy, gz};
          for (int axis = 0; axis < 3; ++axis) {
            atomicAdd(force + 3 * j + axis, -f[axis]);
            state.force[axis] += f[axis];
            state.virial[axis] -= d.x * f[axis];
            state.virial[3 + axis] -= d.y * f[axis];
            state.virial[6 + axis] -= d.z * f[axis];
          }
        }
        __syncthreads();
      }
    }
  }
  __device__ void finalize(int i, State &s, void *scratch) {
    int c = threadIdx.x;
    if constexpr (!Backward) {
      float *accum = reinterpret_cast<float *>(static_cast<Scratch *>(scratch) + 1);
      for (int k = 0; k < NC; ++k)
        message[(size_t(k) * g.n + i) * 128 + c] = accum[k * 128 + c];
      if (c == 0)
        density_out[i] = s.density;
    } else if (c == 0) {
      for (int k = 0; k < 3; ++k)
        atomicAdd(force + 3 * i + k, s.force[k]);
      for (int k = 0; k < 9; ++k)
        virial[9 * i + k] += s.virial[k];
    }
  }
};
struct ZBLOp {
  static constexpr int threads = 128, tile_edges = 32, scratch_alignment = 16;
  static const char *name() { return "MH1.ZBL"; }
  struct State {
    float e = 0, f[3] = {}, v[9] = {};
  };
  DeviceGeometry g;
  const float *constants, *radii, *coeff;
  float *energy, *force, *virial;
  WorkspaceSpec plan(int) const { return {0, 128 * 13 * 4, 16}; }
  void setup(Context &, void *) {}
  void postprocess(Context &, void *) {}
  __device__ void prepare(int, State &, void *) {}
  template <class Tile> __device__ void consume(int i, const Tile &tile, State &s, void *) {
    int k = threadIdx.x;
    if (k >= tile.size())
      return;
    int j = tile.sender(k);
    float zi = g.elements[i], zj = g.elements[j];
    float r = sqrtf(tile.squared_distance(k));
    if (r <= 0) {
      atomicCAS(g.status, 0, 1);
      return;
    }
    float rc = radii[int(zi)] + radii[int(zj)];
    if (r >= rc)
      return;
    D1 x(r, 1);
    float a = constants[9] * .529f / (powf(zi, constants[8]) + powf(zj, constants[8]));
    float rates[4] = {3.2f, .9423f, .4028f, .2016f};
    D1 phi;
    for (int t = 0; t < 4; ++t)
      phi = phi + D1(coeff[t]) * exponential(D1(-rates[t] / a) * x);
    D1 e = D1(.5f * 14.3996f * zi * zj * constants[6]) * phi / x * envelope(x, rc, constants[10]);
    s.e += e.v;
    float3 d = tile.displacement(k);
    float v[3] = {d.x, d.y, d.z};
    for (int t = 0; t < 3; ++t)
      s.f[t] += 2 * e.d * v[t] / r;
    for (int t = 0; t < 3; ++t)
      for (int u = 0; u < 3; ++u)
        s.v[3 * t + u] -= e.d * v[t] * v[u] / r;
  }
  __device__ void finalize(int i, State &s, void *scratch) {
    auto p = static_cast<float *>(scratch);
    int t = threadIdx.x;
    p[t] = s.e;
    for (int k = 0; k < 3; ++k)
      p[(1 + k) * 128 + t] = s.f[k];
    for (int k = 0; k < 9; ++k)
      p[(4 + k) * 128 + t] = s.v[k];
    __syncthreads();
    for (int step = 64; step; step /= 2) {
      if (t < step)
        for (int k = 0; k < 13; ++k)
          p[k * 128 + t] += p[k * 128 + t + step];
      __syncthreads();
    }
    if (t == 0) {
      energy[i] += p[0];
      for (int k = 0; k < 3; ++k)
        force[3 * i + k] = p[(1 + k) * 128];
      for (int k = 0; k < 9; ++k)
        virial[9 * i + k] = p[(4 + k) * 128];
    }
  }
};
} // anonymous namespace

struct MH1::Model {
  std::map<std::string, Tensor> tensors;
  Buffer lookup_buffer;
  Layer layers[2];
  Linear read0, read1up, read1down;
  cublasHandle_t blas{};
  size_t bytes = 0;
  float constants[14];
  float atomic_energy[119]{};
  int species = 0;
  void require_size(const std::string &key, size_t count, size_t item = sizeof(float)) {
    auto it = tensors.find(key);
    if (it == tensors.end() || it->second.host.size() != count * item)
      throw Error("Invalid MH1 tensor size: " + key);
  }
  const float *f(const std::string &key) {
    auto it = tensors.find(key);
    if (it == tensors.end())
      throw Error("Missing MH1 tensor " + key);
    return it->second.device.as<float>();
  }
  template <class T> std::vector<T> host(const std::string &key) {
    auto it = tensors.find(key);
    if (it == tensors.end() || it->second.host.size() % sizeof(T))
      throw Error("Invalid MH1 tensor " + key);
    auto &h = it->second.host;
    std::vector<T> out(h.size() / sizeof(T));
    memcpy(out.data(), h.data(), h.size());
    return out;
  }
  void transpose(const std::string &key, int rows, int cols) {
    auto a = host<float>(key);
    if (a.size() != size_t(rows) * cols)
      throw Error("Invalid radial matrix dimensions: " + key);
    std::vector<float> b(a.size());
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        b[j * rows + i] = a[i * cols + j];
    check(cudaMemcpy(tensors.at(key).device.data, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  }
  Linear get_linear(const std::string &key, int input, int output) {
    auto dims = host<int>(key + ".dims");
    if (dims.size() != 2 || dims[0] != input || dims[1] != output)
      throw Error("Unsupported MH1 linear dimensions: " + key);
    auto paths = host<LinearPath>(key + ".paths");
    size_t weights = host<float>(key + ".weight").size();
    for (auto p : paths)
      if (p.in_offset < 0 || p.out_offset < 0 || p.in_mul <= 0 || p.out_mul <= 0 ||
          p.in_mul > 2048 || p.out_mul > 2048 || p.dim < 1 || p.dim > 7 || p.weight_offset < 0 ||
          int64_t(p.in_offset) + int64_t(p.in_mul) * p.dim > dims[0] ||
          int64_t(p.out_offset) + int64_t(p.out_mul) * p.dim > dims[1] ||
          size_t(p.weight_offset) + size_t(p.in_mul) * p.out_mul > weights)
        throw Error("Invalid linear path: " + key);
    return {dims[0], dims[1], paths, f(key + ".weight")};
  }
  RadialWeights get_radial(const std::string &key, int depth, int outputs) {
    require_size(key + ".source", size_t(species) * 64);
    require_size(key + ".target", size_t(species) * 64);
    for (int k = 0; k < depth; ++k) {
      require_size(key + "." + std::to_string(k * 3) + ".weight", k ? 64 * 64 : 64 * 1034);
      require_size(key + "." + std::to_string(k * 3) + ".bias", 64);
      require_size(key + "." + std::to_string(k * 3 + 1) + ".weight", 64);
      require_size(key + "." + std::to_string(k * 3 + 1) + ".bias", 64);
    }
    require_size(key + "." + std::to_string(depth * 3) + ".bias", outputs);
    transpose(key + ".radial", 64, 10);
    for (int k = 1; k < depth; ++k)
      transpose(key + "." + std::to_string(k * 3) + ".weight", 64, 64);
    transpose(key + "." + std::to_string(depth * 3) + ".weight", outputs, 64);
    RadialWeights r{};
    r.source = f(key + ".source");
    r.target = f(key + ".target");
    r.radial = f(key + ".radial");
    r.bias = f(key + ".0.bias");
    r.depth = depth;
    r.outputs = outputs;
    for (int k = 0; k < depth; ++k) {
      r.weight[k] = f(key + "." + std::to_string(k * 3) + ".weight");
      r.affine_bias[k] = f(key + "." + std::to_string(k * 3) + ".bias");
      r.gamma[k] = f(key + "." + std::to_string(k * 3 + 1) + ".weight");
      r.beta[k] = f(key + "." + std::to_string(k * 3 + 1) + ".bias");
    }
    r.output_weight = f(key + "." + std::to_string(depth * 3) + ".weight");
    r.output_bias = f(key + "." + std::to_string(depth * 3) + ".bias");
    return r;
  }
  Model(const std::string &path, cudaStream_t stream, const std::string &head) {
    std::istringstream in(mace_detail::weight_data(path, head), std::ios::binary|std::ios::in);
    if (!in)
      throw Error("Cannot open MH1 archive: " + path);
    char magic[8]{};
    in.read(magic, 8);
    if (!in || memcmp(magic, "PWMH1001", 8))
      throw Error("Invalid MH1 archive magic");
    uint32_t count = 0;
    in.read(reinterpret_cast<char *>(&count), 4);
    if (!in || count == 0 || count > 10000)
      throw Error("Invalid MH1 tensor count");
    for (uint32_t k = 0; k < count; ++k) {
      uint32_t len = 0;
      uint64_t size = 0;
      in.read(reinterpret_cast<char *>(&len), 4);
      in.read(reinterpret_cast<char *>(&size), 8);
      if (!in || len > 1024 || !len || size > 512 * 1024 * 1024)
        throw Error("Invalid MH1 archive record");
      std::string name(len, '\0');
      in.read(name.data(), len);
      if (tensors.count(name))
        throw Error("Duplicate MH1 tensor");
      auto &t = tensors[name];
      t.host.resize(size);
      in.read(t.host.data(), size);
      if (!in)
        throw Error("Truncated MH1 archive");
      bool indices = name == "atomic_numbers" ||
                     name.compare(name.size() > 5 ? name.size() - 5 : 0, 5, ".dims") == 0 ||
                     name.compare(name.size() > 6 ? name.size() - 6 : 0, 6, ".paths") == 0 ||
                     name.compare(name.size() > 6 ? name.size() - 6 : 0, 6, ".terms") == 0;
      if (size % 4)
        throw Error("Invalid MH1 tensor byte alignment: " + name);
      if (!indices)
        for (size_t i = 0; i < size; i += 4) {
          float value;
          memcpy(&value, t.host.data() + i, 4);
          if (!std::isfinite(value))
            throw Error("Nonfinite MH1 tensor: " + name);
        }
      t.device.reserve(size);
      if (size)
        check(cudaMemcpy(t.device.data, t.host.data(), size, cudaMemcpyHostToDevice));
      bytes += size;
    }
    auto c = host<float>("constants");
    if (c.size() != 14 || c[0] != 6)
      throw Error("Unsupported MH1 constants");
    std::copy(c.begin(), c.end(), constants);
    auto zs = host<int>("atomic_numbers");
    species = int(zs.size());
    if (species < 1 || species > 118)
      throw Error("Invalid MH1 element count");
    require_size("embedding", size_t(species) * 512);
    require_size("bessel", 10);
    require_size("radii", 119);
    require_size("zbl.radii", 119);
    require_size("zbl.c", 4);
    auto e0 = host<float>("e0");
    if (e0.size() != zs.size())
      throw Error("Invalid atomic energy table");
    std::vector<int> lookup(119, -1);
    for (size_t k = 0; k < zs.size(); ++k) {
      if (zs[k] < 1 || zs[k] > 118 || lookup[zs[k]] >= 0)
        throw Error("Invalid MH1 element map");
      lookup[zs[k]] = int(k);
      atomic_energy[zs[k]] = e0[k];
    }
    lookup_buffer.reserve(119 * 4);
    check(cudaMemcpy(lookup_buffer.data, lookup.data(), 119 * 4, cudaMemcpyHostToDevice));
    bytes += 119 * 4;
    for (int k = 0; k < 2; ++k) {
      auto &l = layers[k];
      std::string p = "layer" + std::to_string(k);
      l.up = get_linear(p + ".linear_up", k ? 2048 : 512, k ? 512 : 128);
      l.res = get_linear(p + ".linear_res", k ? 512 : 128, 9728);
      l.one = get_linear(p + ".linear_1", k ? 5120 : 2048, 9728);
      l.two = get_linear(p + ".linear_2", 8192, 8192);
      l.skip = get_linear(p + ".skip_tp", k ? 2048 : 512, k ? 512 : 2048);
      auto norm = host<float>(p + ".norm");
      l.alpha = norm.at(0);
      l.beta = norm.at(1);
      auto dims = host<int>(p + ".tp.dims");
      l.paths = dims.at(0);
      l.in_comp = dims.at(1);
      l.out_comp = dims.at(2);
      if (l.paths != (k ? 10 : 4) || l.in_comp != (k ? 4 : 1) || l.out_comp != (k ? 40 : 16))
        throw Error("Unsupported MH1 tensor product shape");
      l.tp = reinterpret_cast<const TPEntry *>(f(p + ".tp.terms"));
      l.cg = f(p + ".tp.values");
      l.terms = int(host<float>(p + ".tp.values").size());
      require_size(p + ".tp.terms", l.terms, sizeof(TPEntry));
      for (auto t : host<TPEntry>(p + ".tp.terms"))
        if (t.path < 0 || t.path >= l.paths || t.input < 0 || t.input >= l.in_comp || t.sh < 0 ||
            t.sh >= 16 || t.output < 0 || t.output >= l.out_comp)
          throw Error("Invalid MH1 tensor-product index");
      l.radial = get_radial(p + ".radial", 3, l.paths * 128);
      l.density = get_radial(p + ".density", 1, 1);
      l.product = {reinterpret_cast<const PolyTerm *>(f(p + ".product.terms")),
                   f(p + ".product.coefficients"), int(host<PolyTerm>(p + ".product.terms").size()),
                   k ? 1 : 4, get_linear(p + ".product.linear", k ? 512 : 2048, k ? 512 : 2048)};
      require_size(p + ".product.coefficients", size_t(l.product.count) * 512);
      for (auto t : host<PolyTerm>(p + ".product.terms"))
        if (t.degree < 1 || t.degree > 3 || t.output < 0 || t.output >= l.product.outputs ||
            t.a < 0 || t.a >= 16 || (t.degree >= 2 && (t.b < 0 || t.b >= 16)) ||
            (t.degree == 3 && (t.c < 0 || t.c >= 16)))
          throw Error("Invalid MH1 polynomial index");
    }
    read0 = get_linear("readout0", 2048, 1);
    read1up = get_linear("readout1.up", 512, 16);
    read1down = get_linear("readout1.down", 16, 1);
    blas_check(cublasCreate(&blas));
    blas_check(cublasSetStream(blas, stream));
    blas_check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
  }
  ~Model() {
    if (blas)
      cublasDestroy(blas);
  }
};
MH1::MH1(const std::string &archive, Options o, const std::string &head)
    : Calculator(6, o), model_(std::make_unique<Model>(archive, context_.stream, head)) {}
MH1::MH1(const std::string &archive, Options o) : MH1(archive, o, {}) {}
MH1::~MH1() = default;
size_t MH1::parameter_bytes() const {
  return model_->bytes;
}
Result MH1::evaluate() {
  auto &c = context_;
  auto &m = *model_;
  int n = c.geometry.n;
  StageTimer timer(c.stream);
  if (n > INT_MAX / 9728)
    throw Error("MH1 node dimensions exceed v1 int32 capacity");
  auto array = [&](std::string name, int dim) { return c.array<float>(name, size_t(n) * dim); };
  auto launch_add = [&](int dim, const float *src, float *dst, float scale = 1) {
    if (n)
      add_kernel<<<(size_t(n) * dim + 255) / 256, 256, 0, c.stream>>>(size_t(n) * dim, src, dst,
                                                                      scale);
  };
  float *energy = array("mh.e", 1), *force = array("mh.f", 3), *virial = array("mh.v", 9),
        *embedding = array("mh.embedding", 2048);
  if (n)
    embed_kernel<<<n, 128, 0, c.stream>>>(c.geometry, m.lookup_buffer.as<int>(), m.f("embedding"),
                                          m.f("e0"), m.constants[7], embedding, energy);
  execute(c, ZBLOp{c.geometry, m.f("constants"), m.f("zbl.radii"), m.f("zbl.c"), energy, force,
                   virial});
  timer.mark("embedding/ZBL");
  struct Saved {
    float *up, *density, *raw, *pre, *product_in, *hidden;
  };
  Saved saved[2];
  // Common temporaries have stable maximum capacity before either layer runs.
  float *message = array("mh.temp.message", 5120), *res = array("mh.temp.res", 9728),
        *gate = array("mh.temp.gate", 8192), *poly = array("mh.temp.poly", 2048),
        *skip = array("mh.temp.skip", 2048), *node_e = array("mh.temp.energy", 1);
  float *input = embedding;
  for (int k = 0; k < 2; ++k) {
    auto &l = m.layers[k];
    auto &s = saved[k];
    std::string p = "mh.saved" + std::to_string(k);
    s.up = array(p + ".up", l.in_comp * 128);
    s.density = array(p + ".density", 1);
    s.raw = array(p + ".raw", 9728);
    s.pre = array(p + ".pre", 9728);
    s.product_in = array(p + ".product_in", 8192);
    s.hidden = array(p + ".hidden", k ? 512 : 2048);
    linear(c, m.blas, l.up, input, s.up);
    linear(c, m.blas, l.res, s.up, res);
    linear(c, m.blas, l.skip, input, skip);
    timer.mark("forward" + std::to_string(k) + ".up/res/skip");
    if (k == 0)
      execute(c, MH1Pair<0, false>{c.geometry, l, m.lookup_buffer.as<int>(), m.f("constants"),
                                   m.f("radii"), m.f("bessel"), s.up, nullptr, nullptr, message,
                                   s.density, nullptr, nullptr, nullptr});
    else
      execute(c, MH1Pair<1, false>{c.geometry, l, m.lookup_buffer.as<int>(), m.f("constants"),
                                   m.f("radii"), m.f("bessel"), s.up, nullptr, nullptr, message,
                                   s.density, nullptr, nullptr, nullptr});
    timer.mark("forward" + std::to_string(k) + ".pairs");
    linear(c, m.blas, l.one, message, s.raw);
    if (n) {
      normalize_kernel<<<(size_t(n) * 9728 + 255) / 256, 256, 0, c.stream>>>(
          n, s.raw, res, s.density, l.alpha, l.beta, s.pre);
      gate_kernel<false><<<(n * 512 + 255) / 256, 256, 0, c.stream>>>(
          n, s.pre, nullptr, gate, m.constants[11], m.constants[12]);
    }
    linear(c, m.blas, l.two, gate, s.product_in);
    timer.mark("forward" + std::to_string(k) + ".linear/gate");
    if (n)
      product_kernel<false>
          <<<(n * 512 + 127) / 128, 128, 0, c.stream>>>(n, l.product, s.product_in, nullptr, poly);
    timer.mark("forward" + std::to_string(k) + ".product");
    linear(c, m.blas, l.product.linear, poly, s.hidden);
    launch_add(k ? 512 : 2048, skip, s.hidden);
    input = s.hidden;
    if (k == 0) {
      linear(c, m.blas, m.read0, s.hidden, node_e);
      launch_add(1, node_e, energy, m.constants[6]);
    }
  }
  float *read_pre = array("mh.read.pre", 16), *read_hidden = array("mh.read.hidden", 16),
        *read_grad = array("mh.read.grad", 16), *read_dpre = array("mh.read.dpre", 16);
  linear(c, m.blas, m.read1up, saved[1].hidden, read_pre);
  if (n)
    read_act<false><<<(n * 16 + 255) / 256, 256, 0, c.stream>>>(n * 16, read_pre, nullptr,
                                                                read_hidden, m.constants[13]);
  linear(c, m.blas, m.read1down, read_hidden, node_e);
  launch_add(1, node_e, energy, m.constants[6]);
  // Forward temporaries are dead here. Reuse them for adjoints, and overwrite
  // saved node tensors only after their final read in the analytical backward.
  float *one = node_e, *hidden_grad = skip, *previous_grad = embedding, *product_grad = poly,
        *density_grad = array("mh.adj.density", 1), *up_grad = array("mh.adj.up", 512);
  float *gate_grad = gate, *pre_grad = res, *message_grad = message;
  if (n)
    fill_kernel<<<(n + 255) / 256, 256, 0, c.stream>>>(n, one, m.constants[6]);
  linear(c, m.blas, m.read1down, one, read_grad, true);
  if (n)
    read_act<true><<<(n * 16 + 255) / 256, 256, 0, c.stream>>>(n * 16, read_pre, read_grad,
                                                               read_dpre, m.constants[13]);
  linear(c, m.blas, m.read1up, read_dpre, hidden_grad, true);
  timer.mark("readouts");
  for (int k = 1; k >= 0; --k) {
    auto &l = m.layers[k];
    auto &s = saved[k];
    float *pin_grad = s.product_in;
    float *raw_grad = s.raw;
    if (k == 0)
      linear(c, m.blas, m.read0, one, hidden_grad, true, true);
    linear(c, m.blas, l.product.linear, hidden_grad, product_grad, true);
    timer.mark("backward" + std::to_string(k) + ".product.linear");
    if (n)
      product_kernel<true><<<(n * 512 + 127) / 128, 128, 0, c.stream>>>(n, l.product, s.product_in,
                                                                        product_grad, pin_grad);
    timer.mark("backward" + std::to_string(k) + ".product");
    linear(c, m.blas, l.two, pin_grad, gate_grad, true);
    if (n) {
      gate_kernel<true><<<(n * 512 + 255) / 256, 256, 0, c.stream>>>(
          n, s.pre, gate_grad, pre_grad, m.constants[11], m.constants[12]);
      norm_backward<<<n, 128, 0, c.stream>>>(n, pre_grad, s.raw, s.density, l.alpha, l.beta,
                                             raw_grad, density_grad);
    }
    linear(c, m.blas, l.one, raw_grad, message_grad, true);
    timer.mark("backward" + std::to_string(k) + ".linear/gate");
    if (k == 1) {
      linear(c, m.blas, l.res, pre_grad, up_grad, true);
      execute(c, MH1Pair<1, true>{c.geometry, l, m.lookup_buffer.as<int>(), m.f("constants"),
                                  m.f("radii"), m.f("bessel"), s.up, message_grad, density_grad,
                                  nullptr, nullptr, up_grad, force, virial});
      timer.mark("backward1.pairs");
      linear(c, m.blas, l.skip, hidden_grad, previous_grad, true);
      linear(c, m.blas, l.up, up_grad, previous_grad, true, true);
      std::swap(hidden_grad, previous_grad);
    } else
      execute(c, MH1Pair<0, true>{c.geometry, l, m.lookup_buffer.as<int>(), m.f("constants"),
                                  m.f("radii"), m.f("bessel"), s.up, message_grad, density_grad,
                                  nullptr, nullptr, nullptr, force, virial});
    if (k == 0)
      timer.mark("backward0.pairs");
  }
  check(cudaGetLastError());
  auto result = c.finish(energy, force, virial);
  timer.report();
  // Keep large atomic baselines separate from small interaction energies until
  // the host's double-precision output reduction. Model kernels remain float32.
  for (int i = 0; i < n; ++i) {
    float base = m.atomic_energy[c.original_elements[i]];
    result.energy += double(base);
    result.energies[i] += base;
  }
  return result;
}
} // namespace mace
