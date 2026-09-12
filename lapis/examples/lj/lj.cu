#include "lj.hpp"
#include "lapis/pair.cuh"
#include <cmath>
namespace lapis {
struct LJOp {
  static constexpr int threads = 128, tile_edges = 32, scratch_alignment = 16;
  static const char *name() { return "LennardJones"; }
  struct State {
    float e = 0, f[3] = {}, v[9] = {};
  };
  float epsilon, sigma, rc, uc, duc;
  float *energy, *force, *virial;
  int *status;
  WorkspaceSpec plan(int) const { return {0, threads * 13 * sizeof(float), 16}; }
  void setup(Context &, void *) {}
  void postprocess(Context &, void *) {}
  __device__ void prepare(int, State &, void *) {}
  template <class Tile> __device__ void consume(int, const Tile &tile, State &s, void *) {
    int e = threadIdx.x;
    if (e >= tile.size())
      return;
    float r2 = tile.squared_distance(e);
    if (r2 <= 0) {
      atomicCAS(status, 0, 1);
      return;
    }
    float r = sqrtf(r2), a = sigma * sigma / r2, a6 = a * a * a;
    // Force-shifted LJ: energy and radial first derivative vanish at rc.
    float u = 4 * epsilon * (a6 * a6 - a6) - uc - (r - rc) * duc;
    float du = 24 * epsilon * (a6 - 2 * a6 * a6) / r - duc;
    float3 d = tile.displacement(e);
    float x[3] = {d.x, d.y, d.z};
    s.e += .5f * u;
    for (int k = 0; k < 3; ++k)
      s.f[k] += du * x[k] / r;
    for (int k = 0; k < 3; ++k)
      for (int l = 0; l < 3; ++l)
        s.v[3 * k + l] -= .5f * du * x[k] * x[l] / r;
  }
  __device__ void finalize(int i, State &s, void *scratch) {
    float *p = static_cast<float *>(scratch);
    int t = threadIdx.x;
    p[t] = s.e;
    for (int k = 0; k < 3; ++k)
      p[(k + 1) * threads + t] = s.f[k];
    for (int k = 0; k < 9; ++k)
      p[(k + 4) * threads + t] = s.v[k];
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
      if (t < stride)
        for (int k = 0; k < 13; ++k)
          p[k * threads + t] += p[k * threads + t + stride];
      __syncthreads();
    }
    if (t == 0) {
      energy[i] = p[0];
      for (int k = 0; k < 3; ++k)
        force[3 * i + k] = p[(k + 1) * threads];
      for (int k = 0; k < 9; ++k)
        virial[9 * i + k] = p[(k + 4) * threads];
    }
  }
};
LennardJones::LennardJones(float e, float s, float c, Options o)
    : Calculator(c, o), epsilon_(e), sigma_(s) {
  if (!std::isfinite(e) || e <= 0 || !std::isfinite(s) || s <= 0 || !std::isfinite(c) || c <= 0)
    throw Error("Invalid LJ parameters");
}
Result LennardJones::evaluate() {
  int n = context_.geometry.n;
  auto e = context_.array<float>("lj.e", n);
  auto f = context_.array<float>("lj.f", 3 * size_t(n));
  auto v = context_.array<float>("lj.v", 9 * size_t(n));
  float a = std::pow(sigma_ / cutoff_, 6), uc = 4 * epsilon_ * (a * a - a),
        duc = 24 * epsilon_ * (a - 2 * a * a) / cutoff_;
  execute(context_, LJOp{epsilon_, sigma_, cutoff_, uc, duc, e, f, v, context_.geometry.status});
  return context_.finish(e, f, v);
}
} // namespace lapis
