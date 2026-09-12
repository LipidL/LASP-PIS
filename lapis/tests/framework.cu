#include "lj.hpp"
#include "lapis/pair.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <random>
#include <set>
#include <thread>
#include <tuple>
using namespace lapis;
void require(bool b, const char *what) {
  if (!b)
    throw Error(what);
}
using Edge = std::array<int, 5>;
struct Collect {
  static constexpr int threads = 128, tile_edges = 17, scratch_alignment = 64;
  static dim3 block_shape() { return dim3(32, 4, 1); }
  static constexpr int edge_threads = 1;
  static const char *name() { return "test.collect"; }
  struct State {};
  int *edges;
  int *count;
  int capacity;
  const int *map;
  WorkspaceSpec plan(int) const { return {0, 64, 64}; }
  void setup(Context &c, void *) { c.clear(count, 4); }
  void postprocess(Context &, void *) {}
  __device__ void prepare(int, State &, void *p) {
    if (thread_rank() == 0)
      *static_cast<int *>(p) = 1234;
  }
  template <class T> __device__ void consume(int i, const T &tile, State &, void *p) {
    int t = thread_rank();
    if (t < tile.size()) {
      int k = atomicAdd(count, 1);
      if (k < capacity) {
        auto shift = tile.shift(t);
        edges[5 * k] = map[i];
        edges[5 * k + 1] = map[tile.sender(t)];
        edges[5 * k + 2] = shift.x;
        edges[5 * k + 3] = shift.y;
        edges[5 * k + 4] = shift.z;
      }
    }
    __syncthreads();
    if (*static_cast<int *>(p) != 1234)
      asm("trap;");
  }
  __device__ void finalize(int, State &, void *) {}
};
std::vector<Edge> collect(const Structure &s, float rc, Options o) {
  Context c;
  c.prepare(s, rc, o);
  int cap = 2000000;
  int *edges = c.array<int>("test.edges", size_t(cap) * 5);
  int *count = c.array<int>("test.count", 1);
  execute(c, Collect{edges, count, cap, c.internal_to_original});
  c.synchronize();
  int n = 0;
  check(cudaMemcpy(&n, count, 4, cudaMemcpyDeviceToHost));
  require(n <= cap, "collector overflow");
  std::vector<Edge> out(n);
  if (n)
    check(cudaMemcpy(out.data(), edges, size_t(n) * sizeof(Edge), cudaMemcpyDeviceToHost));
  std::sort(out.begin(), out.end());
  return out;
}
std::vector<Edge> reference(const Structure &s, float rc, int bound = 5) {
  std::vector<Edge> out;
  for (int i = 0; i < s.size; ++i)
    for (int j = 0; j < s.size; ++j)
      for (int a = -bound; a <= bound; ++a)
        for (int b = -bound; b <= bound; ++b)
          for (int c = -bound; c <= bound; ++c) {
            if (i == j && a == 0 && b == 0 && c == 0)
              continue;
            double r2 = 0;
            for (int k = 0; k < 3; ++k) {
              double d = double(s.positions[3 * j + k]) - s.positions[3 * i + k] + a * s.cell[k] +
                         b * s.cell[3 + k] + c * s.cell[6 + k];
              r2 += d * d;
            }
            if (r2 <= double(rc) * rc)
              out.push_back({i, j, a, b, c});
          }
  std::sort(out.begin(), out.end());
  return out;
}
struct Excessive : Collect {
  static const char *name() { return "test.excessive"; }
  WorkspaceSpec plan(int) const { return {0, 1024 * 1024, 64}; }
};
class HeldCalculator : public Calculator {
  std::promise<void> &entered_;
  std::shared_future<void> release_;
  Result evaluate() override {
    entered_.set_value();
    release_.wait();
    return {};
  }

public:
  HeldCalculator(std::promise<void> &e, std::shared_future<void> r)
      : Calculator(2.5), entered_(e), release_(r) {}
};
int main() {
  try {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u(.01f, .99f);
    for (int scenario = 0; scenario < 8; ++scenario) {
      int n = scenario == 7 ? 0 : scenario == 6 ? 145 : 11;
      std::vector<float> x(n * 3);
      std::vector<int> z(n, 18);
      Structure s;
      s.size = n;
      s.elements = z.data();
      s.positions = x.data();
      float h = scenario == 0 ? 1.6f : scenario == 1 ? 2.7f : scenario == 2 ? 5.3f : 9.1f;
      s.cell = {h,
                0,
                0,
                scenario == 3 ? 2.f : 0.f,
                h,
                0,
                scenario == 3 ? -1.f : 0.f,
                scenario == 3 ? 1.f : 0.f,
                h};
      if (scenario == 5)
        s.cell = {200, 0, 0, 0, 210, 0, 0, 0, 220};
      for (int i = 0; i < n; ++i) {
        float f[3] = {u(rng), u(rng), u(rng)};
        if (scenario == 6)
          for (float &v : f)
            v *= .1f;
        if (scenario == 4)
          f[0] += i % 3 - 1;
        for (int k = 0; k < 3; ++k)
          x[3 * i + k] = f[0] * s.cell[k] + f[1] * s.cell[3 + k] + f[2] * s.cell[6 + k];
      }
      Options brute;
      brute.backend = NeighborBackend::BruteForce;
      auto got = collect(s, 2.5, brute), ref = reference(s, 2.5);
      require(got == ref, "brute edge multiset mismatch");
      require(std::adjacent_find(got.begin(), got.end()) == got.end(), "duplicate edges");
      Options automatic;
      if (scenario == 5)
        automatic.grid_byte_budget = 1024;
      require(collect(s, 2.5, automatic) == ref, "auto edge multiset mismatch");
      if (h >= 2.5 && scenario != 0) {
        Options grid = automatic;
        grid.backend = NeighborBackend::CellList;
        require(collect(s, 2.5, grid) == ref, "grid edge multiset mismatch");
      }
      std::cout << "neighbor scenario " << scenario << " passed (" << got.size() << " edges)\n";
    }
    float xyz[6] = {.2f, .3f, .4f, 1.4f, .3f, .4f};
    int zs[2] = {18, 18};
    Structure s;
    s.size = 2;
    s.positions = xyz;
    s.elements = zs;
    s.cell = {8, 0, 0, 0, 8, 0, 0, 0, 8};
    LennardJones lj;
    auto a = lj.calculate(s);
    float step = .001f;
    xyz[3] += step;
    double ep = lj.calculate(s).energy;
    xyz[3] -= 2 * step;
    double em = lj.calculate(s).energy;
    xyz[3] += step;
    xyz[3] = 1.4f;
    require(std::abs(a.forces[3] + (ep - em) / (2 * step)) < .001, "LJ finite difference mismatch");
    require(std::abs(a.forces[0] + a.forces[3]) < 1e-6, "LJ momentum mismatch");
    auto capacity = lj.diagnostics().allocated_bytes;
    auto b = lj.calculate(s);
    require(capacity == lj.diagnostics().allocated_bytes, "capacity not reused");
    require(a.energy == b.energy, "blocking repeat mismatch");
    {
      std::promise<void> entered, release;
      auto ready = entered.get_future();
      HeldCalculator held(entered, release.get_future().share());
      std::exception_ptr worker_error;
      std::thread worker([&]() {
        try {
          held.calculate(s);
        } catch (...) {
          worker_error = std::current_exception();
        }
      });
      ready.wait();
      bool rejected = false;
      try {
        held.calculate(s);
      } catch (const Error &e) {
        rejected = std::string(e.what()).find("single-flight") != std::string::npos;
      }
      release.set_value();
      worker.join();
      if (worker_error)
        std::rethrow_exception(worker_error);
      require(rejected, "concurrent call not rejected");
    }
    {
      Context c;
      c.prepare(s, 2.5, {});
      bool rejected = false;
      try {
        execute(c, Excessive{});
      } catch (const Error &) {
        rejected = true;
      }
      require(rejected, "excessive scratch accepted");
    }
    // All candidates accepted: tile boundaries, full batches and carried remainders.
    for (int count : {1, 2, 17, 18, 33, 34, 35, 128, 129, 144, 145}) {
      std::vector<float> x(count * 3, 1.f);
      std::vector<int> z(count, 1);
      for (int i = 0; i < count; ++i)
        x[3 * i] += i * .0001f;
      Structure q = s;
      q.size = count;
      q.positions = x.data();
      q.elements = z.data();
      auto edges = collect(q, .5, {});
      require(edges.size() == size_t(count) * (count - 1), "queue boundary count mismatch");
      require(std::adjacent_find(edges.begin(), edges.end()) == edges.end(), "queue duplicate");
    }
    {
      float x[6] = {0, 0, 0, 2.5, 0, 0};
      Structure q = s;
      q.positions = x;
      require(collect(q, 2.5, {}).size() == 2, "inclusive cutoff rejected");
      x[3] = std::nextafter(2.5f, 3.f);
      require(collect(q, 2.5, {}).empty(), "above cutoff accepted");
      x[3] = std::nextafter(2.5f, 2.f);
      require(collect(q, 2.5, {}).size() == 2, "below cutoff rejected");
      Options grid;
      grid.backend = NeighborBackend::CellList;
      q.cell = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      bool rejected = false;
      try {
        collect(q, 2.5, grid);
      } catch (const Error &) {
        rejected = true;
      }
      require(rejected, "invalid explicit grid accepted");
    }
    s.cell[8] = 0;
    bool threw = false;
    try {
      lj.calculate(s);
    } catch (const Error &) {
      threw = true;
    }
    require(threw, "singular cell accepted");
    std::cout << "LJ derivatives, reuse, concurrency, resource limits, queue boundaries and "
                 "geometry errors passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
