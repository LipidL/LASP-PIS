#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace lapis {
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};
inline void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw Error(cudaGetErrorString(e));
}
enum class NeighborBackend { Auto, BruteForce, CellList };
struct Options {
  NeighborBackend backend = NeighborBackend::Auto;
  size_t grid_byte_budget = 64 * 1024 * 1024;
  int grid_to_atom_budget = 8;
};
struct Structure {
  int size = 0;
  const float *positions = nullptr; // Cartesian, N x 3, Angstrom
  const int *elements = nullptr;    // atomic numbers
  std::array<float, 9> cell{};      // lattice vectors are rows
  std::array<bool, 3> pbc{true, true, true};
};
struct Result {
  double energy = 0;
  std::vector<float> energies, forces;
  std::array<float, 9> virial{}; // -dE/d(strain), eV, row major
};
struct Diagnostics {
  NeighborBackend selected = NeighborBackend::BruteForce;
  std::array<int, 3> grid{1, 1, 1};
  int max_occupancy = 0, occupied_bins = 0;
  double mean_occupancy = 0, grid_candidates = 0, brute_candidates = 0;
  size_t allocated_bytes = 0, peak_bytes = 0;
  float device_ms = 0;
  std::string reason;
};
struct KernelResources {
  std::string name;
  int registers = 0, block_threads = 0, active_blocks_per_sm = 0;
  size_t static_shared = 0, dynamic_shared = 0;
};
struct Buffer {
  void *data = nullptr;
  size_t capacity = 0;
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() {
    if (data)
      cudaFree(data);
  }
  bool reserve(size_t bytes);
  template <class T> T *as() const { return static_cast<T *>(data); }
};
struct DeviceGeometry {
  int n, grid[3], images[3];
  float h[9], cutoff2;
  const float *xyz;
  const int *elements, *bins, *offsets, *winding;
  int *status;
  NeighborBackend backend;
  const int *explicit_offsets = nullptr; // Used only by the optional benchmark build.
  const int4 *explicit_edges = nullptr;
};
struct WorkspaceSpec {
  size_t global_bytes = 0, scratch_bytes = 0, scratch_alignment = 16;
};
class Context {
  std::map<std::string, Buffer> buffers_;
  void *pinned_ = nullptr;
  size_t pinned_capacity_ = 0;

public:
  cudaStream_t stream{};
  cudaEvent_t start{}, stop{};
  DeviceGeometry geometry{};
  Diagnostics diagnostics;
  std::vector<KernelResources> resources;
  std::vector<int> original_elements;
  int *internal_to_original = nullptr, *original_to_internal = nullptr;
  Context();
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  void *reserve(const std::string &name, size_t bytes);
  template <class T> T *array(const std::string &name, size_t count) {
    if (count > SIZE_MAX / sizeof(T))
      throw Error("Workspace byte count overflow");
    return static_cast<T *>(reserve(name, count * sizeof(T)));
  }
  void clear(void *p, size_t bytes) {
    if (bytes)
      check(cudaMemsetAsync(p, 0, bytes, stream));
  }
  void prepare(const Structure &, float, const Options &);
  Result finish(float *energies, float *forces, float *node_virials);
  void synchronize();
};
class Calculator {
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;

protected:
  Context context_;
  Options options_;
  float cutoff_;
  virtual Result evaluate() = 0;

public:
  Calculator(float cutoff, Options options = {}) : options_(options), cutoff_(cutoff) {}
  virtual ~Calculator() = default;
  Result calculate(const Structure &);
  const Diagnostics &diagnostics() const { return context_.diagnostics; }
  const std::vector<KernelResources> &resources() const { return context_.resources; }
};
} // namespace lapis
