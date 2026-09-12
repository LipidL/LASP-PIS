// Optional benchmark-only materialized connectivity. No edge features are saved.
#include "lapis/pair.cuh"
#include <cub/device/device_scan.cuh>
namespace lapis {
template <bool Store> struct Connectivity {
  static constexpr int threads = 128, tile_edges = 32, scratch_alignment = 16;
  static const char *name() { return Store ? "reference.store_edges" : "reference.count_edges"; }
  struct State {
    int count = 0;
  };
  DeviceGeometry g;
  int *counts;
  const int *offsets;
  int4 *edges;
  WorkspaceSpec plan(int) const { return {}; }
  void setup(Context &, void *) {}
  void postprocess(Context &, void *) {}
  __device__ void prepare(int, State &, void *) {}
  template <class Tile> __device__ void consume(int i, const Tile &tile, State &s, void *) {
    if constexpr (Store) {
      int t = threadIdx.x;
      if (t < tile.size()) {
        int j = tile.sender(t);
        int3 shift = tile.shift(t);
        edges[offsets[i] + s.count + t] =
            make_int4(j, shift.x - g.winding[3 * i] + g.winding[3 * j],
                      shift.y - g.winding[3 * i + 1] + g.winding[3 * j + 1],
                      shift.z - g.winding[3 * i + 2] + g.winding[3 * j + 2]);
      }
    }
    s.count += tile.size();
  }
  __device__ void finalize(int i, State &s, void *) {
    if (threadIdx.x == 0)
      counts[i] = s.count;
  }
};
void build_explicit_reference(Context &c) {
  int n = c.geometry.n;
  int *count = c.array<int>("explicit.count", size_t(n) + 1);
  int *offsets = c.array<int>("explicit.offset", size_t(n) + 1);
  c.clear(count, size_t(n + 1) * 4);
  execute(c, Connectivity<false>{c.geometry, count, nullptr, nullptr});
  size_t bytes = 0;
  check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, count, offsets, n + 1, c.stream));
  void *scratch = c.reserve("explicit.scan", bytes);
  check(cub::DeviceScan::ExclusiveSum(scratch, bytes, count, offsets, n + 1, c.stream));
  int total = 0;
  check(cudaMemcpyAsync(&total, offsets + n, 4, cudaMemcpyDeviceToHost, c.stream));
  c.synchronize();
  if (total < 0)
    throw Error("Explicit reference exceeds int32 edge capacity");
  int4 *edges = c.array<int4>("explicit.edges", total);
  execute(c, Connectivity<true>{c.geometry, count, offsets, edges});
  c.geometry.explicit_offsets = offsets;
  c.geometry.explicit_edges = edges;
}
} // namespace lapis
