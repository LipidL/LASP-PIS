#pragma once
#include "runtime.hpp"
#include <climits>
#include <type_traits>

namespace lapis {
__device__ inline int thread_rank() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}
template <class Op, class = void> struct BlockShape {
  static constexpr bool flat = true;
  static dim3 get() { return dim3(Op::threads, 1, 1); }
};
template <class Op> struct BlockShape<Op, std::void_t<decltype(Op::block_shape())>> {
  static constexpr bool flat = false;
  static dim3 get() { return Op::block_shape(); }
};
template <class Op, class = void> struct EdgeSubgroup {
  static constexpr int size = 1;
};
template <class Op> struct EdgeSubgroup<Op, std::void_t<decltype(Op::edge_threads)>> {
  static constexpr int size = Op::edge_threads;
};
__host__ __device__ constexpr size_t align_up(size_t n, size_t a) {
  return (n + a - 1) & ~(a - 1);
}
template <int Q> struct alignas(16) EdgeQueue {
  alignas(16) int sender[Q], sx[Q], sy[Q], sz[Q];
  alignas(16) float dx[Q], dy[Q], dz[Q], r2[Q];
  __device__ void copy(int dst, int src) {
    sender[dst] = sender[src];
    sx[dst] = sx[src];
    sy[dst] = sy[src];
    sz[dst] = sz[src];
    dx[dst] = dx[src];
    dy[dst] = dy[src];
    dz[dst] = dz[src];
    r2[dst] = r2[src];
  }
};
template <int Q> struct TileView {
  const EdgeQueue<Q> *q;
  int offset, count;
  __device__ int size() const { return count; }
  __device__ int sender(int e) const { return q->sender[offset + e]; }
  __device__ int3 shift(int e) const {
    return make_int3(q->sx[offset + e], q->sy[offset + e], q->sz[offset + e]);
  }
  __device__ float3 displacement(int e) const {
    return make_float3(q->dx[offset + e], q->dy[offset + e], q->dz[offset + e]);
  }
  __device__ float squared_distance(int e) const { return q->r2[offset + e]; }
};
__device__ inline int floor_div(int a, int b) {
  int q = a / b, r = a % b;
  return q - (r < 0);
}
__device__ inline float3 displacement(const DeviceGeometry &g, int i, int j, int x, int y, int z) {
  return make_float3(g.xyz[3 * j] + x * g.h[0] + y * g.h[3] + z * g.h[6] - g.xyz[3 * i],
                     g.xyz[3 * j + 1] + x * g.h[1] + y * g.h[4] + z * g.h[7] - g.xyz[3 * i + 1],
                     g.xyz[3 * j + 2] + x * g.h[2] + y * g.h[5] + z * g.h[8] - g.xyz[3 * i + 2]);
}
template <class Op> __global__ void pair_kernel(DeviceGeometry g, Op op) {
  constexpr int B = Op::threads, TE = Op::tile_edges, Q = B + TE - 1;
  extern __shared__ __align__(256) unsigned char storage[];
  auto &queue = *reinterpret_cast<EdgeQueue<Q> *>(storage);
  void *scratch = storage + align_up(sizeof(EdgeQueue<Q>), Op::scratch_alignment);
  __shared__ int counts[B / 32], prefix[B / 32], accepted;
  int t = BlockShape<Op>::flat ? int(threadIdx.x) : thread_rank(), i = blockIdx.x, carry = 0;
  typename Op::State state;
  op.prepare(i, state, scratch);
  __syncthreads();
#ifdef LAPIS_EXPLICIT_REFERENCE
  if (g.explicit_offsets) {
    int begin = g.explicit_offsets[i], end = g.explicit_offsets[i + 1];
    for (int base = begin; base < end; base += TE) {
      int count = min(TE, end - base);
      for (int e = t; e < count; e += B) {
        int4 edge = g.explicit_edges[base + e];
        int j = edge.x;
        float3 d = displacement(g, i, j, edge.y, edge.z, edge.w);
        queue.sender[e] = j;
        queue.sx[e] = edge.y + g.winding[3 * i] - g.winding[3 * j];
        queue.sy[e] = edge.z + g.winding[3 * i + 1] - g.winding[3 * j + 1];
        queue.sz[e] = edge.w + g.winding[3 * i + 2] - g.winding[3 * j + 2];
        queue.dx[e] = d.x;
        queue.dy[e] = d.y;
        queue.dz[e] = d.z;
        queue.r2[e] = d.x * d.x + d.y * d.y + d.z * d.z;
      }
      __syncthreads();
      op.consume(i, TileView<Q>{&queue, 0, count}, state, scratch);
      __syncthreads();
    }
    op.finalize(i, state, scratch);
    return;
  }
#endif
  int bx = 0, by = 0, bz = 0;
  if (g.backend == NeighborBackend::CellList) {
    int b = g.bins[i];
    bz = b % g.grid[2];
    b /= g.grid[2];
    by = b % g.grid[1];
    bx = b / g.grid[1];
  }
  int nx = g.backend == NeighborBackend::CellList ? 1 : g.images[0];
  int ny = g.backend == NeighborBackend::CellList ? 1 : g.images[1];
  int nz = g.backend == NeighborBackend::CellList ? 1 : g.images[2];
  for (int x = -nx; x <= nx; ++x)
    for (int y = -ny; y <= ny; ++y)
      for (int z = -nz; z <= nz; ++z) {
        int sx = x, sy = y, sz = z, begin = 0, end = g.n;
        if (g.backend == NeighborBackend::CellList) {
          sx = floor_div(bx + x, g.grid[0]);
          sy = floor_div(by + y, g.grid[1]);
          sz = floor_div(bz + z, g.grid[2]);
          int bin = ((bx + x - sx * g.grid[0]) * g.grid[1] + by + y - sy * g.grid[1]) * g.grid[2] +
                    bz + z - sz * g.grid[2];
          begin = g.offsets[bin];
          end = g.offsets[bin + 1];
        }
        for (int64_t base = begin; base < end; base += B) {
          int j = int(base + t);
          float3 d = make_float3(0, 0, 0);
          float r2 = 0;
          bool valid = j < end;
          if (valid) {
            d = displacement(g, i, j, sx, sy, sz);
            r2 = d.x * d.x + d.y * d.y + d.z * d.z;
            valid = r2 <= g.cutoff2 && !(i == j && sx == 0 && sy == 0 && sz == 0);
          }
          unsigned mask = __ballot_sync(0xffffffff, valid);
          int lane = t % 32, warp = t / 32;
          if (lane == 0)
            counts[warp] = __popc(mask);
          __syncthreads();
          if (t == 0) {
            int sum = 0;
            for (int w = 0; w < B / 32; ++w) {
              prefix[w] = sum;
              sum += counts[w];
            }
            accepted = sum;
          }
          __syncthreads();
          if (valid) {
            int e = carry + prefix[warp] + __popc(mask & ((1u << lane) - 1));
            queue.sender[e] = j;
            queue.sx[e] = sx + g.winding[3 * i] - g.winding[3 * j];
            queue.sy[e] = sy + g.winding[3 * i + 1] - g.winding[3 * j + 1];
            queue.sz[e] = sz + g.winding[3 * i + 2] - g.winding[3 * j + 2];
            queue.dx[e] = d.x;
            queue.dy[e] = d.y;
            queue.dz[e] = d.z;
            queue.r2[e] = r2;
          }
          __syncthreads();
          int total = carry + accepted, used = (total / TE) * TE;
          for (int off = 0; off < used; off += TE) {
            op.consume(i, TileView<Q>{&queue, off, TE}, state, scratch);
            __syncthreads();
          }
          carry = total - used;
          // Source and destination do not overlap: used >= TE > carry, unless used=0.
          if (used)
            for (int e = t; e < carry; e += B)
              queue.copy(e, used + e);
          __syncthreads();
        }
      }
  if (carry) {
    op.consume(i, TileView<Q>{&queue, 0, carry}, state, scratch);
    __syncthreads();
  }
  op.finalize(i, state, scratch);
}
// Op supplies plan/setup/postprocess even when a phase is empty. No virtual dispatch.
template <class Op> void execute(Context &ctx, Op op) {
  static_assert(Op::threads > 0 && Op::threads % 32 == 0 && Op::threads <= 1024);
  static_assert(Op::tile_edges > 0);
  static_assert(Op::scratch_alignment >= 16 && Op::scratch_alignment <= 256 &&
                (Op::scratch_alignment & (Op::scratch_alignment - 1)) == 0);
  auto plan = op.plan(ctx.geometry.n);
  if (plan.scratch_alignment != Op::scratch_alignment)
    throw Error("PairOp scratch alignment mismatch");
  static_assert(EdgeSubgroup<Op>::size > 0 && Op::threads % EdgeSubgroup<Op>::size == 0);
  dim3 block = BlockShape<Op>::get();
  if (size_t(block.x) * block.y * block.z != Op::threads)
    throw Error("PairOp block shape does not match thread count");
  if (plan.scratch_bytes > 1024 * 1024)
    throw Error("PairOp dynamic scratch exceeds supported per-block storage");
  size_t shared =
      align_up(sizeof(EdgeQueue<Op::threads + Op::tile_edges - 1>), Op::scratch_alignment) +
      plan.scratch_bytes;
  cudaFuncAttributes attr{};
  check(cudaFuncGetAttributes(&attr, pair_kernel<Op>));
  int device;
  check(cudaGetDevice(&device));
  cudaDeviceProp props{};
  check(cudaGetDeviceProperties(&props, device));
  if (block.x > unsigned(props.maxThreadsDim[0]) || block.y > unsigned(props.maxThreadsDim[1]) ||
      block.z > unsigned(props.maxThreadsDim[2]))
    throw Error("PairOp block dimension exceeds device limit");
  if (props.major < 7 || Op::threads > attr.maxThreadsPerBlock ||
      shared + attr.sharedSizeBytes > props.sharedMemPerBlockOptin ||
      size_t(attr.numRegs) * Op::threads > size_t(props.regsPerBlock))
    throw Error("PairOp exceeds device kernel resource limits");
  if (shared + attr.sharedSizeBytes > props.sharedMemPerBlock)
    check(cudaFuncSetAttribute(pair_kernel<Op>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(shared)));
  int active;
  check(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, pair_kernel<Op>, Op::threads, shared));
  if (active < 1)
    throw Error("PairOp has no resident block capacity");
  ctx.resources.push_back(
      {Op::name(), attr.numRegs, Op::threads, active, attr.sharedSizeBytes, shared});
  void *workspace = ctx.reserve(std::string("op.") + Op::name(), plan.global_bytes);
  op.setup(ctx, workspace);
  if (ctx.geometry.n)
    pair_kernel<Op><<<ctx.geometry.n, block, shared, ctx.stream>>>(ctx.geometry, op);
  check(cudaGetLastError());
  op.postprocess(ctx, workspace);
}
} // namespace lapis
