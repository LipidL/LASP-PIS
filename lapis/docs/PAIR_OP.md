# Expert L1 PairOp interface

The implementation is in `include/lapis/pair.cuh`; `examples/lj/lj.cu` is the small
reference operation. L1 controls its mathematics and CUDA work distribution.

An operation is a trivially device-copyable value containing device pointers and
scalar parameters. Define these static members:

```cpp
static constexpr int threads = 128;        // multiple of 32, up to 1024
static constexpr int tile_edges = 32;
static constexpr int scratch_alignment = 16; // power of two, 16..256
static constexpr int edge_threads = 1;      // optional scheduling declaration
static const char* name();
struct State { /* per-thread state, constructed once for a receiver */ };
```

An optional `static dim3 block_shape()` may declare a multidimensional block;
its dimensions must multiply to `threads`. The default is `(threads, 1, 1)`.
`thread_rank()` gives the flattened CTA rank. Edge subgroups must divide the
thread count; their mapping inside consume remains the operation's responsibility.

Provide the following host methods:

```cpp
WorkspaceSpec plan(int atom_count) const;
void setup(Context&, void* global_workspace);
void postprocess(Context&, void* global_workspace);
```

`WorkspaceSpec` declares grow-only operation workspace bytes, dynamic scratch bytes
and scratch alignment. Supply empty setup/postprocess methods when unused. Setup
runs on the host before traversal and may update the operation's device pointers,
initialize atomic destinations or enqueue node kernels on `Context::stream`.
Postprocess enqueues ordered node reductions or other completion work.

Provide these device methods:

```cpp
__device__ void prepare(int receiver, State&, void* scratch);
template<class Tile>
__device__ void consume(int receiver, const Tile&, State&, void* scratch);
__device__ void finalize(int receiver, State&, void* scratch);
```

Call `execute(context, operation)` from model orchestration. The runtime queries
compiled registers and static shared memory, checks block limits and dynamic
scratch, opts into larger shared allocations when needed, and reports resident
blocks per SM. Launch and asynchronous errors become exceptions at the blocking
calculator boundary. L1-detected errors use `atomicCAS(geometry.status, 0, code)`;
do not return a host CUDA error from a device callback.

One CTA owns each receiver. Every CTA thread calls prepare, every consume and
finalize in the same dynamic order. All threads must reach any block-wide barriers
inside an operation. Threads may skip padded numerical work while other threads
still participate in collectives. Neither a tile pointer nor a reference into it
may survive the consume callback.

`TileView` provides `size()`, `sender(edge)`, `shift(edge)`,
`displacement(edge)` and `squared_distance(edge)`. Only `[0, size())` is defined.
Accessors inline directly into shared-memory SoA loads. Sender and receiver IDs
refer to spatially permuted internal nodes; explicit original identity is available
through the context's two node maps. Lattice vectors are rows.

The queue capacity is `threads + tile_edges - 1`. A warp ballot and prefix rank
compact a candidate batch after a carried remainder. Complete windows are consumed
collectively, then the remainder moves to the front; a final partial tile is emitted
once. Barriers separate candidate publication, callbacks and storage reuse. The
framework queue and operation scratch have distinct aligned regions.

The runtime emits full directed edges, including periodic self-images. It excludes
only `(receiver == sender, image == 0)`, accepts the inclusive cutoff, and does not
promise edge order. Distinct atoms at identical positions are still edges: an
operation with singular mathematics must report that input explicitly.

Cutoff comparisons use the stored float32 geometry. Cross-precision comparisons
can differ near `r == cutoff`: a conservative diagnostic boundary band is
`64 * float_epsilon * S`, where `S` is the sum of absolute Cartesian coordinate
and image-translation terms in the displacement, plus the cutoff. Edge multiset
tests use geometries away from this band, with separate float32 `nextafter` tests
for the inclusive boundary. No exact CPU-double/GPU-float edge agreement is
claimed inside that band.

Lennard-Jones applies a half factor to directed edge energies and virial, and
accumulates the full force at its receiver. Its force-shifted potential makes
energy and its first radial derivative vanish at cutoff.
