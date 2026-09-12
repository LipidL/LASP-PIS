# MH1 kernel implementation

Total energy is reduced in double precision on the host. MH1's float32 atomic
baselines are added after reducing interaction energies, avoiding cancellation
for heads with large atomic reference energies. Per-atom output arrays stay float32.

MH1 stores node matrices as `[irrep component][node][channel]` within each irrep
block. Equivariant channel maps become strided batched cuBLAS SGEMMs in FP32 with
pedantic math mode. Species-only radial affine terms are immutable lookup tables;
radial matrices are transposed at load for coalesced channel access. The force
traversals recompute radial tangents and angular derivatives and scatter sender
adjoints. Forward temporaries and dead saved node tensors are reused by backward
work. No general differentiation engine is present in the runtime.

Each MH1 callback handles four edges. One warp evaluates each edge's radial hidden
network and layer normalization; the CTA then projects all four hidden vectors
together, reusing each loaded output weight across the tile. Shared geometry,
path weights and derivatives are discarded after the callback. Padded warps skip
geometry and initialize their projection inputs, while all CTA threads still reach
the subsequent barriers.

Indexed cubic-product values and adjoints use shared arrays with one channel per
lane. In pair backward, sender inputs are loaded once per edge, and radial scratch
is reused for tensor-product adjoints after the radial phase ends. Cartesian force
components share one block reduction. Scratch capacities specialize to each
layer's path count to preserve residency on SM86.

Set `MACE_PROFILE_MH1=1` for diagnostic CUDA-event intervals around model
stages. This adds event creation and logging and must be disabled for benchmarks.
