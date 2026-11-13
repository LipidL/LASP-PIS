# MACE inference benchmarks and optimization roadmap

The next substantial MACE speedup should target the radial networks and tensor
products inside the fused pair kernels. On the measured Ti/O structures these
stages account for 74–76% of the timed model span. Lapis preparation and output
handling deserve a separate small-system latency effort. Replacing implicit
traversal with the existing explicit-connectivity implementation is not a
compelling first step.

These are new measurements of revision `1dce8d3736db2bf2bd864bdf7fe4e252ff332789`,
taken on 2026-11-12. Production inference sources were not changed. The scope is
the supported MH1 architecture and `matpes_r2scan` head, not arbitrary upstream
MACE models. The deliverables include [compact measurements](measurements.json),
[probe preparation](../../../benchmarks/prepare_mace_study.py), and a
[repeatable benchmark matrix](../../../benchmarks/study_mace.py).

## Environment and method

RTX 3060 Ti, 8 GiB, 38 SMs, SM86 Release build, CUDA Toolkit 12.6.77,
GCC 11.4, WSL2. NVIDIA-SMI reports driver 591.86 with Linux interface 590.57;
the CUDA driver API reports 13.1 compatibility. The desktop uses the GPU.
Clocks were not locked. A mid-run observation was 70 C and 1995 MHz SM clock;
this was not a continuous clock trace.

Each native case constructs a calculator, performs 20 warmup calls and 50 timed
calls. Three rounds run sequentially, with case order reversed in round two.
The reported median is the median of the three round medians; p95 is the
nearest-rank percentile of all 150 samples. No concurrent GPU benchmark jobs
were launched. Each call returns total/atomic energy, forces, and virial to the
host. Input parsing, model loading, and warmups are excluded from steady-state
latency. Construction took about 0.61–0.63 seconds for native MACE, including
process-local CUDA/library initialization; retain the calculator in applications.

The Si fixtures are the repository's seeded, jittered cubic grids, **not diamond
Si**. Ti/O is frame zero of the local `structures/165.arc`; 1,320 atoms is its
2x2x2 replication. The model SHA-256 matches the benchmark manifest:
`ec00a2705854622fbbd898ccfb7701072fcd674709102d009fb919c1b8cc5dde`.
Input hashes are recorded in the compact measurements. Model and Ti/O assets
remain external; the repository contains recipes for the Si fixtures.

## Correctness and review findings

- All six configured CTest entries passed: both Lapis traversal suites, Chemio,
  checkpoint reader, and implicit/explicit MACE. Each MACE executable passed
  120 physics cases, including finite-difference forces and all nine strain
  derivatives, invariances, supercells, backend agreement, and buffer reuse.
  HPNN and D3 were disabled for this focused build.
- Benchmark metadata selftest passed 23 checks. All three existing MACE golden
  cases (8, 64, 512 atoms) passed without updating references.
- Independent comparison used the local official MACE source at
  `59ad3a473ca02101a1cf02db242197e8d616dd11`, PyTorch 2.14.0+cu130, CPU FP64,
  and identical float32-rounded input geometries. It explicitly calls
  `MACECalculator.calculate` and compares every force component and full virial.

| Atoms and fixture | Energy error per atom, eV | Maximum force error, eV/Angstrom | Maximum virial error, eV |
| --- | ---: | ---: | ---: |
| 8 Si | 5.30e-8 | 7.14e-7 | 1.32e-5 |
| 64 Si | 1.11e-7 | 1.44e-6 | 6.31e-5 |
| 165 Ti/O | 3.63e-8 | 9.12e-6 | 1.66e-4 |

All three pass the applied limits: energy per atom <1e-5 eV, force <1e-4
eV/Angstrom, and virial <max(1e-4 eV, 1e-4 times the largest reference virial
component). The 512-atom CPU reference process was killed with exit 137;
independent coverage stops at 165 atoms. Larger supplemental cases establish
performance, not independent-reference correctness. These checks cover one head.

## Full-call latency

| Fixture | Auto backend | MACE median, ms | MACE p95, ms | Round median range, ms | Context workspace, MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 Si | brute | 2.97 | 3.66 | 2.93–3.00 | 2.7 |
| 64 Si | brute | 3.97 | 4.71 | 3.94–3.99 | 21.6 |
| 165 Ti/O | cell | 12.71 | 13.51 | 12.57–13.12 | 55.6 |
| 512 Si | cell | 19.99 | 21.60 | 19.41–20.81 | 172.5 |
| 1,320 Ti/O | cell | 85.13 | 88.48 | 79.43–85.84 | 444.6 |
| 4,096 Si | cell | 160.40 | 173.09 | 146.54–162.33 | 1,379.8 |

Context workspace excludes model weights, cuBLAS/context allocations, and pinned
host memory. It is not total process VRAM. Density and coordination differ between
Si and Ti/O; atom count alone does not explain their latency. Clock/thermal/load
drift is substantial at larger sizes. A 3–5% apparent gain is not decisive here.

| Fixture | MACE Auto, ms | MACE brute, ms | MACE explicit Auto, ms | Lapis plus LJ Auto, ms |
| --- | ---: | ---: | ---: | ---: |
| 8 Si | 2.97 | 2.98 | 2.99 | 0.239 |
| 64 Si | 3.97 | 3.93 | 4.07 | 0.205 |
| 165 Ti/O | 12.71 | 12.67 | 12.94 | 0.232 |
| 512 Si | 19.99 | 20.86 | 20.26 | 0.255 |
| 1,320 Ti/O | 85.13 | 91.77 | 85.55 | 0.393 |
| 4,096 Si | 160.40 | 225.89 | 154.44 | 0.816 |

The LJ probe uses the same blocking framework, timing harness, and 6 Angstrom
cutoff as MH1, with epsilon=sigma=1. It measures **Lapis plus a cheap pair
potential**, not pure framework overhead. At 4,096 atoms LJ brute takes 7.22 ms,
showing that cell traversal matters at scale. Explicit MACE rebuilds its edge
list on every call; the nominal 3.7% advantage at 4,096 atoms is smaller than
the observed drift. It does not justify a universal switch, nor rule out an
MD-specific reusable list with a skin.

## Where MACE spends time

Separate `MACE_PROFILE_MH1=1` runs used 20 warmups and 30 samples. These are
medians of CUDA-event intervals, not isolated kernel durations. They include
host submission gaps and adjacent operations: `backward1.pairs` also includes
the residual-gradient GEMM, and some stage boundaries include preceding
readout/linear work. Sums of medians need not equal uninstrumented wall time.
The public `device_ms` similarly spans events; wall time minus `device_ms` is
not a complete CPU overhead measurement.

| Timed category | 165 Ti/O, ms | 512 Si, ms | 1,320 Ti/O, ms | 4,096 Si, ms |
| --- | ---: | ---: | ---: | ---: |
| Four pair stages | 9.65 | 11.90 | 64.31 | 98.51 |
| Four product stages | 1.13 | 3.29 | 8.53 | 27.97 |
| Other stages | 2.27 | 5.04 | 11.34 | 33.40 |
| Sum | 13.04 | 20.23 | 84.19 | 159.89 |
| Pair share | 74.0% | 58.8% | 76.4% | 61.6% |

At 165 atoms, layer-1 forward/reverse pair stages take 2.83/4.45 ms, together
56% of the timed span. At 8 atoms pair stages still account for 57%, but one
block per receiver launches only eight blocks across 38 SMs.

The current [PairScratch and MH1Pair](../../../mace/src/mh1.cu) process four
edges per 128-thread block. Layer 1 reserves about 47.6 KiB dynamic shared
memory plus 256 static bytes, allowing two resident blocks per SM. Reverse
layer 1 uses 140 registers per thread; reverse layer 0 uses 154. Nsight Compute
2024.3.2 measured one warmed reverse layer-1 kernel on the 165-atom fixture:

- Achieved occupancy 16.01%, versus theoretical 16.67%.
- Shared memory limits residency to two blocks; registers alone would allow three.
- 0.42 eligible warps per scheduler, with 1.92 active warps per scheduler.
- DRAM throughput 0.66% of peak; L1/TEX throughput 67.08% of peak.

This supports investigating latency hiding, instruction dependencies, and internal
data movement. It does not support treating the kernel as DRAM-bandwidth-bound.
Counter collection used 38 replay passes, no clock control and no cache control;
profiled wall times are excluded from the latency tables. See NVIDIA's
[profiling methodology](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Prioritized implementation roadmap

The effort ranges below are planning estimates for one engineer, not delivery
commitments. Acceptance targets are experiment gates, not predicted speedups.

| Priority | Work and location | Acceptance criterion | Initial effort |
| --- | --- | --- | --- |
| P1 | Reduce pair-kernel scratch and live ranges in `MH1Pair`; sweep edge tiles 1/2/4, path chunks, and fixed TP-path specialization | At least 10% repeatable full-call gain on Ti/O, without a small-system regression or increased spills | 3–5 days |
| P1 experimental | Prototype species-pair radial interpolation with a direct fallback | At least 20% full-call gain, explicit accuracy limits and bounded table memory | 5–10 days |
| P2 small systems | Reusable input staging, remove preparation synchronization, then capture device evaluation in CUDA Graphs | At least 15% improvement on 8/64 atoms; correct grow/shrink and cell/backend invalidation | 3–5 days |
| P2 larger Si | Specialize product contractions; fuse normalization/gating; tune node GEMMs | At least 10% gain on 512/4,096 Si; preserve derivative consistency | 3–7 days |
| P3 workloads | MD neighbor reuse, optional precision modes, batching or device-resident integration | Workload-specific benefit demonstrated after the preceding bottlenecks move | Separate experiments |

### Pair kernels with unchanged model arithmetic

Start with [MH1Pair](../../../mace/src/mh1.cu) and
[radial_hidden](../../../mace/src/mh1_math.cuh). The four-edge tile amortizes
projection-weight loads but retains all 1,280 layer-1 radial outputs, and the
reverse kernel retains their derivatives too. Smaller tiles or projection/path
chunks may reduce shared memory and improve residency. Measure full-call latency,
register spills, barrier stalls, and instruction counts; occupancy alone is not
an objective. Avoid replacing shared arrays with dynamically indexed local arrays
that spill. Specialize the fixed tensor-product paths while preserving support
for compatible checkpoints with different weights.

For very small systems, experiment with multiple channel/path blocks per receiver
and a bounded reduction. The extra reductions or repeated radial work may cost
more than the extra parallelism, so retain a shape-based fallback. Receiver force
accumulation, backward0's unused input-gradient elimination, shared radial scratch
reuse, ordinary GEMMs over merged components, and overwrite-aware linear outputs
already exist; they are not new opportunities.

### Radial interpolation as an explicit accuracy option

For fixed weights and head, both radial and density networks depend only on
distance and the **ordered sender/receiver species pair**. A table can replace
work repeated in the forward and reverse passes. Compare two designs:

- Final cutoff-multiplied outputs: 512 + 1,280 radial channels plus two density
  scalars. Four FP32 cubic coefficients per interval cost 14.02 MiB per ordered
  pair at 512 intervals, or 56.06 MiB at 2,048 intervals. Ti/O requires four
  ordered pairs: 56.06 or 224.25 MiB respectively.
- Hidden-output tables: two 64-channel hidden outputs plus two density scalars,
  retaining the final linear projection and analytic cutoff. At 512 intervals,
  four coefficients cost 1.02 MiB per ordered pair, but remove less arithmetic.

Use cubic Hermite values/slopes and derive forces from the same polynomial used
for energy. Do not interpolate energy weights and force derivatives independently.
Allocate only encountered species pairs, key tables by model/head, enforce a
budget, and fall back to direct evaluation outside the validated radial range.
Keep analytic ZBL initially. Validate compressed distances, cutoff continuity,
normalization, force/stress errors, and MD energy drift before enabling this mode.
Caching radial intermediates for reuse during reverse evaluation is an alternative
exact-arithmetic experiment, with an explicit per-edge memory budget.

Halving all pair-stage time would yield about 1.59x speedup of the measured
165-atom model span and 1.45x for 4,096 Si by Amdahl's law. These conditional
calculations do not predict that either implementation will halve pair time.

### Lapis preparation and CUDA Graphs

[Context::prepare](../../../lapis/src/runtime.cu) allocates CPU vectors,
wraps/bins atoms, estimates occupancy, makes four H2D copies from pageable data,
and performs histogram/scan/scatter even for brute force. It synchronizes so
temporary input buffers can be destroyed. Reuse CPU buffers and stable pinned
input staging, and add an identity-order brute path. Keep input validation and
periodic image semantics intact.

[Context::finish](../../../lapis/src/runtime.cu) copies 13 floats per atom,
reduces nine virial components per atom on the CPU, then separately copies status.
Pack status into the asynchronous output and consider device energy/virial
reductions. Preserve double-precision accumulation and the separation of large
atomic reference energies from interaction energies.

Capture only the warmed device evaluation first, after stable allocation and
cuBLAS initialization. Keep host preprocessing and the public blocking API
outside capture. Synchronization inside capture must be removed; geometry
scalars passed by value into kernels need updates when the cell, count, or
backend changes. Cache/invalidate graphs on all relevant pointers, capacities,
shapes, and model configuration, not atom count alone. NVIDIA documents these
[CUDA Graph capture constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

### Products and GEMMs

Product kernels account for 16–17.5% on larger Si grids. Their dynamic polynomial
term loop accesses shared intermediates; test degree/output grouping, fixed-term
specialization, and common-subexpression reuse while checking generated code.
Fuse normalization/gating where saved values needed by reverse evaluation permit.

Node GEMMs currently use `CUBLAS_PEDANTIC_MATH`. Benchmark the standard FP32
default mode before considering reduced precision, and validate every force and
stress component. TF32/BF16 must be separately opt-in: NVIDIA distinguishes
[default, pedantic, and reduced-precision cuBLAS modes](https://docs.nvidia.com/cuda/archive/12.6.1/cublas/index.html#cublasmath-t).
Accelerating node GEMMs alone cannot remove the dominant pair-network arithmetic.

Neighbor lists with a skin are a later MD option: rebuild on displacement and
cell-change criteria, and recompute distances/features from current positions.
Batching and a device-resident API help throughput or GPU-integrated MD, but
require extending the current one-structure, blocking contract. Smaller channel
counts, cutoffs, or interaction counts change the potential and belong to a
separate retraining/distillation project.
