# PERF-0003: MACE staging, CUDA Graphs, and product contractions

Status: accepted for MH1 on SM86
Date: 2026-11-14
Baseline: `f2a5635` (P1 pair kernels)
Hardware/software: RTX 3060 Ti, SM86 Release, CUDA 12.6.77, GCC 11.4, WSL2

This optimization reduces full-call latency by 22.0%/22.8% on 8/64 Si atoms and
16.8%/21.0% on 512/4,096 Si atoms. Both roadmap gates are met on this GPU.

## Hypothesis

Small systems benefit from removing preparation synchronization and repeated
device submission. Larger Si systems benefit from specializing the polynomial
products and removing intermediate normalization/gating traffic.

## Change

Lapis retains pinned input storage and CPU occupancy storage. It copies wrapped
coordinates, elements, bins and winding in one transfer. Brute force uses identity
ordering without histogram/scan/scatter. Preparation enqueues work on the context
stream; an event protects input staging if a direct Context client prepares again
before finishing. Device status shares the asynchronous result transfer. Energy
and all nine virial components still accumulate in host double precision; MACE
adds atomic reference energies separately from interaction energies.

Each immutable MH1 model keeps one evaluation graph. The first successful call
warms allocations, kernels and cuBLAS; the next matching call captures and executes
the graph. Preparation, output transfer and host reductions stay outside capture.
The key includes atom count, cell, grid, periodic-image limits, backend, cutoff,
every DeviceGeometry pointer, and the Context allocation generation. Model/head
and math mode are fixed by the owning model. Geometry or allocation changes discard
the graph and warm the new configuration. Coordinates, elements, cell-list sorting
and explicit edge contents are refreshed outside capture on every call. Empty
inputs and stage profiling use direct launches. Capture exceptions end capture and
release graph resources before the normal calculator error handler synchronizes.

Canonical products use compile-time indices for 410 layer-0 terms and 94 layer-1
terms, keeping components and adjoints in registers. Every coefficient remains
checkpoint data. The loader checks the complete term sequence before specializing;
other valid sequences retain the generic kernel. The generator
`mace/tools/product_indices.py` reproduces the index header from a processed archive.
Forward normalization/gating and their reverse operations each share one kernel,
preserving the preactivations and residual gradients needed by the reverse pass.

Node GEMMs retain `CUBLAS_PEDANTIC_MATH` with a stable 4 MiB cuBLAS workspace.
`MACE_DEFAULT_MATH=1` enables standard FP32 math; `MACE_DISABLE_GRAPHS=1`
retains direct submission. Set either before calculator construction.
`MACE_PROFILE_MH1=1` bypasses graphs while stage timing is active. No reduced
precision mode is enabled. NVIDIA documents
[cuBLAS math modes and capture support](https://docs.nvidia.com/cuda/archive/12.6.1/cublas/index.html#cublasmath-t)
and [stream capture constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## Measurement

The P1 library and its matching benchmark executable were saved before editing.
Matching executables matter because Lapis does not promise C++ binary compatibility.
The runner now accepts `--variant-executable NAME=PATH` for this comparison.

Three sequential rounds use 20 warmups and 50 timed calls per process. Fixture
and variant order reverse in round two. Times include the blocking output transfer,
but exclude construction, parsing, warmup and graph capture. The median is the
median of round medians; p95 uses all 150 samples. Clocks were unlocked and the
desktop shared the GPU. No other GPU test or benchmark ran concurrently.

Inputs are the repository's seeded jittered cubic Si grids, not diamond Si.
The model is the supplied MH1 checkpoint with `matpes_r2scan`. Hashes, round medians,
p95 and component errors are in the [compact measurements](PERF-0003-mace-graphs-products/measurements.json).

| Si atoms | P1 median, ms | P2 median, ms | Reduction | P1 / P2 p95, ms |
| --- | ---: | ---: | ---: | ---: |
| 8 | 2.744 | 2.141 | 22.0% | 3.854 / 2.748 |
| 64 | 3.595 | 2.775 | 22.8% | 4.872 / 3.581 |
| 512 | 16.777 | 13.950 | 16.8% | 18.281 / 15.294 |
| 4,096 | 125.078 | 98.844 | 21.0% | 129.306 / 104.978 |

Every fixture improves in each round: small-case reductions range from 22.0% to
34.1%, and larger-case reductions range from 15.8% to 21.1%. These measurements
establish the two P2 acceptance gates on SM86, not performance on other GPUs.

A separate three-round trial used 10 warmups and 20 samples to compare standard
FP32 graph/direct submission with pedantic FP32 graphs. Median times were:

| Si atoms | Standard FP32, graph / direct, ms | Pedantic FP32 graph, ms |
| --- | ---: | ---: |
| 8 | 2.472 / 2.880 | 2.501 |
| 64 | 3.200 / 3.537 | 2.794 |
| 512 | 13.470 / 13.755 | 13.551 |
| 4,096 | 96.489 / 97.927 | 97.396 |

Graph submission helps the small cases. Math-mode differences on the larger
cases are below 1%, within observed drift; standard math was slower on 64 atoms
in this trial. The main table uses the retained pedantic default.

The compiled specialized products use 40 registers/thread forward and 48 backward,
with zero shared, stack or local bytes in both layers. Generic products use
10,240/16,384 shared bytes forward/backward. Resource counts come from
`cuobjdump --dump-resource-usage`; they are not runtime spill-traffic counters.

## Correctness

The historical validation passed the implicit and explicit Lapis, Chemio,
checkpoint, D3, and implicit/explicit MACE CTest entries.
Each MACE executable passes 125 cases, including all 120 physics cases with
finite-difference forces and nine strain derivatives. New cases compare graph
replay against direct pedantic evaluation through coordinate/species changes,
same-count cell changes, backend changes, grow/shrink/empty transitions and
device-error recovery. Product compatibility cases reorder either or both layers
and modify trained coefficients, checking energies, every force and full virial.
Lapis tests exercise repeated preparation, caller input mutation, and brute identity
ordering.

Every output component was also compared against the saved P1 library for all four
benchmark fixtures, through warmup, capture and replay. Maximum differences were
1.20e-7 eV/atom in total energy, 1.91e-6 eV in atomic energy,
4.66e-7 eV/Angstrom in forces, and 2.24e-6 eV in virial. These comparisons establish
equivalence to the previously validated P1 implementation; they are not a new
independent upstream MACE comparison.

All three unchanged MACE golden references pass, and the benchmark metadata
selftest passes 23 checks. Compute Sanitizer memcheck reports zero errors for
the graph lifecycle test, including captured execution, growth and error recovery.
Source hygiene and formatting checks pass.

## Decision

Keep reusable staging, brute identity ordering, graph replay, fixed products,
fused gating enabled. Standard FP32 cuBLAS math did not show a consistent gain;
keep pedantic math as the default and standard math as an explicit experiment.
Direct submission remains available for diagnosis. Graph memory is bounded to one
executable per model plus the fixed cuBLAS workspace; device and pinned staging retain their
largest required capacities. Frequently changing cells do not amortize capture,
so they use warmed direct evaluations until a configuration repeats. The 15%/10%
targets apply to repeated configurations, as in the benchmark protocol.

Device energy/virial reductions, reduced precision, radial interpolation and
neighbor-list reuse are outside this change. The shared Lapis header changed;
rebuild all consumers together.

## Reproduction

Build revision `f2a5635` in a separate checkout and save its `libmace.so.0` and
`lapis_bench_mace` together in `/tmp/mace-p1`. Build the candidate with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLAPIS_BUILD_BENCHMARKS=ON -DMACE_BUILD_EXPLICIT_REFERENCE=ON \
  -DLAPIS_TEST_DISCOVER_CASES=OFF
cmake --build build -j4
ctest --test-dir build -j4 --output-on-failure
python3 benchmarks/run.py --case mace/mh1/8 --case mace/mh1/64 \
  --case mace/mh1/512 --require-reference
```

Materialize the four canonical inputs with the standard-library fixture generator:

```sh
python3 - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'benchmarks')
from structure import encode, from_recipe
manifest = json.loads(Path('benchmarks/fixtures/manifest.json').read_text())
out = Path('/tmp/mace-p2-inputs')
out.mkdir(exist_ok=True)
for name in ['grid-2', 'grid-4', 'grid-8', 'grid-16']:
    recipe = manifest['fixtures'][name]['recipe']
    (out / (name + '.bin')).write_bytes(encode(from_recipe(recipe)))
PY
python3 benchmarks/notes/PERF-0002-mace-pair-chunks/mace_pair_benchmark.py \
  --model mace/model/mace-mh1.model \
  --variant p1=/tmp/mace-p1 --variant p2=build/mace \
  --variant-executable p1=/tmp/mace-p1/lapis_bench_mace \
  --input /tmp/mace-p2-inputs/grid-2.bin --input /tmp/mace-p2-inputs/grid-4.bin \
  --input /tmp/mace-p2-inputs/grid-8.bin --input /tmp/mace-p2-inputs/grid-16.bin \
  --rounds 3 --warmup 20 --samples 50 --output benchmarks/results/mace-p2
```

For direct/standard-math checks, prefix the candidate invocation with
`MACE_DISABLE_GRAPHS=1` or `MACE_DEFAULT_MATH=1`. Regenerate indices with
`python3 mace/tools/product_indices.py model.pwm mace/src/mh1_product.cuh`, then
apply the repository's `clang-format` configuration.

## Artifacts

- Compact measurements: `PERF-0003-mace-graphs-products/measurements.json`.
- Raw timings, build logs, output comparisons and saved executables were retained
  locally under `/tmp/mace-p2/`; those temporary artifacts are not repository assets.
