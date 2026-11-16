# PERF-0002: MACE pair projection chunks and fixed tensor products

Status: accepted for MH1 on SM86
Date: 2026-11-12
Baseline: `a6d0906a878e3272614382b9537a06718524f9d1`
Hardware/software: RTX 3060 Ti, SM86 Release, CUDA 12.6.77, GCC 11.4, WSL2

The optimization reduces full-call latency by 23.1% on 165-atom Ti/O
and 25.3% on its 1,320-atom supercell. Small Si fixtures also improve.

## Hypothesis

The pair kernels retain all radial projections and use dynamically indexed
tensor-product terms. Consuming projections in smaller chunks and specializing
the fixed connectivity should reduce shared storage and instruction count while
preserving the model's FP32 arithmetic and analytic derivatives.

## Change

`MH1Pair` retains four edges per block and projects two paths at a time. Layer 1
therefore retains 256 radial outputs per edge instead of 1,280, including their
derivatives in reverse evaluation. Density evaluation runs before the radial
network so its hidden values remain available throughout the chunks. Reverse
partials accumulate in shared memory across chunks; there is no per-edge global
cache or additional persistent workspace.

`mh1_tp.cuh` describes the canonical 16/86 tensor-product entries. Compile-time
indices remove dynamic term loads and input selection. All CG coefficients,
projection weights, and biases still come from the checkpoint. Loading checks
every connectivity entry before selecting specialization; other valid layouts
use the dynamic implementation with a complete projection chunk, retaining
their term order. Existing public APIs, precision, cutoff, and ZBL are unchanged.

## Measurement

The saved baseline library was built before editing production sources. The
candidate uses the same compiler and model. The supplied MH1 model selects
`matpes_r2scan`; its SHA-256 and all input hashes are in the
[compact measurements](PERF-0002-mace-pair-chunks/measurements.json).

Three rounds ran sequentially, with 20 warmups and 50 timed calls per process.
Baseline and candidate ran consecutively for each fixture; both fixture and
variant order reversed in round two. The table reports the median of the three
round medians. Calls include host output transfer; loading, parsing, and warmup
are excluded. GPU clocks were not locked, and the desktop shared the GPU.
Every Ti/O round improved by more than 20%, despite the drift in larger cases.

Si uses the repository's seeded jittered cubic grids. Ti/O uses frame zero of
the external `165.arc`, with a 2x2x2 replication for 1,320 atoms. These are the
same binary fixtures as the roadmap measurements.

| Fixture | Baseline median, ms | Optimized median, ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 8 Si | 2.896 | 2.603 | 10.1% |
| 64 Si | 3.880 | 3.427 | 11.7% |
| 165 Ti/O | 12.822 | 9.865 | 23.1% |
| 512 Si | 20.209 | 16.480 | 18.5% |
| 1,320 Ti/O | 83.867 | 62.656 | 25.3% |
| 4,096 Si | 153.585 | 119.842 | 22.0% |

## Kernel resources and counters

Dynamic shared bytes include the Lapis edge queue. Each kernel also uses 256
static shared bytes and 128 threads per block.

| Kernel | Shared bytes before / after | Registers before / after | Resident blocks per SM before / after |
| --- | ---: | ---: | ---: |
| forward0 | 24,116 / 20,024 | 123 / 91 | 4 / 4 |
| forward1 | 48,692 / 32,312 | 128 / 94 | 2 / 3 |
| backward1 | 48,688 / 32,304 | 140 / 145 | 2 / 3 |
| backward0 | 24,112 / 24,112 | 154 / 145 | 3 / 3 |

Nsight Compute 2024.3.2 profiled all four pair kernels on the 165-atom fixture
after 20 warmups, with 37–38 replay passes per kernel and clock/cache control
disabled. All four recorded **zero local-memory load and store sectors**, both
before and after. Compiler resource reports retain a 32-byte stack frame and
zero local allocation for each pair kernel.

Reverse layer-1 executed instructions fell from 359,181,244 to 237,492,156
(33.9%), and achieved occupancy rose from 16.04% to 21.16%. The measured barrier
stall ratio increased from 0.110 to 0.359 as expected from chunk synchronization;
the overall latency improvement is the acceptance measure. That ratio is
`smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, not a fraction
of total wall time. Profiler-affected timings are excluded from the latency table.

## Correctness

All six configured CTest suites passed. Implicit and explicit MACE each passed
122 cases: 120 physics cases plus two new checkpoint-compatibility cases. The
physics suite checks finite-difference forces and all nine strain derivatives,
invariances, supercells, backend agreement, and buffer reuse. The new cases
compare canonical and reordered tensor products in either or both layers,
including changed CG coefficients and trained radial projections.

All three existing golden cases passed without changing references. Comparing
every output component against the saved baseline on all six benchmark fixtures
gave maximum errors of 5.21e-8 eV/atom in total energy, 1.91e-6 eV in atomic
energy, 1.91e-6 eV/Angstrom in forces, and 6.11e-5 eV in virial.

An independent official MACE CPU FP64 comparison, using float32-rounded input
geometry and explicit `MACECalculator.calculate`, passed on 8 Si, 64 Si, and
165 Ti/O. Maximum errors across these cases were 1.11e-7 eV/atom, 8.77e-6
eV/Angstrom, and 1.67e-4 eV in virial. Limits were energy/atom <1e-5, force
<1e-4, and virial <max(1e-4, 1e-4 times the largest reference component).
Independent reference coverage remains limited to those three fixtures and
the `matpes_r2scan` head.

## Decision

Keep four edges and two projection paths per chunk. An initial screening used
10 warmups and 20 samples; these single-round values select candidates rather
than establish the final gain.

| Screening variant | 8 Si, ms | 165 Ti/O, ms | 1,320 Ti/O, ms |
| --- | ---: | ---: | ---: |
| Baseline | 3.418 | 12.354 | 77.950 |
| Full projections and fully unrolled TP, four edges | 2.871 | 9.299 | 58.705 |
| Full projections, two-path TP chunks, four edges | 2.997 | 9.745 | 61.054 |
| Full projections, two-path TP chunks, two edges | 3.255 | 11.765 | 73.740 |
| Full projections, two-path TP chunks, one edge | 4.135 | 15.450 | 94.846 |
| Two-path projection/TP chunks, four edges | 2.886 | 9.464 | 59.549 |
| Two-path projection/TP chunks, two edges | 3.159 | 11.313 | 70.815 |

Full unrolling reached 255 registers and increased the reverse layer-1 stack
frame to 40 bytes, so it was rejected. Streaming one path at a time used less
shared memory but took 11.137 ms on 165 Ti/O in a separate 20/50 screening run.
The selected two-path chunks balance synchronization, scratch, and live ranges.
Smaller edge tiles lost too much projection-weight reuse and radial parallelism.

## Reproduction

Build the baseline revision in a separate checkout/build and save its library
before rebuilding the candidate. Each variant directory must contain
`libmace.so.0`. The benchmark executable can load either library through
`LD_LIBRARY_PATH`; the runner records library and input hashes in its outputs.

From the workspace root, with the external model and report fixtures available:

```sh
cmake --build build -j4
ctest --test-dir build -j4 --output-on-failure
python3 benchmarks/PERF-0002-mace-pair-chunks/mace_pair_benchmark.py \
  --model /absolute/path/to/mace-mh1.model \
  --variant baseline=/absolute/path/to/saved-baseline \
  --variant optimized=build/mace \
  --input benchmarks/results/mace-study/grid-2.bin \
  --input benchmarks/results/mace-study/grid-4.bin \
  --input benchmarks/results/mace-study/tio-165.bin \
  --input benchmarks/results/mace-study/grid-8.bin \
  --input benchmarks/results/mace-study/tio-1320.bin \
  --input benchmarks/results/mace-study/grid-16.bin \
  --output benchmarks/results/mace-optimized/repeat
```

The runner accepts any canonical `LAPISB01` inputs. Si fixture recipes and the
binary encoder are in `benchmarks/fixtures/manifest.json` and
`benchmarks/structure.py`; Ti/O assets remain external. Use a separate configured
build to sweep `MACE_PAIR_TILE_EDGES=1,2,4` and positive
`MACE_PAIR_PATH_CHUNK` values via CUDA definitions, for example:

```sh
cmake -S . -B build-pair-sweep \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=OFF -DLAPIS_BUILD_D3=OFF \
  -DLAPIS_BUILD_BENCHMARKS=ON \
  '-DCMAKE_CUDA_FLAGS=-DMACE_PAIR_TILE_EDGES=2 -DMACE_PAIR_PATH_CHUNK=2'
cmake --build build-pair-sweep -j4 --target lapis_bench_mace
```

To reproduce pair-counter collection for either library, set its directory in
`LD_LIBRARY_PATH` and run:

```sh
ncu --set full --kernel-name-base mangled --kernel-name regex:MH1Pair \
  --launch-skip 80 --launch-count 4 --clock-control none --cache-control none \
  -o benchmarks/results/mace-optimized/repeat-pairs \
  build/lapis_bench_mace --engine mace \
  --model /absolute/path/to/mace-mh1.model \
  --input benchmarks/results/mace-study/tio-165.bin --warmup 20 --samples 1
```

## Artifacts

- Raw profiler output: `PERF-0002-mace-pair-chunks/measurements.json`
