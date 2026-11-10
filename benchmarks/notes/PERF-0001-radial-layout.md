# PERF-0001: radial_tile g/dg shared-memory layout

Status: rejected for the generic path; accepted for `Common=true`
Date: 2026-11-03
Commit: `0c8e2fc` (`perf: optimized bank conflict for common pass in pair kernel`)
Hardware/software: RTX 3060 Ti, SM86, CUDA 12.6, `-O3`

This note restores the conclusion of the deleted `hpnn/docs/RADIAL_LAYOUT.md`.
The original document and its raw JSON were removed in `16a652a`; the historical
data is still recoverable with:

```sh
git show 0c8e2fc:hpnn/docs/RADIAL_LAYOUT.md
git show 0c8e2fc:hpnn/docs/results/radial-layout.json
```

## Hypothesis

The generic producer stores `g[b*4 + edge]`, which gives a four-way shared-memory
bank conflict across a full warp. Transposing to `g[edge*basis + b]` should remove
the conflict and make the radial tile faster.

## Change

Tested both layouts for `g`/`dg` while keeping `s`/`ds` in their original
`[hidden][edge]` layout:

- generic: `[basis][edge]` versus `[edge][basis]`;
- `Common=true`: same comparison with B=48, H=256, P=128, three layers.

## Measurement

Standalone wrapper around the actual `radial_tile` body, synthetic weights,
512 independent four-edge blocks, 128 threads, disjoint outputs. 1000 interleaved
warmup pairs, then 15 alternating timing rounds of 32 launches per variant.
Median CUDA-event milliseconds per launch; GPU clocks were not locked.

| B/H/P | Path | Forward before / after (ms) | Backward before / after (ms) |
| --- | --- | ---: | ---: |
| 48/256/128 | Common | 0.059584 / 0.060064 | 0.078496 / 0.076052 |
| 64/256/128 | Generic | 0.056480 / 0.083703 | 0.085216 / 0.105817 |
| 7/43/37 | Generic | 0.011936 / 0.012064 | 0.013310 / 0.013408 |

## Decision

The generic path keeps the apparently conflicting `[basis][edge]` layout. Its
consumer loads four adjacent edge values, which the compiler combines into
`LDS.128`; the transpose breaks that vectorization and regressed generic B=64 by
48.2% forward and 24.2% backward. For `Common=true`, the compile-time shape lets
the compiler retain `LDS.128` across four consecutive basis entries, so the
edge-major layout removes the store conflict without losing wide reads. The
common path uses edge-major; the generic path does not.

The lesson is to measure the generated instruction mix, not only source-level
bank indices. A future kernel microbenchmark should assert the `LDS.128` count or
the relevant Nsight Compute shared-load metrics in addition to time.

## Reproduction

The historical standalone wrapper is not part of the model-level MVP harness yet.
This note preserves the measurement setup and conclusion until a `radial_tile`
A/B case is added. The current source comment in `hpnn/src/pairs.cuh` points here.

## Artifacts

- Historical document: `0c8e2fc:hpnn/docs/RADIAL_LAYOUT.md`
- Historical raw data: `0c8e2fc:hpnn/docs/results/radial-layout.json`
- Current code: `hpnn/src/pairs.cuh`, `radial_tile`
