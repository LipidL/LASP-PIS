# Model-level benchmarks

This is the minimal reproducible benchmark entry point for MACE CUDA inference.
It deliberately separates four things:

| Artifact | Tracked? | Purpose |
| --- | --- | --- |
| `cases/`, `fixtures/`, `models.json` | yes | what to run, and the exact inputs |
| `references/` | yes | compact golden output summaries per case |
| `baseline/` | yes, per release/environment | accepted median/p95 values |
| `results/`, `.cache/` | no | raw run records and materialized inputs |

The model files themselves are gitignored, so `models.json` pins their SHA-256
instead. A missing or changed model is an error unless explicitly allowed.
`LAPIS_BENCH_MACE_MODEL` overrides the model path; the
manifest hash check still applies unless `--allow-model-mismatch` is given.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DLAPIS_BUILD_BENCHMARKS=ON
cmake --build build -j4 --target lapis_bench_mace
```

`lapis_bench_mace` links the MACE library and requires `LAPIS_BUILD_MACE=ON`.
Cases with missing executables or model files can be skipped with `--skip-missing`.

## Quick start

```sh
# Validate tracked metadata without a GPU.
python3 benchmarks/selftest.py

# First-time fixture hash refresh after editing a recipe or generator.
python3 benchmarks/run.py --update-manifest
python3 benchmarks/selftest.py

# Create compact golden summaries from a known-good build. Do this only when
# the model or fixture intentionally changes.
python3 benchmarks/run.py --tier smoke --update-reference

# Normal run. Raw records go to benchmarks/results/ (gitignored).
python3 benchmarks/run.py --tier smoke

# All tracked MACE cases (8, 64, and 512 atoms).
python3 benchmarks/run.py --tier all

# Freeze the accepted values for this environment as a release baseline.
python3 benchmarks/run.py --tier smoke --warmup 20 --samples 50 \
  --baseline-out benchmarks/baseline/sm86-rtx3060ti-cuda12.6.json

# Compare later runs against the baseline. Exits nonzero above the threshold.
python3 benchmarks/compare.py \
  --baseline benchmarks/baseline/sm86-rtx3060ti-cuda12.6.json \
  --results-dir benchmarks/results --threshold 0.05
```

`run.py` uses only the Python standard library. Fixture authoring scripts may
use numpy/ASE; keep that dependency out of the runner. If an authoring script
needs packages, use a PEP 723 `# /// script` header and `uv run`.

## Case definition

A case is a small JSON file in `cases/`:

```json
{
  "schema": 1,
  "id": "mace/mh1/512",
  "tier": "smoke",
  "engine": "mace",
  "model": "mace/mh1",
  "fixture": "grid-8",
  "backend": "auto",
  "warmup": 3,
  "samples": 10,
  "tolerances": {
    "energy_abs": 1e-4,
    "virial_abs": 1e-3,
    "force_max_abs": 1e-3,
    "force_l2_rel": 1e-3,
    "force_sum_abs": 1e-3,
    "energies_sum_abs": 1e-4
  }
}
```

`tier` selects `smoke` (default), `full`, or `all`. `tolerances` are part of
the case, not the run record. Raw samples, environment, git state, measured
errors, and CUDA resources belong to the run record.

## Fixtures

`fixtures/manifest.json` maps a fixture id to either a committed canonical JSON
structure or a deterministic recipe:

```json
{
  "fixtures": {
    "nacl": { "path": "fixtures/nacl.structure.json", "sha256": "..." },
    "grid-16": { "recipe": { "kind": "grid", "repeat": [16, 16, 16],
                              "spacing": 2.7, "jitter": 0.04,
                              "element": 14, "seed": 32189 },
                 "sha256": "...", "atoms": 4096 }
  }
}
```

Small fixtures can be committed as `*.structure.json`. Large fixtures use a
recipe so the repository does not grow with coordinates. In both cases the
identity is the SHA-256 of the canonical binary stream, not of JSON whitespace:

```text
magic "LAPISB01" | version | atoms | cell(9 x f64) | z(N x i32) | xyz(3N x f32)
```

`run.py` materializes that stream into `benchmarks/.cache/inputs/<sha>.bin` and
the C++ executable reads only the binary. Fixture generation and crystallography
(surfaces, adsorbates, supercells) belong in Python authoring tools, never in
the timed C++ path.

If you change a recipe or generator, run `--update-manifest`. A hash mismatch
at run time is intentional and means the input changed.

## References

`references/<case>.json` stores a compact golden summary, not full force or
virial arrays:

```json
{
  "schema": 1,
  "case_id": "mace/mh1/512",
  "fixture_sha256": "...",
  "model_sha256": "...",
  "tolerances": { "...": 0.0 },
  "output": {
    "energy": 0.0,
    "energies_sum": 0.0,
    "virial": [0.0],
    "force_max_abs": 0.0,
    "force_l2": 0.0,
    "force_sum": [0.0, 0.0, 0.0]
  }
}
```

This keeps the repository small while still checking energy, forces, and virial.
Energy alone is not enough: a broken force or virial path can still return a
plausible energy and a meaningless timing. Exhaustive physical correctness
(finite differences, invariances, supercells) remains in `tests/`; the benchmark
reference is a guard against timing a wrong kernel.

Creating a reference from the current build is intentionally explicit:
`--update-reference` writes or replaces it (`--force-reference` is an alias). Do
that only after an intentional model/fixture change, and review the result.

## Baselines and history

A baseline is a small release-time file keyed by hardware/environment class.
Do not compare absolute numbers across different GPUs or driver/CUDA stacks;
`compare.py` is a median-ratio check within one environment class.

Raw run records include full samples, device information, git state, command,
correctness deltas, and kernel resources. They are written to `results/` and
are gitignored. Keep them in CI artifacts or a separate `perf-data` branch if
long-term history is needed; do not commit them to the main tree.

`compare.py` currently implements a simple median-ratio threshold. A stronger
gate should use paired/interleaved samples and a bootstrap interval; replace
the comparison rule rather than adding more prose to this file.

## Negative results

Rejected optimizations still belong in the repository, but as short decision
records and executable alternatives, not as raw profiling dumps. Use
`notes/TEMPLATE.md`; link the note from the relevant source comment and keep
the raw artifact outside git (or next to the note only if it is a few kilobytes).
See `notes/` for the format.
