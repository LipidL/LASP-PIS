# MACE MH1 inference

A C++17/CUDA implementation of MACE MH1 inference using the independently
packaged [Lapis framework](../lapis/README.md). This project owns model kernels,
weights, analytical derivatives, checkpoint loading, and C/Python frontends.

The project provides complete inference for the
supplied MACE MH1 checkpoint: both density-normalized, gated message layers, cubic
products, residuals, readouts, ZBL, analytical forces and virial. C++ inference does
not link to PyTorch, e3nn, cuEquivariance or Python. The independent `checkpoint`
package reads model data with zlib; `chemio` reads structure files. Official
`mace-ref` is used only for optional export and independent validation.

## Build and run

CUDA Toolkit, cuBLAS, CMake 3.24+, a C++17 compiler and an NVIDIA GPU are required.
The implementation has been executed on SM86 (RTX 3060 Ti) and SM89 (RTX 4090).
See the validation records for numerical qualifications. Select your architecture
when building. First install Lapis, Chemio, and checkpoint (zlib required), or use
the workspace build in the root README.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/dependencies-install
cmake --build build -j4

build/mace-main model/mh1.model /path/to/165.arc
build/mace-main model/mh1.model /path/to/165.arc --head omat_pbe --frame 0 --json

# Optional: export a processed archive using the reference Python environment.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
.venv/bin/python tools/export_mh1.py model/mh1.model model/mh1.pwm --head matpes_r2scan
PYTHONPATH=python .venv/bin/python tools/validate.py --case Si8 --check
```

Export accepts the six head names in the supplied checkpoint. The archive and its
JSON manifest record the chosen head and checkpoint checksum. MH1 here means this
checkpoint's architecture; other MACE architectures are rejected by the exporter.
All six supplied heads have passed numerical comparison with official MACE.

```python
from ase.build import bulk
from mace_inference import MH1Calculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms.calc = MH1Calculator("model/mh1.pwm", backend="auto")
energy = atoms.get_potential_energy()  # eV, blocking
forces = atoms.get_forces()           # eV/Angstrom, caller atom order
stress = atoms.get_stress()           # eV/Angstrom**3, ASE Voigt convention
print(atoms.calc.diagnostics)
atoms.calc.close()
```

Set `PYTHONPATH=python`, or install this Python package with
`uv pip install --python .venv/bin/python -e .`. Set `MACE_LIBRARY` to the
absolute shared-library path when using an installed package outside this checkout.
The original checkpoint or a processed archive is required. Lennard-Jones is an optional Lapis example.

The C++ entry point is `mace::MH1`, derived from `lapis::Calculator`. A C ABI is in `include/mace/c_api.h`.
`Structure` contains Cartesian float32 coordinates, atomic numbers and a row-vector
lattice. The returned `Result` owns its energy, per-atom energies, forces and virial.
Virial is `-dE/d(strain)`; stress is `-virial / abs(det(cell))`.

## Native checkpoint and file APIs

`mace::MH1 model(path, options, "omat_pbe")` accepts the original MH1 Python
checkpoint. An omitted/empty head defaults to `matpes_r2scan`. Processed PWMH1001
archives remain supported and reject explicit head selection because their head
is already fixed. `model.calculate_file(path, frame)` reads a zero-based ARC frame
through Chemio and returns the usual result. The ASE frontend accepts original
checkpoints and an optional `head=` argument; C clients use `mace_create_mh1_head`.
The original C entry point remains available with the default head.

`mace-main` accepts `--frame N`, `--backend auto|brute|cell`, `--head NAME`,
`--json`, and `--output FILE`. It reports the selected head on stderr, and total
energy, forces in input order, and row-major virial on stdout. JSON additionally
includes per-atom energies and units. ARC input must have three periodic axes.

Loading is data-only: no serialized Python or TorchScript graph is executed.
The native loader reconstructs normalized linear maps, species projections,
tensor products, and sparse cubic contractions in host memory before GPU upload.
Supported files are the supplied two-layer, six-head MH1 architecture and
compatible re-saved checkpoints with changed weights. Other MACE architectures
are not supported. Mathematical Wigner coefficients are checked-in constants,
regenerable with `tools/generate_cg.py`; they contain no trained weights.

CPU extraction regression (run in an upstream MACE environment):

```sh
python tests/check_checkpoint.py --checkpoint model/mh1.model \
  --dump build/mace_weights_dump
```

This checks all six heads and a changed-weight checkpoint against the independent
Python exporter. Set `MACE_TEST_CHECKPOINT` to register native C ABI inference
checks against an original model file with CTest.
See [native loading validation](docs/NATIVE_LOADING.md) for the measured results.

## Runtime contract

* One structure, three periodic axes, float32 geometry and model arithmetic.
* Calls block through output transfer. Concurrent calls on one calculator are rejected.
* Device capacity grows and is reused; result transfer uses stable pinned host staging.
* Brute force supports small cells and periodic self-images. The occupancy-aware
  grid backend uses reciprocal plane heights, a 27-offset stencil, histogram,
  prefix scan and scatter. Dense metadata is coarsened under a configurable budget.
* Full directed edges use `sender + image * cell - receiver`, `r2 <= cutoff2`,
  and identity-based exclusion of the central self-edge. Exposed shifts refer to
  the input coordinates even when those coordinates are unwrapped.
* Pair energy counting, reductions, numerical behavior and analytical derivatives
  belong to the operation. See [the L1 guide](../lapis/docs/PAIR_OP.md).

Partial periodicity, persistent edge features, general autodiff, heterogeneous
batches, long-range solvers, CUDA Graph capture and asynchronous public calls are
outside v1. Floating-point atomics are not bitwise deterministic.

The C API exports `mace_create_mh1`, `mace_calculate`, `mace_diagnostics`,
`mace_destroy` and `mace_last_error` from `libmace.so`. Python is packaged as
`lapis-mace-inference` and imports as `mace_inference`, avoiding a collision with
the upstream `mace` reference package.

Set `MACE_TEST_ARCHIVE` at configure time to register native C API tests with CTest.
For standalone explicit benchmarks, install Lapis with
`LAPIS_BUILD_EXPLICIT_REFERENCE=ON` before configuring MACE.

## Validation and benchmarks

```sh
.venv/bin/python tools/validate.py --check --output benchmarks/correctness.json
.venv/bin/python tools/check_invariants.py --archive model/mh1.pwm
.venv/bin/python tools/check_suite.py --sanitizers

# Short development run. Omit --quick for 100 warmups and 100 windows of 100 calls.
.venv/bin/python tools/benchmark.py --quick --repeat 2

# Optional, separately compiled explicit-connectivity comparison; production stays implicit.
cmake -S . -B build -DMACE_BUILD_EXPLICIT_REFERENCE=ON
cmake --build build -j4
.venv/bin/python tools/benchmark.py --quick --implementations framework explicit cueq

# Pinned MPtrj supercell, full 100-warmup / 100-window / 100-call protocol.
.venv/bin/python tools/benchmark.py \
  --structure structures/mptrj-mp-1005792.extxyz --supercell 3 3 1 \
  --dataset-revision 10.6084/m9.figshare.23713842.v2 \
  --structure-id mp-1005792/mp-1012897-0-0 \
  --implementations framework explicit cueq \
  --output benchmarks/replay-mptrj-252-v1.json
```

Replay uses the same perturbed structures for every implementation, includes ASE
entry, transfers, preprocessing, model computation, forces, stress and blocking
completion, and randomizes implementation order between windows. It reports wall
and CUDA event timing, actual directed edge counts, allocation accounting, available
process-memory observations, seeds, checksums and versions. `--structure file.extxyz
--frame N` accepts an external structure; the default is explicitly labeled synthetic
silicon. ASE's geometry cache is bypassed for every timed call.
The bundled MPtrj fixture has [versioned provenance](structures/README.md).

See the [3060 Ti validation record](docs/VALIDATION.md) and
[RTX 4090 cluster results](docs/RTX4090.md). On the RTX 4090, 50,176 atoms completed
at a 16.922-GiB sampled process peak. The 252-atom full replay measured 4.645 ms
for implicit CUDA, 44.198 ms for eager cuEquivariance, and 21.412 ms for compiled
cuEquivariance. The cluster run uses a frozen CUDA 12 environment and records
float32 energy-reduction and virial-repeat limitations. Comparison against an
independent specialized implicit implementation remains open. Docker reproduction
is pending; the original container definition is in `Dockerfile`.

On the RTX 3060 Ti, the full-protocol 252-atom MPtrj replay measured 14.45 ms versus 54.15 ms for
cuEquivariance (3.75×), with 0.3% implicit overhead against our explicit-list build.
The denser 512-atom development case measured 2.19× speedup. All six heads,
analytical derivatives, invariances and CUDA sanitizer checks pass. See the
validation record for hardware, precision and measurement limitations.
