# Periodic Pairwise

A C++17/CUDA implementation of `node -> transient edge -> node` computation in
fully periodic atomic structures. The runtime streams compacted neighbor tiles
through statically specialized `PairOp`s. Production inference stores node tensors,
cell metadata and immutable weights; it never materializes global edge features.

The repository includes force-shifted Lennard-Jones and complete inference for the
supplied MACE MH1 checkpoint: both density-normalized, gated message layers, cubic
products, residuals, readouts, ZBL, analytical forces and virial. C++ inference does
not link to PyTorch, e3nn, cuEquivariance or Python. Official `mace-ref` is used only
for checkpoint export and independent validation.

## Build and run

CUDA Toolkit, cuBLAS, CMake 3.24+, a C++17 compiler and an NVIDIA GPU are required.
The implementation has been executed on SM86 (RTX 3060 Ti) and SM89 (RTX 4090).
See the validation records for numerical qualifications. Select your architecture
when building; the default build targets SM80, SM86, SM89 and SM90.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j4
build/pairwise_tests

# Run with the supplied Python environment. Export one selected checkpoint head.
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
from pairwise import PairwiseCalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms.calc = PairwiseCalculator("model/mh1.pwm", backend="auto")
energy = atoms.get_potential_energy()  # eV, blocking
forces = atoms.get_forces()           # eV/Angstrom, caller atom order
stress = atoms.get_stress()           # eV/Angstrom**3, ASE Voigt convention
print(atoms.calc.diagnostics)
atoms.calc.close()
```

Set `PYTHONPATH=python`, or install this Python package with
`uv pip install --python .venv/bin/python -e .`. Set `PAIRWISE_LIBRARY` to the
absolute shared-library path when using an installed package outside this checkout.
Omit the model argument for Lennard-Jones; its parameters are `epsilon`, `sigma`
and `cutoff`.

The C++ entry points are `pairwise::LennardJones`, `pairwise::MH1` and
`Calculator::calculate(Structure)`. A C ABI is in `include/pairwise/c_api.h`.
`Structure` contains Cartesian float32 coordinates, atomic numbers and a row-vector
lattice. The returned `Result` owns its energy, per-atom energies, forces and virial.
Virial is `-dE/d(strain)`; stress is `-virial / abs(det(cell))`.

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
  belong to the operation. See [the L1 guide](docs/PAIR_OP.md).

Partial periodicity, persistent edge features, general autodiff, heterogeneous
batches, long-range solvers, CUDA Graph capture and asynchronous public calls are
outside v1. Floating-point atomics are not bitwise deterministic.

## Validation and benchmarks

```sh
.venv/bin/python tools/validate.py --check --output benchmarks/correctness.json
.venv/bin/python tools/check_invariants.py --archive model/mh1.pwm
.venv/bin/python tools/check_invariants.py
.venv/bin/python tools/check_suite.py --sanitizers
compute-sanitizer --tool memcheck --error-exitcode 1 build/pairwise_tests
compute-sanitizer --tool racecheck --error-exitcode 1 build/pairwise_tests

# Short development run. Omit --quick for 100 warmups and 100 windows of 100 calls.
.venv/bin/python tools/benchmark.py --quick --repeat 2

# Optional, separately compiled explicit-connectivity comparison; production stays implicit.
cmake -S . -B build -DPAIRWISE_BUILD_EXPLICIT_REFERENCE=ON
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
