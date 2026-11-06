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
Select your architecture when building. 
First install Lapis, Chemio, and checkpoint (zlib required), or use
the workspace build in the root README.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/dependencies-install
cmake --build build -j4

build/mace-main model/mh1.model /path/to/165.arc
build/mace-main model/mh1.model /path/to/165.arc --head omat_pbe --frame 0 --json

# Command-line runner with the same structure and head options.
PYTHONPATH=python .venv/bin/python tools/run_mace.py \
  model/mh1.model /path/to/165.arc --head matpes_r2scan --frame 0
```

Export accepts the six head names in the supplied checkpoint. The archive and its
JSON manifest record the chosen head and checkpoint checksum. MH1 here means this
checkpoint's architecture; other MACE architectures are rejected by the exporter.
All six supplied heads have passed numerical comparison with official MACE.

The NumPy frontend takes Cartesian positions, atomic numbers and a row-vector
lattice, and returns total/atomic energies, forces, virial and stress:

```python
# PYTHONPATH=python
from mace_inference import MH1

with MH1("model/mh1.pwm", backend="auto") as model:
    result = model.calculate(positions, atomic_numbers, cell)
    print(result["energy"], result["forces"], result["stress"])
    print(model.diagnostics)
```

With ASE installed, `python/mace_inference_ase.py` provides `MH1Calculator`:

```python
from ase.build import bulk
from mace_inference_ase import MH1Calculator

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
The original checkpoint or a processed archive is required; original checkpoints
select a head with `head=` (default `matpes_r2scan`), while processed archives have
a fixed head. `tools/run_mace.py` reads ARC and NPZ directly, and other formats
through ASE when available. Lennard-Jones is an optional Lapis example.

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
