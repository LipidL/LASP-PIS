# Native potential tests

MACE uses GoogleTest and the public C++ calculator API. It runs the
shared parameterized suite in `physics.cpp`; the model adapter lives in
`mace/tests/physics.cpp`. The native workspace test in `c_api.cpp` also checks
dynamic library loading, interleaved MACE calculators, and their independent
lifetimes through the C API.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j4
ctest --test-dir build -L native --output-on-failure
# Faster local iteration: load MACE once and run its entire suite.
build/mace/mace_tests
# GoogleTest filters and shuffle/repeat work normally.
build/mace/mace_tests --gtest_filter='*NumericalStress*'
```

By default, CTest registers each GoogleTest case separately. Each case launches
a new process, repeating CUDA initialization and model loading. Direct execution
runs cases sequentially in one process and reuses cached models; its speedup is
not due to parallel GoogleTest execution.

For fast whole-suite runs through CTest (also used by the pre-push hook):

```sh
cmake -S . -B build -DLAPIS_TEST_DISCOVER_CASES=OFF
cmake --build build -j4
ctest --test-dir build -j4 --output-on-failure
```

This registers one CTest test per model executable, retaining all of its cases.
CPU tests can run alongside GPU tests, but the `lapis_gpu` lock serializes GPU
executables even with `-j4`. Set `LAPIS_TEST_DISCOVER_CASES=ON` again
to restore individual case selection/reporting in CTest. In whole-suite mode,
use the executable's `--gtest_filter` option for an individual case. The pre-push
check resets this option to `OFF` on each run.

Tests require a CUDA GPU and model weights. The default is
`mace/model/mace-mh1.model`. Override it with `-DMACE_TEST_MODEL=/absolute/model`;
original supported checkpoints and processed archives both work. The old
`MACE_TEST_CHECKPOINT` and `MACE_TEST_ARCHIVE` options
provide defaults when `MACE_TEST_MODEL` is first configured. Missing weights
are a configuration error. Use `BUILD_TESTING=OFF` for inference-only builds.
Standalone model builds use the same tests and options.

CMake builds pinned GoogleTest v1.17.0 with the project's compiler settings.
This avoids accidentally selecting an incompatible installation, such as one
built without RTTI by another project. For an offline build, provide
`-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`. Test discovery does
not initialize CUDA; execution fails visibly when GPU access is unavailable.
CTest serializes these GPU tests using the `lapis_gpu` resource lock.
To explicitly use a compatible installed GoogleTest (1.14+), set
`-DLAPIS_TEST_USE_SYSTEM_GTEST=ON` and optionally `-DGTest_DIR=/path/to/cmake/GTest`.
That installation must support RTTI, as the parameterized suites use it.

## Adding structures

Drop a fully periodic ARC file into `tests/structures/` and rebuild. Each frame
automatically participates in **all** MACE physics tests. No per-test lists need updating.
Alternatively, add a named `Structure` to `structures()` in `physics.cpp`.
Use small, physically reasonable structures with elements supported by MACE.
Negative energy is a fixture/model expectation, not a universal law
for arbitrary trained potentials or highly compressed configurations.

Built-in cases cover empty cells, isolated atoms, distorted water and methane,
BCC Fe, triclinic TiO2, periodic self-images, a Si slab, and separated molecules
across grid bins; the ARC fixture adds distorted rocksalt NaCl.

## Coverage and numerical conventions

- Negative energy for nonempty fixtures; exactly zero energy/virial and empty
  outputs for empty cells; finite outputs and atomic-energy summation.
- Translation, atom permutation, and independent lattice-image shifts preserve
  energy, atomic energies, forces, and virial (with atom order restored).
- Proper three-dimensional rotation preserves energy, rotates forces, and
  transforms the full virial tensor as `R^T virial R` for row-vector geometry.
- Every Cartesian force and every one of nine strain derivatives matches
  fourth-order central finite differences. Strain deforms both atoms and cell
  at fixed fractional coordinates. `stress = -virial / abs(det(cell))`.
- `2x1x1`, `1x2x3`, and `2x2x2` supercells scale total energy and preserve each
  atomic energy, each force, and volume-normalized stress relative to the
  original cell, including the first copy.
- Zero net force, symmetric stress, repeatability, grow/empty/shrink buffer
  reuse, Auto/brute/grid agreement, explicit rejection of unsupported small-cell
  grid geometry, invalid-input errors and recovery.

Invariance tolerances use absolute energy error per atom (no relative allowance
on large atomic baselines), plus small absolute/relative force and virial
tolerances. Numerical forces use 0.004/0.002 Angstrom steps; strain uses
0.002/0.001. Richardson extrapolation reduces truncation without tiny steps
dominated by float32 roundoff. Derivative allowances are 0.003 eV/Angstrom and
0.004/volume eV/Angstrom^3 plus 0.5% of the analytical component. These physics
checks complement, but do not replace, independent reference-model comparisons.

The unrelated checkpoint reader still uses Python to generate serialization
fixtures; the MACE native suite requires neither Python nor PyTorch.
