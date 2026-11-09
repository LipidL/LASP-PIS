# Two-body DFT-D3(BJ)

This CUDA consumer implements the coordination-dependent C6 and C8 two-body
dispersion correction with Becke–Johnson damping. It returns energy, per-atom
energies, forces and virial, including derivatives through coordination numbers.
The ATM three-body term and other damping families are not implemented.

```cpp
#include <d3/d3.hpp>
d3::Calculator dispersion; // PBE-D3(BJ), CN cutoff 40 bohr, dispersion 95 bohr
lapis::Result result = dispersion.calculate(structure);

d3::Parameters bj{1.0, 0.7875, 0.4289, 4.4407}; // s6, s8, a1, a2 (bohr)
d3::Cutoffs cutoffs{10.f, 25.f}; // coordination, dispersion, both in Angstrom
d3::Calculator custom(bj, cutoffs, {lapis::NeighborBackend::BruteForce});
```

Link `d3::d3`. The workspace builds D3 by default; disable it with
`-DLAPIS_BUILD_D3=OFF`. A standalone build uses an installed Lapis package:

```sh
cmake -S d3 -B build-d3 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH=/path/to/installed/lapis
cmake --build build-d3 -j4
ctest --test-dir build-d3 --output-on-failure
```

Atomic numbers 1–94 (H–Pu) are supported. Parameters are bundled; native use
requires no Python, model file, or network access. See [parameter provenance](data/NOTICE.md)
for the source, license and regeneration command. Functional damping parameters
can be supplied directly; the built-in default is PBE-D3(BJ).

The input and output conventions match Lapis: float32 positions and cell in
Angstrom, three periodic axes, energy in eV, force in eV/Angstrom, and virial
`-dE/dstrain` in eV. A molecule can be placed in a cell large enough to keep all
periodic replicas beyond both cutoffs. Coincident atoms are rejected. Calculators
are blocking and single-flight, reuse allocations, and restore input atom order.

Geometry is prepared at the maximum of the two cutoffs. Three pair traversals
perform coordination accumulation, dispersion with `dE/dCN`, and coordination
backpropagation. A node kernel constructs stable normalized Gaussian reference
weights between the first two traversals. Tables, weights and accumulations use
double precision; geometry and returned per-atom arrays follow Lapis float32
conventions. Directed energies and virials carry a half factor, while periodic
self-images contribute virial and coordination dependence without a position force.
No persistent edge arrays or edge features are used by this consumer.

Both cutoffs are independent, positive, inclusive hard cutoffs, with no smoothing
or energy shifting. Energy and derivatives can jump at a cutoff; finite differences
must avoid crossing those boundaries. The shorter traversal filters before tile
compaction. Brute-force traversal tightens periodic-image bounds to its own
radius; cell-list traversal still enumerates candidates from the grid prepared
at the largest radius. Backend selection remains tied to that preparation.

`d3.physics` compares energies, forces and all virial entries against frozen
simple-dftd3 1.2.1 results with `s9=0`. Fixtures cover molecules, mixed elements,
triclinic periodic cells, self-images, heavy elements and either ordering of the
two cutoffs. Tests also check coordinate/strain finite differences, conservation,
permutation and wrapping invariance, backend agreement, reuse and invalid inputs.
Reference generation is separate from test execution:

```sh
python -m pip install dftd3==1.2.1 numpy
python d3/tools/generate_reference.py > d3/tests/reference.inc
```
