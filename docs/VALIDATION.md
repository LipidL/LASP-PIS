# Extraction validation

Validated on 2026-09-12 with CUDA Toolkit 12.6.77, GCC 11.4 and an RTX 3060 Ti
(SM86). GPU tests run outside the filesystem sandbox because it blocks CUDA access.

- Combined Release build: Lapis, MACE, HPNN and the explicit-connectivity MACE variant.
- Framework conformance, native MACE C ABI tests (implicit and explicit), and HPNN's
  self-contained full validation with CPU PyTorch reference, including both generated
  model sizes, analytical derivatives, invariances and radial interpolation.
- Installed-package check: copied standalone consumer sources build against a moved
  Lapis installation; installed MACE/HPNN packages compile external C, C++ and CUDA
  PairOp clients. A C++ client also performs empty-structure inference with both models.
- Original-versus-extracted libraries: the supplied MH1 omat archive and HPNN archive,
  all three backend selections, mixed elements, periodic self-images and empty inputs.
  Forced CellList in a small cell is checked for the expected rejection.
- Both global library load orders: 64 output arrays per order agree with the originals
  within atol=2e-5 and rtol=2e-6. Maximum observed absolute difference: 1.9073486328125e-6.
- Renamed MACE ASE frontend and HPNN frontend: library discovery, energy, forces,
  stress, close/context-manager behavior, and simultaneous use.
- Python syntax compilation and cluster shell syntax checks.

Reproduce the installed-package check with `python3 tools/check_packages.py`.
Configure the workspace with `MACE_TEST_ARCHIVE` pointing at an exported MH1 archive,
then run CTest to include both library load-order tests. MACE native tests require
NumPy; HPNN and coexistence tests additionally require PyTorch. The workspace README
contains the build commands.

The independent upstream MACE checkpoint comparison, Docker image build, full
performance campaign and CUDA sanitizer campaign were not rerun for this extraction.
The upstream MACE reference checkout/environment is absent here. Existing historical
reports remain unchanged and do not substitute for new measurements.
