# Extraction validation

Validated on 2026-09-12 with CUDA Toolkit 12.6.77, GCC 11.4 and an RTX 3060 Ti
(SM86). GPU tests run outside the filesystem sandbox because it blocks CUDA access.

- Combined Release build: Lapis, MACE and the explicit-connectivity MACE variant.
- Framework conformance and native MACE C ABI tests (implicit and explicit).
- Installed-package check: copied standalone consumer sources build against a moved
  Lapis installation; the installed MACE package compiles external C, C++ and CUDA
  PairOp clients. A C++ client also performs empty-structure inference with MACE.
- Original-versus-extracted libraries: the supplied MH1 omat archive,
  all three backend selections, mixed elements, periodic self-images and empty inputs.
  Forced CellList in a small cell is checked for the expected rejection.
- Renamed MACE ASE frontend: library discovery, energy, forces,
  stress, and close/context-manager behavior.
- Python syntax compilation and cluster shell syntax checks.

Reproduce the installed-package check with `python3 tools/check_packages.py`.
Configure the workspace with `MACE_TEST_ARCHIVE` pointing at an exported MH1 archive
or `MACE_TEST_CHECKPOINT` pointing at the original MH1 checkpoint, then run CTest
to include native inference checks. MACE native tests require NumPy. The workspace
README contains the build commands.

The independent upstream MACE checkpoint comparison, Docker image build, full
performance campaign and CUDA sanitizer campaign were not rerun for this extraction.
The upstream MACE reference checkout/environment is absent here. Existing historical
reports remain unchanged and do not substitute for new measurements.
