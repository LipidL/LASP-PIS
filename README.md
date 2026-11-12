# Lapis workspace

Six independently buildable projects share this repository:

- [Lapis](lapis/README.md): periodic geometry, neighbor traversal, PairOp execution and reusable device workspaces.
- [Chemio](chemio/README.md): C++17 ARC structure parsing, with no model or CUDA dependencies.
- [checkpoint](checkpoint/README.md): C++17 data-only PyTorch ZIP64/pickle reading, depending only on zlib.
- [MACE](mace/README.md): MACE MH1 inference, native checkpoint loading, executable, C API and ASE frontend.
- [HPNN](hpnn/README.md): simplified HPNN inference, native checkpoint loading, executable, C API and NumPy/ASE frontends.
- [D3](d3/README.md): two-body DFT-D3(BJ) dispersion with coordination-dependent forces and virial.

MACE and HPNN consume the public Lapis, Chemio, and checkpoint CMake targets.
Lapis and Chemio remain independent of each other and of both models. Each model
library embeds position-independent static dependencies; internal symbols are
hidden. CUDA runtime, cuBLAS, and zlib remain external runtime dependencies.
PairOp templates compile with the consuming model. No Python or LibTorch is
needed for native loading or inference. Python export tools remain available.

## Build together

Requires CMake 3.24+, a C++17 compiler, CUDA Toolkit, and zlib development headers.
Select your GPU architecture:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j4
build/mace-main mace/model/mh1.model structures/165.arc
build/hpnn-main hpnn/model/ggnn.pth structures/165.arc
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/install"
```

HPNN and MACE use native GoogleTest suites and require GPU access, GoogleTest,
and their model weights. Run them with `ctest --test-dir build -L native
--output-on-failure`. See [native testing](tests/README.md) for model overrides,
coverage, and adding structures. The checkpoint reader tests still use Python.
Use `-DBUILD_TESTING=OFF` for a library-only build.

Outputs are `build/lapis/liblapis.a`, `build/mace/libmace.so` and
`build/hpnn/libhpnn.so`, plus `build/mace-main` and `build/hpnn-main`.
The independent CPU archives are `build/chemio/libchemio.a` and
`build/checkpoint/libcheckpoint.a`. Set `MACE_LIBRARY` and `HPNN_LIBRARY` for the Python
frontends, or use their `library=` argument. The frontend also searches these
workspace build locations and each project's own `build/` directory.

`LAPIS_BUILD_MACE`, `LAPIS_BUILD_HPNN` and `LAPIS_BUILD_D3` select consumers. D3
builds `build/d3/libd3.a` and needs only Lapis. Set
`MACE_BUILD_EXPLICIT_REFERENCE=ON` to build the optional MACE benchmark library and
its separate Lapis runtime. Production inference uses implicit connectivity.

## Build independently

```sh
cmake -S lapis -B build-lapis -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=OFF
cmake --build build-lapis -j4
cmake --install build-lapis --prefix "$PWD/install-lapis"
cmake -S chemio -B build-chemio -DBUILD_TESTING=OFF
cmake --build build-chemio -j4
cmake --install build-chemio --prefix "$PWD/install-lapis"
cmake -S checkpoint -B build-checkpoint -DBUILD_TESTING=OFF
cmake --build build-checkpoint -j4
cmake --install build-checkpoint --prefix "$PWD/install-lapis"
cmake -S mace -B build-mace -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH="$PWD/install-lapis" -DBUILD_TESTING=OFF
cmake --build build-mace -j4
cmake -S hpnn -B build-hpnn -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH="$PWD/install-lapis" -DBUILD_TESTING=OFF
cmake --build build-hpnn -j4
```

Both consumers use `find_package(... 0.1.0 EXACT CONFIG REQUIRED)` for Lapis,
Chemio, and checkpoint when the workspace has not supplied those targets. The
projects never compile a sibling's sources. Installed model packages export
`mace::mace` and `hpnn::hpnn`; installed executables go in `bin/`.

## Native inference

Both commands accept `MODEL STRUCTURE.arc`, followed by `--frame N` (zero based,
default 0), `--backend auto|brute|cell`, `--json`, and `--output FILE`.
MACE additionally accepts `--head NAME`, defaulting to `matpes_r2scan` for an
original checkpoint. Processed archives retain their selected head. Output
contains total energy (eV), forces in input atom order (eV/Angstrom), and virial
(eV, `-dE/dstrain`, row major). JSON also contains atomic energies and units.
Errors go to stderr with nonzero exit status.

Support is limited to the supplied MH1 and simplified HPNN architectures and
compatible re-saved checkpoints, including changed trained weights. Native
loading never executes serialized Python or TorchScript code. The checkpoint
utility is a separate package so neither chemical structure I/O nor Lapis needs
to know PyTorch serialization. All preprocessing happens in host memory at model
construction; inference uses the existing CUDA kernels. ARC parsing supports
multiple frames and triclinic cells, while inference requires three periodic axes.

Original model and example structure assets must be provided separately when
extracting individual repositories. Each component owns its build, implementation,
tests, and package configuration. Workspace integration tests stay at the root.

Native loading validation records are in [MACE](mace/docs/NATIVE_LOADING.md) and
[HPNN](hpnn/docs/NATIVE_LOADING.md). Enable checkpoint-specific CTest checks with
`MACE_TEST_CHECKPOINT` and `HPNN_TEST_CHECKPOINT` set to the original model paths.

## Development contract

### Local checks

Install the Git hooks once per clone using Python 3.10+:

```sh
python3 -m pip install 'pre-commit>=3.2,<5'
pre-commit install
```

The configuration installs both `pre-commit` and `pre-push` hooks. In linked Git
worktrees, hook installation is shared with the other worktrees; install only
when those branches also contain this configuration. All hook implementations
are local and run without downloading additional hook environments.

Pre-commit checks staged text for trailing whitespace, missing final newlines,
conflict markers, and Python/JSON syntax. It also checks C++/CUDA formatting.
Checks never rewrite files; fix the reported problems and stage the corrections.
The formatter is pinned to **clang-format 23.1.4**, matching the existing style
configuration. Put it on `PATH`, or set `CLANG_FORMAT` to its executable path.
Use that same executable with `-i` to fix formatting. Changing `.clang-format`
checks all tracked C++/CUDA files.

Pre-push repeats those checks for files selected by pre-commit from the pushed
commits, then builds and tests **chemio** and **checkpoint** independently in
Debug mode with `-Wall -Wextra -Wpedantic -Werror`. It requires CMake 3.24+,
a GCC/Clang-compatible C++17 compiler, and zlib development headers. It needs
neither CUDA, a GPU, model weights, nor Python packages such as PyTorch.
Incremental build trees are kept in the ignored `build-host-hooks/` directory.
Remove that directory when changing compilers/toolchains. Builds use two jobs;
each test has a 120-second timeout, and discovering zero tests is an error.

Run checks manually, including on a CPU-only CI runner:

```sh
pre-commit run --all-files
pre-commit run --hook-stage pre-push --all-files
# The same underlying checks, without installing pre-commit:
python3 tools/checks.py source --all
python3 tools/checks.py format --all
python3 tools/checks.py host
```

Local hooks are developer feedback and can be bypassed. The host suite does not
compile or test the CUDA components, and does not yet include clang-tidy or
sanitizers. Those remain separate additions to the verification pipeline.

### Package boundaries

The API and ABI are unreleased and may change. Rebuild consumers against matching
Lapis headers and library with a compatible CUDA/C++ toolchain. Dependency versions
are checked exactly until a compatibility policy is established.

See [migration provenance](docs/MIGRATION.md) for the imported histories and naming
changes. Historical benchmark reports retain their original names and measurements;
they are not new measurements of this extraction.

See [inference optimization measurements](docs/INFERENCE_OPTIMIZATION.md) for
the current SM86 baseline comparison, validation, and reproduction commands.
