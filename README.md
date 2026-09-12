# Lapis workspace

Three independently buildable C++17/CUDA projects share this repository:

- [Lapis](lapis/README.md): periodic geometry, neighbor traversal, PairOp execution and reusable device workspaces.
- [MACE](mace/README.md): MACE MH1 inference, model export, C API and ASE frontend.
- [HPNN](hpnn/README.md): simplified HPNN inference, model export, C API and NumPy/ASE frontends.

MACE and HPNN depend only on Lapis's public CMake target and headers. Lapis has no
model dependency. Each model library embeds a position-independent static Lapis
runtime; internal symbols are hidden. CUDA runtime and cuBLAS remain external
runtime dependencies. PairOp templates compile with the consuming model.

## Build together

Requires CMake 3.24+, a C++17 compiler and CUDA Toolkit. Select your GPU architecture:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j4
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/install"
```

Tests require GPU access. HPNN validation additionally requires Python, NumPy and
PyTorch. Use `-DBUILD_TESTING=OFF` for a library-only build. MACE's archive-dependent
native checks are enabled by `-DMACE_TEST_ARCHIVE=/absolute/path/model.pwm`.
Upstream MACE validation additionally requires its separate reference environment.

Outputs are `build/lapis/liblapis.a`, `build/mace/libmace.so` and
`build/hpnn/libhpnn.so`. Set `MACE_LIBRARY` and `HPNN_LIBRARY` for the Python
frontends, or use their `library=` argument. The frontend also searches these
workspace build locations and each project's own `build/` directory.

`LAPIS_BUILD_MACE` and `LAPIS_BUILD_HPNN` select consumers. Set
`MACE_BUILD_EXPLICIT_REFERENCE=ON` to build the optional MACE benchmark library and
its separate Lapis runtime. Production inference uses implicit connectivity.

## Build independently

```sh
cmake -S lapis -B build-lapis -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=OFF
cmake --build build-lapis -j4
cmake --install build-lapis --prefix "$PWD/install-lapis"
cmake -S mace -B build-mace -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH="$PWD/install-lapis" -DBUILD_TESTING=OFF
cmake --build build-mace -j4
cmake -S hpnn -B build-hpnn -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH="$PWD/install-lapis" -DBUILD_TESTING=OFF
cmake --build build-hpnn -j4
```

Both consumers use `find_package(lapis 0.1.0 EXACT CONFIG REQUIRED)` when the
workspace has not already supplied `lapis::lapis`. The projects never compile a
sibling's sources. Installed model packages export `mace::mace` and `hpnn::hpnn`.
New executables can link those targets without duplicating inference code.

## Development contract

The API and ABI are unreleased and may change. Rebuild consumers against matching
Lapis headers and library with a compatible CUDA/C++ toolchain. Dependency versions
are checked exactly until a compatibility policy is established.

Keep implementation, tests, documentation and packaging inside the owning project.
Framework changes must pass framework and consumer correctness checks. The package
boundary check in `tools/check_packages.py` builds copied consumer source trees
against a relocated Lapis installation, then compiles C and C++ clients using the
installed model packages. No GPU is required for that build check.

See [migration provenance](docs/MIGRATION.md) for the imported histories and naming
changes. Historical benchmark reports retain their original names and measurements;
they are not new measurements of this extraction.
