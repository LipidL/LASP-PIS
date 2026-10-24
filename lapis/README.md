# Lapis

Lapis is a C++17/CUDA framework for `node -> transient edge -> node` computation
in fully periodic atomic structures. It owns geometry preprocessing, implicit
neighbor traversal, PairOp execution, atom-order restoration, result transfer,
diagnostics and reusable device workspaces. It has no MACE, cuBLAS or Python
dependency.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j4
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/install"
```

Consumers link `lapis::lapis`. `include/lapis/runtime.hpp` provides the C++ host
interface and `include/lapis/pair.cuh` provides the CUDA extension interface.
`lapis::Calculator` owns one context; models implement its `evaluate()` method and
specialize PairOps. See [the PairOp contract](docs/PAIR_OP.md). Internal queue and
kernel implementation details are not extension points.

The runtime supports one structure, three periodic axes, float32 geometry,
blocking calls, and brute-force or cell-list traversal. Results retain input atom
order. Cell vectors are rows, displacement is sender minus receiver including the
periodic shift, and virial is `-dE/dstrain`. A calculator rejects concurrent calls.
Energy units are supplied by the model. Floating-point atomics are not bitwise
deterministic. Partial periodicity, batching, autodiff and long-range solvers are
outside the current contract.

The library is static and position-independent, with hidden internal symbols.
CUDA templates compile inside consumers. Public CUDA compilation requirements and
CUDA runtime linkage propagate through the CMake target. Rebuild consumers when
changing Lapis; binary compatibility is not promised.

`LAPIS_BUILD_EXAMPLES` builds the small Lennard-Jones example in `examples/lj`.
Framework tests use it to check forces and virial alongside traversal invariants.
It is not part of the installed runtime API. `LAPIS_BUILD_EXPLICIT_REFERENCE=ON`
adds the optional `lapis::explicit` target for materialized-connectivity benchmarks.
Its compile definition propagates to consumers; link exactly one runtime variant
into a model library.

The [original framework specification](docs/FRAMEWORK_SPEC.md) records design
motivation and wider requirements; this README and the PairOp contract describe
the current supported implementation.
