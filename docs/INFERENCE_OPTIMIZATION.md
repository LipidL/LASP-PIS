# Inference optimization measurements

Measured November 1, 2026 on an RTX 3060 Ti with CUDA Toolkit 12.6.77.
Both baseline commit `3463a92d5335167a03cd283753afca5b43f0107f` and optimized
libraries were built in Release mode for SM86. The previous local build used
SM52; these comparisons exclude that architecture change.

The changes preserve FP32 arithmetic and the default direct radial evaluation:

- HPNN specializes radial loop bounds and matrix strides for B=48, H=256,
  P=128, three layers, while retaining the general implementation for other
  supported dimensions. The supplied weights are not baked into the kernels.
- MACE combines contiguous component matrices sharing weights into ordinary
  GEMMs. It skips destination clearing only when validated paths cover every
  destination exactly once; overlapping or incomplete paths retain accumulation.
  Radial network depth and output strides use the supported architecture's constants.
- Lapis caches kernel resource checks per calculator, device, kernel, block shape,
  and dynamic shared-memory request. Diagnostics remain available on every call.
  Shared-memory opt-in cannot be lowered by another calculator's smaller request.

Median complete blocking NumPy/C ABI latency, including input conversion,
preprocessing, transfers, energy, forces, and virial:

| Model | Atoms | Baseline ms | Optimized ms | Speedup |
|---|---:|---:|---:|---:|
| HPNN direct | 8 (synthetic) | 2.226 | 1.941 | 1.15× |
| HPNN direct | 64 (synthetic) | 2.598 | 2.147 | 1.21× |
| HPNN direct | 512 (synthetic) | 8.811 | 6.169 | 1.43× |
| HPNN direct | 165 (ARC) | 5.269 | 3.783 | 1.39× |
| HPNN table | 8 (synthetic) | 1.218 | 1.209 | 1.01× |
| HPNN table | 64 (synthetic) | 1.567 | 1.561 | 1.00× |
| HPNN table | 512 (synthetic) | 2.975 | 2.983 | 1.00× |
| HPNN table | 165 (ARC) | 1.885 | 1.836 | 1.03× |
| MACE | 8 (synthetic) | 3.253 | 3.219 | 1.01× |
| MACE | 64 (synthetic) | 5.286 | 4.211 | 1.26× |
| MACE | 512 (synthetic) | 21.651 | 20.773 | 1.04× |
| MACE | 165 (ARC) | 14.135 | 12.880 | 1.10× |

Each result has five warmup calls per library, followed by five windows of
20 calls. Baseline/optimized order is randomized per window; both receive
identical perturbed coordinates. The ARC case uses the first frame of the
user-supplied `structures/165.arc`. Other cases are synthetic silicon grids,
using the supplied trained checkpoints. Raw samples and checkpoint hashes are in
[the measurement artifact](results/inference-optimization-sm86.json).
The GPU also drives a display; changes of a few percent are not conclusive.
HPNN table mode and eight-atom MACE show essentially unchanged performance.
These measurements do not establish an absolute performance ceiling or results
on other GPUs, checkpoints, or structure sizes.

All paired comparisons passed. Maximum observed force differences were below
4e-6 eV/Angstrom, atomic-energy differences below 2e-6 eV, and virial differences
below 2e-5 eV. Floating-point accumulation is not bitwise deterministic.
The table comparisons compare baseline table mode with optimized table mode;
they do not measure interpolation error against direct evaluation.

Validation passed:

- All four CTest tests, including HPNN's independent float64/autograd oracle,
  odd dimensions, derivatives, invariances, and framework geometry tests.
- A new framework regression for changing shared-memory requests, cache reuse,
  and resource-limit rejection following a cached successful launch.
- MACE native C ABI, both neighbor backends, permutation, empty input, and recovery.
- Both global library load orders in the coexistence test.
- CUDA memcheck for HPNN and MACE, and synccheck for HPNN: zero errors.

The upstream MACE Python oracle was not rerun in this environment; MACE validation
here uses baseline numerical comparisons and its native regression suite.
Rebuild consumers against the updated Lapis headers and library.

To reproduce, build the baseline revision separately with the same compiler,
Release mode, and `-DCMAKE_CUDA_ARCHITECTURES=86`. Copy its `libhpnn.so` and
`libmace.so` into a baseline directory, then run from the optimized workspace:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j4
python3 tools/compare_inference.py --baseline /path/to/baseline-libraries \
  --calls 20 --windows 5 --structure structures/165.arc \
  --output docs/results/inference-optimization-sm86.json
ctest --test-dir build --output-on-failure
```

Omit `--structure` if the user-supplied ARC fixture is unavailable.
