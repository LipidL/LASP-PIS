# RTX 4090 cluster results

Generated from the adjacent JSON reports. Times are blocking ASE calls including energy, forces, and stress. Capacity runs use one implementation per process.

## Matched replay

| Case | Implementation | Median (ms) | p95 (ms) | Calls |
| --- | --- | ---: | ---: | ---: |
| 28 atoms | Implicit CUDA | 3.776 | 4.217 | 10000 |
| 28 atoms | Explicit CUDA | 4.164 | 4.706 | 10000 |
| 28 atoms | PyTorch / e3nn | 36.971 | 38.038 | 10000 |
| 28 atoms | cuEquivariance | 37.967 | 47.660 | 10000 |
| 28 atoms | PyTorch / e3nn + compile | 7.098 | 7.409 | 10000 |
| 28 atoms | cuEquivariance + compile | 15.534 | 15.880 | 10000 |
| 252 atoms | Implicit CUDA | 4.645 | 5.076 | 10000 |
| 252 atoms | Explicit CUDA | 5.047 | 5.660 | 10000 |
| 252 atoms | PyTorch / e3nn | 47.867 | 49.234 | 10000 |
| 252 atoms | cuEquivariance | 44.198 | 45.599 | 10000 |
| 252 atoms | PyTorch / e3nn + compile | 28.823 | 29.645 | 10000 |
| 252 atoms | cuEquivariance + compile | 21.412 | 23.068 | 10000 |
| 512 atoms | Implicit CUDA | 9.871 | 10.171 | 10000 |
| 512 atoms | Explicit CUDA | 10.105 | 10.654 | 10000 |
| 512 atoms | PyTorch / e3nn | 118.439 | 119.422 | 10000 |
| 512 atoms | cuEquivariance | 51.165 | 58.446 | 10000 |
| 512 atoms | PyTorch / e3nn + compile | 70.985 | 71.750 | 10000 |
| 512 atoms | cuEquivariance + compile | 30.072 | 31.307 | 10000 |

## Isolated capacity and development timing

These runs use 10 warmups and 50 measured calls (5 windows of 10). They establish capacity and provide development timing; they are not the full v1 timing protocol.

| Atoms | Implementation | Status | Median (ms) | Sampled process peak (GiB) | Tracked context peak (GiB) |
| ---: | --- | --- | ---: | ---: | ---: |
| 4096 | Implicit CUDA | completed | 44.655 | 1.758 | 1.347 |
| 4096 | Explicit CUDA | completed | 44.072 | 1.762 | 1.350 |
| 4096 | PyTorch / e3nn | OOM | — | 23.223 | — |
| 4096 | cuEquivariance | completed | 161.218 | 8.098 | — |
| 4096 | PyTorch / e3nn + compile | OOM | — | 18.666 | — |
| 4096 | cuEquivariance + compile | completed | 111.083 | 13.635 | — |
| 12544 | Implicit CUDA | completed | 133.343 | 4.547 | 4.126 |
| 12544 | Explicit CUDA | completed | 131.364 | 4.557 | 4.144 |
| 12544 | PyTorch / e3nn | OOM | — | 23.480 | — |
| 12544 | cuEquivariance | OOM | — | 23.316 | — |
| 12544 | PyTorch / e3nn + compile | OOM | — | 19.748 | — |
| 12544 | cuEquivariance + compile | OOM | — | 20.279 | — |
| 25088 | Implicit CUDA | completed | 265.681 | 8.674 | 8.253 |
| 25088 | Explicit CUDA | completed | 261.493 | 8.693 | 8.270 |
| 25088 | PyTorch / e3nn | OOM | — | 21.797 | — |
| 25088 | cuEquivariance | OOM | — | 23.307 | — |
| 25088 | PyTorch / e3nn + compile | OOM | — | 14.441 | — |
| 25088 | cuEquivariance + compile | OOM | — | 23.621 | — |
| 50176 | Implicit CUDA | completed | 529.312 | 16.922 | 16.506 |
| 50176 | Explicit CUDA | completed | 522.150 | 16.959 | 16.574 |
| 50176 | PyTorch / e3nn | OOM | — | 23.482 | — |
| 50176 | cuEquivariance | OOM | — | 23.461 | — |
| 50176 | PyTorch / e3nn + compile | OOM | — | 14.238 | — |
| 50176 | cuEquivariance + compile | OOM | — | 23.441 | — |

## Numerical qualifications

The strict total-energy comparison uses the float64 official reference. Float32 baseline reports also retain their raw total-energy error and separately compare the sum of returned per-atom energies in float64. The benchmark retains the stock baseline's energy arithmetic.

The 252-atom native repeat test exceeds the strict 1e-5 eV virial repeat target: implicit range 1.1563e-5 eV, explicit range 1.5259e-5 eV over 100 calls. Energy and force ranges meet their strict targets. See `repeat-variability.json` and the correctness reports.

The environment is bare-metal Linux with a pinned CUDA 12 stack. Docker reproduction is pending. Comparisons with the earlier WSL2/3060 Ti campaign include software and host-platform changes.
