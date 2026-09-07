# Validation status

This page preserves the RTX 3060 Ti / WSL2 validation record. The later
[RTX 4090 cluster campaign](RTX4090.md) measures 50,176-atom capacity and all six
eager/compiled timing entries in a CUDA 12 environment, with its numerical
qualifications reported separately.

These results do not certify every v1 release gate
in `FRAMEWORK_SPEC.md`. Independent development used the supplied official
`mace-ref` and checkpoint; other implementations on the device were not consulted.

## Correctness

GPU execution: NVIDIA RTX 3060 Ti, 8 GiB, SM86, CUDA Toolkit 12.6.2,
nvcc 12.6.77, GCC 11.4. Official MACE revision:
`59ad3a473ca02101a1cf02db242197e8d616dd11`.
Checkpoint SHA256:
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`.
Reference model arithmetic is float64; both implementations receive identical
float32-rounded coordinates and lattice. CUDA model arithmetic is float32.

| Structure, matpes_r2scan head | Energy error, eV/atom | Maximum force error, eV/Å | Maximum virial error, eV |
| --- | ---: | ---: | ---: |
| Perturbed silicon, 8 atoms | 1.07e-7 | 3.51e-6 | 9.95e-6 |
| Perturbed silicon, 64 atoms | 2.76e-8 | 9.21e-6 | 3.46e-5 |
| Perturbed silicon, 216 atoms, grid | 7.51e-9 | 1.62e-5 | 1.37e-4 |
| Triclinic Si/C/O/H, 4 atoms | 2.59e-7 | 9.39e-6 | 4.55e-6 |
| MPtrj As/Cu/O/Sm, 252 atoms, grid | 3.26e-8 | 8.39e-6 | 4.94e-4 |

Assertions require absolute energy error per atom below `1e-5`, componentwise
force error below `1e-4`, and maximum-component virial error below
`max(1e-4 eV, 1e-4 * max(abs(reference virial)))`. The larger structures pass the
relative virial tolerance. Repeated evaluations check energy per atom below
`1e-6` and force/virial maximum-component changes below `1e-5`.

All other checkpoint heads (`omat_pbe`, `spice_wB97M`, `mp_pbe_refit_add`,
`oc20_usemppbe`, `omol`) pass the mixed-element comparison: energy errors below
`2.1e-7 eV/atom` and force errors below `9.5e-6 eV/Å`. This covers all six heads,
with substantially more geometry coverage for `matpes_r2scan`. The separately
compiled explicit-list backend passes the 216-atom reference comparison.

`tools/check_suite.py --sanitizers` passes:

* Exact directed edge multisets against an independent double-precision CPU
  enumerator: orthogonal/triclinic cells, repeated images, one/two-bin axes,
  unwrapped coordinates, dense/sparse/coarsened grids, and empty structures.
* Queue boundaries, carried remainders, inclusive cutoffs, duplicate prevention,
  padded lanes, shared scratch boundaries, and a 2-D CTA shape.
* LJ analytical derivatives, original order, caller ownership, allocation reuse,
  singular cells, invalid explicit grids, resource rejection and concurrent calls.
* Rejection of truncated, malformed, nonfinite and incompatible MH1 archives,
  including out-of-range tensor-product and polynomial indices.
* MH1 translation, wrapping, permutation, rotation, repeated calls and force/strain
  finite differences with a step sweep.
* CUDA memcheck, racecheck and synccheck on framework conformance tests and on two
  complete MH1 energy/force/virial calls, including the optimized scratch reuse.

On the MH1 invariance case, maximum transformed-force error is `9.1e-6 eV/Å`;
maximum transformed-virial error is `2.7e-5 eV`. The best force finite-difference
error over the step sweep is `5.5e-5 eV/Å`. Strain finite differences use a separate
`0.01 eV` absolute tolerance because float32 energy cancellation is more severe;
the best observed error is `0.00151 eV`. These derivative checks sample selected
components and are complemented by full reference force and virial comparisons.

SM80, SM89 and SM90 compile successfully; those architectures have not been
executed here. A short NVE diagnostic runs 200 steps of 0.5 fs on 8-atom silicon
at an initial 300 K. Maximum total-energy deviation is `9.38e-6 eV/atom` and
final drift is `3.80e-6 eV/atom`. This is a short stability check, not evidence
of long-run drift. Machine-readable results and logs are under `benchmarks/`.
An additional 10,000-step / 5-ps run measured maximum deviation `9.26e-6 eV/atom`
and final drift `4.15e-6 eV/atom` (`benchmarks/md-5ps.json`). It remained finite
throughout; larger systems and substantially longer physical trajectories remain
outside this validation record.

## Performance

The dense development replay uses 512 silicon atoms with lattice constant
4.6 Å, about 86 directed neighbors per atom, and identical changing coordinates
and cells for all implementations. The final development run uses 100 warmups
and 5 windows of 20 calls:

| Implementation | Median blocking latency |
| --- | ---: |
| Implicit framework | 35.91 ms |
| Separately compiled explicit-list CUDA build | 35.76 ms |
| Official MACE + cuEquivariance, no torch.compile | 78.64 ms |

This gives 2.19× speedup over cuEquivariance and 0.4% implicit overhead. See
`benchmarks/replay-dense-reused-scratch.json`. It is a development measurement;
the required release window count is larger.

The dense case initially took 62.6 ms. Four-edge radial projection reuse, warp
normalization, shared product intermediates, sender-input caching, fused Cartesian
reductions and layer-specific scratch capacities brought it to 35.9 ms. Earlier
optimization measurements are retained under `benchmarks/`; their warmup counts,
trajectory versions and source schedules differ, so use them as development
history rather than release comparisons.

The full v1 protocol (100 warmups, 100 windows of 100 calls per implementation)
on the pinned 252-atom MPtrj supercell measured:

| Implementation | Median blocking latency | p95 blocking latency |
| --- | ---: | ---: |
| Implicit framework | 14.45 ms | 15.24 ms |
| Explicit-list CUDA build | 14.41 ms | 15.24 ms |
| Official MACE + cuEquivariance | 54.15 ms | 62.54 ms |

This is 3.75× speedup and 0.3% implicit overhead across 10,000 measured calls per
implementation. Directed edge counts range from 11,452 to 11,472 over the fixed
trajectory. Results are in `benchmarks/replay-mptrj-252-v1.json`. The fixture is
the first frame in the versioned official dataset, with no performance-based
selection. Provenance and supercell construction are in `structures/README.md`.

The 28-atom source cell also completed the full protocol, with 1,272–1,276 directed
edges: framework median/p95 `4.96 / 9.02 ms`, explicit `5.04 / 8.83 ms`, and
cuEquivariance `48.17 / 56.91 ms`. This is 9.71× speedup with the implicit build
1.5% faster than explicit. See `benchmarks/replay-mptrj-28-v1.json`. The relatively
wide native tail illustrates why both distributions and medians are retained.

The optional baseline check on the same 28-atom structure uses 20 warmups and
3 windows of 5 calls. Medians were `5.08 ms` for the framework, `46.25 ms` for
official e3nn MACE, and `21.25 ms` for cuEquivariance with `torch.compile`.
Compilation occurs during warmup. This is development evidence, not the full
release protocol (`benchmarks/replay-optional-baselines.json`).

Wall timing includes ASE entry, host preprocessing, transfers, neighbor discovery,
model computation, forces/stress, synchronization and output copies. Native CUDA
events span device preprocessing through output transfer; PyTorch events bracket
the calculator call on its current stream and can include CPU submission gaps.
End-to-end claims use blocking wall latency. Implementation order is randomized
between windows. Full samples, window means, p95 latency, actual edge counts,
seeds, trajectory/checkpoint/archive/source/library hashes and versions are saved.

The installed cuEquivariance package reports an NVML query failure on WSL and
falls back to its bundled SM86 / RTX A6000 cache metadata. The baseline uses the
stock package behavior, with no local source modifications. Its actual execution
device is the RTX 3060 Ti recorded by CUDA and `nvidia-smi`; results should be
rechecked with working hardware discovery on target deployment devices.

## Memory

Forward temporary reuse reduced execution state from about 535 kB to 353 kB per
atom. A 4,096-atom / 188,416-edge case completed using 1,446,777,631 tracked context
bytes plus 25,860,716 parameter bytes. That earlier development run measured
280.4 ms before the final four-edge scheduling optimizations; it is retained as
capacity evidence, not the current performance number.

The final isolated 4,096-atom run measured `171.9 ms`, with the same tracked
capacity (`benchmarks/replay-capacity-isolated.json`). A separate matched run
measured `177.6 ms` native and `845.0 ms` cuEquivariance, with no OOM. That combined
run observed 7,996 of 8,192 MiB in use, and PyTorch peak reserved capacity plus the
native context exceeded physical VRAM. WSL eviction/paging may affect this larger
comparison; it is not used to certify the throughput gate. The isolated run
observed 2,445 MiB total GPU usage, which includes other GPU/driver allocations
and is not a process peak.

Scaling the allocation formula to 50,000 atoms predicts approximately 16.5 GiB
plus CUDA/cuBLAS overhead. This is an estimate, not a measured 50,000-atom run.
Production allocates no global edge connectivity or edge features. The optional
explicit benchmark library stores only sender/image connectivity, then runs the
same model mathematics over transient tiles.

`context_bytes` is live grow-only device capacity. `context_peak_bytes` additionally
accounts for growth while an old buffer is still live. Parameter bytes, PyTorch
allocated/reserved peaks and process observations are distinct fields. With several
PyTorch comparators in one process, the reported PyTorch peak is explicitly combined.
On this WSL device, process-level GPU memory queries return `[N/A]`; no process
peak is claimed. Host pinned staging is excluded from GPU allocation counts.
The benchmark now includes an optional NVML observer with 20-ms polling. It reports
an explicitly sampled process peak on supported systems, or an unavailable/partial
status. Polling can miss shorter transients. Run one implementation per process
when measuring isolated process memory; interleaved timing retains every loaded
calculator. Container PID namespaces may prevent NVML from identifying the process.

## Remaining release gates

* The 50,000-atom capacity target has since been measured on the RTX 4090:
  50,176 atoms and a 16.922-GiB sampled process peak. See the cluster record for
  the protocol and remaining numerical qualifications.
* Evaluate the 10% overhead gate against a separately supplied specialized
  baseline. The user's instruction to avoid existing implementations was preserved.
* Extend matched replay to large MPtrj cases and more source structures. Small
  and medium supercells pass the full protocol; one fixture does not establish
  dataset-wide performance.
* Run SM80/90 hardware and longer NVE trajectories before broad release claims.
  SM89 execution is covered by the cluster record.
* Build and execute the Docker recipe. Package pins mirror the supplied environment.
  `docker version` reports that Docker is unavailable in this WSL distro and asks
  for Docker Desktop WSL integration; the image could not be built here.
