# Native checkpoint validation — 2026-09-12

Validated `model/mh1.model` with CUDA 12.6 on an RTX 3060 Ti. All 117 normalized
records match the Python exporter byte for byte for every supplied head:
`matpes_r2scan`, `mp_pbe_refit_add`, `spice_wB97M`, `oc20_usemppbe`, `omol`, and
`omat_pbe`. A re-saved checkpoint with a changed embedding also matches.
The independent exporter uses upstream MACE, e3nn 0.4.4, PyTorch 2.11.0, and NumPy
2.2.6. These are validation dependencies only.

The CLI passes native-versus-exported inference comparisons for all six heads on
frame zero of the 165-atom triclinic ARC trajectory. The largest force difference
is 1.91e-6 eV/Angstrom and the largest virial difference is 1.53e-5 eV. Small
differences between repeated evaluations reflect float32 atomic reductions; the
normalized input records themselves are identical.

An independent official MACE float64 calculation on perturbed Si8 passes the
existing correctness thresholds: energy error 1.07e-7 eV/atom, maximum force error
3.57e-6 eV/Angstrom, and maximum virial error 9.95e-6 eV. Native C ABI checks also
pass for all neighbor backends, permutation, empty input, and error recovery.

Raw measurements and checkpoint/structure checksums are in
[benchmarks/native-loading.json](../benchmarks/native-loading.json).

```sh
# Run in the upstream MACE Python environment.
python tests/check_checkpoint.py --checkpoint model/mh1.model --dump build/mace_weights_dump
python tools/validate.py --archive model/mh1.model --checkpoint model/mh1.model \
  --library build/libmace.so --case Si8 --check
```

For a workspace build, the dump utility and library are under `build/mace/`.
The workspace `tests/check_cli.py` additionally checks frame selection, JSON file
output, and malformed input. The example ARC contains 100 frames; CLI frame zero
is the default. Independent copied-source builds and relocated installations pass
for Lapis, Chemio, checkpoint, the MACE library/executable, and C/C++/CUDA
clients. The C++ two-argument constructor and original C entry point are retained.
