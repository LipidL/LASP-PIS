# Pinned MPtrj fixture

`mptrj-mp-1005792.extxyz` contains the 28-atom As12Cu4O4Sm8 frame
`mp-1012897-0-0` under material `mp-1005792` from Bowen Deng's
[Materials Project Trajectory dataset, version 2](https://figshare.com/articles/dataset/23713842/2),
DOI `10.6084/m9.figshare.23713842.v2` (MIT license). It is the first frame in
the pinned official file, selected without screening for model performance.
No training labels are used in the benchmark.

The adjacent JSON records provenance and checksums. The source was extracted
from bytes 0–262143 of Figshare file 41619375; the 12-GB dataset was not downloaded
or independently checked against its whole-file MD5. The exact downloaded byte
range and derived structure have independent SHA256 checksums.

Recreate the fixture with:

```sh
.venv/bin/python tools/fetch_mptrj.py
```

The benchmark's `--supercell 3 3 1` produces 252 atoms before seeded coordinate
and lattice perturbations. An approximately 50,000-atom extension is
`--supercell 14 14 9` (49,392 atoms); that extension has not been executed on the
available 8-GiB GPU. One fixture does not establish broad dataset coverage.
