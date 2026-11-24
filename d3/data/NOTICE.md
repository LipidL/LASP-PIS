# D3 parameter tables

Lapis distributes the tables embedded in `libd3.a` under the Apache License,
Version 2.0. The generated source is `d3/src/parameters.inc`. The upstream
license text accompanies this notice in [Apache-2.0.txt](Apache-2.0.txt).

| Upstream release | Input files | Data used |
| --- | --- | --- |
| [tad-dftd3 0.5.0](https://github.com/dftd3/tad-dftd3/tree/v0.5.0/src/tad_dftd3) | `reference.py`, `reference-c6.pt`, `data/r4r2.py` | Reference coordination numbers, pair C6 coefficients, and atomic r⁴/r² expectation values |
| [tad-mctc 0.5.3](https://pypi.org/project/tad-mctc/0.5.3/) | `data/radii.py` (`COV_D3`) | D3 covalent radii |

tad-dftd3 credits Sebastian Ehlert and Marvin Friede. Its r⁴/r² table
credits S. Grimme and J. Mewes for the atomic calculations. The tad-mctc radii
file bears the notice “Copyright (C) 2024 Grimme Group” and credits the radii
of Pyykkö and Atsumi, with reduced values for metals. Both projects publish
these releases under Apache-2.0. The basis for using the binary C6 tensor
is the tad-dftd3 package license. Its history records the conversion of the
upstream C6 table to the distributed tensor. [tad-dftd3 license](https://github.com/dftd3/tad-dftd3/blob/v0.5.0/LICENSE)
and [C6 conversion commit](https://github.com/dftd3/tad-dftd3/commit/d8d76d98457c2f465cb2030d397fb77066e64b5f).

Lapis selects elements H–Pu, omits unused reference slots, and stores the
39,621 distinct C6 coefficients for the 281 reference environments in a
lower-triangular array. It scales the covalent radii by 4/3 and converts
ångströms to bohr using `d3::bohr`. It stores the atomic factors as
`sqrt(0.5 * (r⁴/r²) * sqrt(Z))`.

The scientific references are the [D3 method](https://doi.org/10.1063/1.3382344),
[BJ damping](https://doi.org/10.1002/jcc.21759), and the
[revised actinide parameters](https://doi.org/10.1039/D4CP01514B).

## Regenerating the tables

From the Lapis repository root, unpack the two pinned Python packages in a
temporary directory and run the converter. It verifies the SHA-256 of each
input before writing the output and uses only the Python standard library.

```sh
sources=$(mktemp -d)
python -m pip install --no-deps --target "$sources" tad-dftd3==0.5.0 tad-mctc==0.5.3
python d3/tools/import_parameters.py "$sources/tad_dftd3" "$sources/tad_mctc" d3/src/parameters.inc
```

The converter also accepts the corresponding `src/tad_dftd3` and
`src/tad_mctc` directories from tagged source checkouts. Native builds
and calculations use the checked-in tables.
