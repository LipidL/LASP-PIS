# D3 parameter provenance

`src/parameters.inc` is a repacking of the numerical parameter tables from
[simple-dftd3 v1.2.1](https://github.com/dftd3/simple-dftd3/tree/v1.2.1),
licensed under LGPL-3.0-or-later. The license texts are included here.
It includes H–Pu reference C6 values and coordination numbers, scaled covalent
radii, and r2/r4 factors, including the revised actinide references.

Regenerate with:

```sh
python d3/tools/import_parameters.py /path/to/simple-dftd3 d3/src/parameters.inc
```

The converter requires no third-party Python packages. It checks the following
SHA256 hashes under `src/dftd3/` before packing reference pairs into a lower
triangular matrix, checking dimensions, positive coefficients, and symmetry:

- `reference.f90`: `76368b2d3aa682216e82292b4170edf704b8e88838b32d033722f18710217c0b`
- `data/covrad.f90`: `f9c24ff34d591dfe8a05c746b63a64b4dfabc3160be3cd0dcce3fdaf35eb0a8f`
- `data/r4r2.f90`: `c339eec5d337fed885fbb3bff521f9c972bedcf368a435e24e58c110f6dd1925`

Covalent radii are multiplied by 4/3 and converted to bohr with the public
`d3::bohr` conversion constant. Native execution requires neither source files
nor downloads. The CUDA evaluation code is independently implemented.

Method: S. Grimme et al., J. Chem. Phys. 132, 154104 (2010),
https://doi.org/10.1063/1.3382344; BJ damping: S. Grimme et al.,
J. Comput. Chem. 32, 1456–1465 (2011), https://doi.org/10.1002/jcc.21759.
