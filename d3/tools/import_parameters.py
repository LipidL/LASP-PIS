"""Pack the pinned tad-dftd3 and tad-mctc data into the native D3 tables."""

import argparse
import ast
import hashlib
import io
import math
from pathlib import Path
import struct
import zipfile


MAX_ELEMENT = 94
MAX_REFERENCE = 7
BOHR = 0.52917726  # d3::bohr, in angstrom
SHA256 = {
    "reference.py": "26bacb1767862a00bdc96515962aa5795405a8b1db259fdb713dac03e04be3fa",
    "reference-c6.pt": "81cf0f16c5558b55f53647b2e90d98b78eefeca40a9efbf2fc7b4b425493de5f",
    "data/r4r2.py": "80053770ab77712924fbdcb99089c76dd6d7d94cb20a2635d6c20f2d1e49599d",
    "data/radii.py": "96179f5585a903a04dfe9574a1d9378032759943194ed4e67e10863a6ae3ca35",
}


def pinned_file(package, name):
    path = package / name
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise ValueError(f"Unexpected SHA-256 for {path}")
    return data


def table_literal(source, function_name):
    """Read the outer list literal in a pinned upstream table function."""
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    return ast.literal_eval(next(node for node in ast.walk(function) if isinstance(node, ast.List)))


def pack_tables(dftd3, mctc):
    cn_by_element = table_literal(pinned_file(dftd3, "reference.py"), "_load_cn")
    tensor = pinned_file(dftd3, "reference-c6.pt")
    ratios = table_literal(pinned_file(dftd3, "data/r4r2.py"), "R4R2")
    radii = table_literal(pinned_file(mctc, "data/radii.py"), "COV_D3")

    # The pinned tensor is contiguous float64[Z1][Z2][ref1][ref2], little endian.
    # Read its storage directly so regeneration only needs the Python standard library.
    elements, slots = len(cn_by_element), len(cn_by_element[0])
    if (elements, slots) != (104, MAX_REFERENCE) or any(len(row) != slots for row in cn_by_element):
        raise ValueError("Unexpected reference coordination table shape")
    if len(radii) != 119 or len(ratios) != 119:
        raise ValueError("Unexpected atomic data table size")
    with zipfile.ZipFile(io.BytesIO(tensor)) as archive:
        count = elements * elements * slots * slots
        values = struct.unpack(f"<{count}d", archive.read("archive/data/0"))

    def coefficient(z, w, a, b):
        return values[((z * elements + w) * slots + a) * slots + b]

    offsets = [0]
    references = []
    cn = []
    c6 = []
    for z in range(1, MAX_ELEMENT + 1):
        offsets.append(len(references))
        for a, coordination in enumerate(cn_by_element[z]):
            if coordination < 0:  # unused reference slot
                continue
            references.append((z, a))
            cn.append(coordination)
            for w, b in references:
                value = coefficient(z, w, a, b)
                if value <= 0 or value != coefficient(w, z, b, a):
                    raise ValueError(f"Invalid C6 coefficient for {(z, a)} and {(w, b)}")
                c6.append(value)
    offsets.append(len(references))
    rcov = [v * 4 / 3 / BOHR for v in radii[:MAX_ELEMENT + 1]]
    r2r4 = [math.sqrt(0.5 * ratios[z] * math.sqrt(z)) for z in range(MAX_ELEMENT + 1)]
    return [("offsets", "int", offsets), ("reference_cn", "double", cn), ("c6", "double", c6),
            ("rcov", "double", rcov), ("r2r4", "double", r2r4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tad_dftd3", type=Path, help="tad_dftd3 package directory, version 0.5.0")
    parser.add_argument("tad_mctc", type=Path, help="tad_mctc package directory, version 0.5.3")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tables = pack_tables(args.tad_dftd3, args.tad_mctc)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        f.write("// SPDX-License-Identifier: Apache-2.0\n"
                "// Repacked and scaled for Lapis from tad-dftd3 0.5.0 and tad-mctc 0.5.3.\n"
                "// Covalent radii: Copyright (C) 2024 Grimme Group. See data/NOTICE.md.\n"
                f"static constexpr int max_element = {MAX_ELEMENT}, max_reference = {MAX_REFERENCE};\n")
        for name, kind, data in tables:
            f.write(f"static constexpr {kind} {name}[] = {{\n")
            for start in range(0, len(data), 8):
                f.write("    " + ", ".join(map(repr, data[start:start + 8])) + ",\n")
            f.write("};\n")


if __name__ == "__main__":
    main()
