"""Pack simple-dftd3 v1.2.1 reference tables into src/parameters.inc.

Usage: python d3/tools/import_parameters.py /path/to/simple-dftd3 d3/src/parameters.inc
Uses the Python standard library only. Source hashes are pinned in data/NOTICE.md.
"""
import argparse
import hashlib
import math
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
expected = {
    "reference.f90": "76368b2d3aa682216e82292b4170edf704b8e88838b32d033722f18710217c0b",
    "data/covrad.f90": "f9c24ff34d591dfe8a05c746b63a64b4dfabc3160be3cd0dcce3fdaf35eb0a8f",
    "data/r4r2.f90": "c339eec5d337fed885fbb3bff521f9c972bedcf368a435e24e58c110f6dd1925",
}
source = {}
for name, digest in expected.items():
    path = args.source / "src/dftd3" / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected source hash for {name}")
    source[name] = re.sub(r"!.*", "", path.read_text())

def numbers(text):
    return [float(v) for v in re.findall(r"([+-]?[\d.]+)_wp", text)]

ref = source["reference.f90"]
counts = [int(v) for v in re.findall(r"\d+", ref.split("number_of_references(max_elem) = [")[1].split("]")[0])]
assert len(counts) == 103 and max(counts) == 7
cn_table = numbers(ref.split("reference_cn(max_ref, max_elem) = reshape([")[1].split("]")[0])
assert len(cn_table) == 103 * 7
raw = []
for first, last, body in re.findall(r"c6ab_view\((\d+):(\d+)\) = \[(.*?)\]", ref, re.S):
    assert int(first) == len(raw) + 1
    raw.extend(numbers(body))
    assert len(raw) == int(last)
assert len(raw) == 7 * 7 * 103 * 104 // 2
refs = [(z, a) for z in range(1, 95) for a in range(counts[z - 1])]
offsets = [sum(z < element for z, _ in refs) for element in range(96)]
cn = [cn_table[(z - 1) * 7 + a] for z, a in refs]
assert all(v >= 0 for v in cn)
c6 = []
for i, (z, a) in enumerate(refs):
    for w, b in refs[:i + 1]:
        pair = w - 1 + z * (z - 1) // 2
        value = raw[pair * 49 + b * 7 + a]
        assert value > 0
        if z == w:
            assert value == raw[pair * 49 + a * 7 + b]
        c6.append(value)
radii = numbers(source["data/covrad.f90"].split("= aatoau * [")[1].split("]")[0])
ratios = numbers(source["data/r4r2.f90"].split("r4_over_r2(max_elem) = [")[1].split("]")[0])
assert len(radii) == len(ratios) == 118
rcov = [0.] + [v * 4 / 3 / 0.52917726 for v in radii[:94]]
r2r4 = [0.] + [math.sqrt(.5 * ratios[z - 1] * math.sqrt(z)) for z in range(1, 95)]
with args.output.open("w") as f:
    f.write("// Generated from simple-dftd3 v1.2.1 (LGPL-3.0-or-later).\n"
            "// See tools/import_parameters.py and data/NOTICE.md.\n"
            "static constexpr int max_element = 94, max_reference = 7;\n")
    for name, data, kind in [("offsets", offsets, "int"), ("reference_cn", cn, "double"),
                             ("c6", c6, "double"), ("rcov", rcov, "double"),
                             ("r2r4", r2r4, "double")]:
        f.write(f"static constexpr {kind} {name}[] = {{\n")
        for start in range(0, len(data), 8):
            f.write("    " + ", ".join(str(int(v)) if kind == "int" else repr(float(v))
                                       for v in data[start:start + 8]) + ",\n")
        f.write("};\n")
