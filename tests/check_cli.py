#!/usr/bin/env python3
"""Workspace integration: native checkpoint and archive inference through the MACE CLI.

Reference archives must come from the independent Python export paths. This test
also exercises frame selection, output files, and diagnostic exit statuses.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def run(executable, model, structure, *flags):
    result = subprocess.run([str(executable), str(model), str(structure), "--json", *flags],
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def compare(a, b):
    assert a["atoms"] == b["atoms"]
    errors = {}
    for key in ("energy", "atomic_energies", "forces", "virial"):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        errors[key] = float(np.max(np.abs(x-y)))
        # Existing float32 inference uses atomic reductions, so repeated calls
        # need not be bitwise identical even for identical normalized weights.
        np.testing.assert_allclose(x, y, atol=2e-4, rtol=2e-5, err_msg=key)
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--mace", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--mace-reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = {}
    heads = ("matpes_r2scan", "mp_pbe_refit_add", "spice_wB97M", "oc20_usemppbe", "omol", "omat_pbe")
    for head in heads:
        direct = run(args.build/"mace-main", args.mace, args.structure, "--head", head)
        archive = run(args.build/"mace-main", args.mace_reference_dir/(head+".pwm"), args.structure)
        report[head] = compare(direct, archive)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames = tmp/"frames.arc"
        lines, ends = [], 0
        for line in args.structure.read_text().splitlines():
            lines.append(line)
            ends += line.strip().lower() == "end"
            if ends == 2:
                break
        assert ends == 2
        text = "\n".join(lines)+"\n"
        # Both full ARC headers and unwrapped Cartesian coordinates are accepted.
        frames.write_text(text + "\n" + text)
        exe, model = args.build/"mace-main", args.mace
        first, second = run(exe, model, frames), run(exe, model, frames, "--frame", "1")
        assert first["frame"] == 0 and second["frame"] == 1
        compare(first, second)
        out = tmp/"mace.json"
        subprocess.run([str(exe), str(model), str(frames), "--json", "--output", str(out)], check=True, capture_output=True)
        compare(first, json.loads(out.read_text()))
        bad = tmp/"bad.arc"
        bad.write_text(text.rsplit("end", 1)[0])
        cases = [(frames, ["--frame", "2"], "frame index"),
                 (frames, ["--frame", "-1"], "nonnegative"),
                 (frames, ["--invalid"], "unknown option"),
                 (bad, [], "incomplete")]
        nonperiodic = tmp/"nonperiodic.arc"
        nonperiodic.write_text(text.replace("PBC=ON", "PBC=OFF"))
        cases.append((nonperiodic, [], "three periodic axes"))
        for path, flags, diagnostic in cases:
            failure = subprocess.run([str(exe), str(model), str(path), *flags], text=True, capture_output=True)
            assert failure.returncode and diagnostic in failure.stderr, (flags, diagnostic, failure.stderr)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
