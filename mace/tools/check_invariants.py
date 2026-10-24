#!/usr/bin/env python3
"""End-to-end invariance, analytical derivative and reusable-calculator checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import numpy as np
from ase.build import bulk
from mace_inference import MH1Calculator


def run(args):
    calc = MH1Calculator(args.archive)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rattle(0.02, seed=691)
    first = calc.evaluate(atoms)
    capacity = calc.diagnostics["context_bytes"]
    again = calc.evaluate(atoms)
    assert calc.diagnostics["context_bytes"] == capacity
    assert abs(again["energy"] - first["energy"]) / len(atoms) < 1e-6
    assert np.max(np.abs(again["forces"] - first["forces"])) < 1e-5
    assert np.max(np.abs(again["virial"] - first["virial"])) < 1e-5
    if args.smoke:
        print("Blocking energy, force, virial and repeat checks passed", flush=True)
        return
    report = {}
    rng = np.random.default_rng(513)
    order = rng.permutation(len(atoms))
    permuted = atoms[order]
    p = calc.evaluate(permuted)
    report["permutation_force"] = float(
        np.max(np.abs(p["forces"] - first["forces"][order]))
    )
    assert report["permutation_force"] < 1e-4
    for name in ("translation", "wrapping", "rotation"):
        a = atoms.copy()
        rotation = np.eye(3)
        if name == "translation":
            a.positions += [1.3, -0.7, 2.1]
        elif name == "wrapping":
            a.positions += rng.integers(-1, 2, (len(a), 3)) @ a.cell.array
        else:
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            a.positions = a.positions @ rotation
            a.cell = a.cell.array @ rotation
        b = calc.evaluate(a)
        e = abs(b["energy"] - first["energy"]) / len(a)
        f = float(np.max(np.abs(b["forces"] - first["forces"] @ rotation)))
        v = float(np.max(np.abs(b["virial"] - rotation.T @ first["virial"] @ rotation)))
        report[name] = {"energy_per_atom": e, "force_max": f, "virial_max": v}
        assert e < 1e-4 and f < 1e-4 and v < 1e-4, (name, e, f, v)
    report["force_fd"] = {}
    for step in (0.01, 0.003, 0.001):
        a = atoms.copy()
        b = atoms.copy()
        a.positions[0, 0] += step
        b.positions[0, 0] -= step
        fd = -(calc.evaluate(a)["energy"] - calc.evaluate(b)["energy"]) / (2 * step)
        report["force_fd"][str(step)] = abs(fd - first["forces"][0, 0])
    assert min(report["force_fd"].values()) < 1e-3, report["force_fd"]
    report["strain_fd"] = {}
    for step in (0.001, 0.0003, 0.0001):
        ep = []
        for sign in (-1, 1):
            a = atoms.copy()
            matrix = np.eye(3)
            matrix[0, 0] += sign * step
            a.positions = a.positions @ matrix
            a.cell = a.cell.array @ matrix
            ep.append(calc.evaluate(a)["energy"])
        fd = -(ep[1] - ep[0]) / (2 * step)
        report["strain_fd"][str(step)] = abs(fd - first["virial"][0, 0])
    assert min(report["strain_fd"].values()) < 0.01, report["strain_fd"]
    # The same public call through ASE must return host-valid caller-owned arrays.
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    assert np.isfinite(atoms.get_stress()).all()
    original_forces = first["forces"].copy()
    calc.evaluate(atoms)
    assert np.array_equal(original_forces, first["forces"])
    # An invalid invocation must not poison the calculator for subsequent calls.
    a = atoms.copy()
    a.cell[2] = 0
    try:
        calc.evaluate(a)
    except RuntimeError:
        pass
    else:
        raise AssertionError("singular cell accepted")
    assert np.isfinite(calc.evaluate(atoms)["energy"])
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--output")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Two blocking calls for CUDA instrumentation",
    )
    run(p.parse_args())
