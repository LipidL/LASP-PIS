#!/usr/bin/env python3
"""Measure native repeat variability without changing its strict release targets."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
from ase.io import read
from mace_inference import MH1Calculator, library_path

atoms = read(ROOT / "structures/mptrj-mp-1005792.extxyz").repeat((3, 3, 1))
atoms.positions = np.asarray(atoms.positions, np.float32).astype(float)
atoms.cell = np.asarray(atoms.cell.array, np.float32).astype(float)
report = {}
for implementation, filename in [("framework", "libmace.so"), ("explicit", "libmace_explicit.so")]:
    calc = MH1Calculator(ROOT / "model/mh1.pwm", library=str(library_path(filename)))
    values = [calc.evaluate(atoms) for _ in range(100)]
    energy = np.asarray([r["energy"] for r in values])
    forces = np.asarray([r["forces"] for r in values])
    virial = np.asarray([r["virial"] for r in values])
    report[implementation] = {
        "calls": len(values), "atoms": len(atoms),
        "energy_range_per_atom": float(np.ptp(energy) / len(atoms)),
        "force_component_max_range": float(np.ptp(forces, axis=0).max()),
        "virial_component_max_range": float(np.ptp(virial, axis=0).max()),
        "virial_magnitude": float(np.abs(virial).max()),
        "virial_values": virial.tolist(),
        "strict_targets": {"energy_per_atom": 1e-6, "force_component": 1e-5, "virial_component": 1e-5},
    }
    calc.close()
path = ROOT / "benchmarks/rtx4090/repeat-variability.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({name: {k: v for k, v in item.items() if k != "virial_values"} for name, item in report.items()}, indent=2))
