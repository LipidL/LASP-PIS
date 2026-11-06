#!/usr/bin/env python3
"""Evaluate MACE MH1 energy, forces and stress for one structure frame."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from mace_inference import MH1
from structure_io import read

p = argparse.ArgumentParser(description="Evaluate MACE MH1 energy, forces and stress")
p.add_argument("model", help="MH1 checkpoint or PWMH1001 archive")
p.add_argument("structure", help="ARC, NPZ, or an ASE-readable structure")
p.add_argument("--backend", choices=("auto", "brute", "cell"), default="auto")
p.add_argument("--head", help="Checkpoint head; processed archives fix their own head")
p.add_argument("--frame", type=int, default=0, help="Zero-based frame index")
p.add_argument("--output", help="Save complete arrays as NPZ")
a = p.parse_args()
xyz, z, cell = read(a.structure, a.frame)
with MH1(a.model, a.backend, a.head) as model:
    result = model.calculate(xyz, z, cell)
    if a.output:
        np.savez(a.output, **result)
    print(
        json.dumps(
            {
                "frame": a.frame,
                "atoms": len(z),
                "energy": result["energy"],
                "max_abs_force": float(np.max(abs(result["forces"]), initial=0)),
                "stress": result["stress"].tolist(),
                "diagnostics": model.diagnostics,
            },
            indent=2,
        )
    )
