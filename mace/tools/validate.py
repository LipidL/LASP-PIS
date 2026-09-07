#!/usr/bin/env python3
"""Public ASE seam comparison with official MACE, including force/virial checks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pairwise-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mace-ref"))
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read
from pairwise import PairwiseCalculator


def structures():
    silicon = bulk("Si", "diamond", a=5.43, cubic=True)
    silicon.rattle(0.03, seed=173)
    yield "Si8", silicon
    crystal = silicon.repeat((2, 2, 2))
    crystal.rattle(0.01, seed=891)
    yield "Si64", crystal
    crystal = silicon.repeat((3, 3, 3))
    crystal.rattle(0.01, seed=2891)
    yield "Si216", crystal
    mixed = Atoms(
        "SiCOH",
        positions=[[1, 1, 1], [2.5, 1.2, 1], [2.3, 2.7, 1.3], [3.1, 3.1, 1.5]],
        cell=[[8.4, 0, 0], [1.1, 8.7, 0], [-0.8, 0.4, 9.2]],
        pbc=True,
    )
    yield "mixed4", mixed


def main(args):
    import torch

    torch.serialization.add_safe_globals([slice])
    torch.set_num_threads(4)
    from mace.calculators import MACECalculator

    new = PairwiseCalculator(args.archive, backend=args.backend, library=args.library)
    ref = MACECalculator(
        model_paths=args.checkpoint,
        device=args.device,
        default_dtype=args.dtype,
        head=args.head,
        enable_cueq=args.cueq,
    )
    report = []
    cases = (
        [(args.structure, read(args.structure, index=args.frame))]
        if args.structure
        else structures()
    )
    for name, atoms in cases:
        if args.case and name != args.case:
            continue
        atoms = atoms.repeat(args.supercell)
        # Match float32 public geometry exactly for both implementations.
        atoms.positions = np.asarray(atoms.positions, np.float32).astype(float)
        atoms.cell = np.asarray(atoms.cell.array, np.float32).astype(float)
        t = time.perf_counter()
        a = new.evaluate(atoms)
        elapsed = time.perf_counter() - t
        ref.calculate(atoms, properties=["energy", "forces", "stress"])
        b = ref.results
        bv = -atoms.get_volume() * __import__(
            "ase.stress", fromlist=["voigt_6_to_full_3x3_stress"]
        ).voigt_6_to_full_3x3_stress(b["stress"])
        errors = dict(
            energy_per_atom=abs(a["energy"] - b["energy"]) / len(atoms),
            force_max=float(np.max(np.abs(a["forces"] - b["forces"]))),
            virial_max=float(np.max(np.abs(a["virial"] - bv))),
        )
        item = dict(
            case=name,
            atoms=len(atoms),
            energy=float(a["energy"]),
            reference_energy=float(b["energy"]),
            errors=errors,
            wall_seconds=elapsed,
            diagnostics=new.diagnostics,
        )
        print(json.dumps(item, indent=2), flush=True)
        if args.verbose:
            print(
                "forces new",
                a["forces"],
                "reference",
                b["forces"],
                "virial",
                a["virial"],
                bv,
                flush=True,
            )
        repeat = new.evaluate(atoms)
        assert abs(repeat["energy"] - a["energy"]) / len(atoms) < 1e-6
        assert np.max(np.abs(repeat["forces"] - a["forces"])) < 1e-5
        assert np.max(np.abs(repeat["virial"] - a["virial"])) < 1e-5
        report.append(item)
        if args.check:
            assert errors["energy_per_atom"] < 1e-5, errors
            assert errors["force_max"] < 1e-4, errors
            assert errors["virial_max"] < max(1e-4, 1e-4 * np.max(np.abs(bv))), errors
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default="model/mh1.pwm")
    p.add_argument("--checkpoint", default="model/mh1.model")
    p.add_argument("--head", default="matpes_r2scan")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--backend", choices=["auto", "brute", "cell"], default="auto")
    p.add_argument("--library")
    p.add_argument("--cueq", action="store_true")
    p.add_argument("--case")
    p.add_argument("--structure")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 1])
    p.add_argument("--check", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output")
    main(p.parse_args())
