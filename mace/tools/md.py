#!/usr/bin/env python3
"""Separate NVE stability diagnostic; never used for matched performance ratios."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from mace_inference import MH1Calculator


def run(args):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((args.repeat,) * 3)
    atoms.calc = MH1Calculator(args.archive)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=args.temperature, rng=np.random.default_rng(args.seed)
    )
    Stationary(atoms)
    dynamics = VelocityVerlet(atoms, timestep=args.timestep * units.fs)
    total = []

    def sample():
        total.append(float(atoms.get_total_energy()))

    dynamics.attach(sample, interval=1)
    dynamics.run(args.steps)
    energy = np.asarray(total)
    assert np.isfinite(energy).all()
    report = {
        "diagnostic": "closed-loop NVE, independent of replay timing",
        "atoms": len(atoms),
        "steps": args.steps,
        "timestep_fs": args.timestep,
        "seed": args.seed,
        "initial_temperature_K": args.temperature,
        "energy_eV": total,
        "max_energy_deviation_eV_per_atom": float(
            np.max(np.abs(energy - energy[0])) / len(atoms)
        ),
        "final_drift_eV_per_atom": float((energy[-1] - energy[0]) / len(atoms)),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "energy_eV"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default="model/mh1.pwm")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--timestep", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=300)
    p.add_argument("--seed", type=int, default=11037)
    p.add_argument("--output", default="benchmarks/md.json")
    run(p.parse_args())
