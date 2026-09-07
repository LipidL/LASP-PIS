"""Smoke test: MH1 inference via MACECalculator with cuEquivariance and torch.compile.

Usage:
    python scripts/mace_cueq_smoke.py [--cueq] [--compile {default,reduce-overhead}] \
        [--dtype float64|float32] [--head HEAD] [--device cuda|cpu]

Default: pure e3nn model on GPU in float64 (reference path).
With --cueq the model is converted to cuEquivariance; with --compile it is
additionally wrapped in torch.compile.
"""

import argparse
import logging
import time

from ase import Atoms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MODEL = "model/mh1.model"
AVAILABLE_HEADS = [
    "matpes_r2scan",
    "mp_pbe_refit_add",
    "spice_wB97M",
    "oc20_usemppbe",
    "omol",
    "omat_pbe",
]


def make_structure() -> Atoms:
    """Small periodic diamond-Si supercell (2x2x2, 64 atoms)."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms = atoms.repeat((2, 2, 2))
    atoms.rattle(0.001, seed=42)  # break symmetry so forces are non-trivial
    atoms.center()
    return atoms


def run(args) -> None:
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=MODEL,
        device=args.device,
        default_dtype=args.dtype,
        enable_cueq=args.cueq,
        compile_mode=args.compile,
        head=args.head,
    )

    # Warm-up call: includes cueq graph setup and/or torch.compile tracing.
    # A fresh Atoms object per call: ASE caches results on unchanged atoms, so
    # reusing one object would turn every later call into a cache hit.
    atoms_a = make_structure()
    atoms_a.calc = calc
    t0 = time.perf_counter()
    energy_a = atoms_a.get_potential_energy()
    forces_a = atoms_a.get_forces()
    t1 = time.perf_counter()

    # Steady-state call on an identical structure: real inference, and a
    # determinism check against the warm-up result. The info marker forces ASE
    # to recompute: its result cache keys on geometry equality, so an identical
    # structure would otherwise be a cache hit.
    atoms_b = make_structure()
    atoms_b.info["_smoke_run"] = "repeat"
    atoms_b.calc = calc
    t2 = time.perf_counter()
    energy_b = atoms_b.get_potential_energy()
    forces_b = atoms_b.get_forces()
    t3 = time.perf_counter()

    # Displaced structure: exercise the (possibly padded/compiled) graph on a
    # slightly different neighbor layout.
    atoms_c = make_structure()
    atoms_c.rattle(0.001, seed=7)
    atoms_c.calc = calc
    energy_c = atoms_c.get_potential_energy()
    forces_c = atoms_c.get_forces()

    import numpy as np

    d_energy = abs(energy_b - energy_a)
    d_force = np.abs(forces_b - forces_a).max()
    print(f"\n=== {args.label} ===")
    print(f"atoms           : {len(atoms_a)}")
    print(f"head            : {args.head}")
    print(f"dtype           : {args.dtype}")
    print(f"cueq            : {args.cueq}")
    print(f"compile         : {args.compile}")
    print(f"energy  (eV)    : {energy_a:.10f}  (repeat {energy_b:.10f})")
    print(f"force max (eV/A): {np.abs(forces_a).max():.6f}")
    print(f"time warm-up    : {t1 - t0:.3f} s")
    print(f"time per step   : {t3 - t2:.3f} s")
    finite = (
        np.isfinite(energy_a)
        and np.isfinite(forces_a).all()
        and np.isfinite(energy_b)
        and np.isfinite(forces_b).all()
        and np.isfinite(energy_c)
        and np.isfinite(forces_c).all()
    )
    deterministic = d_energy < 1e-8 and d_force < 1e-6
    print(f"finite          : {finite}")
    print(f"deterministic   : {deterministic}  (|dE|={d_energy:.2e}, max|dF|={d_force:.2e})")
    if not (finite and deterministic):
        raise SystemExit("SMOKE TEST FAILED")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cueq", action="store_true")
    p.add_argument("--compile", choices=["default", "reduce-overhead"], default=None)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--head", choices=AVAILABLE_HEADS, default="matpes_r2scan")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    label = "cueq" if args.cueq else "e3nn"
    if args.compile:
        label += f"+compile({args.compile})"
    label += f" {args.dtype}"
    args.label = label
    run(args)
