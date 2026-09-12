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
from mace_inference import MH1Calculator


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from mace.calculators import MACECalculator

    new = MH1Calculator(args.archive, backend=args.backend, library=args.library)
    ref = MACECalculator(
        model_paths=args.checkpoint,
        device=args.device,
        default_dtype=args.dtype,
        head=args.head,
        enable_cueq=args.cueq,
        compile_mode=args.compile,
    )
    report = []
    failed_checks = []
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
            energy_node_sum_per_atom=abs(a["energy"] - np.sum(b["energies"], dtype=np.float64)) / len(atoms),
            force_max=float(np.max(np.abs(a["forces"] - b["forces"]))),
            virial_max=float(np.max(np.abs(a["virial"] - bv))),
        )
        item = dict(
            case=name,
            atoms=len(atoms),
            energy=float(a["energy"]),
            reference_energy=float(b["energy"]),
            reference_energy_node_sum=float(np.sum(b["energies"], dtype=np.float64)),
            energy_check=args.energy_check,
            errors=errors,
            wall_seconds=elapsed,
            diagnostics=new.diagnostics,
            reference_dtype=args.dtype,
            reference_device=args.device,
            reference_cueq=args.cueq,
            reference_compile=args.compile,
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
        item["repeat_errors"] = {
            "energy_per_atom": abs(repeat["energy"] - a["energy"]) / len(atoms),
            "force_max": float(np.max(np.abs(repeat["forces"] - a["forces"]))),
            "virial_max": float(np.max(np.abs(repeat["virial"] - a["virial"]))),
        }
        item["repeat_strict_pass"] = all(item["repeat_errors"][k] < limit for k, limit in [("energy_per_atom", 1e-6), ("force_max", 1e-5), ("virial_max", 1e-5)])
        if not item["repeat_strict_pass"] and args.repeat_check == "strict":
            failed_checks.append({"case": name, "repeat_errors": item["repeat_errors"]})
        report.append(item)
        if args.check:
            energy_key = "energy_per_atom" if args.energy_check == "total" else "energy_node_sum_per_atom"
            item["comparison_pass"] = bool(errors[energy_key] < 1e-5 and errors["force_max"] < 1e-4 and errors["virial_max"] < max(1e-4, 1e-4 * np.max(np.abs(bv))))
            if not item["comparison_pass"]:
                failed_checks.append({"case": name, "errors": errors})
    if args.compile:
        from torch._dynamo.utils import counters

        print("Compilation counters:", {k: dict(v) for k, v in counters.items()}, flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if failed_checks:
        raise AssertionError(failed_checks)


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
    p.add_argument("--compile", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--case")
    p.add_argument("--structure")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 1])
    p.add_argument("--check", action="store_true")
    p.add_argument("--energy-check", choices=["total", "node-sum"], default="total", help="node-sum diagnoses FP32 total reduction using the reference's returned per-atom energies; raw total error is always reported")
    p.add_argument("--repeat-check", choices=["strict", "report"], default="strict", help="Always record repeat errors; report leaves the strict repeat gate as a reported result")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output")
    main(p.parse_args())
