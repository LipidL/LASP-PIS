#!/usr/bin/env python3
"""Matched, changing-coordinate ASE replay. Defaults implement the v1 protocol.

Use --quick for an explicitly non-release development measurement.
"""
import argparse
import hashlib
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pairwise-matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "mace-ref")]
import numpy as np
from ase.build import bulk
from ase.io import read
from process_memory import ProcessMemory


def main(args):
    from pairwise import PairwiseCalculator

    manifest_path = Path(str(args.archive) + ".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["head"] != args.head:
            raise ValueError("Archive head differs from the requested reference head")
        if (
            manifest["archive_sha256"]
            != hashlib.sha256(Path(args.archive).read_bytes()).hexdigest()
        ):
            raise ValueError("Archive differs from its export manifest")
        if (
            manifest["checkpoint_sha256"]
            != hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        ):
            raise ValueError("Archive and reference checkpoint differ")

    if args.structure:
        base = read(args.structure, index=args.frame)
    else:
        base = bulk("Si", "diamond", a=args.lattice_constant, cubic=True).repeat(
            (args.repeat,) * 3
        )
    base = base.repeat(args.supercell)
    base.positions = np.asarray(base.positions, np.float32).astype(float)
    base.cell = np.asarray(base.cell.array, np.float32).astype(float)
    if args.quick:
        args.warmup, args.windows, args.steps = 3, 3, 3
    rng = np.random.default_rng(args.seed)
    # Fixed trajectory segment reused by every implementation and window.
    frames = []
    for _ in range(max(args.steps, args.warmup)):
        a = base.copy()
        a.positions = base.positions + rng.normal(0, 0.005, base.positions.shape)
        strain = rng.normal(0, 1e-4, (3, 3))
        strain = 0.5 * (strain + strain.T)
        a.set_cell(base.cell.array @ (np.eye(3) + strain), scale_atoms=True)
        a.positions = np.asarray(a.positions, np.float32).astype(float)
        a.cell = np.asarray(a.cell.array, np.float32).astype(float)
        frames.append(a)
    trajectory_hash = hashlib.sha256(
        b"".join(a.positions.tobytes() + a.cell.array.tobytes() for a in frames)
    ).hexdigest()
    implementations = {}
    native = {"framework", "explicit"}
    for name in args.implementations:
        if name in native:
            library = (
                str(ROOT / "build/libpairwise_explicit.so")
                if name == "explicit"
                else None
            )
            implementations[name] = PairwiseCalculator(
                args.archive, backend=args.backend, library=library
            )
    torch = None
    if any(name not in native for name in args.implementations):
        import torch

        torch.serialization.add_safe_globals([slice])
        torch.set_num_threads(4)
        from mace.calculators import MACECalculator

        for name in args.implementations:
            if name in native:
                continue
            implementations[name] = MACECalculator(
                model_paths=args.checkpoint,
                device="cuda",
                default_dtype="float32",
                head=args.head,
                enable_cueq=name in ("cueq", "compile"),
                compile_mode="default" if name == "compile" else None,
            )

    def call(name, atoms):
        calc = implementations[name]
        if name in native:
            result = calc.evaluate(atoms)
        else:
            calc.calculate(atoms, properties=["energy", "forces", "stress"])
            result = calc.results
        return result

    process_memory = ProcessMemory()
    failures = {}
    for name in list(implementations):
        try:
            for step in range(args.warmup):
                call(name, frames[step % len(frames)])
        except RuntimeError as exc:
            if (
                "out of memory" not in str(exc).lower()
                and "memory allocation" not in str(exc).lower()
            ):
                raise
            failures[name] = {"status": "OOM", "phase": "warmup", "error": str(exc)}
            del implementations[name]
            gc.collect()
            if torch:
                torch.cuda.empty_cache()
    timing = {name: [] for name in implementations}
    device = {name: [] for name in implementations}
    events = {}
    if torch:
        torch.cuda.reset_peak_memory_stats()
        events = {
            name: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for name in implementations
            if name not in native
        }
    for window in range(args.windows):
        order = list(implementations)
        rng.shuffle(order)
        for name in order:
            for step in range(args.steps):
                if name in events:
                    events[name][0].record()
                start = time.perf_counter()
                result = call(name, frames[step % len(frames)])
                elapsed = time.perf_counter() - start
                assert (
                    np.isfinite(result["energy"])
                    and np.isfinite(result["forces"]).all()
                )
                timing[name].append(elapsed * 1000)
                if name in native:
                    device[name].append(implementations[name].diagnostics["device_ms"])
                else:
                    events[name][1].record()
                    events[name][1].synchronize()
                    device[name].append(events[name][0].elapsed_time(events[name][1]))
        print(f"window {window+1}/{args.windows} complete", flush=True)
    results = {}
    for name, samples in timing.items():
        windows = np.asarray(samples).reshape(args.windows, args.steps).mean(axis=1)
        results[name] = {
            "wall_median_ms": float(np.median(samples)),
            "wall_p95_ms": float(np.percentile(samples, 95)),
            "window_means_ms": windows.tolist(),
            "samples_ms": samples,
        }
        if device[name]:
            results[name]["device_mean_ms"] = float(np.mean(device[name]))
        if name in native:
            results[name]["diagnostics"] = implementations[name].diagnostics
    from matscipy.neighbours import neighbour_list

    edge_counts = [len(neighbour_list("i", a, 6.0)) for a in frames]
    versions = {}
    for package in (
        "torch",
        "e3nn",
        "cuequivariance",
        "cuequivariance-torch",
        "ase",
        "numpy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    def output(command):
        try:
            return subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return str(exc)

    report = {
        "protocol": "v1"
        if (args.warmup, args.windows, args.steps) == (100, 100, 100)
        else "development",
        "atoms": len(base),
        "edges": edge_counts,
        "warmup": args.warmup,
        "windows": args.windows,
        "steps": args.steps,
        "seed": args.seed,
        "trajectory_sha256": trajectory_hash,
        "structure": args.structure
        or f"synthetic diamond Si a={args.lattice_constant} {args.repeat}x{args.repeat}x{args.repeat}",
        "frame": args.frame,
        "supercell": args.supercell,
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
        "head": args.head,
        "dataset_revision": args.dataset_revision
        or ("synthetic-v2" if not args.structure else None),
        "structure_id": args.structure_id,
        "structure_file_sha256": hashlib.sha256(
            Path(args.structure).read_bytes()
        ).hexdigest()
        if args.structure
        else None,
        "mace_revision": output(
            ["git", "-C", str(ROOT / "mace-ref"), "rev-parse", "HEAD"]
        ),
        "versions": versions,
        "python": platform.python_version(),
        "nvcc": output(["/usr/local/cuda/bin/nvcc", "--version"]),
        "gpu": output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ]
        ),
        "process_memory_observation": output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader",
            ]
        ),
        "results": results,
        "failures": failures,
        "process_memory_sampling": process_memory.stop(),
    }
    source_files = sorted(
        p
        for folder in ("src", "include", "python/pairwise", "tools")
        for p in (ROOT / folder).rglob("*")
        if p.suffix in (".cpp", ".cu", ".cuh", ".hpp", ".h", ".py")
    )
    source_files.append(ROOT / "CMakeLists.txt")
    report["source_sha256"] = hashlib.sha256(
        b"".join(
            str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes()
            for p in source_files
        )
    ).hexdigest()
    report["library_sha256"] = {
        name: hashlib.sha256(
            Path(implementations[name].lib._name).read_bytes()
        ).hexdigest()
        for name in implementations
        if name in native
    }
    report["compiler"] = output(["c++", "--version"])
    report["environment_lock_sha256"] = hashlib.sha256(
        (ROOT / "docker/requirements.lock").read_bytes()
    ).hexdigest()
    if torch:
        report["combined_torch_memory"] = {
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }
    if "framework" in results and "cueq" in results:
        report["speedup_over_cueq"] = (
            results["cueq"]["wall_median_ms"] / results["framework"]["wall_median_ms"]
        )
    if "framework" in results and "explicit" in results:
        report["implicit_over_explicit"] = (
            results["framework"]["wall_median_ms"]
            / results["explicit"]["wall_median_ms"]
        )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a != "samples_ms"}
                for k, v in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default="model/mh1.pwm")
    p.add_argument("--checkpoint", default="model/mh1.model")
    p.add_argument("--head", default="matpes_r2scan")
    p.add_argument("--backend", choices=["auto", "brute", "cell"], default="auto")
    p.add_argument(
        "--implementations",
        nargs="+",
        choices=["framework", "explicit", "e3nn", "cueq", "compile"],
        default=["framework", "cueq"],
    )
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--structure")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 1])
    p.add_argument("--lattice-constant", type=float, default=5.43)
    p.add_argument("--dataset-revision")
    p.add_argument("--structure-id")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--windows", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=83071)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", default="benchmarks/replay.json")
    main(p.parse_args())
