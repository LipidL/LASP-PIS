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
import shutil
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
    from mace_inference import MH1Calculator, library_path

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
    failures = {}
    construction_seconds = {}
    native = {"framework", "explicit"}
    compiled = {"compile", "cueq_compile", "e3nn_compile"}
    torch = None
    if any(name not in native for name in args.implementations):
        import torch

        torch.serialization.add_safe_globals([slice])
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        from mace.calculators import MACECalculator

    def is_oom(exc):
        return any(s in str(exc).lower() for s in ("out of memory", "memory allocation"))

    def counters():
        if torch is None or not any(n in compiled for n in args.implementations):
            return {}
        from torch._dynamo.utils import counters as dynamo_counters

        return {k: dict(v) for k, v in dynamo_counters.items()}

    for name in args.implementations:
        start = time.perf_counter()
        try:
            if name in native:
                library = str(library_path("libmace_explicit.so")) if name == "explicit" else None
                implementations[name] = MH1Calculator(args.archive, backend=args.backend, library=library)
            else:
                implementations[name] = MACECalculator(
                    model_paths=args.checkpoint,
                    device="cuda",
                    default_dtype="float32",
                    head=args.head,
                    enable_cueq=name in ("cueq", "compile", "cueq_compile"),
                    compile_mode="default" if name in compiled else None,
                )
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            failures[name] = {"status": "OOM", "phase": "construction", "error": str(exc)}
            gc.collect()
            if torch:
                torch.cuda.empty_cache()
        construction_seconds[name] = time.perf_counter() - start

    def call(name, atoms):
        calc = implementations[name]
        if name in native:
            result = calc.evaluate(atoms)
        else:
            calc.calculate(atoms, properties=["energy", "forces", "stress"])
            result = calc.results
        return result

    process_memory = ProcessMemory()
    warmup_ms = {}
    for name in list(implementations):
        warmup_ms[name] = []
        try:
            for step in range(args.warmup):
                start = time.perf_counter()
                call(name, frames[step % len(frames)])
                warmup_ms[name].append((time.perf_counter() - start) * 1000)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            failures[name] = {"status": "OOM", "phase": "warmup", "error": str(exc)}
            del implementations[name]
            gc.collect()
            if torch:
                torch.cuda.empty_cache()
    timing = {name: [] for name in implementations}
    device = {name: [] for name in implementations}
    events = {}
    after_warmup_counters = counters()
    startup_torch_memory = None
    if torch:
        startup_torch_memory = {"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}
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
                try:
                    result = call(name, frames[step % len(frames)])
                except RuntimeError as exc:
                    if not is_oom(exc):
                        raise
                    failures[name] = {"status": "OOM", "phase": "measurement", "error": str(exc), "completed_calls": len(timing[name])}
                    del implementations[name], timing[name], device[name]
                    gc.collect()
                    if torch:
                        torch.cuda.empty_cache()
                    break
                elapsed = time.perf_counter() - start
                assert (
                    np.isfinite(result["energy"])
                    and np.isfinite(result["forces"]).all()
                    and np.isfinite(result["stress"]).all()
                )
                timing[name].append(elapsed * 1000)
                if name in native:
                    device[name].append(implementations[name].diagnostics["device_ms"])
                else:
                    events[name][1].record()
                    events[name][1].synchronize()
                    device[name].append(events[name][0].elapsed_time(events[name][1]))
        print(f"window {window+1}/{args.windows} complete", flush=True)
    after_measurement_counters = counters()
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
        "cuequivariance-ops-torch-cu12",
        "cuequivariance-ops-cu12",
        "triton",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-ml-py",
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
        "requested_implementations": args.implementations,
        "construction_seconds": construction_seconds,
        "warmup_samples_ms": warmup_ms,
        "compile_counters_after_warmup": after_warmup_counters,
        "compile_counters_after_measurement": after_measurement_counters,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "SLURM_JOB_ID", "SLURM_JOB_GPUS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS")},
        "precision": {"model": "float32", "torch_tf32": False if torch else None, "torch_threads": args.threads if torch else None},
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
        "nvcc": output([shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc", "--version"]),
        "gpu": output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,power.limit,clocks.sm,clocks.mem,temperature.gpu",
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
        for folder in ("src", "include", "python/mace_inference", "tools")
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
    report["native_library_dependencies"] = {
        name: output(["ldd", implementations[name].lib._name])
        for name in implementations if name in native
    }
    report["compiler"] = output(["c++", "--version"])
    report["environment_lock_file"] = args.environment_lock
    report["environment_lock_sha256"] = hashlib.sha256(Path(args.environment_lock).read_bytes()).hexdigest()
    if torch:
        report["torch_cuda"] = torch.version.cuda
        report["torch_build"] = torch.__version__
        report["torch_device"] = str(torch.cuda.get_device_properties(torch.cuda.current_device()))
        report["startup_combined_torch_memory"] = startup_torch_memory
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
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
        choices=["framework", "explicit", "e3nn", "cueq", "compile", "cueq_compile", "e3nn_compile"],
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
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--environment-lock", default="docker/requirements.lock")
    p.add_argument("--output", default="benchmarks/replay.json")
    main(p.parse_args())
