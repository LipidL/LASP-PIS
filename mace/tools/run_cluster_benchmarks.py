#!/usr/bin/env python3
"""Run the RTX 4090 campaign inside a one-GPU Slurm allocation.

Each command has its own process and persistent log. The environment is prepared
on gpunode9 before submission; execution requires no network access.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from mace_inference.library import library_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/rtx4090"
PYTHON = sys.executable
START_AT = None
IMPLEMENTATIONS = ["framework", "explicit", "e3nn", "cueq", "e3nn_compile", "cueq_compile"]
MPTRJ = ["--structure", "structures/mptrj-mp-1005792.extxyz", "--dataset-revision", "10.6084/m9.figshare.23713842.v2", "--structure-id", "mp-1005792/mp-1012897-0-0"]


def run(name, command):
    global START_AT
    if START_AT is not None:
        if name != START_AT:
            return
        START_AT = None
    log = OUT / "logs" / (name + ".log")
    print(f"START {name}: {' '.join(command)}", flush=True)
    start = time.time()
    with log.open("w") as f:
        completed = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    record = {"name": name, "command": command, "returncode": completed.returncode, "elapsed_seconds": time.time() - start, "job_id": os.environ.get("SLURM_JOB_ID")}
    with (OUT / "commands.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"END {name}: exit={completed.returncode}, seconds={record['elapsed_seconds']:.1f}", flush=True)
    if completed.returncode:
        print(log.read_text()[-10000:], flush=True)
        raise SystemExit(completed.returncode)


def benchmark(name, options, implementations, protocol=()):
    run(name, [PYTHON, "tools/benchmark.py", *options, "--implementations", *implementations, *protocol, "--environment-lock", "benchmarks/rtx4090/environment.freeze.txt", "--output", f"benchmarks/rtx4090/{name}.json"])


def smoke():
    run("device", [PYTHON, "-c", "import torch, cuequivariance_ops_torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_properties(0)); x=torch.ones(16, device='cuda'); print((x*x).sum().item())"])
    run("correctness-cpu64", [PYTHON, "tools/validate.py", "--check", "--output", "benchmarks/rtx4090/correctness-cpu64.json"])
    run("correctness-explicit", [PYTHON, "tools/validate.py", "--check", "--case", "Si216", "--library", str(library_path("libmace_explicit.so")), "--output", "benchmarks/rtx4090/correctness-explicit.json"])
    run("correctness-mptrj-cpu64", [PYTHON, "tools/validate.py", "--check", "--structure", "structures/mptrj-mp-1005792.extxyz", "--supercell", "3", "3", "1", "--output", "benchmarks/rtx4090/correctness-mptrj-cpu64.json"])
    run("invariants-mh1", [PYTHON, "tools/check_invariants.py", "--archive", "model/mh1.pwm", "--output", "benchmarks/rtx4090/invariants-mh1.json"])
    run("invariants-lj", [PYTHON, "tools/check_invariants.py", "--output", "benchmarks/rtx4090/invariants-lj.json"])
    for name, flags in [("e3nn", []), ("cueq", ["--cueq"]), ("e3nn_compile", ["--compile", "default"]), ("cueq_compile", ["--cueq", "--compile", "default"])]:
        run("correctness-" + name, [PYTHON, "tools/validate.py", "--check", "--energy-check", "node-sum", "--repeat-check", "report", "--device", "cuda", "--dtype", "float32", "--structure", "structures/mptrj-mp-1005792.extxyz", "--supercell", "3", "3", "1", *flags, "--output", f"benchmarks/rtx4090/correctness-{name}.json"])
    run("repeat-variability", [PYTHON, "tools/check_cluster_repeats.py"])
    benchmark("pilot-mptrj-28", MPTRJ, IMPLEMENTATIONS, ["--warmup", "20", "--windows", "3", "--steps", "5"])


def replay():
    benchmark("replay-mptrj-28-v1", MPTRJ, IMPLEMENTATIONS)
    benchmark("replay-mptrj-252-v1", [*MPTRJ, "--supercell", "3", "3", "1"], IMPLEMENTATIONS)
    benchmark("replay-dense-512-v1", ["--repeat", "4", "--lattice-constant", "4.6"], IMPLEMENTATIONS)


def capacity():
    # A fresh process per implementation and size avoids retained comparator memory.
    sizes = [(4096, ["--repeat", "8"]), (12544, [*MPTRJ, "--supercell", "8", "8", "7"]), (25088, [*MPTRJ, "--supercell", "8", "14", "8"]), (50176, [*MPTRJ, "--supercell", "14", "16", "8"])]
    for atoms, options in sizes:
        for implementation in IMPLEMENTATIONS:
            benchmark(f"capacity-{atoms}-{implementation}", options, [implementation], ["--warmup", "10", "--windows", "5", "--steps", "10"])


def campaign():
    replay()
    capacity()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["smoke", "replay", "capacity", "campaign"])
    parser.add_argument("--start-at", help="Resume from this named command after inspecting an earlier failure")
    args = parser.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    START_AT = args.start_at
    globals()[args.phase]()
    if START_AT is not None:
        raise SystemExit(f"Unknown command name: {START_AT}")
