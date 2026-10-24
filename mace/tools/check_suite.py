#!/usr/bin/env python3
"""Run the project's GPU checks sequentially, saving complete logs."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from mace_inference.library import library_path


def main(args):
    root = Path(__file__).resolve().parents[1]
    logs = root / "benchmarks/logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(name, command):
        print(f"Running {name}", flush=True)
        with (logs / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=root, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print((logs / (name + ".log")).read_text()[-6000:], flush=True)
            raise SystemExit(f"{name} failed; see {logs/(name+'.log')}")
        print(f"Passed {name}", flush=True)

    py = sys.executable
    run("archive", [py, "tools/check_archive.py"])
    run(
        "mh1-reference",
        [py, "tools/validate.py", "--check", "--output", "benchmarks/correctness.json"],
    )
    run(
        "mh1-explicit",
        [
            py,
            "tools/validate.py",
            "--case",
            "Si216",
            "--check",
            "--library",
            str(library_path("libmace_explicit.so")),
            "--output",
            "benchmarks/correctness-explicit.json",
        ],
    )
    for head, archive in (
        ("omat_pbe", "model/mh1-omat.pwm"),
        ("spice_wB97M", "model/mh1-spice.pwm"),
        ("mp_pbe_refit_add", "model/mh1-mp.pwm"),
        ("oc20_usemppbe", "model/mh1-oc20.pwm"),
        ("omol", "model/mh1-omol.pwm"),
    ):
        if not (root / archive).exists():
            run(
                "export-" + head,
                [py, "tools/export_mh1.py", "model/mh1.model", archive, "--head", head],
            )
        run(
            "head-" + head,
            [
                py,
                "tools/validate.py",
                "--case",
                "mixed4",
                "--check",
                "--head",
                head,
                "--archive",
                archive,
                "--output",
                f"benchmarks/correctness-{head}.json",
            ],
        )
    run(
        "mh1-invariants",
        [
            py,
            "tools/check_invariants.py",
            "--archive",
            "model/mh1.pwm",
            "--output",
            "benchmarks/invariants.json",
        ],
    )
    run("md", [py, "tools/md.py"])
    if args.sanitizers:
        for tool in ("memcheck", "racecheck", "synccheck"):
            run(
                "mh1-" + tool,
                [
                    "/usr/local/cuda/bin/compute-sanitizer",
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "1",
                    py,
                    "tools/check_invariants.py",
                    "--archive",
                    "model/mh1.pwm",
                    "--smoke",
                ],
            )
    print("All selected checks passed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sanitizers", action="store_true")
    main(p.parse_args())
