#!/usr/bin/env python3
"""Compare saved MACE libraries sequentially, alternating order each round."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/lapis_bench_mace"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=LIBRARY_DIR")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0 or args.samples < 1:
        parser.error("rounds/samples must be positive and warmup nonnegative")
    variants = {}
    for value in args.variant:
        name, directory = value.split("=", 1)
        library = Path(directory).resolve() / "libmace.so.0"
        if name in variants or not name or any(c not in "-_abcdefghijklmnopqrstuvwxyz0123456789" for c in name):
            parser.error("variant names must be unique lowercase identifiers")
        variants[name] = (library, hashlib.sha256(library.read_bytes()).hexdigest())
    if len({p.stem for p in args.input}) != len(args.input):
        parser.error("input file stems must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_sha256": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input},
    }
    records = []
    for round_index in range(args.rounds):
        inputs = args.input if round_index % 2 == 0 else args.input[::-1]
        names = list(variants) if round_index % 2 == 0 else list(variants)[::-1]
        for fixture in inputs:
            for name in names:
                library, digest = variants[name]
                env = dict(os.environ)
                env.pop("MACE_PROFILE_MH1", None)
                env.pop("LD_PRELOAD", None)
                env["LD_LIBRARY_PATH"] = str(library.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                command = [str(args.executable.resolve()), "--engine", "mace",
                           "--model", str(args.model.resolve()), "--input", str(fixture.resolve()),
                           "--backend", "auto", "--warmup", str(args.warmup), "--samples", str(args.samples)]
                done = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                record = json.loads(done.stdout)
                record.update(variant=name, fixture=fixture.stem, round=round_index,
                              library=str(library), library_sha256=digest, command=command)
                records.append(record)
                path = args.output / f"{fixture.stem}-{name}-r{round_index}.json"
                path.write_text(json.dumps(record, indent=2) + "\n")
                print(f"{path.stem}: {record['summary']['median_ms']:.3f} ms", flush=True)
    summary = {}
    for fixture in args.input:
        summary[fixture.stem] = {}
        for name in variants:
            selected = [r for r in records if r["fixture"] == fixture.stem and r["variant"] == name]
            medians = [r["summary"]["median_ms"] for r in selected]
            samples = sorted(v for r in selected for v in r["samples_ms"])
            summary[fixture.stem][name] = {
                "median_ms": statistics.median(medians), "round_medians_ms": medians,
                "p95_ms": samples[(95 * len(samples) + 99) // 100 - 1],
                "resources": selected[0]["resources"],
            }
    (args.output / "summary.json").write_text(json.dumps({**metadata, "summary": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
