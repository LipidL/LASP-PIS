#!/usr/bin/env python3
"""Compare run records against a release baseline.

This is intentionally a small median-ratio check, not a statistical test.  A
real regression gate should use paired samples and a bootstrap interval; the
minimal harness keeps the rule explicit and easy to replace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_records(directory: Path) -> list[Path]:
    records = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in ("summary.json", "baseline.json"):
            continue
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        case_id = data.get("case_id")
        if case_id:
            records[case_id] = path
    return [records[key] for key in sorted(records)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="fail when the median is slower by more than this fraction",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail when a baseline case has no candidate record",
    )
    parser.add_argument(
        "--allow-environment-mismatch",
        action="store_true",
        help="compare even when GPU/compute capability/CUDA runtime differ",
    )
    args = parser.parse_args()

    baseline = read_json(args.baseline)
    if "cases" not in baseline:
        raise SystemExit(f"{args.baseline}: not a baseline file")
    baseline_cases = baseline["cases"]
    baseline_env = baseline.get("environment", {})
    env_keys = ("name", "compute_capability", "runtime_version")

    results = list(args.results)
    if args.results_dir:
        results += latest_records(args.results_dir)
    if not results:
        raise SystemExit("no run records given; use positional files or --results-dir")

    failed = False
    seen = set()
    print(f"{'case':30} {'baseline':>10} {'candidate':>10} {'delta':>9} status")
    for path in results:
        record = read_json(path)
        case_id = record.get("case_id")
        if not case_id:
            print(f"{path.name:30} {'':>10} {'':>10} {'':>9} not-a-record")
            continue
        if case_id not in baseline_cases:
            print(f"{case_id:30} {'':>10} {'':>10} {'':>9} not-in-baseline")
            continue
        device = record.get("device", {})
        mismatch = [
            key for key in env_keys
            if baseline_env.get(key) not in (None, device.get(key))
        ]
        if mismatch and not args.allow_environment_mismatch:
            print(
                f"{case_id:30} {'':>10} {'':>10} {'':>9} environment-mismatch "
                f"({', '.join(mismatch)})"
            )
            failed = True
            continue
        seen.add(case_id)
        if record.get("status") != "ok":
            print(f"{case_id:30} {'':>10} {'':>10} {'':>9} {record.get('status')}")
            failed = True
            continue
        base = baseline_cases[case_id].get("median_ms")
        candidate = record.get("summary", {}).get("median_ms")
        if base is None or candidate is None:
            print(f"{case_id:30} {'':>10} {'':>10} {'':>9} incomplete")
            failed = True
            continue
        delta = candidate / base - 1.0
        status = "regression" if delta > args.threshold else "ok"
        if status == "regression":
            failed = True
        print(
            f"{case_id:30} {base:10.4f} {candidate:10.4f} "
            f"{delta * 100:8.2f}% {status}"
        )
    missing = sorted(set(baseline_cases) - seen)
    for case_id in missing:
        print(f"{case_id:30} {'':>10} {'':>10} {'':>9} missing")
    if missing and args.require_all:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, KeyError) as error:
        print(f"compare.py: {error}", file=sys.stderr)
        raise SystemExit(1)
