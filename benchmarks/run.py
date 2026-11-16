#!/usr/bin/env python3
"""Run the minimal Lapis benchmark suite.

The driver is deliberately standard-library only.  It resolves tracked case
definitions, validates fixture and model hashes, materializes the canonical
binary input, invokes one model-specific executable, verifies the output
against a compact reference, and writes an untracked run record.

Typical flow:
    python3 benchmarks/run.py --update-manifest
    python3 benchmarks/run.py --tier smoke --update-reference
    python3 benchmarks/run.py --tier smoke
    python3 benchmarks/run.py --tier smoke --baseline-out benchmarks/baseline/sm86.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))
import structure  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in case_id)


def git_state() -> dict:
    state = {"sha": None, "dirty": None}
    try:
        state["sha"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        state["dirty"] = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return state


def case_sha256(case: dict) -> str:
    payload = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def load_manifest() -> dict:
    return read_json(BENCH / "fixtures" / "manifest.json")


def fixture_structure(entry: dict) -> dict:
    if "recipe" in entry:
        return structure.from_recipe(entry["recipe"])
    if "path" in entry:
        return structure.load_json(BENCH / entry["path"])
    raise ValueError("fixture entry needs recipe or path")


def resolve_fixture(entry: dict, cache: Path) -> tuple[str, Path, int, dict]:
    fixture = fixture_structure(entry)
    data = structure.encode(fixture)
    digest = sha256_bytes(data)
    expected = entry.get("sha256")
    if expected and digest != expected:
        raise RuntimeError(
            f"fixture hash mismatch: expected {expected}, generated {digest}"
        )
    path = cache / f"{digest}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or sha256_file(path) != digest:
        path.write_bytes(data)
    return digest, path, len(fixture["atomic_numbers"]), fixture


def update_manifest(manifest: dict) -> None:
    for fixture_id, entry in manifest["fixtures"].items():
        fixture = fixture_structure(entry)
        payload = structure.encode(fixture)
        entry["sha256"] = sha256_bytes(payload)
        entry["atoms"] = len(fixture["atomic_numbers"])
    target = BENCH / "fixtures" / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"updated {target} ({len(manifest['fixtures'])} fixtures)")


def resolve_model(
    model_id: str, models: dict, allow_mismatch: bool
) -> tuple[Path, str, str]:
    if model_id not in models:
        raise RuntimeError(f"unknown model id: {model_id}")
    entry = models[model_id]
    override = os.environ.get(
        f"LAPIS_BENCH_{entry.get('engine', model_id.split('/')[0]).upper()}_MODEL"
    )
    path = Path(override) if override else ROOT / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(f"model not found: {path}")
    actual = sha256_file(path)
    expected = entry.get("sha256", "")
    if expected and actual != expected and not allow_mismatch:
        raise RuntimeError(
            f"model hash mismatch for {model_id}: expected {expected}, got {actual}"
        )
    return path, actual, expected


def load_cases(tier: str, selected: list[str] | None) -> list[dict]:
    cases = []
    for path in sorted((BENCH / "cases").glob("*.json")):
        case = read_json(path)
        if case.get("schema") != 1:
            raise RuntimeError(f"{path}: unsupported case schema")
        if selected and case["id"] not in selected:
            continue
        if not selected and tier != "all" and case.get("tier", "smoke") != tier:
            continue
        cases.append(case)
    return cases


def compare_output(reference: dict, output: dict, tolerances: dict) -> dict:
    required = ("energy", "energies_sum", "virial", "force_max_abs", "force_l2", "force_sum")
    for key in required:
        if key not in output or key not in reference:
            raise RuntimeError(f"output/reference is missing {key}")
    if len(output["virial"]) != 9 or len(reference["virial"]) != 9:
        raise RuntimeError("virial must contain nine components")
    if len(output["force_sum"]) != 3 or len(reference["force_sum"]) != 3:
        raise RuntimeError("force_sum must contain three components")

    errors = {
        "energy_abs": abs(float(output["energy"]) - float(reference["energy"])),
        "energies_sum_abs": abs(
            float(output["energies_sum"]) - float(reference["energies_sum"])
        ),
        "virial_abs": max(
            abs(float(a) - float(b))
            for a, b in zip(output["virial"], reference["virial"])
        ),
        "force_max_abs": abs(
            float(output["force_max_abs"]) - float(reference["force_max_abs"])
        ),
        "force_l2_rel": abs(float(output["force_l2"]) - float(reference["force_l2"]))
        / max(abs(float(reference["force_l2"])), 1e-12),
        "force_sum_abs": max(
            abs(float(a) - float(b))
            for a, b in zip(output["force_sum"], reference["force_sum"])
        ),
    }
    limits = {
        "energy_abs": float(tolerances.get("energy_abs", 1e-4)),
        "energies_sum_abs": float(tolerances.get("energies_sum_abs", 1e-4)),
        "virial_abs": float(tolerances.get("virial_abs", 1e-3)),
        "force_max_abs": float(tolerances.get("force_max_abs", 1e-3)),
        "force_l2_rel": float(tolerances.get("force_l2_rel", 1e-3)),
        "force_sum_abs": float(tolerances.get("force_sum_abs", 1e-3)),
    }
    ok = all(errors[key] <= limits[key] for key in errors)
    return {
        "status": "ok" if ok else "mismatch",
        "max_errors": errors,
        "limits": limits,
        "failed": [key for key in errors if errors[key] > limits[key]],
    }


def bench_command(args, case: dict, exe: Path, model: Path, input_path: Path) -> list[str]:
    command = [
        str(exe),
        "--engine",
        case["engine"],
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--backend",
        case.get("backend", "auto"),
        "--warmup",
        str(args.warmup if args.warmup is not None else case.get("warmup", 5)),
        "--samples",
        str(args.samples if args.samples is not None else case.get("samples", 20)),
    ]
    if case.get("head"):
        command += ["--head", case["head"]]
    return command


def write_record(args, case: dict, record: dict) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = (record.get("git", {}).get("sha") or "nogit")[:12]
    base = f"{stamp}-{sha}-{safe_name(case['id'])}"
    path = args.out_dir / f"{base}.json"
    suffix = 1
    while path.exists():
        path = args.out_dir / f"{base}-{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    summary_path = args.out_dir / "summary.jsonl"
    summary = {
        "timestamp_utc": record["timestamp_utc"],
        "case_id": case["id"],
        "git": record.get("git", {}).get("sha"),
        "status": record.get("status"),
        "median_ms": record.get("summary", {}).get("median_ms"),
        "p95_ms": record.get("summary", {}).get("p95_ms"),
        "atoms": record.get("atoms"),
        "verification": record.get("verification", {}).get("status"),
        "record": path.name,
    }
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
    return path


def write_baseline(path: Path, records: list[dict]) -> None:
    usable = [
        r
        for r in records
        if r.get("status") == "ok"
        and r.get("verification", {}).get("status") == "ok"
    ]
    if not usable:
        raise RuntimeError("no verified records available for a baseline")
    environment = usable[0].get("device", {})
    cases = {}
    for record in usable:
        cases[record["case_id"]] = {
            "median_ms": record["summary"]["median_ms"],
            "p95_ms": record["summary"]["p95_ms"],
            "atoms": record.get("atoms"),
            "fixture_sha256": record.get("fixture_sha256"),
            "model_sha256": record.get("model_sha256"),
            "record_sha256": sha256_file(Path(record["_record_path"])),
        }
    baseline = {
        "schema": 1,
        "generated_utc": utc_now(),
        "git": usable[0].get("git", {}),
        "environment": environment,
        "environment_class": (
            f"sm{environment.get('compute_capability', 'unknown').replace('.', '')}"
            f"-{str(environment.get('name', 'unknown')).replace(' ', '-').lower()}"
            f"-cuda{environment.get('runtime_version', 'unknown')}"
        ),
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"wrote baseline {path} ({len(cases)} cases)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("smoke", "full", "all"), default="smoke")
    parser.add_argument("--case", action="append", default=None, help="case id (repeatable)")
    parser.add_argument("--list", action="store_true", help="list matching cases")
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--bench", type=Path, help="explicit executable (single engine)")
    parser.add_argument("--out-dir", type=Path, default=BENCH / "results")
    parser.add_argument("--cache-dir", type=Path, default=BENCH / ".cache" / "inputs")
    parser.add_argument("--warmup", type=int, help="override case warmup")
    parser.add_argument("--samples", type=int, help="override case samples")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--update-reference", action="store_true")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--baseline-out", type=Path)
    args = parser.parse_args()
    if args.force_reference:
        args.update_reference = True

    manifest = load_manifest()
    if args.update_manifest:
        update_manifest(manifest)
        return 0

    cases = load_cases(args.tier, args.case)
    if not cases:
        parser.error("no cases matched")
    if args.list:
        for case in cases:
            print(f"{case['id']:28} tier={case.get('tier', 'smoke'):5} "
                  f"engine={case['engine']:4} fixture={case['fixture']}")
        return 0
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")

    models = read_json(BENCH / "models.json")["models"]
    git = git_state()
    host = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    records = []
    failed = False

    for case in cases:
        case_id = case["id"]
        print(f"== {case_id} ==", flush=True)
        fixture_entry = manifest["fixtures"].get(case["fixture"])
        if fixture_entry is None:
            raise RuntimeError(f"unknown fixture: {case['fixture']}")
        try:
            fixture_sha, input_path, atoms, _ = resolve_fixture(
                fixture_entry, args.cache_dir
            )
            model_path, model_sha, model_expected = resolve_model(
                case["model"], models, args.allow_model_mismatch
            )
            exe = args.bench or (args.bench_dir / f"lapis_bench_{case['engine']}")
            if not exe.is_file() and not args.bench:
                raise FileNotFoundError(f"benchmark executable not found: {exe}")
        except (FileNotFoundError, RuntimeError) as error:
            if args.skip_missing:
                print(f"skip {case_id}: {error}")
                continue
            raise

        command = bench_command(args, case, exe, model_path, input_path)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            record = {
                "schema": 1,
                "status": "error",
                "error": f"benchmark timed out after {args.timeout}s",
                "case_id": case_id,
                "command": command,
            }
        else:
            try:
                record = json.loads(completed.stdout)
            except json.JSONDecodeError:
                record = {
                    "schema": 1,
                    "status": "error",
                    "error": "benchmark did not emit JSON",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                    "case_id": case_id,
                    "command": command,
                }
            if completed.returncode and record.get("status") == "ok":
                record["status"] = "error"
                record["error"] = f"benchmark exited with {completed.returncode}"
            if record.get("status") == "error" and completed.stderr:
                record.setdefault("stderr", completed.stderr[-4000:])

        record.update(
            {
                "timestamp_utc": utc_now(),
                "case_id": case_id,
                "case_sha256": case_sha256(case),
                "tier": case.get("tier", "smoke"),
                "fixture_id": case["fixture"],
                "fixture_sha256": fixture_sha,
                "atoms": atoms,
                "model_id": case["model"],
                "model_sha256": model_sha,
                "model_expected_sha256": model_expected,
                "git": git,
                "host": host,
                "command": command,
            }
        )
        record.pop("_record_path", None)

        reference_path = BENCH / "references" / f"{safe_name(case_id)}.json"
        verification = {"status": "unverified", "reference": reference_path.name}
        if record.get("status") == "ok" and "output" in record:
            if args.update_reference:
                reference = {
                    "schema": 1,
                    "case_id": case_id,
                    "fixture_sha256": fixture_sha,
                    "model_sha256": model_sha,
                    "tolerances": case.get("tolerances", {}),
                    "output": record["output"],
                    "generated_utc": utc_now(),
                    "note": "Compact golden summary, not a full force/virial array.",
                }
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                reference_path.write_text(
                    json.dumps(reference, indent=2) + "\n", encoding="utf-8"
                )
                verification = {
                    "status": "ok",
                    "reference": reference_path.name,
                    "written": True,
                }
                record["status"] = "ok"
            elif reference_path.is_file():
                reference = read_json(reference_path)
                if reference.get("fixture_sha256") not in (None, fixture_sha):
                    raise RuntimeError(f"{reference_path}: fixture hash no longer matches")
                if reference.get("model_sha256") not in (None, model_sha):
                    raise RuntimeError(f"{reference_path}: model hash no longer matches")
                comparison = compare_output(
                    reference["output"],
                    record["output"],
                    case.get("tolerances", {}),
                )
                verification = {
                    "status": comparison["status"],
                    "reference": reference_path.name,
                    "max_errors": comparison["max_errors"],
                    "limits": comparison["limits"],
                    "failed": comparison["failed"],
                }
                record["status"] = comparison["status"]
            elif args.require_reference:
                verification = {
                    "status": "missing",
                    "reference": reference_path.name,
                }
                record["status"] = "missing-reference"

        record["verification"] = verification
        path = write_record(args, case, record)
        record["_record_path"] = str(path)
        records.append(record)
        if record["status"] not in ("ok", "unverified"):
            failed = True
        median = record.get("summary", {}).get("median_ms")
        p95 = record.get("summary", {}).get("p95_ms")
        shown = f"{median:10.4f} {p95:10.4f}" if median is not None else f"{'-':>10} {'-':>10}"
        print(f"{case_id:28} {atoms:>6} {shown} {record['status']:10} "
              f"{verification['status']:10} -> {path.name}", flush=True)

    if args.baseline_out:
        write_baseline(args.baseline_out, records)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"run.py: {error}", file=sys.stderr)
        raise SystemExit(1)
