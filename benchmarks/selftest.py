#!/usr/bin/env python3
"""Validate tracked benchmark metadata without a GPU.

This catches the common failures that make a benchmark unreproducible:
a fixture recipe drifted from its manifest hash, a case points at a missing
fixture/model, duplicate case ids, or a reference was generated for a different
fixture/model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
import structure  # noqa: E402


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    errors: list[str] = []
    checks = 0
    manifest = read_json(BENCH / "fixtures" / "manifest.json")
    models = read_json(BENCH / "models.json")["models"]

    fixture_hashes = {}
    for fixture_id, entry in manifest["fixtures"].items():
        try:
            if "recipe" in entry:
                fixture = structure.from_recipe(entry["recipe"])
            else:
                fixture = structure.load_json(BENCH / entry["path"])
            payload = structure.encode(fixture)
            digest = sha256_bytes(payload)
            atoms = len(fixture["atomic_numbers"])
            checks += 1
            if entry.get("sha256") != digest:
                errors.append(
                    f"fixture {fixture_id}: manifest hash {entry.get('sha256')} "
                    f"!= generated {digest}"
                )
            if entry.get("atoms") not in (None, atoms):
                errors.append(
                    f"fixture {fixture_id}: manifest atoms {entry.get('atoms')} != {atoms}"
                )
            fixture_hashes[fixture_id] = digest
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"fixture {fixture_id}: {error}")

    case_ids = set()
    cases = {}
    for path in sorted((BENCH / "cases").glob("*.json")):
        try:
            case = read_json(path)
            case_id = case["id"]
            checks += 1
            if case_id in case_ids:
                errors.append(f"{path.name}: duplicate case id {case_id}")
            case_ids.add(case_id)
            cases[case_id] = case
            if case.get("fixture") not in fixture_hashes:
                errors.append(f"{path.name}: unknown fixture {case.get('fixture')}")
            if case.get("model") not in models:
                errors.append(f"{path.name}: unknown model {case.get('model')}")
            if case.get("engine") not in ("hpnn", "mace"):
                errors.append(f"{path.name}: invalid engine {case.get('engine')}")
            if case.get("tier", "smoke") not in ("smoke", "full"):
                errors.append(f"{path.name}: invalid tier {case.get('tier')}")
            for key, value in case.get("tolerances", {}).items():
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"{path.name}: invalid tolerance {key}={value}")
        except (OSError, json.JSONDecodeError, KeyError) as error:
            errors.append(f"{path.name}: {error}")

    required_output = {
        "energy",
        "energies_sum",
        "virial",
        "force_max_abs",
        "force_l2",
        "force_sum",
    }
    for path in sorted((BENCH / "references").glob("*.json")):
        try:
            reference = read_json(path)
            case_id = reference.get("case_id")
            checks += 1
            if case_id not in cases:
                errors.append(f"{path.name}: reference for unknown case {case_id}")
                continue
            case = cases[case_id]
            fixture_sha = fixture_hashes.get(case["fixture"])
            if reference.get("fixture_sha256") != fixture_sha:
                errors.append(f"{path.name}: fixture hash no longer matches")
            if reference.get("model_sha256") != models[case["model"]].get("sha256"):
                errors.append(f"{path.name}: model hash no longer matches")
            if not required_output <= set(reference.get("output", {})):
                errors.append(f"{path.name}: incomplete output summary")
        except (OSError, json.JSONDecodeError, KeyError) as error:
            errors.append(f"{path.name}: {error}")

    if errors:
        for error in errors:
            print(f"selftest: {error}", file=sys.stderr)
        return 1
    print(f"selftest: {checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
