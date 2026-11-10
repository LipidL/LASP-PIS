"""Canonical benchmark structure generation and encoding.

Only the Python standard library is used here.  This module is imported by
run.py and can also be used from fixture-authoring scripts.  The binary format
is deliberately tiny so the C++ benchmark only needs a few reads and no JSON
parser:

    magic      8 bytes   b"LAPISB01"
    version    uint32    1
    reserved   uint32    0
    atoms      uint64
    cell       9 x float64 (row-major)
    elements   atoms x int32
    positions  3 * atoms x float32

The SHA-256 of this byte stream is the stable fixture identity.  JSON layout,
whitespace, and key order therefore do not affect a fixture hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

MAGIC = b"LAPISB01"
VERSION = 1
_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    """SplitMix64 finalizer; used for deterministic, dependency-free jitter."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _unit(seed: int, index: int, component: int) -> float:
    """Deterministic uniform in [-1, 1] for (seed, atom, component)."""
    key = (
        (seed & _MASK64)
        ^ ((index + 1) * 0x9E3779B97F4A7C15)
        ^ ((component + 1) * 0xD1B54A32D192ED03)
    ) & _MASK64
    return (_mix64(key) >> 11) / float(1 << 53) * 2.0 - 1.0


def _as_cell(value) -> list[float]:
    if len(value) == 3 and all(not isinstance(x, (int, float)) for x in value):
        rows = [list(map(float, row)) for row in value]
    else:
        flat = list(map(float, value))
        if len(flat) != 9:
            raise ValueError("cell must contain nine numbers or three rows")
        rows = [flat[0:3], flat[3:6], flat[6:9]]
    return [x for row in rows for x in row]


def _as_positions(value) -> list[float]:
    if value and isinstance(value[0], (list, tuple)):
        flat = [float(x) for row in value for x in row]
    else:
        flat = [float(x) for x in value]
    if len(flat) % 3:
        raise ValueError("positions must contain three coordinates per atom")
    return flat


def generate_grid(recipe: dict) -> dict:
    """Generate a periodic, jittered cubic grid.

    Recipe fields:
      repeat: [nx, ny, nz]
      spacing: scalar Angstrom (or [sx, sy, sz])
      jitter: maximum absolute displacement in Angstrom
      element: atomic number, or [numbers] cycled over atoms
      seed: integer
    """
    repeat = [int(x) for x in recipe["repeat"]]
    if len(repeat) != 3 or any(x < 1 for x in repeat):
        raise ValueError("repeat must contain three positive integers")
    spacing = recipe.get("spacing", 2.7)
    if isinstance(spacing, (int, float)):
        spacing = [float(spacing)] * 3
    spacing = [float(x) for x in spacing]
    if len(spacing) != 3 or any(not math.isfinite(x) or x <= 0 for x in spacing):
        raise ValueError("spacing must be positive and finite")
    jitter = float(recipe.get("jitter", 0.0))
    if not math.isfinite(jitter) or jitter < 0 or jitter >= min(spacing) / 2:
        raise ValueError("jitter must be finite, nonnegative, and less than half the spacing")
    elements = recipe.get("element", 14)
    if isinstance(elements, (int, float)):
        elements = [int(elements)]
    elements = [int(x) for x in elements]
    if not elements or any(not 1 <= x <= 118 for x in elements):
        raise ValueError("element must contain atomic numbers in [1, 118]")
    seed = int(recipe.get("seed", 0))

    nx, ny, nz = repeat
    positions = []
    numbers = []
    index = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = [
                    (ix + 0.5) * spacing[0],
                    (iy + 0.5) * spacing[1],
                    (iz + 0.5) * spacing[2],
                ]
                for component in range(3):
                    base[component] += jitter * _unit(seed, index, component)
                positions.append(base)
                numbers.append(elements[index % len(elements)])
                index += 1
    cell = [
        nx * spacing[0], 0.0, 0.0,
        0.0, ny * spacing[1], 0.0,
        0.0, 0.0, nz * spacing[2],
    ]
    return {
        "format": "lapis-structure-v1",
        "cell": cell,
        "positions": positions,
        "atomic_numbers": numbers,
        "pbc": [True, True, True],
    }


def from_json(data: dict) -> dict:
    numbers = data.get("atomic_numbers", data.get("elements"))
    if numbers is None:
        raise ValueError("structure JSON needs atomic_numbers")
    cell = _as_cell(data["cell"])
    positions = _as_positions(data["positions"])
    numbers = [int(x) for x in numbers]
    if len(positions) != 3 * len(numbers):
        raise ValueError("positions and atomic_numbers disagree")
    if not numbers:
        raise ValueError("structure must contain at least one atom")
    if any(not 1 <= x <= 118 for x in numbers):
        raise ValueError("atomic numbers must be in [1, 118]")
    if any(not math.isfinite(x) for x in cell + positions):
        raise ValueError("structure contains a non-finite value")
    if abs(
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6])
    ) < 1e-12:
        raise ValueError("cell is singular")
    return {
        "format": "lapis-structure-v1",
        "cell": cell,
        "positions": [positions[i : i + 3] for i in range(0, len(positions), 3)],
        "atomic_numbers": numbers,
        "pbc": [bool(x) for x in data.get("pbc", [True, True, True])],
    }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return from_json(json.load(handle))


def from_recipe(recipe: dict) -> dict:
    kind = recipe.get("kind", "grid")
    if kind == "grid":
        return generate_grid(recipe)
    raise ValueError(f"unknown fixture recipe kind: {kind}")


def encode(structure: dict) -> bytes:
    cell = _as_cell(structure["cell"])
    positions = _as_positions(structure["positions"])
    numbers = [int(x) for x in structure["atomic_numbers"]]
    count = len(numbers)
    if len(positions) != 3 * count:
        raise ValueError("positions and atomic_numbers disagree")
    if count == 0:
        raise ValueError("structure must contain at least one atom")
    if not all(structure.get("pbc", [True, True, True])):
        raise ValueError("benchmark fixtures must be three-dimensional periodic")
    return b"".join(
        (
            struct.pack("<8sIIQ", MAGIC, VERSION, 0, count),
            struct.pack("<9d", *cell),
            struct.pack(f"<{count}i", *numbers),
            struct.pack(f"<{3 * count}f", *positions),
        )
    )


def sha256(structure: dict) -> str:
    return hashlib.sha256(encode(structure)).hexdigest()


def write_binary(structure: dict, path: Path) -> str:
    payload = encode(structure)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
