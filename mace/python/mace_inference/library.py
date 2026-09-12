import os
from pathlib import Path


def library_path(filename="libmace.so"):
    project = Path(__file__).resolve().parents[2]
    variable = "MACE_EXPLICIT_LIBRARY" if filename == "libmace_explicit.so" else "MACE_LIBRARY"
    override = os.environ.get(variable)
    if override:
        return Path(override)
    candidates = [project / "build" / filename, project.parent / "build/mace" / filename]
    return next((p for p in candidates if p.is_file()), candidates[0])
