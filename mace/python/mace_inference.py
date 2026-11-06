"""Blocking NumPy frontend for the MACE MH1 shared library."""
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np

_BACKENDS = {"auto": 0, "brute": 1, "cell": 2}


def library_path():
    """Locate libmace.so in a source checkout, honoring MACE_LIBRARY."""
    override = os.environ.get("MACE_LIBRARY")
    if override:
        return Path(override)
    project = Path(__file__).resolve().parents[1]
    candidates = [project / "build/libmace.so", project.parent / "build/mace/libmace.so"]
    return next((p for p in candidates if p.is_file()), candidates[0])


class MH1:
    """Single-flight MACE MH1 calculator; calls block through output transfer."""

    def __init__(self, model, backend="auto", head=None, library=None):
        self._handle = None
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; expected one of {', '.join(_BACKENDS)}")
        self._lib = C.CDLL(str(library or library_path()))
        lib = self._lib
        lib.mace_create_mh1_head.argtypes = [C.c_char_p, C.c_int, C.c_char_p]
        lib.mace_create_mh1_head.restype = C.c_void_p
        lib.mace_destroy.argtypes = [C.c_void_p]
        lib.mace_last_error.restype = C.c_char_p
        lib.mace_calculate.argtypes = [C.c_void_p, C.c_int] + [C.c_void_p] * 7
        lib.mace_calculate.restype = C.c_int
        lib.mace_diagnostics.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        lib.mace_diagnostics.restype = C.c_int
        self._handle = lib.mace_create_mh1_head(
            os.fsencode(model), _BACKENDS[backend], (head or "").encode()
        )
        if not self._handle:
            raise RuntimeError(lib.mace_last_error().decode())

    def calculate(self, positions, elements, cell):
        """Return energy, atomic energies, forces, virial and stress for one structure."""
        if not self._handle:
            raise RuntimeError("MH1 calculator is closed")
        xyz = np.ascontiguousarray(positions, dtype=np.float32)
        z = np.ascontiguousarray(elements, dtype=np.int32)
        lattice = np.ascontiguousarray(cell, dtype=np.float32)
        if (
            xyz.ndim != 2
            or xyz.shape[1:] != (3,)
            or z.ndim != 1
            or len(z) != len(xyz)
            or lattice.shape != (3, 3)
        ):
            raise ValueError("Expected positions (N,3), elements (N,), cell (3,3)")
        energies = np.empty(len(z), np.float32)
        forces = np.empty_like(xyz)
        virial = np.empty((3, 3), np.float32)
        energy = C.c_double()
        rc = self._lib.mace_calculate(
            self._handle,
            len(z),
            xyz.ctypes.data,
            z.ctypes.data,
            lattice.ctypes.data,
            C.byref(energy),
            energies.ctypes.data,
            forces.ctypes.data,
            virial.ctypes.data,
        )
        if rc:
            raise RuntimeError(self._lib.mace_last_error().decode())
        volume = abs(np.linalg.det(lattice.astype(np.float64)))
        return {
            "energy": energy.value,
            "energies": energies,
            "forces": forces,
            "virial": virial,
            "stress": -virial / volume,
        }

    @property
    def diagnostics(self):
        if not self._handle:
            raise RuntimeError("MH1 calculator is closed")
        buffer = C.create_string_buffer(16384)
        if self._lib.mace_diagnostics(self._handle, buffer, len(buffer)):
            raise RuntimeError(self._lib.mace_last_error().decode())
        return json.loads(buffer.value)

    def close(self):
        if self._handle:
            self._lib.mace_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


__all__ = ["MH1", "library_path"]
