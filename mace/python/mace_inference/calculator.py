import ctypes as ct
import json
import os
from pathlib import Path
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress


class PairwiseCalculator(Calculator):
    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(
        self,
        model=None,
        *,
        backend="auto",
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.5,
        library=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        path = (
            library
            or os.environ.get("PAIRWISE_LIBRARY")
            or Path(__file__).resolve().parents[2] / "build/libpairwise.so"
        )
        self.lib = ct.CDLL(str(path))
        self.lib.pw_last_error.restype = ct.c_char_p
        self.lib.pw_create_lj.argtypes = [ct.c_float] * 3 + [ct.c_int]
        self.lib.pw_create_lj.restype = ct.c_void_p
        self.lib.pw_create_mh1.argtypes = [ct.c_char_p, ct.c_int]
        self.lib.pw_create_mh1.restype = ct.c_void_p
        self.lib.pw_destroy.argtypes = [ct.c_void_p]
        self.lib.pw_calculate.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p] * 7
        self.lib.pw_diagnostics.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        b = {"auto": 0, "brute": 1, "cell": 2}[backend]
        self.handle = (
            self.lib.pw_create_mh1(os.fsencode(model), b)
            if model is not None
            else self.lib.pw_create_lj(epsilon, sigma, cutoff, b)
        )
        if not self.handle:
            raise RuntimeError(self.lib.pw_last_error().decode())

    def close(self):
        if getattr(self, "handle", None):
            self.lib.pw_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def evaluate(self, atoms):
        """Always execute, bypassing ASE's geometry cache (for trajectory replay)."""
        if not self.handle:
            raise RuntimeError("Calculator is closed")
        if not np.all(atoms.pbc):
            raise ValueError("v1 requires three periodic axes")
        xyz = np.ascontiguousarray(atoms.positions, dtype=np.float32)
        z = np.ascontiguousarray(atoms.numbers, dtype=np.int32)
        cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float32)
        energy = np.empty(1, np.float64)
        energies = np.empty(len(atoms), np.float32)
        forces = np.empty((len(atoms), 3), np.float32)
        virial = np.empty((3, 3), np.float32)
        arrays = [xyz, z, cell, energy, energies, forces, virial]
        if self.lib.pw_calculate(
            self.handle, len(atoms), *[a.ctypes.data for a in arrays]
        ):
            raise RuntimeError(self.lib.pw_last_error().decode())
        return dict(
            energy=energy[0],
            energies=energies,
            forces=forces,
            stress=full_3x3_to_voigt_6_stress(-virial / atoms.get_volume()),
            virial=virial,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = self.evaluate(self.atoms)

    @property
    def diagnostics(self):
        buf = ct.create_string_buffer(16384)
        if self.lib.pw_diagnostics(self.handle, buf, len(buf)):
            raise RuntimeError(self.lib.pw_last_error().decode())
        return json.loads(buf.value)
