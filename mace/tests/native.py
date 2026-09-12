#!/usr/bin/env python3
"""Exercise the public MACE C ABI without the upstream reference environment."""
import argparse
import ctypes as ct
import json
import numpy as np


class NativeMH1:
    def __init__(self, library, archive, backend):
        self.lib = ct.CDLL(str(library))
        self.lib.mace_create_mh1.argtypes = [ct.c_char_p, ct.c_int]
        self.lib.mace_create_mh1.restype = ct.c_void_p
        self.lib.mace_last_error.restype = ct.c_char_p
        self.lib.mace_destroy.argtypes = [ct.c_void_p]
        self.lib.mace_destroy.restype = None
        self.lib.mace_calculate.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p] * 7
        self.handle = self.lib.mace_create_mh1(str(archive).encode(), backend)
        if not self.handle:
            raise RuntimeError(self.lib.mace_last_error().decode())

    def calculate(self, xyz, z, cell):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
        z = np.ascontiguousarray(z, dtype=np.int32)
        cell = np.ascontiguousarray(cell, dtype=np.float32).reshape(3, 3)
        energy = np.zeros(1, np.float64)
        atomic = np.zeros(len(z), np.float32)
        forces = np.zeros_like(xyz)
        virial = np.zeros((3, 3), np.float32)
        arrays = (xyz, z, cell, energy, atomic, forces, virial)
        if self.lib.mace_calculate(self.handle, len(z), *[a.ctypes.data for a in arrays]):
            raise RuntimeError(self.lib.mace_last_error().decode())
        return dict(energy=energy, energies=atomic, forces=forces, virial=virial)

    def close(self):
        if self.handle:
            self.lib.mace_destroy(self.handle)
            self.handle = None


def main(args):
    cell = np.array([[8.4, 0, 0], [1.1, 8.7, 0], [-0.8, 0.4, 9.2]], np.float32)
    xyz = np.array([[1, 1, 1], [2.5, 1.2, 1], [2.3, 2.7, 1.3], [3.1, 3.1, 1.5]], np.float32)
    z = np.array([14, 6, 8, 1], np.int32)
    reference = None
    for backend in (1, 2, 0):
        model = NativeMH1(args.library, args.archive, backend)
        try:
            result = model.calculate(xyz, z, cell)
            for value in result.values():
                assert np.isfinite(value).all()
            if reference is None:
                reference = result
            for key in result:
                np.testing.assert_allclose(result[key], reference[key], atol=2e-4, rtol=2e-5)
            order = np.array([2, 0, 3, 1])
            permuted = model.calculate(xyz[order], z[order], cell)
            for key in ('energies', 'forces'):
                np.testing.assert_allclose(permuted[key], result[key][order], atol=2e-4, rtol=2e-5)
            assert model.calculate([], [], cell)['energy'][0] == 0
            try:
                model.calculate(xyz, z, np.zeros((3, 3)))
            except RuntimeError:
                pass
            else:
                raise AssertionError('Singular cell accepted')
            recovered = model.calculate(xyz, z, cell)
            for key in result:
                np.testing.assert_allclose(recovered[key], result[key], atol=2e-4, rtol=2e-5)
        finally:
            model.close()
    print(json.dumps({'status': 'passed', 'checks': ['C ABI', 'backends', 'permutation', 'empty', 'error recovery']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--archive', required=True)
    main(parser.parse_args())
