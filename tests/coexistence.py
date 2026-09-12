#!/usr/bin/env python3
"""Check that embedded runtimes coexist even with global dynamic-library loading."""
import argparse
import ctypes as ct
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in
               ("mace/tests", "hpnn/python", "hpnn/tools", "hpnn/tests")]
from native import NativeMH1
from hpnn import HPNN
from model_io import export
from reference import generated_state


def main(args):
    paths = {"mace": args.mace, "hpnn": args.hpnn}
    order = [args.first, "hpnn" if args.first == "mace" else "mace"]
    libraries = [ct.CDLL(paths[name], mode=ct.RTLD_GLOBAL) for name in order]
    with tempfile.TemporaryDirectory(prefix="lapis-coexistence-") as directory:
        archive = Path(directory) / "generated.hpnn"
        export(archive, generated_state())
        m = NativeMH1(args.mace, args.archive, 0)
        try:
            with HPNN(archive, library=args.hpnn) as h:
                xyz = np.array([[1, 1, 1], [2.2, 1.1, 1.3]], np.float32)
                z = np.array([1, 6], np.int32)
                cell = np.diag([12., 12., 12.]).astype(np.float32)
                first_m = m.calculate(xyz, z, cell)
                first_h = h.calculate(xyz, z, cell)
                for model, expected in ((m, first_m), (h, first_h)):
                    actual = model.calculate(xyz, z, cell)
                    for key in ("energy", "energies", "forces", "virial"):
                        assert np.isfinite(actual[key]).all()
                        np.testing.assert_allclose(actual[key], expected[key], atol=2e-5, rtol=2e-6)
        finally:
            m.close()
    assert len(libraries) == 2  # Keep both global handles alive through inference.
    print("Both model libraries passed with load order:", ", ".join(order))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace", required=True)
    parser.add_argument("--hpnn", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--first", choices=("mace", "hpnn"), required=True)
    main(parser.parse_args())
