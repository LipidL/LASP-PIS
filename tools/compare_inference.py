#!/usr/bin/env python3
"""Compare blocking inference against saved baseline libraries on identical replays."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'hpnn/python'), str(ROOT / 'mace/tests'), str(ROOT / 'hpnn/tools')]
from hpnn import HPNN
from native import NativeMH1
from structure_io import read as read_structure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True, type=Path,
                   help='Directory containing baseline libhpnn.so and libmace.so')
    p.add_argument('--build', type=Path, default=ROOT / 'build')
    p.add_argument('--repeats', nargs='+', type=int, default=[2, 4, 8])
    p.add_argument('--calls', type=int, default=12)
    p.add_argument('--windows', type=int, default=3)
    p.add_argument('--structure', type=Path, help='Also benchmark the first ARC/NPZ frame')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.calls < 1 or a.windows < 1 or any(r < 2 for r in a.repeats):
        p.error("Require positive calls/windows and repeats >= 2")
    rng = np.random.default_rng(32189)
    records = []
    for name, intervals in [('hpnn', 0), ('hpnn', 2048), ('mace', 0)]:
        paths = [a.baseline / ('lib' + name + '.so'),
                 a.build / name / ('lib' + name + '.so')]
        models = [HPNN(ROOT / 'hpnn/model/ggnn.pth', intervals=intervals, library=path)
                  if name == 'hpnn' else NativeMH1(path, ROOT / 'mace/model/mh1.model', 0)
                  for path in paths]
        try:
            for repeat in a.repeats + ([None] if a.structure else []):
                if repeat is None:
                    xyz, z, cell = read_structure(a.structure)
                    fixture = str(a.structure)
                else:
                    xyz = np.indices((repeat,) * 3).reshape(3, -1).T * 2.7 + .4
                    xyz += rng.normal(0, .04, xyz.shape)
                    z = np.full(len(xyz), 14, np.int32)
                    cell = np.eye(3, dtype=np.float32) * repeat * 2.7
                    fixture = 'synthetic silicon grid'

                frames = [(xyz + rng.normal(0, .001, xyz.shape)).astype(np.float32)
                          for _ in range(a.calls)]
                results = [m.calculate(frames[0], z, cell) for m in models]
                errors = {k: float(np.max(np.abs(results[0][k] - results[1][k])))
                          for k in ('energy', 'energies', 'forces', 'virial')}
                for k in errors:
                    np.testing.assert_allclose(results[1][k], results[0][k], atol=2e-4, rtol=2e-5)
                for m in models:
                    for _ in range(5):
                        m.calculate(frames[0], z, cell)
                times = [[], []]
                for _ in range(a.windows):
                    for index in rng.permutation(2):
                        for frame in frames:
                            start = time.perf_counter()
                            models[index].calculate(frame, z, cell)
                            times[index].append((time.perf_counter() - start) * 1000)
                medians = [float(np.median(t)) for t in times]
                record = dict(fixture=fixture, model=name, intervals=intervals, atoms=len(z),
                              baseline_ms=medians[0], optimized_ms=medians[1],
                              speedup=medians[0] / medians[1], max_abs_errors=errors,
                              samples_ms=times)
                records.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != 'samples_ms'}), flush=True)
        finally:
            for m in models:
                m.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [ROOT / 'hpnn/model/ggnn.pth', ROOT / 'mace/model/mh1.model']}
    a.output.write_text(json.dumps(dict(seed=32189, calls=a.calls, windows=a.windows, model_sha256=hashes,
                                       records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
