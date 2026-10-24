#!/usr/bin/env python3
"""Compare every native MH1 record against the official Python export path.

Run in the upstream MACE environment (its package must be on PYTHONPATH).
No GPU is needed. Every head is checked, plus a changed-weight checkpoint.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_mh1 import export


def records(path):
    out = {}
    with path.open("rb") as f:
        assert f.read(8) == b"PWMH1001"
        count, = struct.unpack("<I", f.read(4))
        for _ in range(count):
            length, size = struct.unpack("<IQ", f.read(12))
            name = f.read(length).decode()
            out[name] = f.read(size)
        assert not f.read(1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder = args.reference_dir or Path(tmp)
        folder.mkdir(exist_ok=True, parents=True)
        model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        def check(checkpoint, head, name):
            reference = folder / (name + ".pwm")
            native = Path(tmp) / "native.pwm"
            export(SimpleNamespace(checkpoint=checkpoint, head=head, output=reference))
            subprocess.run([str(args.dump), str(checkpoint), str(native), head], check=True)
            a, b = records(native), records(reference)
            assert a.keys() == b.keys()
            for key in a:
                assert a[key] == b[key], f"{name}: native/reference mismatch in {key}"
            print(f"MH1 {name}: {len(a)} records match byte for byte", flush=True)

        for head in model.heads:
            check(args.checkpoint, head, head)
        with torch.no_grad():
            model.node_embedding.linear.weight[0] += 0.125
        changed = Path(tmp) / "changed.model"
        torch.save(model, changed)
        check(changed, "matpes_r2scan", "changed")
        bad = subprocess.run([str(args.dump), str(args.checkpoint), str(Path(tmp)/"bad.pwm"), "missing_head"], capture_output=True, text=True)
        assert bad.returncode and "unknown head" in bad.stderr


if __name__ == "__main__":
    main()
