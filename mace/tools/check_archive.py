#!/usr/bin/env python3
"""Check that malformed MH1 archives fail before submitting model kernels."""
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mace_inference import MH1Calculator


def records(blob):
    offset = 12
    result = {}
    for _ in range(struct.unpack_from("<I", blob, 8)[0]):
        length, size = struct.unpack_from("<IQ", blob, offset)
        header = offset
        offset += 12
        name = blob[offset : offset + length].decode()
        offset += length
        result[name] = (header, offset, size)
        offset += size
    return result


def main():
    original = (ROOT / "model/mh1.pwm").read_bytes()
    locations = records(original)
    with tempfile.TemporaryDirectory(prefix="pairwise-archive-") as tmp:
        path = Path(tmp) / "invalid.pwm"

        def rejected(blob, expected):
            path.write_bytes(blob)
            try:
                calc = MH1Calculator(path)
            except RuntimeError as exc:
                assert expected in str(exc), str(exc)
                print("Rejected:", expected)
            else:
                calc.close()
                raise AssertionError("Malformed archive accepted")

        rejected(b"", "magic")
        rejected(original[:10], "tensor count")
        rejected(original[:-4], "Truncated")
        for name, index, value, expected in (
            ("layer1.linear_up.dims", 1, 4096, "linear dimensions"),
            ("layer0.tp.terms", 2, 16, "tensor-product index"),
            ("layer1.product.terms", 2, -1, "polynomial index"),
            ("layer1.product.linear.paths", 0, 100000, "linear path"),
            ("atomic_numbers", 0, 119, "element map"),
        ):
            blob = bytearray(original)
            struct.pack_into("<i", blob, locations[name][1] + 4 * index, value)
            rejected(blob, expected)
        blob = bytearray(original)
        struct.pack_into("<f", blob, locations["embedding"][1], float("nan"))
        rejected(blob, "Nonfinite")
        # Truncate a validly framed tensor, retaining every following record.
        name = "layer0.radial.source"
        header, start, size = locations[name]
        blob = bytearray(original)
        del blob[start + size - 4 : start + size]
        struct.pack_into("<Q", blob, header + 4, size - 4)
        rejected(blob, "tensor size")
    print("Archive rejection checks passed")


if __name__ == "__main__":
    main()
