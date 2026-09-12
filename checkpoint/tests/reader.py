"""Exercise stored/deflated records, ZIP64 end records, CRC, and truncation."""
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zipfile

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "fixture.zip"
    for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, "w", compression=compression) as z:
            with z.open("archive/data.pkl", "w", force_zip64=True) as f:
                f.write(b"\x80\x02N.")
            z.writestr("archive/byteorder", "little")
        data = memory.getvalue()
        path.write_bytes(data)
        subprocess.run([sys.argv[1], str(path)], check=True)
        end = len(data) - 22
        _, _, _, count, _, size, offset, _ = struct.unpack("<IHHHHIIH", data[end:])
        # Use a ZIP64 EOCD even though this fixture is small.
        zip64 = struct.pack("<IQHHIIQQQQ", 0x06064b50, 44, 45, 45, 0, 0, count, count, size, offset)
        locator = struct.pack("<IIQI", 0x07064b50, 0, end, 1)
        eocd = struct.pack("<IHHHHIIH", 0x06054b50, 0, 0, 65535, 65535, 0xffffffff, 0xffffffff, 0)
        path.write_bytes(data[:end] + zip64 + locator + eocd)
        subprocess.run([sys.argv[1], str(path)], check=True)
        for bad in (data[:-8], data[:offset+16] + b"\0\0\0\0" + data[offset+20:]):
            path.write_bytes(bad)
            result = subprocess.run([sys.argv[1], str(path)], capture_output=True)
            assert result.returncode, "corrupt ZIP was accepted"
