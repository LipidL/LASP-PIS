#!/usr/bin/env python3
"""Extract a fixed MPtrj v2 frame using a bounded 256-KiB HTTP range request."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from ase import Atoms
from ase.io import write

URL = "https://ndownloader.figshare.com/files/41619375"
REVISION = "10.6084/m9.figshare.23713842.v2"


def main(args):
    if args.prefix:
        prefix = Path(args.prefix).read_bytes()
    else:
        request = Request(URL, headers={"Range": "bytes=0-262143"})
        with urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise RuntimeError("Server did not honor the bounded range request")
            prefix = response.read(262145)
    if len(prefix) != 262144:
        raise ValueError("Expected exactly the first 256 KiB")
    text = prefix.decode()
    match = re.match(r'\{"([^"]+)": \{"([^"]+)": ', text)
    if not match or match.groups() != ("mp-1005792", "mp-1012897-0-0"):
        raise ValueError("Pinned MPtrj file has unexpected first-frame identifiers")
    frame, _ = json.JSONDecoder().raw_decode(text, match.end())
    structure = frame["structure"]
    sites = structure["sites"]
    if any(len(s["species"]) != 1 or s["species"][0]["occu"] != 1 for s in sites):
        raise ValueError("Disordered structure requires a different frontend")
    atoms = Atoms(
        [s["species"][0]["element"] for s in sites],
        positions=[s["xyz"] for s in sites],
        cell=structure["lattice"]["matrix"],
        pbc=True,
    )
    atoms.info.update(
        dataset_revision=REVISION, material_id=match[1], frame_id=match[2]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, atoms, format="extxyz")
    metadata = {
        "dataset": "Materials Project Trajectory (MPtrj) Dataset",
        "revision": REVISION,
        "url": "https://figshare.com/articles/dataset/23713842/2",
        "file_url": URL,
        "file_md5_from_figshare": "50ead5f27f9a4f6beb7564c4188f1e9f",
        "full_file_downloaded_or_verified": False,
        "prefix_bytes": len(prefix),
        "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "material_id": match[1],
        "frame_id": match[2],
        "license": "MIT",
        "author": "Bowen Deng",
        "structure_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", help="Previously downloaded bytes 0 through 262143")
    parser.add_argument("--output", default="structures/mptrj-mp-1005792.extxyz")
    main(parser.parse_args())
