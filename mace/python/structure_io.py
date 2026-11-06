"""Read a periodic structure from ARC or NPZ without requiring ASE."""
import math
from pathlib import Path

import numpy as np

SYMBOLS = "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()


def read(path, frame=0):
    """Return positions (N, 3), atomic numbers (N,), and cell (3, 3)."""
    if frame < 0:
        raise ValueError("frame must be a nonnegative integer")
    path = Path(path)
    if path.suffix == ".npz":
        if frame:
            raise ValueError("NPZ files hold a single structure; frame must be 0")
        with np.load(path, allow_pickle=False) as data:
            return _checked(data["positions"], data["elements"], data["cell"])
    if path.suffix.lower() == ".arc":
        return _read_arc(path, frame)
    try:
        from ase.io import read as ase_read
    except ImportError as error:
        raise ValueError("Install ASE for formats other than ARC and NPZ") from error
    atoms = ase_read(path, index=frame)
    if not atoms.pbc.all():
        raise ValueError("MH1 requires three periodic axes")
    return _checked(atoms.positions, atoms.numbers, atoms.cell.array)


def _read_arc(path, frame):
    xyz, z, cell, index = [], [], None, 0
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "PBC" and len(fields) >= 7:
            cell = _cell(*map(float, fields[1:7]))
        elif len(fields) >= 5 and fields[4] == "CORE":
            if cell is None:
                raise ValueError("ARC atom precedes its PBC line")
            try:
                z.append(SYMBOLS.index(fields[0]))
            except ValueError:
                raise ValueError(f"Unknown element symbol: {fields[0]}") from None
            xyz.append([float(value) for value in fields[1:4]])
        elif fields[0].lower() == "end" and xyz:
            if cell is None:
                raise ValueError("ARC frame has no PBC line")
            if index == frame:
                return _checked(xyz, z, cell)
            index += 1
            xyz, z, cell = [], [], None
    raise IndexError(f"ARC file has no frame {frame}")


def _cell(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = map(math.radians, (alpha, beta, gamma))
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    return [
        [a, 0.0, 0.0],
        [b * math.cos(gamma), b * math.sin(gamma), 0.0],
        [cx, cy, math.sqrt(c * c - cx * cx - cy * cy)],
    ]


def _checked(positions, elements, cell):
    positions = np.asarray(positions, np.float32)
    elements = np.asarray(elements, np.int32)
    cell = np.asarray(cell, np.float32)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or elements.ndim != 1
        or len(positions) != len(elements)
        or cell.shape != (3, 3)
    ):
        raise ValueError("Expected positions (N,3), elements (N,), cell (3,3)")
    return positions, elements, cell
