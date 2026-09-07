#!/usr/bin/env python3
"""Export the supplied official MH1 checkpoint; inference has no Python dependency.

The archive stores normalized linear maps, immutable radial coefficients and sparse
polynomials. It contains no generated CUDA or serialized executable Python.
"""
import argparse
import hashlib
import itertools
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mace-ref"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pairwise-matplotlib")
import numpy as np
import torch

torch.serialization.add_safe_globals([slice])
from e3nn import o3


class Archive:
    def __init__(self):
        self.items = {}

    def add(self, name, value, integer=False):
        self.items[name] = np.asarray(value, dtype="<i4" if integer else "<f4").copy()

    def write(self, path):
        with open(path, "wb") as f:
            f.write(b"PWMH1001")
            f.write(struct.pack("<I", len(self.items)))
            for name, a in self.items.items():
                key = name.encode()
                f.write(struct.pack("<IQ", len(key), a.nbytes))
                f.write(key)
                f.write(a.tobytes())


def linear(ar, name, module, select_in=None, select_out=None):
    ins, outs = module.irreps_in, module.irreps_out
    ioff = np.cumsum([0] + [mul * ir.dim for mul, ir in ins])
    ooff = np.cumsum([0] + [mul * ir.dim for mul, ir in outs])
    desc, weights = [], []
    for instruction, w in zip(module.instructions, module.weight_views()):
        i, o = instruction.i_in, instruction.i_out
        a = w.detach().numpy() * instruction.path_weight
        if select_in is not None:
            assert len(ins) == 1
            a = a[select_in, :]
        if select_out is not None:
            assert len(outs) == 1
            a = a[:, select_out]
        if a.ndim == 1:
            a = a.reshape(w.shape[0] if select_in is None else len(select_in), -1)
        desc.append(
            [
                ioff[i] if select_in is None else 0,
                ooff[o] if select_out is None else 0,
                a.shape[0],
                a.shape[1],
                ins[i].ir.dim,
                sum(x.size for x in weights),
            ]
        )
        weights.append(a)
    ar.add(name + ".paths", desc, True)
    ar.add(
        name + ".dims",
        [
            int(ioff[-1]) if select_in is None else len(select_in),
            int(ooff[-1]) if select_out is None else len(select_out),
        ],
        True,
    )
    ar.add(name + ".weight", np.concatenate([a.ravel() for a in weights]))


def radial(ar, name, module, source, target):
    net = module.net
    w = net[0].weight.detach().numpy()
    # Species-only portions of the first affine transform are immutable lookups.
    ar.add(name + ".source", source @ w[:, 10:522].T)
    ar.add(name + ".target", target @ w[:, 522:].T)
    ar.add(name + ".radial", w[:, :10])
    for k, block in enumerate(net):
        if hasattr(block, "weight"):
            ar.add(name + f".{k}.weight", block.weight.detach().numpy())
            ar.add(name + f".{k}.bias", block.bias.detach().numpy())


def polynomial(ar, name, product):
    terms, coefficients = [], []
    out_offset = 0
    for contraction in product.symmetric_contractions.contractions:
        assert contraction.correlation == 3
        assert contraction.weights_max.shape[0] == 1
        for degree in (1, 2, 3):
            u = contraction.U_tensors(degree).detach().numpy()
            w = (
                (
                    contraction.weights_max
                    if degree == 3
                    else contraction.weights[2 - degree]
                )
                .detach()
                .numpy()[0]
            )
            if u.ndim == degree + 1:
                u = u[None]
            # Sum permutations of commuting inputs before multiplying by weights.
            # This removes antisymmetric paths that cancel in the polynomial.
            for out in range(u.shape[0]):
                for indices in itertools.combinations_with_replacement(
                    range(16), degree
                ):
                    coupling = sum(
                        (u[(out,) + p] for p in set(itertools.permutations(indices)))
                    )
                    coeff = coupling @ w
                    if np.max(np.abs(coeff)) < 1e-12:
                        continue
                    terms.append(
                        [out_offset + out, degree, *indices, *([-1] * (3 - degree))]
                    )
                    coefficients.append(coeff)
        out_offset += (
            1
            if contraction.U_tensors(1).ndim == 2
            else contraction.U_tensors(1).shape[0]
        )
    ar.add(name + ".terms", terms, True)
    ar.add(name + ".coefficients", coefficients)
    print(name, len(terms), "monomials")


def export(args):
    torch.set_num_threads(4)
    model = (
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        .double()
        .eval()
    )
    assert (
        len(model.interactions) == 2
        and model.node_embedding.linear.irreps_out.dim == 512
    )
    assert not model.radial_embedding.apply_cutoff
    assert float(model.r_max) == 6.0
    head = model.heads.index(args.head)
    ar = Archive()
    ar.add("atomic_numbers", model.atomic_numbers.numpy(), True)
    ar.add(
        "embedding",
        model.node_embedding(torch.eye(len(model.atomic_numbers), dtype=torch.float64))
        .detach()
        .numpy(),
    )
    ar.add("e0", model.atomic_energies_fn.atomic_energies[head].detach().numpy())
    re = model.radial_embedding
    zbl = model.pair_repulsion_fn
    ar.add(
        "constants",
        [
            model.r_max,
            re.bessel_fn.prefactor,
            re.cutoff_fn.p,
            re.distance_transform.a,
            re.distance_transform.q,
            re.distance_transform.p,
            model.scale_shift.scale[head],
            model.scale_shift.shift[head],
            zbl.a_exp,
            zbl.a_prefactor,
            zbl.p,
            model.interactions[0].equivariant_nonlin.act_scalars.acts[0].cst,
            model.interactions[0].equivariant_nonlin.act_gates.acts[0].cst,
            model.readouts[1].non_linearity.acts[0].cst,
        ],
    )
    ar.add("bessel", re.bessel_fn.bessel_weights.numpy())
    ar.add("radii", re.distance_transform.covalent_radii.numpy())
    ar.add("zbl.radii", zbl.covalent_radii.numpy())
    ar.add("zbl.c", zbl.c.numpy())
    for layer, block in enumerate(model.interactions):
        assert type(block).__name__ == "RealAgnosticResidualNonLinearInteractionBlock"
        prefix = f"layer{layer}"
        for key in ("linear_up", "linear_res", "linear_1", "linear_2", "skip_tp"):
            linear(ar, prefix + "." + key, getattr(block, key))
        ar.add(prefix + ".norm", [block.alpha.item(), block.beta.item()])
        eye = torch.eye(len(model.atomic_numbers), dtype=torch.float64)
        source = block.source_embedding(eye).detach().numpy()
        target = block.target_embedding(eye).detach().numpy()
        radial(ar, prefix + ".radial", block.conv_tp_weights, source, target)
        radial(ar, prefix + ".density", block.density_fn, source, target)
        tp = block.conv_tp
        terms, values = [], []
        in_offsets = np.cumsum([0] + [ir.dim for _, ir in tp.irreps_in1])
        out_offsets = np.cumsum([0] + [ir.dim for _, ir in tp.irreps_out])
        for path, inst in enumerate(tp.instructions):
            assert inst.connection_mode == "uvu" and inst.path_shape == (128, 1)
            l1, l2, lo = (
                tp.irreps_in1[inst.i_in1].ir.l,
                tp.irreps_in2[inst.i_in2].ir.l,
                tp.irreps_out[inst.i_out].ir.l,
            )
            cg = o3.wigner_3j(l1, l2, lo).numpy() * inst.path_weight
            for a, b, c in np.argwhere(np.abs(cg) > 1e-14):
                terms.append(
                    [
                        path,
                        in_offsets[inst.i_in1] + a,
                        l2 * l2 + b,
                        out_offsets[inst.i_out] + c,
                    ]
                )
                values.append(cg[a, b, c])
        ar.add(prefix + ".tp.terms", terms, True)
        ar.add(prefix + ".tp.values", values)
        ar.add(
            prefix + ".tp.dims",
            [len(tp.instructions), int(in_offsets[-1]), int(out_offsets[-1])],
            True,
        )
        polynomial(ar, prefix + ".product", model.products[layer])
        linear(ar, prefix + ".product.linear", model.products[layer].linear)
    linear(ar, "readout0", model.readouts[0].linear, select_out=[head])
    selection = list(range(head * 16, (head + 1) * 16))
    linear(ar, "readout1.up", model.readouts[1].linear_1, select_out=selection)
    linear(
        ar,
        "readout1.down",
        model.readouts[1].linear_2,
        select_in=selection,
        select_out=[head],
    )
    ar.write(args.output)
    manifest = {
        "format": "PWMH1001",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "head": args.head,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dtype": "float32",
        "units": {"energy": "eV", "length": "Angstrom"},
    }
    Path(str(args.output) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--head", default="matpes_r2scan")
    export(p.parse_args())
