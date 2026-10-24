#!/usr/bin/env python3
"""Render the saved cluster reports without running GPU workloads."""
import argparse
import json
import os
from pathlib import Path

LABELS = {
    "framework": "Implicit CUDA",
    "explicit": "Explicit CUDA",
    "e3nn": "PyTorch / e3nn",
    "cueq": "cuEquivariance",
    "e3nn_compile": "PyTorch / e3nn + compile",
    "cueq_compile": "cuEquivariance + compile",
}


def main(root):
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(root.glob("*.json")) if p.name.startswith(("replay-", "capacity-"))}
    timing = [(name, d) for name, d in reports.items() if name.startswith("replay-")]
    capacity = [(name, d) for name, d in reports.items() if name.startswith("capacity-")]
    lines = ["# RTX 4090 cluster results", "", "Generated from the adjacent JSON reports. Times are blocking ASE calls including energy, forces, and stress. Capacity runs use one implementation per process.", "", "## Matched replay", "", "| Case | Implementation | Median (ms) | p95 (ms) | Calls |", "| --- | --- | ---: | ---: | ---: |"]
    audit = {}
    for name, d in sorted(timing, key=lambda x: x[1]["atoms"]):
        for implementation in LABELS:
            if implementation in d["results"]:
                r = d["results"][implementation]
                lines.append(f"| {d['atoms']} atoms | {LABELS[implementation]} | {r['wall_median_ms']:.3f} | {r['wall_p95_ms']:.3f} | {len(r['samples_ms'])} |")
        before = d.get("compile_counters_after_warmup", {}).get("stats", {}).get("unique_graphs", 0)
        after = d.get("compile_counters_after_measurement", {}).get("stats", {}).get("unique_graphs", 0)
        audit[name] = {
            "protocol": d["protocol"], "failures": d["failures"],
            "graphs_created_during_measurement": after - before,
            "memory_sampling": d["process_memory_sampling"],
            "trajectory_sha256": d["trajectory_sha256"],
            "source_sha256": d["source_sha256"],
            "environment_lock_sha256": d["environment_lock_sha256"],
        }
    lines += ["", "## Isolated capacity and development timing", "", "These runs use 10 warmups and 50 measured calls (5 windows of 10). They establish capacity and provide development timing; they are not the full v1 timing protocol.", "", "| Atoms | Implementation | Status | Median (ms) | Sampled process peak (GiB) | Tracked context peak (GiB) |", "| ---: | --- | --- | ---: | ---: | ---: |"]
    for name, d in sorted(capacity, key=lambda x: (x[1]["atoms"], list(LABELS).index(x[1]["requested_implementations"][0]))):
        implementation = d["requested_implementations"][0]
        r = d["results"].get(implementation)
        failure = d["failures"].get(implementation)
        peak = d["process_memory_sampling"].get("sampled_peak_bytes")
        peak_text = f"{peak / 2**30:.3f}" if peak is not None else "unavailable"
        context = r.get("diagnostics", {}).get("context_peak_bytes") if r else None
        context_text = f"{context / 2**30:.3f}" if context is not None else "—"
        latency = f"{r['wall_median_ms']:.3f}" if r else "—"
        status = failure["status"] if failure else "completed"
        lines.append(f"| {d['atoms']} | {LABELS[implementation]} | {status} | {latency} | {peak_text} | {context_text} |")
        audit[name] = {"failures": d["failures"], "memory_sampling": d["process_memory_sampling"], "environment_lock_sha256": d["environment_lock_sha256"]}
    lines += ["", "## Numerical qualifications", "", "The strict total-energy comparison uses the float64 official reference. Float32 baseline reports also retain their raw total-energy error and separately compare the sum of returned per-atom energies in float64. The benchmark retains the stock baseline's energy arithmetic.", "", "The 252-atom native repeat test exceeds the strict 1e-5 eV virial repeat target: implicit range 1.1563e-5 eV, explicit range 1.5259e-5 eV over 100 calls. Energy and force ranges meet their strict targets. See `repeat-variability.json` and the correctness reports.", "", "The environment is bare-metal Linux with a pinned CUDA 12 stack. Docker reproduction is pending. Comparisons with the earlier WSL2/3060 Ti campaign include software and host-platform changes.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not timing:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pairwise-report-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, len(timing), figsize=(5 * len(timing), 4.8), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for ax, (_, d) in zip(axes[0], sorted(timing, key=lambda x: x[1]["atoms"])):
        names = [n for n in LABELS if n in d["results"]]
        medians = [d["results"][n]["wall_median_ms"] for n in names]
        bars = ax.barh(range(len(names)), medians, color=colors[:len(names)])
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xlim(0, max(medians) * 1.18)
        ax.set_xlabel("Median blocking latency (ms)")
        ax.set_title(f"{d['atoms']} atoms · {d['protocol']}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "latency.png")
    fig.savefig(root / "latency.svg")
    plt.close(fig)
    if capacity:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, implementation in enumerate(LABELS):
            points = sorted((d["atoms"], d["process_memory_sampling"].get("sampled_peak_bytes")) for _, d in capacity if implementation in d["results"])
            points = [(n, b / 2**30) for n, b in points if b is not None]
            if points:
                x, y = np.asarray(points).T
                ax.plot(x, y, "x--" if implementation == "explicit" else "o-", label=LABELS[implementation], color=colors[i])
        ax.axhline(24564 / 1024, color="black", linestyle="--", label="Reported GPU memory")
        ax.set_xlabel("Atoms (4,096: silicon; larger cases: MPtrj)")
        ax.set_ylabel("Sampled process peak GPU memory (GiB)")
        ax.text(0.02, 0.92, "Both e3nn variants: OOM at every tested size\nBoth cuEquivariance variants: OOM at 12,544+ tested atoms", transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(root / "memory.png")
        fig.savefig(root / "memory.svg")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("benchmarks/rtx4090"))
    main(parser.parse_args().root)
