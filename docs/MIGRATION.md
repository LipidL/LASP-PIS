# Framework extraction provenance

The MACE history is imported under `mace/`, with the framework relocated into
`lapis/` in a separate commit. Original authors, messages and timestamps are
retained; path rewriting changes commit hashes. The original-to-imported MACE
commit map is in `docs/history/mace-commit-map.txt`.

The imported MACE development branch is retained separately from the selected
checkout revision when they differ. Future extraction of Lapis history must include
its old paths under `mace/src`, `mace/include/pairwise`, `mace/tests`, and the original
framework documentation; path filtering does not automatically follow moves.

Unreleased names now follow ownership:

| Before | After |
|---|---|
| `pairwise::Context`, `Calculator`, `execute` | `lapis::Context`, `Calculator`, `execute` |
| `pairwise::MH1` | `mace::MH1` |
| `libpairwise.so`, `pw_*` | `libmace.so`, `mace_*` |
| Python `pairwise.PairwiseCalculator` | `mace_inference.MH1Calculator` |
| `PAIRWISE_LIBRARY` | `MACE_LIBRARY` |

Lennard-Jones is a Lapis example/test fixture and is no longer exposed by MACE.
Model archive formats and numerical kernels are unchanged. Model-specific
cuBLAS dependencies remain in the consumers. The optional explicit-connectivity
runtime is owned and packaged by Lapis; consumers receive its template compile
definition through `lapis::explicit`.

Historical validation reports, logs and lock files remain evidence of the original
implementations and retain their original paths, identifiers and environment data.
