# Release baselines

A baseline is a small, reviewed snapshot of accepted medians for one hardware
and software environment class. It is not a history of every run.

- Create or replace one only at a release or after a deliberate performance
  change, using `run.py --baseline-out ...`.
- Keep the filename tied to the environment class, for example
  `sm86-rtx3060ti-cuda12.6.json`.
- Do not compare absolute values across different GPUs, drivers, CUDA versions,
  or clock policies.
- `compare.py` reads these files and applies a median-ratio threshold. It is a
  starting point; paired samples and bootstrap intervals are the next step when
  the gate needs to be stricter than run-to-run noise.

The included `sm86-rtx3060ti-cuda12.6.json` is an example snapshot from the
development RTX 3060 Ti with unlocked clocks and a display attached. Regenerate
it after committing on the target machine; do not treat its absolute values as a
hardware guarantee.
