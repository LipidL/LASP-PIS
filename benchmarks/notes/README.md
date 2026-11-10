# Performance decision notes

One page per experiment. The goal is to preserve the *reasoning* and a
reproduction path, not the raw profiler output.

Rules:

- Copy `TEMPLATE.md` to `PERF-NNNN-short-title.md`.
- Never delete a note. If a later result supersedes it, add
  `superseded-by PERF-NNNN` to the status.
- Link the note from the relevant source comment by note id, not by a long
  path that can move.
- Raw run records, ncu reports, and SASS dumps stay in CI artifacts or release
  assets. If a note needs to survive artifact expiry, attach only a small
  (a few kilobytes) summary next to the note.
- A rejected optimization should remain executable when practical: keep the
  alternative behind a template/policy/flag and add an A/B case, rather than
  deleting the code and relying on prose.
