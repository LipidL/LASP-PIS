# PERF-NNNN: short title

Status: accepted | rejected | superseded-by PERF-NNNN
Date: YYYY-MM-DD
Commit: <sha or branch>
Hardware/software: <GPU, SM, CUDA, driver, compiler flags>

## Hypothesis

One or two sentences. What did you expect to get faster or more stable, and why?

## Change

The smallest description that lets someone reproduce the patch or switch. Name
the template parameter, layout, launch shape, or algorithm.

## Measurement

Fixed fixture/case, warmup, samples, clocks (locked or not), interleaving, and
statistic. Link the run record or artifact by content hash.

## Result

| Variant | Metric | Before | After | Delta |
| --- | --- | ---: | ---: | ---: |
| ... | median ms | ... | ... | ... |

Include correctness deltas. A speedup with a failed output is not a result.

## Decision

What was kept, what was rejected, and under which dimensions (architecture,
`Common` vs generic, compiler version). Explain the mechanism, not just the
number.

## Reproduction

Exact command, including how to select the rejected variant. If the variant is
kept in the code behind a policy/template, show both sides of the A/B.

## Artifacts

- Run record: `<sha256 or release asset>`
- Raw profiler output: `<CI artifact / external location>`
- SASS/PTX excerpt: `<optional, small>`
