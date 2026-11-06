# Temporary issue #3367 experiment

This fork evaluates the two u-estimation ideas discussed at
https://github.com/moj-analytical-services/splink/issues/3367.

- `u-performance-arrow`: 64-row Arrow batches for the already sampled records.
  The logical sample also feeds the two-input link-only split, so both linkage
  paths benefit. Registered scans and materialised samples are cleaned on error.
- `u-performance-aggregation`: count gamma values directly, without unused EM
  probability arithmetic or the discarded global-prior row.
- `u-performance-combined`: both changes and seeded equivalence tests.

The experiment preserves pair selection, null handling, fixed probabilities,
chunking and early stopping. Tests compare per-chunk counts and populated model
probabilities against the v5.0.0 tag. Three SQL logging checks retain their usual
sample naming contract.

`benchmark.py` accepts an untrained Linker and a directory containing
`control.py`, `arrow.py`, `aggregation.py`, and `combined.py` extracted from the
corresponding pinned Git revisions. It runs each twice in reversed order, measures
wall/CPU/RSS, and asserts identical count and u-probability fingerprints.
It requires no changes to normal application dependency checks.

Real-data notebooks and results stay outside this public repository. The fork is
for temporary benchmarking, not a released dependency or an upstream submission.
