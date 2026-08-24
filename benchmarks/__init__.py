"""Benchmark suites for the warpSPH frontend.

`benchmarks/wave/` is the first suite: it benchmarks the explicit and
implicit time integrators (warpSPHIntegrators) on the registered
`waveEquation` case, the same case `examples/wave/
waveCase_implicit_vs_explicit.ipynb` explores by hand, against both
computational cost (runtime vs. particle count, memory) and numerical
accuracy (error, convergence order, energy drift, stability limit).

Each suite is a standalone CLI script (run from the repo root, in the
`warp` conda env) that writes a timestamped `results/` directory with
`results.json`, `summary.md` and PNG plots. See `benchmarks/README.md`.
"""
