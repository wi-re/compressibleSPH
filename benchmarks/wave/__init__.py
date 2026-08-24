"""Benchmark suite for the registered `waveEquation` case.

Three CLIs, all built on `benchmarks.common`:

* `bench_accuracy.py`    -- temporal error and measured convergence order of
  explicit and implicit integrators against a converged-RK4 reference, at a
  fixed resolution, across a `dt` refinement grid;
* `bench_performance.py` -- runtime (ms/step, ms/RHS-evaluation) and memory
  (peak allocated/reserved, static state footprint) vs. particle count;
* `bench_stability.py`   -- the notebook's "payoff" experiment, systematized:
  `dt` pushed past the explicit stability limit, and the internal-solver
  loop-limit matrix (Picard count, JFNK matvec/tol/max_iterations) mapped
  for the implicit schemes.

`examples/wave/waveCase_implicit_vs_explicit.ipynb` is the hand-worked
version of exactly these measurements; this suite is what makes them repeat
across schemes, resolutions, and solver settings.
"""
