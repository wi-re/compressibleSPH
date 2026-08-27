"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 4, 2026-08-27): end-to-end A/B of
`ShiftPressureGauge` through the **real** `solveIncompressible`, as opposed
to `probe_incompressibleGaugeDrift.py`'s standalone reimplementation of its
loop (which is where the gauge was prototyped and compared against the
alternatives).

`nonNegativeClamp` is the historical `clamp(p, min=0)`: a floor, so nothing
pins the constant null-space mode of an operator whose source term
(`rho0 - rhoStar`) carries a mean no pressure field can remove -- the SPH
summation density's particle average is bounded below by its lattice value
and rises quadratically with disorder (`probe_densityBiasVsDisorder.py`), so
`mean_i rho_i == rho0` is simply unattainable. `minShift` subtracts the fluid
minimum instead: still non-negative, but gauge-fixed, and it translates the
field's negative part rather than discarding it.

Runs both gauges on a case and reports the density-error diagnostics the
case itself already emits, plus whether the run stayed finite. Both cases
this matters for are covered:

  --case kolmogorovIncompressible   periodic, no wall, no free surface
  --case randomFlowIncompressible   mDBC walls (Part 2's case) -- the one
                                    where a gauge that shifts fluid
                                    pressures but not the frozen boundary
                                    rows could plausibly introduce a
                                    spurious fluid/boundary pressure jump

Usage: `python scripts/probe_shiftPressureGauge.py [--case ...] [--nx 128]
[--nsteps 1000] [--gauges nonNegativeClamp minShift]`
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', default='kolmogorovIncompressible',
                    choices=['kolmogorovIncompressible', 'randomFlowIncompressible'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--nsteps', type=int, default=1000)
parser.add_argument('--tLimit', type=float, default=1000.0)
parser.add_argument('--gauges', nargs='*', default=['nonNegativeClamp', 'minShift'])
parser.add_argument('--extra', nargs='*', default=[],
                    help="extra argv forwarded to the case (e.g. --extra --bounded)")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

from warpSPH.runner.cli import caseMain
from warpSPH.configurations import ShiftPressureGauge

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

results = {}
for gaugeName in args.gauges:
    gauge = ShiftPressureGauge[gaugeName]
    _orig = case.configureScheme

    def _wrapped(ctx, _orig=_orig, gauge=gauge):
        _orig(ctx)
        ctx.schemeConfig.solverConfig.shiftPressureGauge = gauge

    case.configureScheme = _wrapped
    try:
        result = caseMain(case, argv=[
            '--nx', str(args.nx), '--nSteps', str(args.nsteps),
            '--tLimit', str(args.tLimit), '--quiet', '--no-store', '--no-plot',
        ] + args.extra)
    finally:
        case.configureScheme = _orig
    results[gaugeName] = result

print(f"\n=== {args.case} nx={args.nx} nSteps={args.nsteps} ===")
# `densityStd` is `kolmogorovIncompressible`'s own extra diagnostic;
# `randomFlowIncompressible` reports the shared weakly-compressible set, which
# has the density bounds but not the std.
print(f"{'gauge':>18} {'steps':>7} {'diverged':>9} {'minRho':>10} {'maxRho':>10} "
      f"{'|rho-1| (2nd half)':>19} {'maxRho-1 (worst)':>17} {'t_final':>9} {'wall s':>8}")
for name, r in results.items():
    tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
    tail = tr[len(tr) // 2:] or tr
    band = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0)) for row in tail]
    bandMean = sum(band) / max(1, len(band))
    worst = max(max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0))
                for row in tr) if tr else float('nan')
    print(f"{name:>18} {len(tr):7d} {str(r.diverged):>9} "
          f"{min(row['minDensity'] for row in tr):10.5f} "
          f"{max(row['maxDensity'] for row in tr):10.5f} "
          f"{bandMean:19.4e} {worst:17.4e} "
          f"{(tr[-1].get('t', float('nan')) if tr else float('nan')):9.4f} {r.wallTime:8.1f}")
    if len(tr) < len(r.trajectory):
        print(f"{'':>18} (non-finite from step {len(tr)} of {len(r.trajectory)})")
