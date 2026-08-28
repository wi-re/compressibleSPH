"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 12, 2026-08-28): is the advective
CFL condition this scheme applies the published one, and is the published one
strict enough for a *bounded* case?

Bender & Koschier state `dt <= 0.4 * d / |v_max|` with `d` the particle
**diameter**. `kolmogorovIncompressibleTimestep` used to apply `cflFactor` to
the support radius `h = n_h * dx` instead, so the shipped `cflFactor=0.3`
allowed `1.2 dx` of travel per step -- 3x the published limit -- and the same
`cflFactor` meant different physics at different `n_h`. It now multiplies `dx`,
so the number in the config is the published constant.

  --mode verify  Per step: `dt`, `|v_max|`, and the dimensionless travel
      `dt |v_max| / dx`, which must equal `cflFactor` exactly whenever the
      advective term is the binding one. This is the check that the
      implemented condition *is* the published condition.

  --mode sweep   The second question. The 0.4 constant was validated on
      free-surface scenes, where a large fraction of particles sit below rest
      density because their support is truncated; an average density error
      over such a scene is diluted by them. A bounded case has complete
      support everywhere, so nothing dilutes it, and the same `dt` may be far
      more permissive than the constant's calibration implies. The sweep runs
      each `cflFactor` to the *same simulated time* (not the same step count,
      which would not be comparable) and reports what accuracy each buys and
      what it costs, alongside the sub-rest-density fraction that makes the
      dilution argument concrete.

Usage:
  python scripts/probe_cflCondition.py --mode verify
  python scripts/probe_cflCondition.py --mode sweep --tLimit 3.0
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='verify', choices=['verify', 'sweep'])
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--steps', type=int, default=40, help="--mode verify")
parser.add_argument('--tLimit', type=float, default=3.0, help="--mode sweep")
parser.add_argument('--cfls', nargs='*', type=float, default=[0.4, 0.2, 0.1, 0.05])
parser.add_argument('--every', type=int, default=10)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

import torch

from warpSPH.runner.cli import caseMain
import warpSPH.systems.incompressible as sysmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')


def runVerify():
    rows = []
    _real = sysmod.solveIncompressible

    def _watch(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
        vMax = float(particles.velocities.norm(dim=1).max())
        rows.append((dt, vMax, config.dx, float(particles.supports.mean())))
        return _real(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)

    sysmod.solveIncompressible = _watch
    try:
        r = caseMain(case, argv=['--nx', str(args.nx), '--nSteps', str(args.steps),
                                 '--tLimit', '1000.0', '--quiet', '--no-store',
                                 '--no-plot'] + args.extra)
    finally:
        sysmod.solveIncompressible = _real

    cfl = r.ctx.config.cflFactor
    dx = rows[0][2]
    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} ===")
    print(f"cflFactor = {cfl}   dx = {dx:.6g}   h = {rows[0][3]:.6g} = {rows[0][3]/dx:.3f} dx")
    print(f"\npublished condition: dt <= {cfl} * d / |v_max|, d = particle diameter = dx")
    print(f"{'step':>6} {'dt':>12} {'|v_max|':>10} {'dt|v|/dx':>10} {'binding':>12}")
    for i, (dt, vMax, dx_, _h) in enumerate(rows):
        if i % max(1, len(rows) // 12) and i != len(rows) - 1:
            continue
        travel = dt * vMax / dx_
        print(f"{i:6d} {dt:12.6g} {vMax:10.5f} {travel:10.5f} "
              f"{'advective' if abs(travel - cfl) < 2e-2 * cfl else 'viscous/cap':>12}")
    travels = [dt * v / d for dt, v, d, _ in rows]
    # `dt` is computed from the previous step's `|v_max|`, so the ratio
    # measured against the current one wobbles by a fraction of a percent even
    # when the advective term is the binding one.
    binding = [t for t in travels if abs(t - cfl) < 2e-2 * cfl]
    print(f"\n{len(binding)}/{len(travels)} steps advection-limited; on those, "
          f"dt|v_max|/dx = {sum(binding)/max(1,len(binding)):.4f} against the "
          f"configured {cfl}.")
    print("Particles travel at most that fraction of a spacing per step, which is")
    print("the published statement. Before this was fixed the same number meant "
          f"{rows[0][3]/dx:.0f}x more.")


def runSweep():
    # Rows print as they finish rather than at the end: a sweep at small
    # `cflFactor` is a long run, and a partial table is worth more than a
    # complete one that a killed process never gets to print.
    def mean(xs):
        xs = [x for x in xs if math.isfinite(x)]
        return sum(xs) / len(xs) if xs else float('nan')

    def half(xs):
        return xs[len(xs) // 2:] or xs

    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx}, "
          f"all runs to t={args.tLimit} ===", flush=True)
    print(f"{'cflFactor':>10} {'steps':>7} {'reached t':>10} {'maxRho':>9} {'minRho':>9} "
          f"{'mean|rho-1|':>12} {'mean(rho-1)':>12} {'frac rho<rho0':>14} {'wall s':>8}",
          flush=True)

    def emit(cfl, r, st):
        tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
        if not tr:
            print(f"{cfl:10.4g}      -  diverged immediately", flush=True)
            return
        print(f"{cfl:10.4g} {len(tr):7d} {tr[-1].get('t', float('nan')):10.4f} "
              f"{max(row['maxDensity'] for row in tr):9.5f} "
              f"{min(row['minDensity'] for row in tr):9.5f} "
              f"{mean(half(st['abs'])):12.4e} {mean(half(st['signed'])):12.4e} "
              f"{mean(half(st['below'])):13.1%} {r.wallTime:8.1f}", flush=True)

    out = []
    for cfl in args.cfls:
        stats = {'signed': [], 'abs': [], 'below': []}
        step = {'n': 0}
        _real = sysmod.solveIncompressible

        def _watch(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
            step['n'] += 1
            if step['n'] % args.every == 0:
                fluid = particles.kinds == 0
                rho = particles.densities[fluid]
                stats['signed'].append(float((rho - 1.0).mean()))
                stats['abs'].append(float((rho - 1.0).abs().mean()))
                stats['below'].append(float((rho < 1.0).to(rho.dtype).mean()))
            return _real(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)

        sysmod.solveIncompressible = _watch
        try:
            # **Do not pass `--nSteps` here.** `runner.py:246` sets
            # `timeLimited = spec.nSteps is None and case.timestep is not None`,
            # so supplying a step count silently turns `--tLimit` off and runs
            # the full count instead -- which would make every row a
            # *step*-matched comparison, the exact thing this mode exists not to
            # do. The loop breaks on NaN (`runner.py:313`), so an uncapped
            # time-limited run still terminates on a divergent configuration.
            r = caseMain(case, argv=['--nx', str(args.nx),
                                     '--tLimit', str(args.tLimit), '--cflFactor', str(cfl),
                                     '--quiet', '--no-store', '--no-plot'] + args.extra)
        finally:
            sysmod.solveIncompressible = _real
        out.append((cfl, r, stats))
        emit(cfl, r, stats)

    print("\n`frac rho<rho0` is the dilution term: in a free-surface scene it is large")
    print("(truncated support pulls those particles below rest density) and drags any")
    print("average density error down with it. A bounded case has no such population,")
    print("so the same dt is graded on a stricter scale than the constant was tuned on.")


runVerify() if args.mode == 'verify' else runSweep()
