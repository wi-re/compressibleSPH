"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 21, 2026-08-29): is the dam break's
over-dissipation (Part 19) the free-surface compaction §1.10 describes, or
something else that happens at impact?

§1.10 found the free-surface density deficit vanishes while the surface stays
geometrically a surface, and called that "the constant-density solve
compacting the surface layer to reach a setpoint the geometry forbids" --
plausible, but explicitly not established as the cause of the 88% kinetic
energy loss Part 19 measures at the same moment.

`solveIncompressible` already has a hook for exactly this: `ShiftPressureGauge
.minShift` (the shipped default) downgrades to the historical
`nonNegativeClamp` on any solve with free-surface particles present (§1.5),
and this is not a workaround this codebase invented -- it is [BK]'s own
published remedy for free-surface clustering (bender2015, discussion section:
"In our implementation this problem is solved by clamping negative pressures
to zero. However, a better solution would be to introduce ghost particles as
suggested by Schechter and Bridson [2012]"). So the shipped configuration is
already running the literature's answer, not omitting it -- the commented-out
`pressureB[surfaceIndicators == 1] = 0.0` line in `divergenceFree.py` is a
third, unpublished, more aggressive intervention (it zeros the *pressure*
outright rather than merely its negative part, and it sits in the
divergence-free solve, not the constant-density one that actually drives the
compaction), not a documented next step.

The clean, code-free A/B is `forceShiftPressureGauge` (`solver.py:513`),
explicitly named there as "untested" for exactly the free-surface half of the
guard. Forcing it keeps `minShift` at the surface instead of falling back to
the clamp -- i.e. it removes the mechanism §1.10 suspects, without touching
any kernel. If the KE trajectory and front speed are unchanged, the
free-surface gauge fallback is not the driver of the dissipation and the
search moves to the impact itself; if they change, the surface treatment is a
live contributor and worth pursuing (with the ghost-particle route (bender2015
+ schechter2012) as the literature-sanctioned next step, not the commented-out
line).

Usage:
  python scripts/probe_dambreakSurfaceGauge.py --nx 64 --tLimit 1.0
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--tLimit', type=float, default=1.0)
parser.add_argument('--cflFactor', type=float, default=0.2,
                    help="Part 20: the published 0.4 diverges on this case; 0.2 is safe")
parser.add_argument('--samples', type=int, default=15)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import math

from warpSPH.runner.cli import caseMain
from warpSPH.cases.dambreak import dambreakCase as case

_origConfigureScheme = case.configureScheme
_origDiag = case.diagnostics


def diagnostics(ctx, state):
    out = dict(_origDiag(ctx, state))
    p = state.state
    fluid = p.kinds == 0
    pos = p.positions[fluid]
    rho = p.densities[fluid]
    out.update(
        front=pos[:, 0].max().detach().cpu().item(),
        nLow=int((rho < 0.9).sum().detach().cpu().item()),
        nFluid=int(fluid.sum().detach().cpu().item()),
    )
    return out


def runVariant(forceGauge: bool):
    def configureScheme(ctx, _orig=_origConfigureScheme, forceGauge=forceGauge):
        _orig(ctx)
        ctx.schemeConfig.solverConfig.forceShiftPressureGauge = forceGauge

    case.configureScheme = configureScheme
    case.diagnostics = diagnostics
    try:
        return caseMain(case, argv=[
            '--nx', str(args.nx), '--tLimit', str(args.tLimit),
            '--scheme', 'divergenceFree', '--integrationScheme', 'semiImplicitEuler',
            '--cflFactor', str(args.cflFactor), '--quiet', '--no-store', '--no-plot',
        ])
    finally:
        case.configureScheme = _origConfigureScheme
        case.diagnostics = _origDiag


results = {}
for label, forceGauge in [('shipped (clamp at surface)', False),
                          ('forced minShift at surface', True)]:
    r = runVariant(forceGauge)
    results[label] = r
    tr = [row for row in r.trajectory
          if all(not isinstance(v, float) or math.isfinite(v) for v in row.values())]
    print(f"\n=== dambreak --scheme divergenceFree, {label} "
          f"({len(tr) - 1} steps, {r.wallTime:.0f}s, diverged={r.diverged}) ===")
    hdr = (f"{'t':>7} {'KE':>9} {'front':>8} {'nLow':>7} {'nLow/N':>8} "
           f"{'rhoMin':>9} {'rhoMax':>9}")
    print(hdr)
    print('-' * len(hdr))
    step = max(1, (len(tr) - 1) // args.samples)
    for row in tr[::step] + [tr[-1]]:
        print(f"{row['t']:7.3f} {row['kineticEnergy']:9.4f} {row['front']:8.4f} "
              f"{row['nLow']:7d} {row['nLow'] / row['nFluid']:8.4f} "
              f"{row['minDensity']:9.5f} {row['maxDensity']:9.5f}", flush=True)

print("\n=== side by side (KE) ===")
labels = list(results)
hdr = f"{'t':>7}" + ''.join(f"{lbl[:24]:>26}" for lbl in labels)
print(hdr)
series = {}
for lbl, r in results.items():
    tr = [row for row in r.trajectory
          if all(not isinstance(v, float) or math.isfinite(v) for v in row.values())]
    series[lbl] = tr
n = min(len(v) for v in series.values())
step = max(1, n // args.samples)
for i in range(0, n, step):
    line = f"{series[labels[0]][i]['t']:7.3f}"
    for lbl in labels:
        line += f"{series[lbl][i]['kineticEnergy']:26.4f}"
    print(line)
