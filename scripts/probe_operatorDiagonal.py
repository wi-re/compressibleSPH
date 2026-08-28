"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 8 item 2, 2026-08-27): does
`computeAlpha` actually equal the diagonal of the operator the relaxed-Jacobi
loop iterates?

The loop in `modules/incompressible/incompressible.py` is

    dx_p = dt**2 * computePressureShiftIISPH(computePressureAccelIISPH(p))
    residual = sourceTerm - dx_p
    p <- p + omega * residual / alphas          # alphas = dt**2 * computeAlpha(...)

which is Jacobi on `A p = sourceTerm` with `A p := dx_p`, and `alphas`
standing in for `diag(A)`. Jacobi's stability window scales inversely with
how wrong that stand-in is: if `alphas = c * diag(A)` then the effective
relaxation is `omega / c`. Part 1 measured this solver's window as
`omega < 0.355`, while **both** primary sources report 0.5 as optimal without
qualification (Ihmsen et al. 2014 §3.1.1, "We observed an optimal convergence
for the relaxation factor omega = 0.5 in all settings"; Bender & Koschier 2015
§3.2 uses the same). A diagonal off by ~1.4x would put 0.5 exactly at the
observed edge, so this is worth measuring rather than deriving.

Part 8 predicts a specific discrepancy from expanding Ihmsen Eq. 12 against
`wp_alpha.py`: identical brackets, but the codebase carries **one extra power
of rho** in the prefactor (`1/rho^3` against Ihmsen's `1/rho^2`), invisible at
`rho = rho0 = 1` and worth 30-60% in the wall band where Part 5 measured
`rho` reaching 1.30-1.60. If that is right, `diag(A) / alphas` should come out
proportional to `rho_i`, and the script fits exactly that.

Method: the diagonal is extracted **exactly**, not estimated -- for a sampled
particle `i`, apply the real operator to the unit vector `e_i` and read entry
`i` of the result. One matvec per sampled particle. Particles are sampled
across the density range (not uniformly at random) so the `rho` dependence has
something to fit against.

Usage:
  python scripts/probe_operatorDiagonal.py [--nx 64] [--warmup 80] [--nprobe 48]
  python scripts/probe_operatorDiagonal.py --case randomFlowIncompressible \
      --extra --bounded          # includes kind==1 rows, for the neighbour-set question
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', default='kolmogorovIncompressible',
                    choices=['kolmogorovIncompressible', 'randomFlowIncompressible'])
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--warmup', type=int, default=80,
                    help="steps to run before probing, so the configuration is "
                         "developed rather than a perfect lattice (on a lattice "
                         "rho is uniform, so a rho-dependent error is invisible)")
parser.add_argument('--nprobe', type=int, default=48,
                    help="particles to probe; one matvec each")
parser.add_argument('--extra', nargs='*', default=[],
                    help="extra argv forwarded to the case (e.g. --extra --bounded)")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import torch

from warpSPH.runner.cli import caseMain
from warpSPHCore import SupportScheme
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH
from warpSPH.modules.incompressible.drift import computePressureShiftIISPH

import warpSPH.systems.incompressible as sysmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')

captured = {}
_orig = sysmod.solveIncompressible


def _capturing(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
    """Grab the exact state/adjacency/dt the real solve runs on, then hand off
    to the real solver so the trajectory is unperturbed."""
    if 'state' not in captured:
        captured.update(particles=particles, config=config, schemeConfig=schemeConfig,
                        adjacency=adjacency, dt=dt, state=True)
    return _orig(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)


steps = {'n': 0}
_origStep = None


def run():
    sysmod.solveIncompressible = _capturing
    try:
        caseMain(case, argv=[
            '--nx', str(args.nx), '--nSteps', str(args.warmup), '--tLimit', '1000.0',
            '--quiet', '--no-store', '--no-plot',
        ] + args.extra)
    finally:
        sysmod.solveIncompressible = _orig


# The last capture wins: re-arm each step so we end up with the final step's
# state rather than step 0's lattice.
def _capturingLast(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
    captured.update(particles=particles, config=config, schemeConfig=schemeConfig,
                    adjacency=adjacency, dt=dt, state=True)
    return _orig(particles, config, schemeConfig, adjacency, dvdt, dt, verbose)


_capturing = _capturingLast
run()

particles = captured['particles']
config = captured['config']
schemeConfig = captured['schemeConfig']
adjacency = captured['adjacency']
dt = captured['dt']

apparentArea = particles.masses / particles.densities
alphas = dt ** 2 * computeAlpha(
    currentState=particles, config=config, schemeConfig=schemeConfig,
    adjacency=adjacency, apparentVolumes=apparentArea)


def applyOperator(p):
    ap = computePressureAccelIISPH(state=particles, pressureValues=p, config=config,
                                   supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    return dt ** 2 * computePressureShiftIISPH(state=particles, config=config, pressureAccels=ap,
                                               supportScheme=SupportScheme.Scatter, adjacency=adjacency)


rho = particles.densities
kinds = particles.kinds
n = rho.shape[0]

# Sample across the density range rather than uniformly, so a rho-dependent
# error has spread to show up against.
fluidIdx = torch.nonzero(kinds == 0).flatten()
order = fluidIdx[torch.argsort(rho[fluidIdx])]
pick = order[torch.linspace(0, len(order) - 1, args.nprobe).long()]

rows = []
for idx in pick.tolist():
    e = torch.zeros_like(rho)
    e[idx] = 1.0
    diagTrue = applyOperator(e)[idx].item()
    rows.append((idx, rho[idx].item(), diagTrue, alphas[idx].item()))

print(f"\n=== {args.case} nx={args.nx} after {args.warmup} steps, "
      f"{len(rows)} probed rows of {n} ===")
print(f"dt={dt:.6g}  omega={schemeConfig.solverConfig.pressureSolver.relaxationFactor}")
print(f"\n{'idx':>7} {'rho':>10} {'diag(A)':>13} {'alphas':>13} {'ratio':>9} {'ratio/rho':>10}")
for idx, r, d, a in rows[::max(1, len(rows) // 24)]:
    ratio = d / a if a != 0 else float('nan')
    print(f"{idx:7d} {r:10.5f} {d:13.5e} {a:13.5e} {ratio:9.4f} {ratio / r:10.4f}")

ratios = torch.tensor([d / a for _, _, d, a in rows], dtype=torch.float64)
rhos = torch.tensor([r for _, r, _, _ in rows], dtype=torch.float64)

print(f"\ndiag(A)/alphas : mean={ratios.mean():.5f} std={ratios.std():.5f} "
      f"min={ratios.min():.5f} max={ratios.max():.5f}")
print(f"(diag/alphas)/rho: mean={(ratios / rhos).mean():.5f} std={(ratios / rhos).std():.5f}")
print(f"rho range       : [{rhos.min():.5f}, {rhos.max():.5f}]")

# Which of the two candidate scalings explains the ratio better: a constant, or
# one proportional to rho? Compare the residual spread of each fit.
cBest = ratios.mean()
constResid = (ratios - cBest).abs().mean()
kBest = (ratios / rhos).mean()
rhoResid = (ratios - kBest * rhos).abs().mean()
print(f"\nfit 'ratio = c'      : c={cBest:.5f}  mean|resid|={constResid:.3e}")
print(f"fit 'ratio = k * rho': k={kBest:.5f}  mean|resid|={rhoResid:.3e}")
print(f"-> {'rho-proportional' if rhoResid < constResid else 'constant'} fits better"
      f" ({constResid / max(rhoResid, 1e-300):.2f}x)")

print(f"\nImplied effective relaxation: omega_eff = omega / c = "
      f"{schemeConfig.solverConfig.pressureSolver.relaxationFactor / cBest:.4f} "
      f"(papers report omega=0.5 optimal)")

if bool((kinds != 0).any()):
    bIdx = torch.nonzero(kinds != 0).flatten()[:args.nprobe]
    bRows = []
    for idx in bIdx.tolist():
        e = torch.zeros_like(rho)
        e[idx] = 1.0
        bRows.append((applyOperator(e)[idx].item(), alphas[idx].item()))
    bRatios = torch.tensor([d / a for d, a in bRows if a != 0], dtype=torch.float64)
    print(f"\nboundary rows (kind!=0), {len(bRatios)} probed: "
          f"diag/alphas mean={bRatios.mean():.5f} std={bRatios.std():.5f}")
    print("  (Bender & Koschier §3.2: static boundary particles take no reaction "
          "force, so they belong in alpha's first sum and not its second; "
          "`computeAlpha` sums both over one AllToAll neighbour set.)")
