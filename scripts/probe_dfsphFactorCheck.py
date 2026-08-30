"""Verify the new DFSPH factor kernel (`wp_dfsph_factor.computeDFSPHFactor`)
against the known-correct IISPH diagonal (`wp_alpha.computeAlpha`), and the
composed pressure acceleration `computePressureAccelIISPH` against a direct
O(N^2) torch reference of the standard SPH pressure acceleration.

Check 1 (the factor): in the bulk (every neighbour a fluid particle, rho ~ 1,
equal masses) the two are algebraically identical, so they must agree to float
precision. At the wall (the boundary carries an Akinci apparent volume and
rho != 1) they diverge -- that divergence is the whole point of step 2, so we
report it too.

Check 2 (the a_p): `computePressureAccelIISPH` is `-warpOperation(Symmetric
gradient)/rho`, i.e. `-sum_j m_j (k_i/rho_i^2 + k_j/rho_j^2) gradW_ij` -- the
standard SPH pressure acceleration. We verify that empirically against a direct
torch double-loop over all in-support pairs (Wendland2, 2D:
  W(q) = (1-q)^4 (1+4q), dW/dq = -20 q (1-q)^3, C_2 = 7/pi,
  gradW = (C_2/h^3) (dW/dq)(q) (x/|x|),  q = |x|/h, support |x| <= h).

Both are computed *inside* the runner at the first CD `_jacobiSolve` -- inside
the `applyConsistentCoupling` context, so boundary masses are already the
Akinci values -- and the results are cloned out, so the live state being
advanced afterwards does not matter.

Usage: python scripts/probe_dfsphFactorCheck.py [--nx 32]
"""
import argparse

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=32)
args = args.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import math
import torch
from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run
import warpSPH.schemes.dfsphReference as ref
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.incompressible.wp_dfsph_factor import computeDFSPHFactor
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH
from warpSPHCore.enumTypes import SupportScheme

captured = {}
_orig_js = ref._jacobiSolve


def _spy(state, config, schemeConfig, adjacency, **kw):
    if 'aii' not in captured:
        apparent = state.masses / state.densities
        # Check 1 inputs: both factors at this (CD) state.
        captured['aii'] = computeAlpha(
            state, config, schemeConfig, adjacency,
            apparentVolumes=apparent, includeBoundaryReaction=False).clone()
        captured['diag'] = computeDFSPHFactor(
            state, config, schemeConfig, adjacency,
            apparentVolumes=apparent).clone()
        # Check 2 inputs: the a_p for a fixed linear test field p = y
        # (hydrostatic-like; a_p is linear in p so any field tests the
        # operator). Computed inside the context so boundary masses/densities
        # are the Akinci values.
        ptest = state.positions[:, 1]
        captured['apWarp'] = computePressureAccelIISPH(
            state, ptest, config,
            supportScheme=SupportScheme.Scatter, adjacency=adjacency).clone()
        # Shared tensors for the torch reference.
        captured['pos'] = state.positions.clone()
        captured['masses'] = state.masses.clone()
        captured['densities'] = state.densities.clone()
        captured['support'] = state.supports.clone()
        captured['kinds'] = state.kinds.clone()
        captured['ptest'] = ptest.clone()
        captured['dx'] = config.dx
    return _orig_js(state, config, schemeConfig, adjacency, **kw)


ref._jacobiSolve = _spy
try:
    run(hydrostaticColumn.hydrostaticColumnCase, nx=args.nx, nSteps=1,
        scheme='dfsphReference', quiet=True, plot=False, store=False,
        progress=False, integrationScheme='semiImplicitEuler')
finally:
    ref._jacobiSolve = _orig_js

print(f'nx={args.nx}  n={int(captured["kinds"].numel())}')
print()

# ---- Check 1: factor vs IISPH diagonal -------------------------------------
aii = captured['aii']
diag = captured['diag']
pos = captured['pos']
kinds = captured['kinds']
dx = captured['dx']

fluid = kinds == 0
ylo = pos[fluid, 1].min()
surfaceY = pos[fluid, 1].max()
wall = fluid & (pos[:, 1] < ylo + 3 * dx)
bulk = fluid & (pos[:, 1] < surfaceY - 3 * dx) & (pos[:, 1] > ylo + 3 * dx)

iisphPos = -aii   # computeAlpha returns negated; make both positive
dfsphPos = diag

print('Check 1: DFSPH factor vs IISPH diagonal')


def rel(mask, tag):
    if not bool(mask.any()):
        print(f'  {tag}: (no particles)')
        return
    a = iisphPos[mask].double()
    b = dfsphPos[mask].double()
    relerr = (a - b).abs() / b.abs().clamp(min=1e-12)
    print(f'  {tag:6s} n={int(mask.sum()):6d}  '
          f'mean|IISPH-DFSPH|/DFSPH={relerr.mean():.3e}  '
          f'max={relerr.max():.3e}   '
          f'IISPH mean={a.mean():.5g}  DFSPH mean={b.mean():.5g}')


rel(bulk, 'bulk')
rel(wall, 'wall')
print()

# ---- Check 2: a_p vs direct O(N^2) torch reference -------------------------
print('Check 2: computePressureAccelIISPH vs O(N^2) torch reference')

pos = captured['pos'].to('cuda').double()
masses = captured['masses'].to('cuda').double()
densities = captured['densities'].to('cuda').double()
support = captured['support'].to('cuda').double()
kappa = captured['ptest'].to('cuda').double()
kinds = captured['kinds'].to('cuda')
fm = kinds == 0

# All in-support pairs (i, j): 0 < r_ij <= h_i (Scatter => reference h_i), i!=j.
xij = pos[:, None, :] - pos[None, :, :]          # (N, N, 2): x_i - x_j
r = xij.norm(dim=-1)                             # (N, N)
h = support.clamp(min=1e-12)[:, None]            # (N, 1), guard h=0 rows
# The a_p sums over fluid + boundary neighbours; ghosts (kind=2) are excluded
# by the warp kernel (AllToAll directionality), so exclude them here too.
mask = (r > 1e-12) & (r <= h + 1e-12)
mask = mask & (kinds[None, :] != 2)              # drop ghost neighbours j

C2 = torch.tensor(7.0 / math.pi)                 # 2D Wendland2 constant
h3 = h ** 3                                      # (N, 1)
q = torch.where(mask, r / h.clamp(min=1e-12), torch.zeros_like(r))
dWdq = -20.0 * q * (1.0 - q) ** 3                # (N, N)
invr = torch.where(mask, 1.0 / r.clamp(min=1e-12), torch.zeros_like(r))
gradW = (C2 / h3) * dWdq * invr                  # (N, N) scalar factor
gradW = torch.where(mask, gradW, torch.zeros_like(gradW))
gradWvec = gradW[:, :, None] * xij               # (N, N, 2)

# a_p_i = -sum_j m_j (k_i/rho_i^2 + k_j/rho_j^2) gradW_ij. Clamp densities so
# ghost (kind=2) rows with rho=0 stay finite (they are excluded from the sum
# anyway); fluid (rho>0.6) and boundary (rho=1) are untouched, so the a_p for
# compared particles is unchanged.
kiRi2 = (kappa / densities.clamp(min=1e-12) ** 2)   # (N,)
coef = (kiRi2[:, None, None] + kiRi2[None, :, None])   # (N, N, 1)
term = coef * gradWvec                             # (N, N, 2)
apRef = -(term * masses[None, :, None]).sum(dim=1)    # (N, 2)

apW = captured['apWarp'].to('cuda').double()
valid = fm & torch.isfinite(apRef).all(dim=1)
diff = (apW - apRef).norm(dim=1)[valid]
refMag = apRef.norm(dim=1)[valid].clamp(min=1e-12)
relerr = diff / refMag
print(f'  fluid n={int(valid.sum())}/{int(fm.sum())}  '
      f'mean rel|a_p^warp - a_p^ref|/|a_p^ref|={relerr.mean():.3e}  '
      f'max={relerr.max():.3e}')
print(f'  |a_p|: warp mean={apW[valid].norm(dim=1).mean():.5g}  '
      f'ref mean={apRef[valid].norm(dim=1).mean():.5g}')
