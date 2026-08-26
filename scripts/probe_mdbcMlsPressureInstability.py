"""Probe (`DFSPH_IMPROVEMENT_PLAN.md`'s "New" open question, 2026-08-26):
`BoundaryPressureMode.mdbcMlsPressure` is now actually live (the boundary-row
masking bug that silently zeroed it was fixed this session), but on
`randomFlowIncompressibleCase --bounded` at nx=128 it NaNs by step 7, where
`plain`/`mdbcDensity` stay bounded over the same steps. This script drives the
same case manually (not through `runner.run`, which only exposes the
diagnostics dict) so it can inspect `computeMdbcPressure`'s own internal
quantities -- `numNeighbors`, the raw MLS fit (`A_g`, `b`), the projected
pressure `p_proj` and its Taylor gradient term -- per boundary particle, on
the exact steps leading up to the blowup, to find out *which* boundary
particles run away and why (a low-neighbor-count corner point taking the
Shepard fallback, a well-sampled point with a noisy/large pressure gradient,
or something else).

Usage: `python scripts/probe_mdbcMlsPressureInstability.py [--nx 128]
[--nsteps 8] [--tlimit None]`
"""

from __future__ import annotations

import argparse

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.cases.randomFlowIncompressible import randomFlowIncompressibleCase
from warpSPH.runner.runner import buildContext
from warpSPH.runner.caseSpec import CaseSpec
from warpSPH.modules.liu import interpolateLiuLiu
from warpSPHCore import AdjacencyList, OperationDirection


def inspectMdbcPressure(state, config, adjacency, tag):
    """Re-run `computeMdbcPressure`'s own math (not the production function
    itself -- that one returns a merged tensor, not the intermediate
    diagnostics) to see numNeighbors/A_g/b/p_proj per boundary particle."""
    p_interp, p_interp_grad, numNeighbors, A_g, b = interpolateLiuLiu(
        state.positions[state.kinds == 2],
        referenceParticles=state,
        referenceQuantities=state.pressures,
        config=config,
        neighbor_threshold=4,
        direction=OperationDirection.FluidToGhost,
        supportScale=1.0,
        adjacency=adjacency.hashMap if isinstance(adjacency, AdjacencyList) else None,
    )
    ghostMask = state.kinds == 2
    bIndices = state.ghostIndices[ghostMask]
    relPos = -state.ghostOffsets[ghostMask]
    dP = -torch.einsum('nu, nu -> n', relPos, p_interp_grad)
    p_proj = p_interp + dP
    gradNorm = p_interp_grad.norm(dim=-1)

    print(f'  [{tag}] computeMdbcPressure internals over {bIndices.numel()} boundary particles:')
    print(f'    numNeighbors: min={numNeighbors.min().item():.1f} max={numNeighbors.max().item():.1f} '
         f'mean={numNeighbors.float().mean().item():.2f}')
    print(f'    p_interp: min={p_interp.min().item():.4g} max={p_interp.max().item():.4g}')
    print(f'    |grad p|: min={gradNorm.min().item():.4g} max={gradNorm.max().item():.4g} '
         f'mean={gradNorm.mean().item():.4g}')
    print(f'    p_proj (final, before fallback ladder): min={p_proj.min().item():.4g} '
         f'max={p_proj.max().item():.4g}')

    worst = torch.argmax(p_proj.abs())
    print(f'    worst |p_proj| particle: idx_in_ghost={worst.item()} p_proj={p_proj[worst].item():.4g} '
         f'numNeighbors={numNeighbors[worst].item():.0f} |grad|={gradNorm[worst].item():.4g} '
         f'relPos={relPos[worst].cpu().tolist()}')
    return dict(numNeighborsMin=numNeighbors.min().item(), pProjAbsMax=p_proj.abs().max().item(),
               gradNormMax=gradNorm.max().item())


def run(nx, nSteps, tag='mdbcMlsPressure'):
    spec = CaseSpec(caseName=randomFlowIncompressibleCase.name,
                    scheme=randomFlowIncompressibleCase.scheme,
                    params=dict(randomFlowIncompressibleCase.params))
    spec = spec.merged(**randomFlowIncompressibleCase.defaults)
    spec = spec.merged(nx=nx, nSteps=nSteps, store=False, plot=False, quiet=True,
                       progress=False, params={'bounded': True, 'boundaryPressureMode': tag})

    ctx = buildContext(randomFlowIncompressibleCase, spec)
    randomFlowIncompressibleCase.configureScheme(ctx)
    system = randomFlowIncompressibleCase.buildSystem(ctx)
    randomFlowIncompressibleCase.initialConditions(ctx, system)
    runningState = system.initializeNewState()

    nBoundary = int((runningState.state.kinds == 1).sum().item())
    nGhost = int((runningState.state.kinds == 2).sum().item())
    nFluid = int((runningState.state.kinds == 0).sum().item())
    print(f'nx={nx} nFluid={nFluid} nBoundary={nBoundary} nGhost={nGhost}')

    for i in range(1, nSteps + 1):
        stepResult = ctx.integrator.function(
            state=runningState, f=ctx.stepFunction, dt=ctx.config.dt,
            config=ctx.config, verbose=False, schemeConfig=ctx.schemeConfig)
        runningState = stepResult.state
        if randomFlowIncompressibleCase.timestep is not None:
            ctx.config.dt = randomFlowIncompressibleCase.timestep(ctx, runningState)

        s = runningState.state
        rho = s.densities
        v = s.velocities
        bMask = s.kinds == 1
        pB = s.pressures[bMask]
        print(f'step {i}: t={runningState.t:.5f} dt={ctx.config.dt:.4e} '
             f'rho=[{rho.min().item():.4g},{rho.max().item():.4g}] '
             f'vMax={v.norm(dim=-1).max().item():.4g} '
             f'boundaryPressure=[{pB.min().item():.4g},{pB.max().item():.4g}]')

        if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
            print(f'  DIVERGED at step {i}')
            break

        if i >= nSteps - 3 or not torch.isfinite(rho).all():
            inspectMdbcPressure(s, ctx.config, runningState.adjacency, tag=f'step{i}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--nsteps', type=int, default=8)
    args = p.parse_args()
    run(args.nx, args.nsteps)
