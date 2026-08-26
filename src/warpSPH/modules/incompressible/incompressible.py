"""IISPH constant-density pressure solver: relaxed-Jacobi iteration that
solves for a pressure field driving the predicted density back to
`schemeConfig.fluid.restDensity` (source term `rho0 - rhoStar`, the
density-error formulation, as opposed to `divergenceFree.py`'s divergence
formulation).

Each iteration computes the pressure acceleration (`computePressureAccelIISPH`,
scatter mode), its position-drift residual (`dt**2 * computePressureShiftIISPH`),
and updates `pressure += omega * residual / alpha` (`dt**2 * computeAlpha`'s
IISPH diagonal term), clamping pressures to be non-negative each step.
`rhoStar` is clamped to a minimum of 0.9 and `alpha` to a maximum of -1e-6 to
avoid division blow-up. Iterates between `solverConfig.pressureSolver.
{minIterations,maxIterations,tolerance,relaxationFactor}`, stopping early once
past `minIterations` and below `tolerance`.
"""

from warpSPHCore import *
import torch
from warpSPH.systems import *
from warpSPH.modules import *
from warpSPH.configurations import SimulationConfig
from typing import Optional, Union, Any



from warpSPH.configurations import SimulationConfig
from typing import Optional


from typing import Any, Optional, Union
from .wp_alpha import computeAlpha
from ..momentum.incompressible import computeMomentumIncompressible
from ..pressure.iisph import computePressureAccelIISPH
from .drift import computePressureShiftIISPH
from ...configurations import PressureSolverType, JacobiRelaxationMode
from .krylov import solvePressureKrylov

__all__ = ['solveIncompressible']


def solveIncompressible(
    
        particles: Any, 
        config: SimulationConfig, 
        schemeConfig: Any, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        dvdt: torch.Tensor,
        dt : float,
        verbose: bool = False        
):
        minIters = schemeConfig.solverConfig.pressureSolver.minIterations
        maxIters = schemeConfig.solverConfig.pressureSolver.maxIterations
        threshold = schemeConfig.solverConfig.pressureSolver.tolerance
        omega = schemeConfig.solverConfig.pressureSolver.relaxationFactor

        predictedVelocities = particles.velocities + dt * dvdt
        # dt = config.dt

        # print(f'Predicted velocities: mean: {predictedVelocities.mean().cpu().item():.6g}, min: {predictedVelocities.min().cpu().item():.6g}, max: {predictedVelocities.max().cpu().item():.6g}')

        apparentArea = particles.masses / particles.densities    

        # print(f'Apparent area: {apparentArea.mean().cpu().item():.6g}, min: {apparentArea.min().cpu().item():.6g}, max: {apparentArea.max().cpu().item():.6g}')

        divergence = computeMomentumIncompressible(
                currentState = particles, 
                config = config, 
                schemeConfig = schemeConfig, 
                adjacency = adjacency, 
                advectionVelocities = predictedVelocities
        )

        rho0 = schemeConfig.fluid.restDensity
        rhoStar = particles.densities + dt * divergence

        rhoStar = torch.clamp(rhoStar, min = 0.9)  # Clamp to avoid extreme density values

        # sourceTerm = config.dt * divergence
        sourceTerm = (rho0 - rhoStar)# / config.dt

        # sourceTerm = sourceTerm - sourceTerm.mean()  # Remove mean to ensure zero-mean source term
        if verbose:
            print(f'Incompressible Solver')
            print(f'[IS] Source term: {sourceTerm.mean().cpu().item():.6g}, min: {sourceTerm.min().cpu().item():.6g}, max: {sourceTerm.max().cpu().item():.6g} abs mean: {sourceTerm.abs().mean().cpu().item():.6g}')
            print(f'[IS] Mean density error: {(particles.densities - schemeConfig.fluid.restDensity).abs().mean().cpu().item():.6g}')

        # Opt-in Krylov pressure solvers (BiCGStab/GMRES/CG/BiCG/MINRES) share the same
        # matrix-free operator and IISPH-diagonal preconditioner as the relaxed
        # Jacobi path below, which stays the byte-identical default
        # (solverType == relaxedJacobi). The constant-density variant scales the
        # operator by dt**2 and clamps the pressure non-negative (gauge='nonnegative').
        psSolver = schemeConfig.solverConfig.pressureSolver
        if psSolver.solverType != PressureSolverType.relaxedJacobi:
            return solvePressureKrylov(
                particles, config, schemeConfig, adjacency, sourceTerm, dt**2,
                psSolver, gauge='nonnegative', verbose=verbose)

        if psSolver.relaxationMode is JacobiRelaxationMode.optimal:
            raise ValueError(
                "relaxationMode 'optimal' is only supported by the divergenceFree "
                "(IISPH) solver: the constant-density solver clamps pressures to "
                "non-negative each iteration, which breaks the exact residual "
                "recurrence the optimal step relies on")

        alphas = dt**2 * computeAlpha(
                currentState = particles,
                config = config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                apparentVolumes = apparentArea,
        )

        alphas = torch.clamp(alphas, max=-1e-6)  # Avoid division by zero
        # print(f'Alpha: {alphas.mean().cpu().item():.6g}, min: {alphas.min().cpu().item():.6g}, max: {alphas.max().cpu().item():.6g}')

        # kind==1 (boundary) and kind==2 (ghost) particles are not pressure unknowns: their
        # pressure is held fixed at 0 for the duration of the solve (see
        # `BoundaryPressureMode`'s docstring), excluded from the gauge mean
        # (this solver's gauge is a non-negativity clamp, not a mean-center,
        # so there is no mean to exclude them from), and their `a_p` is
        # zeroed post-solve. A no-op when there are no boundary particles
        # (`fluidMask` all-True).
        fluidMask = particles.kinds == 0

        pressureA = particles.pressures.clone() * 0.
        pressureB = pressureA.clone()

        errors = []
        pressures = []
        i = 0
        error = 0.

        # print(f"Solving for divergence-free velocities with maxIters={maxIters}, threshold={threshold:.6g}, omega={omega:.6g}")

        for i in range(maxIters):
                pressureA = pressureB.clone()
                a_p = computePressureAccelIISPH(
                        state = particles,
                        pressureValues = pressureA,
                        config = config,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )
                dx_p = dt**2 * computePressureShiftIISPH(
                        state = particles,
                        config = config,
                        pressureAccels = a_p,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )

                residual = sourceTerm - dx_p
                pressureB = pressureA + omega * residual / alphas
                pressureB = torch.clamp(pressureB, min=0.0)  # Ensure non-negative pressures
                pressureB = torch.where(fluidMask, pressureB, torch.zeros_like(pressureB))
                # pressureB = pressureB - pressureB.mean()  # Fix the pressure gauge without altering the RHS

                residual_clamped = torch.clamp(-residual, min=-threshold)

                error = torch.mean(residual_clamped[fluidMask]).cpu().item()
                # error = torch.mean(torch.abs(residual)).cpu().item()
                errors.append(error)

                pressures.append((pressureB.min().cpu().item(), pressureB.max().cpu().item(), pressureB.mean().cpu().item()))

                if i >= minIters and error < threshold:
                #     print(f"Converged after {i+1} iterations with error: {error:.6g}")
                    break
                
                if verbose:
                    print(f"[IS] Iteration {i+1}/{maxIters}, error: {error:.6g}, pressure min/max/mean: {pressures[-1]}")

                if len(errors) > 1 and error > errors[-2]:
                    if verbose:
                        print(f"!!![IS] Warning: Error increased from {errors[-2]:.6g} to {error:.6g}.!!!")

        a_p = computePressureAccelIISPH(
                state = particles,
                pressureValues = pressureB,
                config = config,
                supportScheme = SupportScheme.Scatter,
                adjacency = adjacency,
        )
        a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))
        # print(f"Final pressure acceleration: mean: {a_p.mean().cpu().item():.6g}, min: {a_p.min().cpu().item():.6g}, max: {a_p.max().cpu().item():.6g}")

        # print(f'final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')
        if verbose:
            print(f'[IS] final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')
            # if residual.mean() > 
        return a_p, pressureB, errors, pressures

