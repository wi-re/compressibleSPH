"""IISPH divergence-free pressure solver: relaxed-Jacobi iteration that
solves for a pressure field whose acceleration cancels the predicted
velocity divergence (source term `-divergence`, not a density error).

Each iteration computes the pressure acceleration (`computePressureAccelIISPH`,
scatter mode), its induced position-drift residual (`computePressureShiftIISPH`),
and updates `pressure += omega * residual / alpha` (`computeAlpha`'s IISPH
diagonal term), re-centering the pressure field to zero mean each step since
this is a pure-Neumann (gauge-free) problem. Iterates between
`solverConfig.divergenceFreeSolver.{minIterations,maxIterations,tolerance,
relaxationFactor}`, stopping early once past `minIterations` and below
`tolerance`; does not clamp non-convergence (increasing-error iterations are
only logged when `verbose`).

Relaxation mode (`relaxationMode`):
  `fixed` (default, byte-identical history): constant `relaxationFactor`.
    The update matrix `I - omega*D^-1*A` converges iff
    `omega < 2/rho(D^-1*A)` (the `D^-1*A` spectrum lies in `[0, rho]`: the
    operator is symmetric negative-semi-definite with a constant gauge null
    space, so `D^-1*A` is similar to the symmetric `|D|^-1/2*(-A)*|D|^-1/2`).
    On this operator family `rho ~= 5.64` (a degenerate high-frequency
    lattice cluster, robust to grid deformation), i.e. the window is
    `omega < 0.355`: the historical dataclass default 0.5 diverges, 0.3 sits
    inside with ~15% margin, and fixed-omega performance is flat inside the
    window (so only the margin matters, not the exact value).
  `optimal`: per-step exact residual minimizer
    `omega_k = (r . A*D^-1*r) / ||A*D^-1*r||^2` (the 1-D minimizer of
    `||r - w*A*D^-1*r||`). Costs the same single accel+shift pair per step
    as `fixed` (the residual is updated by the exact recurrence
    `r <- r - omega_k*A*D^-1*r`, re-verified every 16 steps against the true
    `b - A*p` to bound fp32 drift) and decreases the residual monotonically
    for any starting size -- no stability window, no tuning. Measured on the
    TGV probe state (N=1024, 64 steps): final relative residual ~4.8% vs
    ~5.2% for in-window `fixed` omega=0.3, monotone throughout.
    `optimal` is only defined for this solver: the constant-density
    (`incompressible.py`) variant clamps pressures non-negative, which breaks
    the exact residual recurrence.
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

from typing import Any, Optional, Union

__all__ = ['solveDivergenceFree']


def _solveDivergenceFreeOptimal(
        particles: Any,
        config: SimulationConfig,
        schemeConfig: Any,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
        sourceTerm: torch.Tensor,
        alphas: torch.Tensor,
        dt: float,
        dfSolver: Any,
        verbose: bool = False,
):
        """Optimal-step relaxed Jacobi (see the module docstring): same warm
        start, gauge re-centering, convergence check, and return contract as
        the fixed-omega loop in `solveDivergenceFree`, with the step size
        replaced by the exact per-step residual minimizer
        `omega_k = (r . A D^-1 r) / ||A D^-1 r||^2`.

        Per step this costs one accel+shift pair (the same as one fixed step):
        `q = A (D^-1 r)` replaces `A p`, and the residual is advanced by the
        exact recurrence `r <- r - omega_k * q` instead of recomputed. The
        recurrence is re-verified against the true `b - A p` every 16 steps
        to bound fp32 drift. `relaxationFactor` is ignored.
        """
        minIters = dfSolver.minIterations
        maxIters = dfSolver.maxIterations
        threshold = dfSolver.tolerance

        # kind==1 (boundary) and kind==2 (ghost) particles are not pressure unknowns: their
        # pressure is held fixed at 0 for the duration of the solve (see
        # `BoundaryPressureMode`'s docstring), excluded from the gauge mean,
        # and their `a_p` is zeroed post-solve. A no-op when there are no
        # boundary particles (`fluidMask` all-True).
        fluidMask = particles.kinds == 0

        pressureA = particles.pressures.clone() * 0.75
        pressureB = torch.where(fluidMask, pressureA, torch.zeros_like(pressureA))
        pressureA = pressureB.clone()

        errors = []
        pressures = []
        error = 0.

        def op(pressureValues: torch.Tensor) -> torch.Tensor:
            a = computePressureAccelIISPH(
                    state = particles,
                    pressureValues = pressureValues,
                    config = config,
                    supportScheme = SupportScheme.Scatter,
                    adjacency = adjacency,
            )
            return dt * computePressureShiftIISPH(
                    state = particles,
                    config = config,
                    pressureAccels = a,
                    supportScheme = SupportScheme.Scatter,
                    adjacency = adjacency,
            )

        residual = sourceTerm - op(pressureA)
        for i in range(maxIters):
                # Zero the trial step at boundary rows *before* the matvec:
                # `u` is the pressure field `op()` evaluates, so this both
                # freezes boundary pressure at 0 and keeps `q` (and hence
                # `omega_k`) reflecting only the fluid-unknown subproblem.
                u = torch.where(fluidMask, residual / alphas, torch.zeros_like(residual))
                a_p = computePressureAccelIISPH(
                        state = particles,
                        pressureValues = u,
                        config = config,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )
                q = dt * computePressureShiftIISPH(
                        state = particles,
                        config = config,
                        pressureAccels = a_p,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )
                num = float(torch.dot(residual[fluidMask], q[fluidMask]))
                den = float(torch.dot(q[fluidMask], q[fluidMask]))
                # exact 1-D minimizer of ||r - w*q||^2; clamp at 0 against
                # fp noise (in exact arithmetic num >= 0 for this operator)
                omega_k = max(0.0, num / den) if den > 0.0 else 0.0

                pressureA = pressureB.clone()
                pressureB = pressureA + omega_k * u
                pressureB = pressureB - pressureB[fluidMask].mean()  # Fix the pressure gauge without altering the RHS
                pressureB = torch.where(fluidMask, pressureB, torch.zeros_like(pressureB))
                residual = residual - omega_k * q
                if (i + 1) % 16 == 0:
                        residual = sourceTerm - op(pressureB)  # bound fp32 recurrence drift

                error = torch.mean(torch.abs(residual[fluidMask])).cpu().item()
                errors.append(error)

                pressures.append((pressureB.min().cpu().item(), pressureB.max().cpu().item(), pressureB.mean().cpu().item()))

                if i >= minIters and error < threshold:
                    break

                if verbose:
                    print(f"[DF] Iteration {i+1}/{maxIters} (optimal step, omega={omega_k:.6g}), residual min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}, mean: {residual.mean().cpu().item():.6g}, error: {error:.6g}, pressure min/max/mean: {pressures[-1]}")
                if len(errors) > 1 and error > errors[-2]:
                    if verbose:
                        print(f"!!![DF] Warning: Error increased from {errors[-2]:.6g} to {error:.6g}.!!!")

        a_p = computePressureAccelIISPH(
                state = particles,
                pressureValues = pressureB,
                config = config,
                supportScheme = SupportScheme.Scatter,
                adjacency = adjacency,
        )
        a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))
        if verbose:
            print(f'[DF] final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')

        return a_p, pressureB, errors, pressures


def solveDivergenceFree(
        particles: Any, 
        config: SimulationConfig, 
        schemeConfig: Any, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        dvdt: torch.Tensor,
        dt: float,
        verbose: bool = False,
):
        predictedVelocities = particles.velocities + dt * dvdt
        # dt = dt

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

        # sourceTerm = dt * divergence
        # sourceTerm = (rho0 - rhoStar)# / dt
        sourceTerm =  -divergence# * dt


        # sourceTerm = sourceTerm - sourceTerm.mean()  # Remove mean to ensure zero-mean source term
        if verbose:
            print(f'Divergence Free Solver')
            print(f'[DF] Source term: {sourceTerm.mean().cpu().item():.6g}, min: {sourceTerm.min().cpu().item():.6g}, max: {sourceTerm.max().cpu().item():.6g}')

        # Opt-in Krylov pressure solvers (BiCGStab/GMRES/CG/BiCG/MINRES) share the same
        # matrix-free operator and IISPH-diagonal preconditioner as the relaxed
        # Jacobi path below, which stays the byte-identical default
        # (solverType == relaxedJacobi).
        dfSolver = schemeConfig.solverConfig.divergenceFreeSolver
        if dfSolver.solverType != PressureSolverType.relaxedJacobi:
            return solvePressureKrylov(
                particles, config, schemeConfig, adjacency, sourceTerm, dt,
                dfSolver, gauge='center', verbose=verbose)

        alphas = dt * computeAlpha(
                currentState = particles,
                config = config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                apparentVolumes = apparentArea,
        )

        # print(f'Alpha: {alphas.mean().cpu().item():.6g}, min: {alphas.min().cpu().item():.6g}, max: {alphas.max().cpu().item():.6g}')

        # Opt-in optimal-step relaxed Jacobi (per-step exact residual
        # minimizer, see module docstring): same warm start and convergence
        # contract as the fixed-omega loop below, no stability window.
        if dfSolver.relaxationMode is JacobiRelaxationMode.optimal:
            return _solveDivergenceFreeOptimal(
                particles, config, schemeConfig, adjacency, sourceTerm, alphas, dt,
                dfSolver, verbose=verbose)

        # kind==1 (boundary) and kind==2 (ghost) particles are not pressure unknowns: their
        # pressure is held fixed at 0 for the duration of the solve (see
        # `BoundaryPressureMode`'s docstring), excluded from the gauge mean,
        # and their `a_p` is zeroed post-solve. A no-op when there are no
        # boundary particles (`fluidMask` all-True).
        fluidMask = particles.kinds == 0

        pressureA = particles.pressures.clone() * 0.75
        pressureA = torch.where(fluidMask, pressureA, torch.zeros_like(pressureA))
        pressureB = pressureA.clone()

        errors = []
        pressures = []
        i = 0
        error = 0.
        minIters = schemeConfig.solverConfig.divergenceFreeSolver.minIterations
        maxIters = schemeConfig.solverConfig.divergenceFreeSolver.maxIterations
        threshold = schemeConfig.solverConfig.divergenceFreeSolver.tolerance
        omega = schemeConfig.solverConfig.divergenceFreeSolver.relaxationFactor
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
                dx_p = dt * computePressureShiftIISPH(
                        state = particles,
                        config = config,
                        pressureAccels = a_p,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )

                residual = sourceTerm - dx_p
                pressureB = pressureA + omega * residual / alphas
                pressureB = pressureB - pressureB[fluidMask].mean()  # Fix the pressure gauge without altering the RHS
                pressureB = torch.where(fluidMask, pressureB, torch.zeros_like(pressureB))


                # pressureB[particles.surfaceIndicators == 1] = 0.0  # Set pressures to zero for surface particles
                # residual_clamped = torch.clamp(residual, min=-threshold).abs()

                # error = torch.mean(residual_clamped).cpu().item()

                error = torch.mean(torch.abs(residual[fluidMask])).cpu().item()
                errors.append(error)

                pressures.append((pressureB.min().cpu().item(), pressureB.max().cpu().item(), pressureB.mean().cpu().item()))

                if i >= minIters and error < threshold:
                #     print(f"Converged after {i+1} iterations with error: {error:.6g}")
                    break
                
                if verbose:
                    print(f"[DF] Iteration {i+1}/{maxIters}, residual min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}, mean: {residual.mean().cpu().item():.6g}, error: {error:.6g}, pressure min/max/mean: {pressures[-1]}")
                if len(errors) > 1 and error > errors[-2]:
                    if verbose:
                        print(f"!!![DF] Warning: Error increased from {errors[-2]:.6g} to {error:.6g}.!!!")

        a_p = computePressureAccelIISPH(
                state = particles,
                pressureValues = pressureB,
                config = config,
                supportScheme = SupportScheme.Scatter,
                adjacency = adjacency,
        )
        a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))
        # print(f"Final pressure acceleration: mean: {a_p.mean().cpu().item():.6g}, min: {a_p.min().cpu().item():.6g}, max: {a_p.max().cpu().item():.6g}")

        if verbose:
            print(f'[DF] final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')

        return a_p, pressureB, errors, pressures