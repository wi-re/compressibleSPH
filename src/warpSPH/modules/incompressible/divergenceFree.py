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

from typing import Any, Optional, Union


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

        alphas = dt * computeAlpha(
                currentState = particles,
                config = config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                apparentVolumes = apparentArea,
        )

        # print(f'Alpha: {alphas.mean().cpu().item():.6g}, min: {alphas.min().cpu().item():.6g}, max: {alphas.max().cpu().item():.6g}')

        pressureA = particles.pressures.clone() * 0.75
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
                pressureB = pressureB - pressureB.mean()  # Fix the pressure gauge without altering the RHS


                # pressureB[particles.surfaceIndicators == 1] = 0.0  # Set pressures to zero for surface particles
                # residual_clamped = torch.clamp(residual, min=-threshold).abs()

                # error = torch.mean(residual_clamped).cpu().item()

                error = torch.mean(torch.abs(residual)).cpu().item()
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
        # print(f"Final pressure acceleration: mean: {a_p.mean().cpu().item():.6g}, min: {a_p.min().cpu().item():.6g}, max: {a_p.max().cpu().item():.6g}")

        if verbose:
            print(f'[DF] final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')

        return a_p, pressureB, errors, pressures