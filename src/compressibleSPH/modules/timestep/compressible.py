from compressibleSPH.utils.support import volumeToSupportHelper

from ...systems.compressibleMonaghan import CompressibleState, CompressibleSystem
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from typing import Optional
from sphWarpCore.kernels.wp_kernel import sphKernel_xi
from ...modules.eos import idealGasEOS
import torch
import warp as wp

def computeTimestep(
    system: CompressibleSystem,
    config: SimulationConfig,
    compParams: CompressibleSPHConfig,
    dt: Optional[float] = None,
):
    if config.adaptiveDt:
        # 
        targetCFL = 0.3
        # The initial state contains density and pressure
        # we can get the internal energies for each state via the gas EOS
        u = system.state.internalEnergies
        # c_s = system.soundspeeds.max()            
        A_, _, P_, c_s = idealGasEOS(
            A = None,
            u = system.state.internalEnergies,
            P = None,
            rho = system.state.densities,
            gamma = compParams.gamma,
        )

        # print(f"Initial internal energy: u (min: {u.min()}, max: {u.max()}, mean: {u.mean()})")
        # print(f"Initial sound speed: c_s (min: {c_s.min()}, max: {c_s.max()}, mean: {c_s.mean()})")

        # u_left = leftState.p / ((gamma - 1) * leftState.rho)
        # u_right = rightState.p / ((gamma - 1) * rightState.rho)
        # # this then gives us the speed of sound for each state, which we can use to determine the initial timestep based on the CFL condition
        # c_s_left =  np.sqrt(u_left * gamma * (gamma - 1))
        # c_s_right =  np.sqrt(u_right * gamma * (gamma - 1))

        # We cab then get the minimum support radius based on the initial sampling and use this to get the initial timestep
        # The sampling is done such that with a sampling ratio of 1 th left and right states each have nx particles
        # The sampling ratio then reduces the number of particles by that factor
        # accordingly, the higher resolution is always the unmultiplied L/2 /nx
        # L = config.domain.max - config.domain.min
        # dx = L / (nx * 2)
        h = volumeToSupportHelper(config.dx, config.targetNeighbors, config.dim)
        h = system.state.supports.min()
        xi = float(sphKernel_xi(config.kernel.value, config.dim))
        # xi = 1

        dt_cfl_left = targetCFL * h / (c_s * xi)
        # dt_cfl_right = targetCFL * h / (c_s + h * xi)
        initial_dt = torch.min(dt_cfl_left)
        if initial_dt < config.minDt:
            print(f"Warning: initial dt {initial_dt} is less than minDt {config.minDt}. Clamping to minDt.")
        initial_dt = torch.clamp(initial_dt, min=config.minDt, max=config.maxDt)
        if initial_dt > config.dt:
            initial_dt = torch.clamp(initial_dt, max=config.dtGrowthFactor * config.dt)
        return initial_dt.cpu().item()
    else:
        if dt is not None:
            return dt
        else:
            raise ValueError("dt must be provided if adaptiveDt is False")