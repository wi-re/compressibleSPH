from compressibleSPH.utils.support import volumeToSupportHelper

from ...systems.weaklyCompressible import WeaklyCompressibleState, WeaklyCompressibleSystem, WeaklyCompressibleSystemUpdate
from ...configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from typing import Optional
from sphWarpCore.kernels.wp_kernel import sphKernel_xi
from ...modules.eos import idealGasEOS
import torch
import warp as wp


def computeTimestep(
    system: WeaklyCompressibleSystem,
    config: SimulationConfig,
    compParams: WeaklyCompressibleSPHConfig,
    dt: Optional[float] = None,
    systemUpdate: Optional[WeaklyCompressibleSystemUpdate] = None,
):
# See Sun et al: The delta SPH-model: Simple procedures for a further improvement of the SPH scheme
    if not config.adaptiveDt:
        return dt
    
    timestepCFL = config.cflFactor
    maxDt = config.maxDt
    minDt = config.minDt
    
    alpha = compParams.diffusionParams.C_l
    c_s = compParams.fluid.fixedSoundSpeed
    particleSupport = system.state.supports.min()
    kernelScale = sphKernel_xi(config.kernel.value, config.dim)

    dtype =  config.dtype
    device = config.device

    # if verbose:
    #     print(f'[SPH] - Adaptive Timestep Update')
    #     print(f'\tCFL: {timestepCFL}, maxDt: {maxDt}, minDt: {minDt}')
    #     print(f'\tDiffusion: {diffusionAlpha}, {diffusionScheme}')
    #     print(f'\tFluid: {fluidCs}, {particleSupport}, {kernelScale}')


    # with record_function("[SPH] - Adaptive Timestep Update"):
    state = system.state
    dim = state.positions.shape[1]

    nu = alpha * c_s * particleSupport / (2 * (dim +2))
    # nu = config.get('diffusion', {}).get('nu', nu) if diffusionScheme == 'deltaSPH_viscid' else nu
    dt_v = 0.125 * particleSupport**2 / nu / kernelScale
    dt_v = torch.tensor(dt_v, dtype = dtype, device = device) if not isinstance(dt_v, torch.Tensor) else dt_v

    dt_c = timestepCFL * particleSupport / c_s / kernelScale
    dt_c = torch.tensor(dt_c, dtype = dtype, device = device) if not isinstance(dt_c, torch.Tensor) else dt_c

    # acceleration timestep condition
    if systemUpdate is not None and hasattr(systemUpdate, 'velocities'):
        dudt = systemUpdate.velocities
        max_accel = torch.max(torch.linalg.norm(dudt[~torch.isnan(dudt)], dim = -1))
        dt_a = 0.25 * torch.sqrt(particleSupport / (max_accel + 1e-7)) / kernelScale
    else:
        dt_a = torch.tensor(maxDt, dtype = dtype, device = device)
    dt_a = torch.tensor(dt_a, dtype = dtype, device = device) if not isinstance(dt_a, torch.Tensor) else dt_a
    # if verbose:
        # print(f'\tViscosity: {dt_v}, Acoustic: {dt_c}, Acceleration: {dt_a}')

    # dt = config['timestep']['dt']
    dt = torch.tensor(dt, dtype = dtype, device = device) if not isinstance(dt, torch.Tensor) else dt
    new_dt = dt
    if compParams.dt_viscosityConstraint:
        new_dt = dt_v
    if compParams.dt_accelerationConstraint:
        new_dt = torch.min(new_dt, dt_a)
    if compParams.dt_acousticConstraint:
        new_dt = torch.min(new_dt, dt_c)
    new_dt = torch.min(new_dt, torch.tensor(maxDt, dtype = dtype, device = device))
    new_dt = torch.max(new_dt, torch.tensor(minDt, dtype = dtype, device = device))
    return new_dt
