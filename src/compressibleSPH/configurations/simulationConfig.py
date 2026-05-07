import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from compressibleSPH import config

from ..utils import *
from integrators import *
from sphWarpCore import *

from dataclasses import dataclass, field

# from waves.utils.domain import buildDomainDescription
from ..utils.sampling import SamplingScheme

@dataclass
class SimulationConfig:
    device: torch.device = field(default_factory=lambda: torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    dtype: torch.dtype = torch.float32

    domain: DomainDescription = field(
        default_factory=lambda: buildDomainDescription(l=2, dim=2, periodic=True)
    )
    dim: int = field(default=2, metadata={'description': 'Dimensionality of the simulation'})
    verletScale: float = field(default=2.0**0.5, metadata={'description': 'Scale factor for Verlet list construction'})
    kernel: KernelFunctions = field(default=KernelFunctions.Wendland2, metadata={'description': 'Kernel function to use'})
    
    integrationScheme: IntegrationSchemeType = field(default=IntegrationSchemeType.rungeKutta4, metadata={'description': 'Integration scheme for time stepping'})
    cflFactor: float = field(default=0.3, metadata={'description': ' CFL factor for time step calculation'})
    dt: Optional[float] = None
    adaptiveDt: bool = field(default=True, metadata={'description': 'Whether to use adaptive time stepping'})
    
    dx: Optional[int] = field(default=None, metadata={'description': 'Initial particle spacing'})
    nx: Optional[int] = field(default=None, metadata={'description': 'Number of particles along one dimension'})
    
    targetNeighbors: int = field(default_factory=lambda: n_h_to_nH(4, 2))
    supportMode: SupportScheme = field(default=SupportScheme.SuperSymmetric, metadata={'description': 'Support scheme for neighbor search'})
    
    
    gradientMode: GradientScheme = field(default=GradientScheme.Difference, metadata={'description': 'Gradient scheme for spatial derivatives'})
    laplacianMode: LaplacianScheme = field(default=LaplacianScheme.Brookshaw, metadata={'description': 'Laplacian scheme for spatial derivatives'})
    samplingScheme: SamplingScheme = field(default=SamplingScheme.regular, metadata={'description': 'Sampling scheme for particle distribution'})


def buildConfig(
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    domain: Optional[DomainDescription] = None,
    dim: Optional[int] = None,
    verletScale: Optional[float] = None, 
    kernel: Optional[KernelFunctions] = None,
    integrationScheme: Optional[IntegrationSchemeType] = None,
    cflFactor: Optional[float] = None,
    dt: Optional[float] = None,
    adaptiveDt: Optional[bool] = None,
    targetNeighbors: Optional[int] = None,
    supportMode: Optional[SupportScheme] = None,
    gradientMode: Optional[GradientScheme] = None,
    laplacianMode: Optional[LaplacianScheme] = None,
    samplingScheme: Optional[SamplingScheme] = None,
) -> SimulationConfig:
    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    if dim is None:
        dim = 2
    if domain is None:
        domain = buildDomainDescription(l=2, dim=dim, periodic=True)
    if verletScale is None:
        verletScale = 2.0**(1/dim)
    if kernel is None:
        kernel = KernelFunctions.Wendland2
    if integrationScheme is None:
        integrationScheme = IntegrationSchemeType.rungeKutta2
    if cflFactor is None:
        cflFactor = 0.3
    if adaptiveDt is None:
        adaptiveDt = True
    if targetNeighbors is None:
        targetNeighbors = n_h_to_nH(4, dim)
    if supportMode is None:
        supportMode = SupportScheme.SuperSymmetric
    if gradientMode is None:
        gradientMode = GradientScheme.Difference
    if laplacianMode is None:
        laplacianMode = LaplacianScheme.Brookshaw
    if samplingScheme is None:
        samplingScheme = SamplingScheme.regular

    return SimulationConfig(
        device=device,
        dtype=dtype,
        domain=domain,
        dim=dim,
        verletScale=verletScale,
        kernel=kernel,
        integrationScheme=integrationScheme,
        cflFactor=cflFactor,
        dt=dt,
        adaptiveDt=adaptiveDt,
        targetNeighbors=targetNeighbors,
        supportMode=supportMode,
        gradientMode=gradientMode,
        laplacianMode=laplacianMode,
        samplingScheme=samplingScheme,
    ), getIntegrator(integrationScheme)
