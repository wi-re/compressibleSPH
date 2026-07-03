import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


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
    minDt: Optional[float] = field(default=1e-6, metadata={'description': 'Minimum time step for adaptive time stepping'})
    maxDt: Optional[float] = field(default=1e-2, metadata={'description': 'Maximum time step for adaptive time stepping'})
    dtGrowthFactor: Optional[float] = field(default=1.1, metadata={'description': 'Growth factor for time step increase in adaptive time stepping'})
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
    minDt: Optional[float] = None,
    maxDt: Optional[float] = None,
    dtGrowthFactor: Optional[float] = None,
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
    if minDt is None:
        minDt = 1e-6
    if maxDt is None:        
        maxDt = 1e-2
    if dtGrowthFactor is None:        
        dtGrowthFactor = 1.1

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
        minDt=minDt,
        maxDt=maxDt,
        dtGrowthFactor=dtGrowthFactor,
        adaptiveDt=adaptiveDt,
        targetNeighbors=targetNeighbors,
        supportMode=supportMode,
        gradientMode=gradientMode,
        laplacianMode=laplacianMode,
        samplingScheme=samplingScheme,
    ), getIntegrator(integrationScheme)

import numpy as np

def configurationToDict(config: SimulationConfig) -> Dict[str, Any]:
    exdict = {
        'device': str(config.device),
        'dtype': str(config.dtype),
        'domain': {
            'min': config.domain.min.cpu().numpy().tolist(),
            'max': config.domain.max.cpu().numpy().tolist(),
            'periodic': config.domain.periodic.cpu().numpy().tolist(),
            'dim': config.domain.dim,
        },
        'dim': config.dim,
        'verletScale': config.verletScale,
        'kernel': config.kernel.name,
        'integrationScheme': config.integrationScheme.name,
        'cflFactor': config.cflFactor,
        'dt': config.dt if not isinstance(config.dt, torch.Tensor) else config.dt.detach().cpu().item(),
        'minDt': config.minDt,
        'maxDt': config.maxDt,
        'dtGrowthFactor': config.dtGrowthFactor,
        'adaptiveDt': config.adaptiveDt,
        'targetNeighbors': config.targetNeighbors,
        'supportMode': config.supportMode.name,
        'gradientMode': config.gradientMode.name,
        'laplacianMode': config.laplacianMode.name,
        'samplingScheme': config.samplingScheme.name,
    }
    for key, value in exdict.items():
        if isinstance(value, (torch.Tensor)):
            exdict[key] = value.detach().cpu().numpy().tolist()
    
    return exdict

def dictToConfig(
    configDict: Dict[str, Any]
) -> SimulationConfig:
    device = torch.device(configDict['device'])
    dtype = getattr(torch, configDict['dtype'].split('.')[-1])
    domainDict = configDict['domain']
    domain = DomainDescription(
        min=torch.tensor(domainDict['min'], device=device, dtype=dtype),
        max=torch.tensor(domainDict['max'], device=device, dtype=dtype),
        periodic=torch.tensor(domainDict['periodic'], device=device, dtype=torch.bool),
        dim=domainDict['dim']
    )
    dim = configDict['dim']
    verletScale = configDict['verletScale']
    kernel = KernelFunctions[configDict['kernel']]
    integrationScheme = IntegrationSchemeType[configDict['integrationScheme']]
    cflFactor = configDict['cflFactor']
    dt = configDict['dt']
    minDt = configDict['minDt']
    maxDt = configDict['maxDt']
    dtGrowthFactor = configDict['dtGrowthFactor']
    adaptiveDt = configDict['adaptiveDt']
    targetNeighbors = configDict['targetNeighbors']
    supportMode = SupportScheme[configDict['supportMode']]
    gradientMode = GradientScheme[configDict['gradientMode']]
    laplacianMode = LaplacianScheme[configDict['laplacianMode']]
    samplingScheme = SamplingScheme[configDict['samplingScheme']]

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
        minDt=minDt,
        maxDt=maxDt,
        dtGrowthFactor=dtGrowthFactor,
        adaptiveDt=adaptiveDt,
        targetNeighbors=targetNeighbors,
        supportMode=supportMode,
        gradientMode=gradientMode,
        laplacianMode=laplacianMode,
        samplingScheme=samplingScheme
    )