import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


from ..utils import *
from warpSPHIntegrators import *
from warpSPHCore import *

from dataclasses import dataclass, field

# from waves.utils.domain import buildDomainDescription
from ..geometry import SamplingScheme

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
    dx: Optional[float] = None,
    nx: Optional[int] = None,
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
        dx=dx,
        nx=nx,
        targetNeighbors=targetNeighbors,
        supportMode=supportMode,
        gradientMode=gradientMode,
        laplacianMode=laplacianMode,
        samplingScheme=samplingScheme,
    ), getIntegrator(integrationScheme)

import numpy as np
import enum
import types
import typing
from dataclasses import fields as _dataclassFields

# Bind the domain type explicitly rather than relying on whichever `import *` above
# happens to win. `warpSPHCore.DomainDescription` is the single owner; `..utils.domain`
# re-exports it and `buildDomainDescription` returns it.
from warpSPHCore import DomainDescription as _DomainDescription


def _unwrapOptional(annotation):
    """Strip ``Optional[X]`` / ``X | None`` down to ``X``."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(types, 'UnionType', None):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _encodeValue(value: Any) -> Any:
    """Serialize a single config field to a JSON/HDF5-friendly value.

    Dispatches on the runtime type, so a newly added field of an already
    supported type serializes without touching this function.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.numpy().tolist()
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, _DomainDescription):
        return {
            'min': _encodeValue(value.min),
            'max': _encodeValue(value.max),
            'periodic': _encodeValue(value.periodic),
            'dim': value.dim,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _decodeValue(annotation: Any, value: Any, device, dtype) -> Any:
    """Inverse of :func:`_encodeValue`, dispatching on the declared field type.

    Only *structural* types are converted (enums, device, dtype, domain).
    Numeric fields are passed through rather than coerced to their annotation:
    ``targetNeighbors`` is annotated ``int`` but legitimately holds a float
    (``n_h_to_nH(4, 2) == 50.265...``), and coercing it would silently change
    the support radius.
    """
    if value is None:
        return None
    annotation = _unwrapOptional(annotation)
    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return annotation[value] if isinstance(value, str) else annotation(value)
        if annotation is torch.device:
            return torch.device(value)
        if annotation is torch.dtype:
            return getattr(torch, str(value).split('.')[-1])
        if annotation is _DomainDescription:
            return _DomainDescription(
                min=torch.tensor(value['min'], device=device, dtype=dtype),
                max=torch.tensor(value['max'], device=device, dtype=dtype),
                periodic=torch.tensor(value['periodic'], device=device, dtype=torch.bool),
                dim=int(value['dim']),
            )
        if annotation is bool:
            return bool(value)
    # Normalize numpy scalars that come back out of HDF5 without changing width.
    if isinstance(value, np.generic):
        return value.item()
    return value


def _configFieldTypes() -> Dict[str, Any]:
    """Resolved annotations for :class:`SimulationConfig`, with a safe fallback."""
    try:
        return typing.get_type_hints(SimulationConfig)
    except Exception:
        return {f.name: f.type for f in _dataclassFields(SimulationConfig)}


def configurationToDict(config: SimulationConfig) -> Dict[str, Any]:
    """Serialize every declared field of ``config``.

    Driven by ``dataclasses.fields`` rather than a hand-written list, so a field
    added to :class:`SimulationConfig` can never again be silently dropped from
    the round-trip (this previously lost ``nx`` and ``dx``, i.e. resolution).
    """
    return {
        f.name: _encodeValue(getattr(config, f.name))
        for f in _dataclassFields(config)
    }


def dictToConfig(
    configDict: Dict[str, Any]
) -> SimulationConfig:
    """Rebuild a :class:`SimulationConfig` from :func:`configurationToDict` output.

    Fields absent from ``configDict`` fall back to their dataclass defaults, so
    dicts written by older versions still load.
    """
    fieldTypes = _configFieldTypes()

    device = torch.device(configDict['device']) if 'device' in configDict else None
    dtype = (
        getattr(torch, str(configDict['dtype']).split('.')[-1])
        if 'dtype' in configDict else torch.float32
    )

    kwargs = {}
    for f in _dataclassFields(SimulationConfig):
        if f.name not in configDict:
            continue
        kwargs[f.name] = _decodeValue(
            fieldTypes.get(f.name, f.type), configDict[f.name], device, dtype
        )

    return SimulationConfig(**kwargs)