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
class ShapeSpec:
    """
    Generic shape descriptor used by source and obstacle configuration.

    `kind` supports both primitive and composite shapes currently used in gencase:
    sphere, box, horizontal_line, vertical_line, vesica,
    triangle, equilateral_triangle, single_slit, double_slit, prism.
    """
    kind: str = 'sphere'
    position: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaveSource:
    shapeSpec: Optional[ShapeSpec] = None

    position: Tuple[float, float] = (0.0, 0.0)
    shape: str = 'circle'  # 'point', 'line', 'area'
    radius: float = 0.1  # For circle or area sources
    size: Tuple[float, float] = (1.0, 1.0)  # For line or area sources
    magnitude: Optional[float] = None  # If None, use default from CaseConfig

    randomizeRadius: bool = False
    randomizeMagnitude: bool = False
    radiusRange: Tuple[float, float] = (0.05, 0.15)

@dataclass
class WaveBoundary:
    shapeSpec: Optional[ShapeSpec] = None

    shape: str = 'box'  # 'box', 'circle', 'line'
    position: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (1.0, 1.0)  # For box boundaries
    radius: float = 0.1  # For circle boundaries
    speed: Optional[float] = None  # If None, use default from CaseConfig

    randomizeSpeed: bool = False

@dataclass
class CaseConfig:
    name: str = 'defaultCase'
    description: str = 'This is a default case configuration.'

    domainBox: bool = False
    domainDamping: bool = False

    smoothICs: bool = False
    smoothIterations: int = 4

    defaultSpeed: float = 1.0
    defaultBoundarySpeed: float = 0.01
    defaultObstacleSpeed: Union[float, List[float]] = 0.5
    defaultAmplitude: float = 10.0

    amplitudeRange: Tuple[float, float] = (15.0, 15.0)
    boundarySpeedRange: Tuple[float, float] = (0.01, 0.01)
    obstacleSpeedRange: Tuple[float, float] = (0.3, 0.7)

    noisyICs: bool = False
    noiseAmplitude: float = 0.1
    noiseType: str = 'perlin' 
    noiseSmoothIters: int = 4
    noiseSeed: int = 42

    sources: List[WaveSource] = ()
    obstacles: List[WaveBoundary] = ()