from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum

class ShiftingScheme(Enum):
    none = 0
    deltaSPH = 1
    implicit = 2


class ShiftingProjectionScheme(Enum):
    zero = 0
    dot = 1
    mat = 2

@dataclass 
class ShiftProperties:
    iterations: int = field(default=1, metadata={"description": "Number of iterations for shifting"})
    CFL: float = field(default=0.3, metadata={"description": "CFL number for the delta-SPH shift"})
    computeMach: bool = field(default=False, metadata={"description": "Whether to compute Mach number for the delta-SPH shift"})
    maxC: float = field(default=0.3, metadata={"description": "Maximum sound speed for the delta-SPH shift"})
    active: bool = field(default=True, metadata={"description": "Whether to apply the shifting"})

    scheme: ShiftingScheme = field(default=ShiftingScheme.deltaSPH, metadata={"description": "Shifting scheme to use"})
    projectionScheme: ShiftingProjectionScheme = field(default=ShiftingProjectionScheme.dot, metadata={"description": "Projection scheme to use for shifting"})

    summationDensity: bool = field(default=False, metadata={"description": "Whether to use summation density"})
    surfaceScaling: float = field(default=0.1, metadata={"description": "Scaling factor for the surface detection"})
    threshold: float = field(default=0.5, metadata={"description": "Threshold for shifting magnitude"})

    projectQuantities: bool = field(default=False, metadata={"description": "Whether to project quantities after shifting"})

    correctdrhodt: bool = field(default=False, metadata={"description": "Whether to correct drhodt after shifting"})
    correctdvdt: bool = field(default=False, metadata={"description": "Whether to correct dvdt after shifting"})

    reuseNormals: bool = field(default=True, metadata={"description": "Whether to reuse normals from previous iteration for surface detection"})

def buildDefaultShiftProperties() -> ShiftProperties:
    return ShiftProperties(
        iterations=1,
        CFL=0.3,
        computeMach=True,
        maxC=0.3,
        active=True,
        scheme=ShiftingScheme.deltaSPH,
        projectionScheme=ShiftingProjectionScheme.mat,
        summationDensity=False,
        surfaceScaling=0.1,
        threshold=0.5,
        projectQuantities=False
    )
