from dataclasses import dataclass
from enum import Enum
from .boundaryConditions import BoundaryConditionType, BoundaryCondition
from ..utils.sampling import ParticleSet

class RegionType(Enum):
    Fluid = 1
    Boundary = 2
    Inlet = 4
    Outlet = 5

@dataclass
class ParticleRegion:
    sdf: callable
    type: RegionType
    particles: ParticleSet
    contour: list = None

    initialConditions: dict = None
    kind: BoundaryConditionType = BoundaryConditionType.constant