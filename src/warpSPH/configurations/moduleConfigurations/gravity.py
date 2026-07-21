from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum

class GravityType(Enum):
    Directional = 0
    PointSource = 1
    PotentialField = 2

def buildDefaultDirection() -> List[float]:
    return [0.0, -1.0]

def buildDefaultOrigin() -> List[float]:
    return [0.0, 0.0]

@dataclass
class gravityConfiguration:
    active: bool = field(default=False, metadata={"description": "Whether gravity is active or not"})
    type: GravityType = field(default=GravityType.Directional)

    magnitude: float = field(default=9.81, metadata={"description": "Gravity magnitude"})
    direction: List[float] = field(default_factory=buildDefaultDirection, metadata={"description": "Gravity direction"})

    origin: List[float] = field(default_factory=buildDefaultOrigin, metadata={"description": "Origin of the point source gravity"})


def buildDefaultGravityConfiguration() -> gravityConfiguration:
    return gravityConfiguration(
        active=False,
        type=GravityType.Directional,
        magnitude=9.81,
        direction=buildDefaultDirection(),
        origin=buildDefaultOrigin()
    )

def gravityConfigurationToDict(config: gravityConfiguration) -> dict:
    return {
        "active": config.active,
        "type": config.type.name,
        "magnitude": config.magnitude,
        "direction": config.direction.cpu().tolist() if isinstance(config.direction, torch.Tensor) else config.direction,
        "origin": config.origin if not isinstance(config.origin, torch.Tensor) else config.origin.cpu().tolist()
    }

def dictToGravityConfiguration(config_dict: dict) -> gravityConfiguration:
    return gravityConfiguration(
        active=config_dict.get("active", False),
        type=GravityType[config_dict.get("type", "Directional")],
        magnitude=config_dict.get("magnitude", 9.81),
        direction=config_dict.get("direction", [0.0, -1.0]),
        origin=config_dict.get("origin", [0.0, 0.0])
    )