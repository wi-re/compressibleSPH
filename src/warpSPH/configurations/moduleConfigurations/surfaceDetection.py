from dataclasses import dataclass, field
from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum
import numpy as np


class SurfaceDetectionScheme(Enum):
    ColorField = 0
    ColorFieldGrad = 1
    Barecasco = 2
    Maronne = 3

class NormalSource(Enum):
    Native = 0
    ColorFieldGrad = 1
    LambdaGrad = 2
    Maronne = 3

@dataclass
class SurfaceDetectionConfig:
    active: bool = field(default = True, metadata = {"help": "Whether to use surface detection or not"})
    colorFieldThreshold: float = field(default = 0.75, metadata = {"help": "Threshold for color field based surface detection as a fraction of target neighbors"})
    colorFieldGradThreshold: float = field(default = 10.0, metadata = {"help": "Threshold for color field gradient based surface detection"})
    barecascoThreshold: float = field(default = 1.5, metadata = {"help": "Threshold for Barecasco surface detection"})


    expansionIterations: int = field(default = 1, metadata = {"help": "Number of iterations for surface expansion"})



    scheme: SurfaceDetectionScheme = field(default = SurfaceDetectionScheme.ColorField, metadata = {"help": "Surface detection scheme to use"})

    normalSource: NormalSource = field(default = NormalSource.ColorFieldGrad, metadata = {"help": "Source of normals to use for surface detection"})


def buildDefaultSurfaceDetectionConfig() -> SurfaceDetectionConfig:
    return SurfaceDetectionConfig(
        active = False,
        colorFieldThreshold = 0.75,
        colorFieldGradThreshold = 10.0,
        barecascoThreshold = np.pi/3,
        expansionIterations = 1,

        scheme = SurfaceDetectionScheme.Barecasco,
        normalSource = NormalSource.LambdaGrad
    )