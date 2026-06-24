from dataclasses import dataclass, field

@dataclass
class SurfaceDetectionConfig:
    active: bool = field(default = True, metadata = {"help": "Whether to use surface detection or not"})
    colorFieldThreshold: float = field(default = 0.75, metadata = {"help": "Threshold for color field based surface detection as a fraction of target neighbors"})
    colorFieldGradThreshold: float = field(default = 10.0, metadata = {"help": "Threshold for color field gradient based surface detection"})

    expansionIterations: int = field(default = 1, metadata = {"help": "Number of iterations for surface expansion"})


def buildDefaultSurfaceDetectionConfig() -> SurfaceDetectionConfig:
    return SurfaceDetectionConfig(
        active = True,
        colorFieldThreshold = 0.75,
        colorFieldGradThreshold = 10.0,
        expansionIterations = 1
    )