from ..configurations.region import RegionType, ParticleRegion, BCType
from .sample import sampleParticles
from .contour import find_contour
from typing import Optional

def buildRegion(
    config, schemeConfig,
    sdf: callable,
    regionType: RegionType = RegionType.Fluid,
    initialConditions: dict = None,
    nGrid: int = 255,
    nx: Optional[int] = None,
    kind: Optional[BCType] = BCType.zeros,
    shortEdge: bool = True
):
    nx_ = nx if nx is not None else config.nx

    return ParticleRegion(
        sdf = sdf,
        type = regionType,
        particles = sampleParticles(config, schemeConfig, sdf, nx_, shortEdge = shortEdge)[0],
        contour = find_contour(config, schemeConfig, sdf, nGrid) if config.dim == 2 else None,
        initialConditions = initialConditions,
        kind = kind 
    )
    

