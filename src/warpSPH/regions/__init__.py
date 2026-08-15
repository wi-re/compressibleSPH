"""Spatial regions: SDF-defined shapes that particles are sampled into or
filtered against (`buildRegion`, `sampleParticles`, `filterRegion`), the
default domain-box SDF (`domainSDF`), boundary-contour extraction for
plotting (`find_contour`), and region plotting (`plotRegions`). `inlet.py`
and `outlet.py` are currently-unimplemented placeholders for
`RegionType.Inlet`/`Outlet`-specific logic.
"""

from .contour import find_contour
from .domainSDF import domainSDF, sampleDomainSDF
from .filter import filterRegion
from .plot import plotRegions
from .region import buildRegion
from .sample import sampleParticles

__all__ = [
    'find_contour',
    'domainSDF',
    'sampleDomainSDF',
    'filterRegion',
    'plotRegions',
    'buildRegion',
    'sampleParticles'
]