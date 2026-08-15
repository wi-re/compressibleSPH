"""Particle-position and initial-velocity randomization utilities used by case setup.

Covers Perlin/Simplex-noise-driven divergence-free velocity fields and
delta-shift-based positional jittering for de-correlating initial lattices.
"""

from .sampleDivergenceFree import sampleDivergenceFreeNoise
from .shuffleParticles import shuffleParticles

__all__ = [
    "sampleDivergenceFreeNoise",
    "shuffleParticles",
]