"""Thin positional-argument wrapper around
`noiseFunctions.generator.generateOctaveNoise` (`generateNoise`), re-exporting
`generateOctaveNoise` itself; both are the entry points `warpSPH.math`
exposes for procedural noise generation.
"""

from .noiseFunctions.generator import generateOctaveNoise, sampleVoronoi
import torch

__all__ = ['generateNoise', 'generateOctaveNoise']

def generateNoise(n, dim = 2, octaves = 4, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', device = 'cpu', dtype = torch.float32, seed = 12345, normalized = True):
    return generateOctaveNoise(n, dim, octaves, lacunarity, persistence, baseFrequency, tileable, kind, device, dtype, seed, normalized)