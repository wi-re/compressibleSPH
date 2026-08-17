"""Top-level Perlin (`generatePerlin`) and simplex (`generateSimplex`) noise
grid generators, an octave-summed wrapper (`generateOctaveNoise`, this
package's main entry point, re-exported through `warpSPH.math.noise`), and
a Voronoi-style smooth-value sampler (`sampleVoronoi`). `generateSimplex` is
only reached via `generateOctaveNoise(..., kind='simplex')` -- `'perlin'` is
the default and the only kind used anywhere in this repo, so the simplex path
was untested; it previously called `_init(seed)` (defined in `util.py`)
without importing it, an unconditional `NameError` on every call (fixed
2026-08-15 by adding the missing import).
"""

from .simplex1d import _noise1
from .simplex2d import _noise2
from .simplex3d import _noise3,_noise3periodic
from .simplex4d import _noise4

from .perlin import interpolant, perlinNoise1D, perlinNoise2D, perlinNoise3D
from .util import _init

import math

import numpy as np
from numba import prange
import torch

__all__ = ['generateOctaveNoise', 'sampleVoronoi', 'paddedNoiseResolution',
           'resampleNoise', 'octaveFrequencies']

def generatePerlin(shape, res, tileable, dim = 2, interpolant  = interpolant , seed =42 , device = 'cpu', dtype = torch.float32):
    if dim == 1:
        return perlinNoise1D(shape, res, tileable, interpolant = interpolant, rng = np.random.default_rng(seed=seed), device = device, dtype = dtype)
    if dim == 2:
        return perlinNoise2D([shape, shape], [res, res], [tileable, tileable], interpolant = interpolant, rng = np.random.default_rng(seed=seed), device = device, dtype = dtype)
    if dim == 3:
        return perlinNoise3D([shape, shape, shape], [res, res, res], [tileable, tileable, tileable], interpolant = interpolant, rng = np.random.default_rng(seed=seed), device = device, dtype = dtype)
     

def generateSimplex(shape, freq, dim = 2, seed = 42, device = 'cpu', dtype = torch.float32, tileable = False):    
    dx = 2/shape
    x = torch.linspace(-1 + dx / 2, 1 - dx/2, shape, device = device, dtype = dtype)
    y = torch.linspace(-1 + dx / 2, 1 - dx/2, shape, device = device, dtype = dtype)
    z = torch.linspace(-1 + dx / 2, 1 - dx/2, shape, device = device, dtype = dtype)
    perm, perm_grad = _init(seed)
    # print('generateSimplex', shape, freq, dim, seed, device, dtype, tileable)
    if not tileable:
        if dim == 1:
            xx = x
            noise = []        
            for point in xx.numpy():
                noise.append(_noise1(point * freq, perm))
            return x, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
        if dim == 2:
            # print('generateSimplexPP', shape, freq, dim, seed, device, dtype, tileable)
            xx,yy = torch.meshgrid(x,y, indexing = 'xy')
            p = torch.stack((xx,yy), axis = -1).flatten().reshape((-1,2)).numpy()
            noise = []        
            for point in p:
                noise.append(_noise2(point[0] * freq, point[1] * freq, perm))
            return xx, yy, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
        if dim == 3:
            xx,yy,zz = torch.meshgrid(x,y,z, indexing = 'xy')
            p = torch.stack((xx,yy,zz), axis = -1).flatten().reshape((-1,3)).numpy()
            noise = []        
            for point in p:
                noise.append(_noise3(point[0] * freq, point[1] * freq,point[2] * freq, perm, perm_grad))
            return xx, yy,zz, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
    else:
        if dim == 1:
            xx = np.sin(x.cpu().numpy() * np.pi)
            yy = np.cos(x.cpu().numpy() * np.pi)
            p = np.stack((xx,yy), axis = -1).flatten().reshape((-1,2))
            noise = []                
            for point in p:
#                 print(point.shape)
                noise.append(_noise2(point[0] * freq / np.pi, point[1] * freq / np.pi, perm))
            return x, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
        if dim == 2:
            # print('generateSimplexP', shape, freq, dim, seed, device, dtype, tileable)
            noise = np.zeros((x.shape[0], y.shape[0]))
            frequency = 1
            amplitude = 1
            xx,yy = torch.meshgrid(x,y, indexing = 'xy')
            x = x.numpy()
            y = y.numpy()
            for ix in prange(x.shape[0]):
                for iy in prange(y.shape[0]):
                    nx = np.cos(x[ix] * np.pi + np.pi) 
                    ny = np.cos(y[iy] * np.pi + np.pi)
                    nz = np.sin(x[ix] * np.pi + np.pi)
                    nw = np.sin(y[iy] * np.pi + np.pi)
                    noise[ix,iy] += _noise4(freq * nx / np.pi, freq * ny / np.pi, freq * nz / np.pi, freq * nw / np.pi, perm)
            return xx, yy, torch.tensor(noise, device = device, dtype = dtype)
        if dim == 3:
            raise Exception('Not implemented yet (noise is not periodic)')
            xx,yy,zz = torch.meshgrid(x,y,z, indexing = 'xy')
            p = torch.stack((xx,yy,zz), axis = -1).flatten().reshape((-1,3)).numpy()
            noise = []        
            ifreq = int(freq)
            for point in p:                
                noise.append(_noise3periodic(point[0] * ifreq, point[1] * ifreq,point[2] * ifreq, perm, perm_grad, w6 = ifreq, d6 = ifreq, h6 = ifreq))
            return xx, yy,zz, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
    

def octaveFrequencies(octaves, lacunarity = 2, baseFrequency = 1):
    """The integer lattice frequency each octave is generated at."""
    freqs = []
    freq = baseFrequency
    for _ in range(octaves):
        freqs.append(max(1, int(round(freq))))
        freq *= lacunarity
    return freqs


def paddedNoiseResolution(n, octaves = 4, lacunarity = 2, baseFrequency = 1):
    """The smallest grid resolution >= `n` that every octave can be generated at.

    `perlinNoise*D` lays a `res`-periodic gradient lattice over a `shape`-point
    grid using `shape // res` and `numpy.repeat`, so it only works when `shape` is
    an exact multiple of `res`; otherwise the ramp and gradient arrays come out
    different sizes and it dies several frames deep with a shape mismatch (e.g.
    "The size of tensor a (78) must match the size of tensor b (76)"). With the
    default 4 octaves at baseFrequency 2, lacunarity 2, the highest frequency is
    16, so only multiples of 16 were usable -- `nx=39` simply crashed.

    Rather than push that constraint onto callers, generation is padded up to the
    least common multiple of the octave frequencies and the field is resampled
    back down to `n` (see `resampleNoise`). An `n` that already satisfies the
    constraint is returned unchanged, so those paths stay bit-identical.
    """
    step = math.lcm(*octaveFrequencies(octaves, lacunarity, baseFrequency))
    return int(math.ceil(n / step) * step)


def resampleNoise(field, n, tileable = True):
    """Resample a `[m]*dim` noise field onto a `[n]*dim` grid.

    Both grids are cell-centred over the same extent -- the convention
    `perlinNoise*D` uses, `linspace(-1 + dx/2, 1 - dx/2, shape)` -- so a coarse
    cell centre generally falls between two fine samples, and the resampling is a
    separable linear interpolation applied one axis at a time.

    Index arithmetic wraps for a `tileable` field and clamps otherwise, which is
    what keeps a tileable field tileable across the seam: `F.interpolate` clamps
    at the border in both cases, leaving a discontinuity in noise meant to wrap.
    """
    m = field.shape[0]
    if m == n:
        return field
    u = (torch.arange(n, device = field.device, dtype = torch.float64) + 0.5) * m / n - 0.5
    i0 = torch.floor(u)
    frac = (u - i0).to(field.dtype)
    i0 = i0.to(torch.long)
    i1 = i0 + 1
    if tileable:
        i0, i1 = i0 % m, i1 % m
    else:
        i0, i1 = i0.clamp(0, m - 1), i1.clamp(0, m - 1)

    for axis in range(field.ndim):
        shape = [1] * field.ndim
        shape[axis] = n
        w = frac.reshape(shape)
        field = (field.index_select(axis, i0) * (1 - w)
                 + field.index_select(axis, i1) * w)
    return field


def generateOctaveNoise(n, dim = 2, octaves = 4, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', device = 'cpu', dtype = torch.float32, seed = 12345, normalized = True):
    """Octave-summed noise on a `[n]*dim` grid.

    `n` is unconstrained: generation happens at `paddedNoiseResolution(n, ...)`,
    the smallest resolution every octave's gradient lattice divides, and the
    summed field is resampled down to `n`. When `n` already satisfies that
    constraint nothing is padded or resampled and the result is exactly what it
    has always been.
    """
    nGen = paddedNoiseResolution(n, octaves, lacunarity, baseFrequency)

    freq = baseFrequency
    amplitude = 1
    noise = torch.zeros([nGen] * dim, device = device, dtype = dtype)
    for i in range(octaves):
        result = generatePerlin(nGen, freq, dim = dim, tileable = tileable, device = device, dtype = dtype, seed = seed) if kind == 'perlin' else generateSimplex(nGen, freq = freq, dim = dim, tileable = tileable, device = device, dtype = dtype, seed = seed)
        noise += amplitude * result[-1]
        freq *= lacunarity
        amplitude *= persistence

    coords = result[:-1]
    if nGen != n:
        noise = resampleNoise(noise, n, tileable = tileable)
        # The generator returns its coordinate grids at nGen; rebuild them at n
        # on the same cell-centred convention, so a caller that reads them
        # (sampleVoronoi) stays consistent with the field they accompany.
        dx = 2 / n
        axis = torch.linspace(-1 + dx / 2, 1 - dx / 2, n, device = device, dtype = dtype)
        coords = torch.meshgrid(*([axis] * dim), indexing = 'xy')
    # Normalisation stays last, so the returned field still spans [-1, 1] after
    # resampling has smoothed the extremes.
    if normalized:
        noise = (noise  - torch.min(noise)) / (torch.max(noise) - torch.min(noise)) * 2 - 1
    return *coords, noise


from ..interpolation import RegularGridInterpolator
from .. import getPeriodicPositions

def sampleVoronoi(positions, nGrid, octaves = 2, baseFrequency = 1, kind = 'perlin', tileable=True, seed = 12365, vmin = 0.0, vmax = 1.0, config=None):
    """Sample an octave-noise field, rescaled to `[vmin, vmax]`, at `positions`.

    The field is laid out over `[-1, 1]^2` regardless of the actual domain
    extents, so positions are wrapped into the domain first and are expected to
    fall in that box; anything outside it raises, as it did under scipy.
    Interpolation runs on the positions' device (it used to round-trip through
    the host and unconditionally return a CUDA tensor, which meant this function
    could not run on a CPU-only machine at all).
    """
    positions = getPeriodicPositions(positions, config['domain'])
    xx, yy , noise = generateOctaveNoise(n = nGrid * 4, dim = 2, octaves = octaves, baseFrequency = baseFrequency, kind = kind, tileable=tileable, seed = seed)
    cTarget = noise / 2 + 0.5
    cTarget = vmin + (vmax - vmin) * cTarget
    axes = [torch.linspace(-1, 1, cTarget.shape[d], dtype = torch.float64) for d in range(2)]
    cInterp = RegularGridInterpolator(axes, cTarget)
    return cInterp(positions).to(positions.dtype)