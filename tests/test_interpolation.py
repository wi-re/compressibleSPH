"""`warpSPH.math.interpolation.RegularGridInterpolator` against scipy's.

The torch interpolator replaced scipy's in the noise sampling path (see the
module docstring for why), so what matters is that it stays a faithful port:
the same cell choice, the same linear extrapolation when `fill_value is None`,
the same fill and the same raise otherwise. scipy is a declared dependency, so
it is used directly as the oracle rather than hand-rolling expected values.
"""

import numpy as np
import pytest
import torch

from warpSPH.math.interpolation import RegularGridInterpolator

scipyInterpolate = pytest.importorskip('scipy.interpolate')


def buildPair(axes, values, **kwargs):
    reference = scipyInterpolate.RegularGridInterpolator(axes, values, **kwargs)
    ported = RegularGridInterpolator([torch.as_tensor(axis) for axis in axes], torch.as_tensor(values), **kwargs)
    return reference, ported


def assertMatches(axes, values, points, tol = 1e-11, **kwargs):
    reference, ported = buildPair(axes, values, **kwargs)
    expected = reference(points)
    actual = ported(torch.as_tensor(points)).numpy()
    assert actual.shape == expected.shape
    assert np.array_equal(np.isnan(expected), np.isnan(actual))
    assert np.nanmax(np.abs(expected - actual)) < tol


@pytest.fixture
def rng():
    return np.random.default_rng(20260817)


@pytest.fixture
def uniformGrid(rng):
    axes = (np.linspace(-1.3, 2.7, 61), np.linspace(-0.5, 0.9, 37))
    return axes, rng.random((61, 37))


def test_matchesScipyInsideTheGrid(uniformGrid, rng):
    axes, values = uniformGrid
    points = np.stack([rng.uniform(-1.3, 2.7, 4096), rng.uniform(-0.5, 0.9, 4096)], -1)
    assertMatches(axes, values, points, bounds_error = False, fill_value = None)


def test_extrapolatesLinearlyLikeScipyWhenFillValueIsNone(uniformGrid, rng):
    axes, values = uniformGrid
    points = np.stack([rng.uniform(-3, 4, 4096), rng.uniform(-2, 2, 4096)], -1)
    # Loosened only because far extrapolation amplifies float64 roundoff; inside
    # the grid the two agree to ~1e-14 (test above).
    assertMatches(axes, values, points, tol = 1e-10, bounds_error = False, fill_value = None)


@pytest.mark.parametrize('fillValue', [float('nan'), 0.5])
def test_fillsOutOfBoundsLikeScipy(uniformGrid, rng, fillValue):
    axes, values = uniformGrid
    points = np.stack([rng.uniform(-3, 4, 4096), rng.uniform(-2, 2, 4096)], -1)
    assertMatches(axes, values, points, bounds_error = False, fill_value = fillValue)


def test_picksTheSameCellOnExactNodes(uniformGrid):
    axes, values = uniformGrid
    x, y = np.meshgrid(*axes, indexing = 'ij')
    nodes = np.stack([x.ravel(), y.ravel()], -1)
    assertMatches(axes, values, nodes, bounds_error = False, fill_value = None)


def test_matchesScipyOnNonUniformAxes(rng):
    axes = (np.sort(rng.uniform(-1, 1, 40)), np.sort(rng.uniform(0, 3, 25)))
    points = np.stack([rng.uniform(-1.5, 1.5, 2048), rng.uniform(-1, 4, 2048)], -1)
    assertMatches(axes, rng.random((40, 25)), points, tol = 1e-10, bounds_error = False, fill_value = None)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_matchesScipyInEveryDimension(rng, dim):
    axes = tuple(np.linspace(-1, 1 + d, 7 + d) for d in range(dim))
    points = np.stack([rng.uniform(-1.5, 2 + d, 512) for d in range(dim)], -1)
    assertMatches(axes, rng.random(tuple(len(a) for a in axes)), points, bounds_error = False, fill_value = None)


def test_interpolatesTrailingComponentsAndKeepsQueryShape(uniformGrid, rng):
    axes, _ = uniformGrid
    values = rng.random((61, 37, 3))
    points = np.stack([rng.uniform(-1.3, 2.7, 2048), rng.uniform(-0.5, 0.9, 2048)], -1)
    assertMatches(axes, values, points, bounds_error = False, fill_value = None)
    assertMatches(axes, values, points.reshape(32, 64, 2), bounds_error = False, fill_value = None)


def test_raisesOutOfBoundsByDefault(uniformGrid, rng):
    axes, values = uniformGrid
    _, ported = buildPair(axes, values)
    inside = torch.as_tensor(np.stack([rng.uniform(-1.3, 2.7, 64), rng.uniform(-0.5, 0.9, 64)], -1))
    ported(inside)
    with pytest.raises(ValueError, match='out of bounds'):
        ported(torch.as_tensor(np.stack([rng.uniform(3, 5, 64), rng.uniform(-0.5, 0.9, 64)], -1)))


def test_followsTheQueryDtypeAndDevice(uniformGrid, rng):
    axes, values = uniformGrid
    _, ported = buildPair(axes, values, bounds_error = False, fill_value = None)
    points = np.stack([rng.uniform(-1.3, 2.7, 256), rng.uniform(-0.5, 0.9, 256)], -1)

    assert ported(torch.as_tensor(points, dtype = torch.float32)).dtype == torch.float32
    assert ported(torch.as_tensor(points, dtype = torch.float64)).dtype == torch.float64
    if torch.cuda.is_available():
        query = torch.as_tensor(points, dtype = torch.float32, device = 'cuda')
        result = ported(query)
        assert result.device.type == 'cuda'
        # The point of the port: the field is resident on the device, not copied per call.
        assert ported._cache[(query.device, query.dtype)][1].device == query.device


def test_fieldIsTransferredOncePerDeviceAndDtype(uniformGrid, rng):
    axes, values = uniformGrid
    _, ported = buildPair(axes, values, bounds_error = False, fill_value = None)
    points = torch.as_tensor(np.stack([rng.uniform(-1, 2, 128), rng.uniform(-0.4, 0.8, 128)], -1), dtype = torch.float32)
    ported(points)
    cached = ported._cache[(points.device, points.dtype)][1]
    for _ in range(3):
        ported(points)
    assert ported._cache[(points.device, points.dtype)][1] is cached


def test_isDifferentiableUnlikeScipy(uniformGrid, rng):
    axes, values = uniformGrid
    _, ported = buildPair(axes, values, bounds_error = False, fill_value = None)
    points = torch.as_tensor(np.stack([rng.uniform(-1, 2, 32), rng.uniform(-0.4, 0.8, 32)], -1)).requires_grad_(True)
    ported(points).sum().backward()
    assert torch.isfinite(points.grad).all()
    assert (points.grad != 0).any()


def test_rejectsMalformedInputs(uniformGrid):
    axes, values = uniformGrid
    with pytest.raises(NotImplementedError):
        RegularGridInterpolator([torch.as_tensor(a) for a in axes], torch.as_tensor(values), method = 'cubic')
    with pytest.raises(ValueError, match='strictly ascending'):
        RegularGridInterpolator([torch.as_tensor(axes[0]).flip(0), torch.as_tensor(axes[1])], torch.as_tensor(values))
    with pytest.raises(ValueError, match='but values has'):
        RegularGridInterpolator([torch.as_tensor(axes[0])[:-1], torch.as_tensor(axes[1])], torch.as_tensor(values))
    _, ported = buildPair(axes, values, bounds_error = False, fill_value = None)
    with pytest.raises(ValueError, match=r'shape \(\.\.\., 2\)'):
        ported(torch.zeros(10, 3))
