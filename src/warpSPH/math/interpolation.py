"""A torch-native, device-resident port of `scipy.interpolate.RegularGridInterpolator`
(linear method only), used to sample procedural noise fields at particle positions.

The scipy original is a host-only object: every query ships the position array
device -> host, interpolates on one CPU core, and ships the result back. In the
Kolmogorov forcing (`caseUtils/weaklyCompressible.py`) that happens once per
integrator stage and dominates the step -- at 155k particles it measured 16.4
ms/step inside the scipy call plus 9.1 ms in `aten::copy_` and 3.7 ms in
`cudaStreamSynchronize`, i.e. ~68% of host self-time, against a flat ~4 ms of
actual dispatch.

`RegularGridInterpolator` here keeps the sampled field as a torch tensor and
migrates it to the querying device *once*, caching the migrated copy per
(device, dtype); subsequent queries are pure device work (a `searchsorted` or a
divide per axis, then `2**dim` gathers) with no host round-trip and no sync.

Semantics follow scipy's linear method exactly, including the parts that are
easy to get subtly wrong:

* cell selection is `searchsorted(axis, x, side='left') - 1` clamped to
  `[0, len(axis) - 2]`, so a query exactly on a node picks the cell below it
  (either cell gives the same interpolated value, but the same one scipy picks
  gives the same rounding too);
* with `bounds_error = False` and `fill_value = None` the clamped cell plus an
  unclamped interpolation weight *linearly extrapolates* outside the grid,
  rather than clamping to the border value;
* `fill_value` is applied to points strictly outside `[axis[0], axis[-1]]` on
  any axis, and `bounds_error = True` raises on them.

Unlike scipy's, this interpolator is differentiable: gradients flow to the query
positions (piecewise-linearly, as the interpolant itself is) and to the field
values. Call sites that relied on scipy silently zeroing `d(noise)/d(x)` -- the
per-step forcing -- keep that behaviour by detaching their query positions
explicitly.

Arithmetic runs in the query's dtype rather than scipy's unconditional float64,
which keeps the query free of float64 traffic on the GPU. A float64 query matches
scipy to roundoff (~1e-14 measured on a 61x37 grid); a float32 query carries the
usual float32 grid-coordinate error, ~1e-6 absolute on the same grid, growing
with the number of cells the coordinate spans -- irrelevant for sampling a noise
field that was itself generated in float32, but worth knowing before using this
for anything that wants more than single precision.
"""

import math

import torch

__all__ = ['RegularGridInterpolator']


class RegularGridInterpolator:
    """Linear interpolation of `values` sampled on the rectilinear grid `points`.

    `points` is a sequence of `dim` strictly ascending 1D coordinate axes and
    `values` has shape `(len(points[0]), ..., len(points[dim-1]), *trailing)`;
    trailing dimensions are interpolated componentwise, as in scipy. Axes may be
    torch tensors, numpy arrays or plain sequences, and are held in float64
    internally so that the uniform-spacing fast path stays exact for a grid built
    by `linspace`.

    Calling the interpolator with an `(..., dim)` tensor of query points returns
    a `(..., *trailing)` tensor on the query's device and dtype.
    """

    def __init__(self, points, values, method = 'linear', bounds_error = True, fill_value = float('nan')):
        if method != 'linear':
            raise NotImplementedError(f'RegularGridInterpolator only implements the linear method, got {method!r}')

        values = torch.as_tensor(values)
        axes = [torch.as_tensor(axis).detach().to(torch.float64).flatten() for axis in points]

        self.dim = len(axes)
        if values.ndim < self.dim:
            raise ValueError(f'values has {values.ndim} dimensions, fewer than the {self.dim} grid axes')
        self.gridShape = tuple(values.shape[:self.dim])
        self.trailingShape = tuple(values.shape[self.dim:])

        for d, axis in enumerate(axes):
            if axis.numel() < 2:
                raise ValueError(f'axis {d} needs at least 2 points for linear interpolation, got {axis.numel()}')
            if axis.numel() != self.gridShape[d]:
                raise ValueError(f'axis {d} has {axis.numel()} points but values has {self.gridShape[d]} along that dimension')
            if not bool((axis[1:] > axis[:-1]).all()):
                raise ValueError(f'axis {d} must be strictly ascending')

        self.axes = axes
        # Flattened once here so a query is a single index_select per corner into a
        # (nGridPoints, nComponents) matrix, whatever dim and trailing shape are.
        self.values = values.reshape(-1, math.prod(self.trailingShape))
        self.bounds_error = bounds_error
        self.fill_value = fill_value

        strides = []
        stride = 1
        for size in reversed(self.gridShape):
            strides.append(stride)
            stride *= size
        self.strides = list(reversed(strides))

        # A `linspace` axis lets the cell index come out of a single multiply, which
        # is why the axes are kept in float64: the tolerance below is a couple of
        # float64 ulps, so a genuinely non-uniform axis still takes the searchsorted
        # path rather than silently landing in the wrong cell.
        self.uniform = []
        self.start = []
        self.invStep = []
        for axis in axes:
            step = axis[1:] - axis[:-1]
            mean = step.mean()
            self.uniform.append(bool(((step - mean).abs() <= 1e-9 * mean.abs()).all()))
            self.start.append(axis[0].item())
            self.invStep.append((1.0 / mean).item())

        self.lower = [axis[0].item() for axis in axes]
        self.upper = [axis[-1].item() for axis in axes]

        self._cache = {}

    def _resolve(self, device, dtype):
        """The field and axes on `device` in `dtype`, transferred at most once."""
        key = (device, dtype)
        resolved = self._cache.get(key)
        if resolved is None:
            resolved = (
                [axis.to(device = device, dtype = dtype) for axis in self.axes],
                self.values.to(device = device, dtype = dtype).contiguous(),
            )
            self._cache[key] = resolved
        return resolved

    def to(self, device = None, dtype = None):
        """Pre-place the field, so the first query does not pay the transfer."""
        # Normalised through an empty tensor so that a bare 'cuda' here resolves to the
        # same indexed device ('cuda:0') a query tensor reports, and hits the same cache
        # entry rather than making a second copy of the field.
        device = torch.empty(0, device = device).device if device is not None else self.values.device
        self._resolve(device, dtype if dtype is not None else self.values.dtype)
        return self

    def __call__(self, xi):
        x = torch.as_tensor(xi)
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f'query points must have shape (..., {self.dim}), got {tuple(x.shape)}')
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())

        query = x.reshape(-1, self.dim)
        axes, values = self._resolve(query.device, query.dtype)

        indices = []
        weights = []
        outside = None
        for d in range(self.dim):
            coord = query[:, d]
            size = self.gridShape[d]
            if self.uniform[d]:
                u = (coord - self.start[d]) * self.invStep[d]
                index = torch.floor(u).clamp(0, size - 2)
                frac = u - index
                index = index.to(torch.long)
            else:
                # side='left' minus one, matching scipy's cell choice on exact nodes.
                index = (torch.searchsorted(axes[d], coord.contiguous()) - 1).clamp(0, size - 2)
                lower = axes[d][index]
                frac = (coord - lower) / (axes[d][index + 1] - lower)
            indices.append(index)
            weights.append(frac)

            if self.bounds_error or self.fill_value is not None:
                beyond = (coord < self.lower[d]) | (coord > self.upper[d])
                outside = beyond if outside is None else (outside | beyond)

        if self.bounds_error and bool(outside.any()):
            raise ValueError('one of the requested xi is out of bounds in one of its dimensions')

        base = indices[0] * self.strides[0]
        for d in range(1, self.dim):
            base = base + indices[d] * self.strides[d]

        result = None
        for corner in range(2 ** self.dim):
            weight = None
            offset = 0
            for d in range(self.dim):
                if (corner >> d) & 1:
                    offset += self.strides[d]
                    factor = weights[d]
                else:
                    factor = 1 - weights[d]
                weight = factor if weight is None else weight * factor
            contribution = values.index_select(0, base + offset) * weight.unsqueeze(-1)
            result = contribution if result is None else result + contribution

        if self.fill_value is not None and outside is not None:
            result = torch.where(outside.unsqueeze(-1), torch.as_tensor(self.fill_value, device = result.device, dtype = result.dtype), result)

        return result.reshape(tuple(x.shape[:-1]) + self.trailingShape)
