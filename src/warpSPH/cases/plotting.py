"""Plot hooks the converted example cases share.

Every 2D example notebook built the same `warpSPHPlotting.visualize` call --
one or two particle fields, a mosaic, `export` to `frame_NNNNN.png` -- differing
only in which fields and which colour maps. :func:`particlePlot` is that call as
data, so a case declares the fields and gets a matching `setupPlot` /
`updatePlot` pair back.

The 1D compressible examples instead scattered a few state fields against `x`;
:func:`profilePlot` is that, with an optional analytic overlay.

The plotting backend is chosen by dimension -- **vispy for 2D**, matplotlib for
1D -- because a matplotlib scatter of a large 2D particle set costs more per
frame than the step it is drawing. ``--plotBackend`` overrides, and a vispy
canvas that cannot start (no GL context over ssh, in a container) falls back to
matplotlib rather than taking the run down with it.

Plots are *live* by default when run from a console: :func:`openWindow` puts
matplotlib into interactive mode and shows the figure, and :func:`pumpEvents`
gives the GUI toolkit a chance to repaint after every redraw. Without those two
calls a script builds the figure, writes its PNGs and never opens a window --
which is what a notebook gets away with, because the notebook frontend displays
the figure for it. ``--no-show`` turns the window off and keeps the frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..runner import RunContext

__all__ = ['Field', 'particlePlot', 'profilePlot', 'ProfileAxis', 'figureTitle',
           'openWindow', 'pumpEvents', 'holdWindow', 'closeWindow', 'figureOf']


# -- live display -----------------------------------------------------------
# Implemented in `warpSPH.runner.display` so the runner can tear a figure down
# without importing from `warpSPH.cases`; re-exported here because that is
# where a case looks for them.

from ..runner.display import (closeWindow, figureOf,  # noqa: E402
                              holdWindow, openWindow, pumpEvents,
                              resolvePlotBackend, visualizeWithFallback)


@dataclass
class Field:
    """One panel of a particle plot: what to colour by, and how."""

    #: Attribute of the particle state, e.g. ``'densities'``.
    quantity: str
    title: str
    colorMap: str = 'viridis'
    #: Which colour-map family `colorMap` names.
    colorMapKind: str = 'uniform'          # 'uniform' | 'diverging' | 'cyclic'
    flip: bool = False
    scaling: str = 'Linear'                # 'Linear' | 'Logarithmic'
    mapping: str = 'none'                  # 'none' | 'L2Norm' | 'x' | 'y'
    vMin: Optional[float] = None
    vMax: Optional[float] = None
    midPoint: Optional[float] = 0.0
    #: Resolution of the interpolated-grid rendering; `None` scatters particles.
    gridResolution: Optional[int] = None

    def tensor(self, state) -> torch.Tensor:
        return getattr(state.state, self.quantity)


def _plotOptions(field: Field, markerSize: float):
    from warpSPHPlotting import (CyclicColorMap, DivergingColorMap, GridVisualization,
                                 Mapping, PlotScaling, PlottingOptions, UniformColorMap)

    families = {'uniform': UniformColorMap, 'diverging': DivergingColorMap,
                'cyclic': CyclicColorMap}
    family = families[field.colorMapKind]
    return PlottingOptions(
        colorMap=getattr(family, field.colorMap),
        flipColorMap=field.flip,
        markerSize=markerSize,
        midPoint=field.midPoint,
        quantityScaling=getattr(PlotScaling, field.scaling),
        mapping=getattr(Mapping, field.mapping),
        plotTitle=field.title,
        vMin=field.vMin,
        vMax=field.vMax,
        gridVisualization=(GridVisualization(resolution=field.gridResolution)
                           if field.gridResolution else None),
    )


def figureTitle(ctx: RunContext, state, row: Optional[Dict[str, float]] = None) -> str:
    """The `t = ..., dt = ..., ptcls = ...` banner every notebook wrote."""
    parts = [f'{ctx.case.name}  t = {float(state.t):.4g}',
             f'dt = {float(ctx.config.dt):.3g}',
             f'ptcls = {len(state.state.positions)}']
    if row:
        parts += [f'{k} = {v:.4g}' for k, v in row.items()]
    return ' | '.join(parts)


def particlePlot(fields: Sequence[Field], figsize: Tuple[float, float] = (11, 5),
                 dpi: int = 300) -> Tuple[Callable, Callable]:
    """`(setupPlot, updatePlot)` rendering `fields` side by side.

    Panels are keyed 'A', 'B', ... in order, which is exactly the mosaic string
    the notebooks passed.
    """
    keys = [chr(ord('A') + i) for i in range(len(fields))]

    def setupPlot(ctx: RunContext, state):
        markerSize = ctx.param('markerSize', 2)
        plotter = visualizeWithFallback(
            ctx, resolvePlotBackend(ctx),
            particleState=state.state,
            domain=ctx.config.domain,
            quantities={k: f.tensor(state) for k, f in zip(keys, fields)},
            plotOptions={k: _plotOptions(f, markerSize) for k, f in zip(keys, fields)},
            figTitle=figureTitle(ctx, state),
            mosaic=''.join(keys),
            figsize=figsize,
        )
        _export(ctx, plotter, 0, dpi)
        openWindow(ctx, plotter)
        return plotter

    def updatePlot(ctx: RunContext, state, plotter, step: int) -> None:
        plotter.updateQuantities(
            {k: f.tensor(state) for k, f in zip(keys, fields)},
            newParticleState=state.state,
        )
        # The notebooks never refreshed the title, so every frame after the
        # first showed t = 0 -- which makes the encoded video misleading about
        # what it is showing.
        plotter.updateTitle(figureTitle(ctx, state))
        _export(ctx, plotter, step, dpi)
        pumpEvents(plotter)

    return setupPlot, updatePlot


def _export(ctx: RunContext, plotter, step: int, dpi: int) -> None:
    if ctx.imagePath:
        plotter.export(os.path.join(ctx.imagePath, f'frame_{step:05d}.png'), dpi=dpi)


@dataclass
class ProfileAxis:
    """One panel of a 1D profile plot."""

    #: Attribute of the particle state to plot against ``x``.
    quantity: str
    title: str
    #: Component to take when the quantity is a vector (velocities).
    component: Optional[int] = None
    yscale: str = 'linear'
    ylim: Optional[Tuple[float, float]] = None
    #: Horizontal reference lines, as ``(ctx, state) -> [y, ...]``.
    hlines: Optional[Callable[[RunContext, Any], List[float]]] = None
    #: Vertical reference lines, same shape.
    vlines: Optional[Callable[[RunContext, Any], List[float]]] = None
    #: Analytic overlay, as ``(ctx, state) -> (x, y)`` or `None`.
    reference: Optional[Callable[[RunContext, Any], Optional[Tuple[Any, Any]]]] = None


def profilePlot(axes: Sequence[ProfileAxis], shape: Tuple[int, int],
                figsize: Tuple[float, float] = (10, 6), dpi: int = 150,
                xlim: Optional[Tuple[float, float]] = None) -> Tuple[Callable, Callable]:
    """`(setupPlot, updatePlot)` scattering 1D state fields against position."""
    rows, cols = shape

    def draw(ctx: RunContext, state, handle):
        import matplotlib.pyplot as plt  # noqa: F401  (backend already chosen)

        fig, axis = handle
        flat = axis.flatten()
        positions = state.state.positions.detach().cpu().numpy()[:, 0]
        for ax, spec in zip(flat, axes):
            ax.clear()
            values = getattr(state.state, spec.quantity).detach().cpu().numpy()
            if spec.component is not None and values.ndim > 1:
                values = values[:, spec.component]
            values = values.reshape(len(positions), -1)[:, 0]
            ax.scatter(positions, values, s=1)
            if spec.reference is not None:
                reference = spec.reference(ctx, state)
                if reference is not None:
                    ax.plot(reference[0], reference[1], color='black', ls=':',
                            label='analytic')
            for y in (spec.hlines(ctx, state) if spec.hlines else []):
                ax.axhline(y, color='red', ls='--', alpha=0.4)
            for x in (spec.vlines(ctx, state) if spec.vlines else []):
                ax.axvline(x, color='black', ls=':', alpha=0.4)
            ax.set_title(spec.title)
            ax.set_yscale(spec.yscale)
            if spec.ylim:
                ax.set_ylim(*spec.ylim)
            if xlim:
                ax.set_xlim(*xlim)
        fig.suptitle(figureTitle(ctx, state))
        fig.tight_layout()

    def setupPlot(ctx: RunContext, state):
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        ctx.scratch['plotBackend'] = 'matplotlib'   # profile plots are axes-based
        handle = (fig, axis)
        draw(ctx, state, handle)
        _save(ctx, fig, 0, dpi)
        openWindow(ctx, handle)
        return handle

    def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
        fig, _ = handle
        draw(ctx, state, handle)
        _save(ctx, fig, step, dpi)
        pumpEvents(handle)

    return setupPlot, updatePlot


def _save(ctx: RunContext, fig, step: int, dpi: int) -> None:
    if ctx.imagePath:
        fig.savefig(os.path.join(ctx.imagePath, f'frame_{step:05d}.png'), dpi=dpi)
