"""Channel flow past an obstacle (2D), weakly compressible.

The script forms of these cases were
`examples/weaklyCompressible/11-driven-square.ipynb` and `13-openFlow.ipynb`.

Both are the same construction the dam break already uses -- a `W x L` box, a
boundary region cut from an obstacle SDF, and a freestream forcing band at the
inflow -- assembled by the very same `caseUtils.weaklyCompressible` helpers.
So rather than a second copy of that machinery, these two are the dam break's
hooks under different defaults: no gravity, the box filled with fluid, and the
freestream switched on.

The notebooks reached the same place by hand, writing out the obstacle SDF, the
Dirichlet band and the inflow ramp inline; `13-openFlow.ipynb` had already
started importing `buildObstacleSDF` from the shared helpers.
"""

from __future__ import annotations

from .dambreak import (buildSystem, configureScheme, diagnostics, extraData,
                       initialConditions, setupPlot, updatePlot)
from ..runner import Case, caseMain, registerCase
from .dambreak import dambreakCase

__all__ = ['openFlowCase', 'drivenSquareCase', 'channelCase']


def channelCase(name: str, description: str, defaults, params) -> Case:
    """A dam-break-hooked case with channel defaults."""
    return registerCase(Case(
        name=name,
        scheme='deltaSPH',
        description=description,
        buildSystem=buildSystem,
        configureScheme=configureScheme,
        initialConditions=initialConditions,
        diagnostics=diagnostics,
        setupPlot=setupPlot,
        updatePlot=updatePlot,
        extraData=extraData,
        defaults=dict(dambreakCase.defaults, **defaults),
        params=dict(dambreakCase.params, **params),
    ))


openFlowCase = channelCase(
    'openFlow',
    'Open channel flow past an obstacle (2D), weakly compressible deltaSPH.',
    defaults=dict(
        caseName='13-openFlow',
        nx=128,
        L=2.0,
        tLimit=5.0,
    ),
    params=dict(
        W=4.0,
        band=5,
        # A partially filled, semi-periodic channel with a free surface: this
        # is what separates "open flow" from the closed driven square below.
        fillRatio=0.25,
        fluidWidth=1.0,
        semiPeriodic=True,
        disableGravity=False,
        enableFreestream=True,
        freeStreamVelocity=1.0,
        forcingWidth=2.0 / 16.0,
        obstacleActive=True,
        obstacleType='circleMiddle',
        maxExtent=0.125,
        offsetX=0.0,
    ),
)

drivenSquareCase = channelCase(
    'drivenSquare',
    'Driven channel flow past a cylinder (2D), weakly compressible deltaSPH.',
    defaults=dict(
        caseName='11-drivenSquare',
        nx=256,
        L=2.0,
        tLimit=10.0,
    ),
    params=dict(
        W=8.0,
        band=0,
        # Fully flooded and gravity-free, so the only driver is the freestream.
        fillRatio=1.0,
        fluidWidth=1.0,
        semiPeriodic=False,
        disableGravity=True,
        enableFreestream=True,
        freeStreamVelocity=1.0,
        forcingWidth=2.0 / 16.0,
        obstacleActive=True,
        obstacleType='circleMiddle',
        maxExtent=0.25,
        offsetX=0.0,
        targetDt=0.00025,
    ),
)


if __name__ == '__main__':
    caseMain(openFlowCase)
