"""Compatibility layer for the weakly-compressible plotting helpers.

The source-of-truth implementation now lives in
`warpSPH.caseUtils.weaklyCompressiblePlot`.
"""

from warpSPH.caseUtils.weaklyCompressiblePlot import (
    buildPlotText,
    setupPlotter,
    updatePlot,
)

__all__ = ["buildPlotText", "setupPlotter", "updatePlot"]
