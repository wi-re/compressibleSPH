#!/usr/bin/env python
"""13 rayleigh taylor -- thin wrapper around the shared runner.

The notebook this came from is `13-Rayleigh_Taylor.ipynb`; the case itself is
`warpSPH.cases.rayleighTaylor`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/13-rayleigh-taylor.py
    warpsph-run rayleighTaylor            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.rayleighTaylor import rayleighTaylorCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = []

if __name__ == '__main__':
    caseMain(rayleighTaylorCase, PRESET + sys.argv[1:])
