#!/usr/bin/env python
"""Sod shock tube (1D) -- thin wrapper around the shared runner.

The 264 lines that used to live here (boilerplate cell, argparse block, step
loop, export, ffmpeg) are now `warpSPH.runner`, and the case itself is
`warpSPH.cases.sod`. Equivalent invocations::

    python examples/compressible/01-sod-shock-tube-1d.py --plot --store
    python -m warpSPH.cases.sod --plot --store
    warpsph-run sod --plot --store            # also honours --precision
"""

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.sod import sodCase          # noqa: E402
from warpSPH.runner import caseMain            # noqa: E402

if __name__ == '__main__':
    caseMain(sodCase)
