#!/usr/bin/env python
"""03 rotating square patch -- thin wrapper around the shared runner.

The notebook this came from is `03-rotating-square-patch.ipynb`; the case itself is
`warpSPH.cases.rotatingSquarePatch`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/weaklyCompressible/03-rotating-square-patch.py --scheme deltaSPH
    warpsph-run squarePatch --scheme deltaSPH            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.rotatingSquarePatch import rotatingSquarePatchCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = ['--scheme', 'deltaSPH']

if __name__ == '__main__':
    caseMain(rotatingSquarePatchCase, PRESET + sys.argv[1:])
