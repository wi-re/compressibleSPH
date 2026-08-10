#!/usr/bin/env python
"""14 triple point -- thin wrapper around the shared runner.

The notebook this came from is `14-Triple_point.ipynb`; the case itself is
`warpSPH.cases.triplePoint`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/14-triple-point.py --no-equalMass --nx 256
    warpsph-run triplePoint --no-equalMass --nx 256            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.triplePoint import triplePointCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = ['--no-equalMass', '--nx', '256']

if __name__ == '__main__':
    caseMain(triplePointCase, PRESET + sys.argv[1:])
