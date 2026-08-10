#!/usr/bin/env python
"""15 triple point equal mass -- thin wrapper around the shared runner.

The notebook this came from is `15-Triple_point_equalMass.ipynb`; the case itself is
`warpSPH.cases.triplePoint`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/15-triple-point-equal-mass.py --equalMass --nx 128
    warpsph-run triplePoint --equalMass --nx 128            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.triplePoint import triplePointCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = ['--equalMass', '--nx', '128']

if __name__ == '__main__':
    caseMain(triplePointCase, PRESET + sys.argv[1:])
