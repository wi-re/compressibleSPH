#!/usr/bin/env python
"""02 impact squares -- thin wrapper around the shared runner.

The notebook this came from is `02-impact-squares.ipynb`; the case itself is
`warpSPH.cases.impact`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/weaklyCompressible/02-impact-squares.py --shape box --nx 128 --tLimit 1.5
    warpsph-run impact --shape box --nx 128 --tLimit 1.5            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.impact import impactCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = ['--shape', 'box', '--nx', '128', '--tLimit', '1.5']

if __name__ == '__main__':
    caseMain(impactCase, PRESET + sys.argv[1:])
