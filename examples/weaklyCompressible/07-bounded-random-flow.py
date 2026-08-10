#!/usr/bin/env python
"""07 bounded random flow -- thin wrapper around the shared runner.

The notebook this came from is `07-bounded-random-flow.ipynb`; the case itself is
`warpSPH.cases.randomFlow`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/weaklyCompressible/07-bounded-random-flow.py --bounded --nx 256
    warpsph-run randomFlow --bounded --nx 256            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.randomFlow import randomFlowCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = ['--bounded', '--nx', '256']

if __name__ == '__main__':
    caseMain(randomFlowCase, PRESET + sys.argv[1:])
