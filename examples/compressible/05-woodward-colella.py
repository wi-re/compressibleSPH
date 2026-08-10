#!/usr/bin/env python
"""05 woodward colella -- thin wrapper around the shared runner.

The notebook this came from is `05-Woodward_Colella.ipynb`; the case itself is
`warpSPH.cases.woodwardColella`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/05-woodward-colella.py
    warpsph-run woodwardColella            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.woodwardColella import woodwardColellaCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = []

if __name__ == '__main__':
    caseMain(woodwardColellaCase, PRESET + sys.argv[1:])
