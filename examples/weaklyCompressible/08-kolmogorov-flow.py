#!/usr/bin/env python
"""08 kolmogorov flow -- thin wrapper around the shared runner.

The notebook this came from is `08-kolmogorov-flow.ipynb`; the case itself is
`warpSPH.cases.kolmogorov`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/weaklyCompressible/08-kolmogorov-flow.py
    warpsph-run kolmogorov            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.kolmogorov import kolmogorovCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = []

if __name__ == '__main__':
    caseMain(kolmogorovCase, PRESET + sys.argv[1:])
