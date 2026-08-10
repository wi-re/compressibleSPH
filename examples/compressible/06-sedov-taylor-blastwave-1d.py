#!/usr/bin/env python
"""06 sedov taylor blastwave 1d -- thin wrapper around the shared runner.

The notebook this came from is `06-Sedov_Taylor_Blastwave_1D.ipynb`; the case itself is
`warpSPH.cases.sedov`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/06-sedov-taylor-blastwave-1d.py --dim 1 --nx 800
    warpsph-run sedov --dim 1 --nx 800            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.sedov import sedovCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = ['--dim', '1', '--nx', '800']

if __name__ == '__main__':
    caseMain(sedovCase, PRESET + sys.argv[1:])
