#!/usr/bin/env python
"""11 shearing noh implosion 2d -- thin wrapper around the shared runner.

The notebook this came from is `11-Shearing_Noh_Implosion_2D.ipynb`; the case itself is
`warpSPH.cases.shearingNoh`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/11-shearing-noh-implosion-2d.py
    warpsph-run shearingNoh            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.shearingNoh import shearingNohCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
PRESET = []

if __name__ == '__main__':
    caseMain(shearingNohCase, PRESET + sys.argv[1:])
