#!/usr/bin/env python
"""07 sedov taylor blastwave 2d -- thin wrapper around the shared runner.

The notebook this came from is `07-Sedov_Taylor_Blastwave_2D.ipynb`; the case itself is
`warpSPH.cases.sedov`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/07-sedov-taylor-blastwave-2d.py --dim 2 --nx 200
    warpsph-run sedov --dim 2 --nx 200            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.sedov import sedovCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot', '--dim', '2', '--nx', '200']

if __name__ == '__main__':
    caseMain(sedovCase, PRESET + sys.argv[1:])
