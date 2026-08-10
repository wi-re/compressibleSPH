#!/usr/bin/env python
"""11 driven square -- thin wrapper around the shared runner.

The notebook this came from is `11-driven-square.ipynb`; the case itself is
`warpSPH.cases.channelFlow`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/weaklyCompressible/11-driven-square.py
    warpsph-run drivenSquare            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.channelFlow import drivenSquareCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot']

if __name__ == '__main__':
    caseMain(drivenSquareCase, PRESET + sys.argv[1:])
