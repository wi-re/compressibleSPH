"""Small standalone helpers with no scheme-specific dependencies: domain-box
construction (`domain`), the support-radius re-exports (`support`, documented
separately), a CUDA-aware `TimedBlock` profiling context manager (`timer`),
and this module's own timestamp/verbose-print/debug-print utilities used
throughout the case/runner/config layers.
"""

import datetime
import inspect
import re

from .domain import (DomainDescription, buildDomainDescription)
from .support import n_h_to_nH, volumeToSupport

def getCurrentTimestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def verbosePrint(verbose, *args):
    if verbose:
        print(*args)

def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))

__all__ = ['buildDomainDescription', 'DomainDescription', 'getCurrentTimestamp', 'verbosePrint', 'debugPrint', 'n_h_to_nH', 'volumeToSupport']