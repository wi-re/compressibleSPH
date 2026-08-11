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