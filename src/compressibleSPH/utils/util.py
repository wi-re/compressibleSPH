
import datetime
def getCurrentTimestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def verbosePrint(verbose, *args):
    if verbose:
        print(*args)

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))