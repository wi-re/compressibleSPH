from .schemes.crkSPH import crkSPH_step

from .schemes import *
from .systems import *
from .configurations import *

def buildScheme(
    schemeName: str
):
    if schemeName == 'MonaghanCompressibleSPH':
        return (
            CompressibleSystem, CompressibleState, CompressibleSPHConfig, compressibleSPH_Monaghan
        )
    elif schemeName == 'compSPH':
        return (
            CompSPHSystem, CompSPHState, CompSPHConfig, compSPH_step
        )
    elif schemeName == 'crkSPH':
        return (
            CompSPHSystem, CompSPHState, CompSPHConfig, crkSPH_step
        )
    else:
        raise ValueError(f"Scheme {schemeName} not recognized.")