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
    else:
        raise ValueError(f"Scheme {schemeName} not recognized.")