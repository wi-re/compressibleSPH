from ..systems import *
from ..configurations import *
from .compSPH import compSPH_step
from .deltaSPH import deltaSPH_step
from .crkSPH import crkSPH_step
from .monaghan import compressibleSPH_Monaghan
from ..enumTypes import CompressibleSPHScheme, WeaklyCompressibleSPHScheme
from typing import Union

def buildScheme(
    schemeName: Union[str, CompressibleSPHScheme, WeaklyCompressibleSPHScheme]
):
    if (isinstance(schemeName, str) and schemeName == 'MonaghanCompressibleSPH') or (isinstance(schemeName, CompressibleSPHScheme) and schemeName == CompressibleSPHScheme.Monaghan):
        return (
            CompressibleSystem, CompressibleState, CompressibleSPHConfig, CompressibleSystemUpdate, compressibleSPH_Monaghan, compressibleConfigToDict, dictToCompressibleConfig
        )
    elif (isinstance(schemeName, str) and schemeName == 'compSPH') or (isinstance(schemeName, CompressibleSPHScheme) and schemeName == CompressibleSPHScheme.CompSPH):
        return (
            CompSPHSystem, CompSPHState, CompSPHConfig, CompressibleSystemUpdate, compSPH_step, compSPHConfigToDict, dictToCompSPHConfig
        )
    elif (isinstance(schemeName, str) and schemeName == 'crkSPH') or (isinstance(schemeName, CompressibleSPHScheme) and schemeName == CompressibleSPHScheme.CRKSPH):
        return (
            CompSPHSystem, CompSPHState, CRKSPHConfig, CompressibleSystemUpdate, crkSPH_step, crkSPHConfigToDict, dictToCRKSPHConfig
        )
    elif (isinstance(schemeName, str) and schemeName == 'deltaSPH') or (isinstance(schemeName, WeaklyCompressibleSPHScheme) and schemeName == WeaklyCompressibleSPHScheme.deltaSPH):
        return (
            WeaklyCompressibleSystem, WeaklyCompressibleState, WeaklyCompressibleSPHConfig, CompressibleSystemUpdate, deltaSPH_step, weaklyCompressibleConfigToDict, dictToWeaklyCompressibleConfig
        )
    else:
        raise ValueError(f"Scheme {schemeName} not recognized.")