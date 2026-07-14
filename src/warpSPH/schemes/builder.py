from ..systems import *
from ..configurations import *
from .compSPH import compSPH_step
from .deltaSPH import deltaSPH_step
from .crkSPH import crkSPH_step
from .monaghan import compressibleSPH_Monaghan
from ..enumTypes import CompressibleSPHScheme, WeaklyCompressibleSPHScheme, IncompressibleSPHScheme
# from .dfsph import dfsph_step
from typing import Union

def buildScheme(
    schemeName: Union[str, CompressibleSPHScheme, WeaklyCompressibleSPHScheme, IncompressibleSPHScheme]
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
    elif (isinstance(schemeName, str) and schemeName == 'divergenceFree') or (isinstance(schemeName, IncompressibleSPHScheme) and schemeName == IncompressibleSPHScheme.divergenceFree):
        from .dfsph import dfsph_step, incompressibleConfigToDict, dictToIncompressibleSPHConfig, IncompressibleSystem, IncompressibleState, IncompressibleSystemUpdate
        return (
            IncompressibleSystem, IncompressibleState, IncompressibleSPHConfig, IncompressibleSystemUpdate, dfsph_step, incompressibleConfigToDict, dictToIncompressibleSPHConfig
        )
    else:
        raise ValueError(f"Scheme {schemeName} not recognized.")