from ..systems import *
from ..configurations import *
from .compSPH import compSPH_step
from .crkSPH import crkSPH_step
from .monaghan import compressibleSPH_Monaghan
from ..enumTypes import CompressibleSPHScheme
from typing import Union

def buildScheme(
    schemeName: Union[str, CompressibleSPHScheme]
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
        
    else:
        raise ValueError(f"Scheme {schemeName} not recognized.")