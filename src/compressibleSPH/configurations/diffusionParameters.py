from sphWarpCore.diffusion.viscosity import DiffusionParameters, ViscosityTerms
from typing import Dict, Any

def buildDefaultDiffusionParamsCompressibleSPH():
    diffusionParams = DiffusionParameters()
    diffusionParams.c_s = 1
    diffusionParams.C_l = 1
    diffusionParams.C_q = 0
    diffusionParams.Cu_l = 1
    diffusionParams.Cu_q = 0
    diffusionParams.K = 1.0
    diffusionParams.thermalConductivity = 0.5
    diffusionParams.viscosityTerm = ViscosityTerms.Price2012_98.value
    diffusionParams.thermalConducitiyTerm = ViscosityTerms.Price2008.value
    diffusionParams.scaleBeta = False
    diffusionParams.monaghanSwitch = True
    diffusionParams.correctXi = True
    
    return diffusionParams



def diffusionParamsToDict(diffusionParams: DiffusionParameters) -> Dict[str, Any]:
    return {
        'c_s': diffusionParams.c_s,
        'C_l': diffusionParams.C_l,
        'C_q': diffusionParams.C_q,
        'Cu_l': diffusionParams.Cu_l,
        'Cu_q': diffusionParams.Cu_q,
        'K': diffusionParams.K,
        'thermalConductivity': diffusionParams.thermalConductivity,
        'viscosityTerm': diffusionParams.viscosityTerm.name if isinstance(diffusionParams.viscosityTerm, ViscosityTerms) else diffusionParams.viscosityTerm,
        'thermalConducitiyTerm': diffusionParams.thermalConducitiyTerm.name if isinstance(diffusionParams.thermalConducitiyTerm, ViscosityTerms) else diffusionParams.thermalConducitiyTerm,
        'scaleBeta': diffusionParams.scaleBeta,
        'monaghanSwitch': diffusionParams.monaghanSwitch,
        'correctXi': diffusionParams.correctXi
    }

def dictToDiffusionParams(diffusionParamsDict: Dict[str, Any]) -> DiffusionParameters:
    diffusionParams = DiffusionParameters()
    diffusionParams.c_s = diffusionParamsDict['c_s']
    diffusionParams.C_l = diffusionParamsDict['C_l']
    diffusionParams.C_q = diffusionParamsDict['C_q']
    diffusionParams.Cu_l = diffusionParamsDict['Cu_l']
    diffusionParams.Cu_q = diffusionParamsDict['Cu_q']
    diffusionParams.K = diffusionParamsDict['K']
    diffusionParams.thermalConductivity = diffusionParamsDict['thermalConductivity']
    diffusionParams.viscosityTerm = ViscosityTerms[diffusionParamsDict['viscosityTerm']] if isinstance(diffusionParamsDict['viscosityTerm'], str) else diffusionParamsDict['viscosityTerm']
    diffusionParams.thermalConducitiyTerm = ViscosityTerms[diffusionParamsDict['thermalConducitiyTerm']] if isinstance(diffusionParamsDict['thermalConducitiyTerm'], str) else diffusionParamsDict['thermalConducitiyTerm']
    diffusionParams.scaleBeta = diffusionParamsDict['scaleBeta']
    diffusionParams.monaghanSwitch = diffusionParamsDict['monaghanSwitch']
    diffusionParams.correctXi = diffusionParamsDict['correctXi']
    
    return diffusionParams