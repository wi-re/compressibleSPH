from ...enumTypes import *
from torch.profiler import profile, record_function, ProfilerActivity

def stiffTaitEOS(rho, rho0: float, c_s: float, polytropicExponent: float):

    return rho0 * c_s**2 / polytropicExponent * ((rho / rho0)**polytropicExponent - 1)

def TaitEOS(rho, rho0: float, kappa: float):
    return kappa * (rho - rho0)

def isoThermalEOS(rho, rho0: float, c_s: float):
    return c_s**2 * (rho - rho0)

def polytropicEOS(rho, polytropicExponent : float, kappa : float):
    return kappa * (rho)**polytropicExponent

def murnaghanEOS(rho, rho0: float, kappa: float, exponent: float):
    return kappa / exponent * ((rho / rho0)**exponent - 1)


def weaklyCompressibleEOS(particleState, schemeConfig):
    with record_function("[warpSPH] - weaklyCompressibleEOS"):
        rho0 = schemeConfig.fluid.restDensity
        c_s = schemeConfig.fluid.fixedSoundSpeed
        
        eosType = schemeConfig.fluid.eosType
        kappa = schemeConfig.fluid.kappa
        polytropicExponent = schemeConfig.fluid.polytropicExponent

        rho = torch.clamp(particleState.densities, min=0.8)  # Avoid negative densities

        
        if eosType == EquationOfState.stiffTait:
            return stiffTaitEOS(rho, rho0, c_s, polytropicExponent)
        elif eosType == EquationOfState.Tait:
            return TaitEOS(rho, rho0, kappa)
        elif eosType == EquationOfState.isoThermal:
            return isoThermalEOS(rho, rho0, c_s)
        elif eosType == EquationOfState.Polytropic:
            return polytropicEOS(rho, polytropicExponent, kappa)
        elif eosType == EquationOfState.Murnaghan:
            return murnaghanEOS(rho, rho0, kappa, polytropicExponent)
        else:
            raise ValueError('EOS type not recognized')
