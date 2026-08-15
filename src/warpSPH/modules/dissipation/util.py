"""Per-formulation helpers used by `pi.computePi_actual`: `compute_mu_ij`
computes the velocity-divergence proxy mu_ij together with its
formulation-specific scaling factor, and `compute_bars` selects whether the
averaged ("bar") or one-sided (`i` or `j`, via `useJ`) density/soundspeed/
smoothing-length is used. Both dispatch on the same `ViscosityTerms` enum as
`pi.py`.
"""

from warpSPHCore import *
import warp as wp
from ...configurations.moduleConfigurations.diffusionParameters import ViscosityTerms

__all__ = ['compute_mu_ij', 'compute_bars']


@wp.func
def compute_mu_ij(
    ux_ij: scalar_t, r_ij: scalar_t, h: scalar_t, viscosityTerm: wp.int32, xi: scalar_t
):
    mu_ij = ux_ij / (r_ij + scalar_t(1e-14) * h) # Always start with this as the base

    scalingFactor = h / xi / (r_ij + scalar_t(1e-14) * h)
    scaled_mu_ij = mu_ij * scalingFactor
    
    if viscosityTerm == wp.static(ViscosityTerms.Default.value): # Default to Monaghan1992
        return scaled_mu_ij, h / xi
    elif viscosityTerm == wp.static(ViscosityTerms.MonaghanGingold1983.value): # MonaghanGingold1983
        return scaled_mu_ij, h / xi
    elif viscosityTerm == wp.static(ViscosityTerms.Cleary1998.value): # Cleary1998
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1992.value): # Monaghan1992
        return scaled_mu_ij, scalar_t(1.0) * scalingFactor
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997a.value): # Monaghan1997a
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997b.value): # Monaghan1997b
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Dukowicz.value): # Dukowicz
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012_98.value): # Price2012_98
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012.value): # Price2012
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2008.value): # Price2008
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Wadsley2008.value): # Wadsley2008
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.DeltaSPH.value): # DeltaSPH
        return scaled_mu_ij, h / xi
    else:
        return scaled_mu_ij, h / xi

@wp.func
def compute_bars(
    rho_i : scalar_t, rho_j : scalar_t, rho_bar : scalar_t, 
    c_i : scalar_t, c_j : scalar_t, c_bar : scalar_t,
    h_i : scalar_t, h_j : scalar_t, h_bar : scalar_t,
    viscosityTerm: wp.int32, useJ: bool
):
    use_rho_bar = wp.bool(False)
    use_c_bar = wp.bool(False)
    use_h_bar = wp.bool(False)

    if viscosityTerm == wp.static(ViscosityTerms.Default.value): # Default
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.MonaghanGingold1983.value): # MonaghanGingold1983
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Cleary1998.value): # Cleary1998
        use_rho_bar = False
        use_c_bar = False
        use_h_bar = False
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1992.value): # Monaghan1992
        use_rho_bar = True
        use_c_bar = False
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997a.value): # Monaghan1997a
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997b.value): # Monaghan1997b
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Dukowicz.value): # Dukowicz
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012_98.value): # Price2012_98
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012.value): # Price2012
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Price2008.value): # Price2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Wadsley2008.value): # Wadsley2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.DeltaSPH.value): # DeltaSPH
        use_rho_bar = False
        use_c_bar = True
        use_h_bar = True
    else:        
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True

    rho = scalar_t(scalar_t(0.0))
    c = scalar_t(scalar_t(0.0))
    h = scalar_t(scalar_t(0.0))

    if use_rho_bar:
        rho = rho_bar
    else:
        if useJ:
            rho = rho_j
        else:            
            rho = rho_i    

    if use_c_bar:
        c = c_bar
    else:
        if useJ:
            c = c_j
        else:            
            c = c_i

    if use_h_bar:
        h = h_bar
    else:
        if useJ:
            h = h_j
        else:            
            h = h_i

    return rho, c, h
