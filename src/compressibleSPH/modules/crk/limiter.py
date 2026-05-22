import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi, sphKernelScale
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters, getCRK_j

@wp.func
def limiterVL(x: scalar_t):
    if x <= scalar_t(0.0):
        return scalar_t(0.0)
    # x = wp.min(x, scalar_t(1.0e6))
    vL = scalar_t(2.0) / (scalar_t(1.0) + x)
    # return x * vL*vL
    return torch.where(x > scalar_t(0.0), x * vL**scalar_t(2.0), scalar_t(0.0))

@wp.func
def sgn(x: scalar_t):
    if x >= scalar_t(0.0):
        return scalar_t(1.0)
    else:
        return scalar_t(-1.0)

@wp.func
def computeVanLeer(
    xij_ : vector(length=Any, dtype=scalar_t), # type: ignore
    DvDxi : matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    DvDxj: matrix(shape=(Any, Any), dtype=scalar_t) # type: ignore
):
    xij = scalar_t(0.5) * (xij_)
    # gradi = torch.einsum('na, na -> n', torch.einsum('nab, nb -> na', DvDxi, xij), xij)
    gradi = wp.dot(matmul(DvDxi, xij), xij)
    # gradj = torch.einsum('na, na -> n', torch.einsum('nab, nb -> na', DvDxj, xij), xij)
    gradj = wp.dot(matmul(DvDxj, xij), xij)

    # rif = gradj.sgn() * gradj.abs().clamp(min = 1e-30)
    # rif = wp.sign(gradj) * wp.max(wp.abs(gradj), scalar_t(1.0e-30))
    # rjf = gradi.sgn() * gradi.abs().clamp(min = 1e-30)
    # rjf = wp.sign(gradi) * wp.max(wp.abs(gradi), scalar_t(1.0e-30))
    # denom_i = wp.max(wp.abs(rif), scalar_t(1.0e-30))
    # denom_j = wp.max(wp.abs(rjf), scalar_t(1.0e-30))
    # if rif < scalar_t(0.0):
    #     denom_i = -denom_i
    # if rjf < scalar_t(0.0):
    #     denom_j = -denom_j
    # ri = gradi / denom_i
    # rj = gradj / denom_j
#   from spheral
#   const Scalar ri = gradi/(sgn(gradj)*max(1.0e-30, abs(gradj)));
#   const Scalar rj = gradj/(sgn(gradi)*max(1.0e-30, abs(gradi)));
    ri = gradi / (sgn(gradj) * wp.max(wp.abs(gradj), scalar_t(1.0e-30)))
    rj = gradj / (sgn(gradi) * wp.max(wp.abs(gradi), scalar_t(1.0e-30)))

    rij = wp.min(ri, rj)

    # rij = (gradi + 1e-30) / (gradj + 1e-30)
    phi = limiterVL(rij)
    return phi
    # return scalar_t(0.0)
    # return phi * scalar_t(0.0)


@wp.func
def crkLimiter(
    x_ij: vector(length=Any, dtype=scalar_t), # type: ignore
    hi: scalar_t,
    hj: scalar_t,
    kernel_int: wp.int32,
    dim: wp.int32
):
    w_xi = sphKernel_xi(kernel_int, dim)
    # xi = Kernel_xi(config['kernel'], particles.positions.shape[0])
    # eta_max = getSetConfig(config, 'CRKSPH', 'eta_max', 4.0)
    ks = sphKernelScale(kernel_int, dim)
    eta_max = w_xi
    # eta_max = 1.0


    # The logic here gets a bit messy because of h/H shenanigans in SPH.
    # In our code we define the support of a particle as the cut-off distance, i.e., the distance at which the kernel value goes to zero. 
    # In other codes, such as spheral, they define the cut-off radius as a mulitple of the smoothing scale, i.e., H = ks * h, where ks is the cut-off in terms of smoothing lengths.
    # That means that we are effectively storing H as a property of a particle instead of h.
    # Consequently when the CRK paper refers to eta_ij = r_ij / h, we need to convert this to our definition of support which gives us eta_ij = r_ij / H * ks
    # By scaling eta_crit instead 

    eta_crit = scalar_t(1.0)/scalar_t(4.0)# * ks #  * eta_max
    eta_fold = scalar_t(0.2) #/ ks
    eta_i = x_ij/hi#*ks #* eta_max
    eta_j = x_ij/hj#*ks #* eta_max
        
    eta_i_norm = safe_sqrt(wp.dot(eta_i, eta_i))
    eta_j_norm = safe_sqrt(wp.dot(eta_j, eta_j))
    eta_ij = wp.min(eta_i_norm, eta_j_norm)

    factor = scalar_t(1.0)
    if eta_ij < eta_crit:
        factor = wp.exp(- ((eta_ij - eta_crit)/eta_fold)**scalar_t(2.0))
    
    # return scalar_t(1.0)
    return factor