import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters, getCRK_j

@wp.func
def limiterVL(x: wp.float32):
    if x <= 0.0:
        return 0.0
    x = wp.min(x, 1.0e6)
    vL = 2.0 / (1.0 + x)
    return x * vL*vL
    # return torch.where(x > 0, x * vL**2, 0.0)

@wp.func
def computeVanLeer(
    xij_ : vector(length=Any, dtype=wp.float32), # type: ignore
    DvDxi : matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
    DvDxj: matrix(shape=(Any, Any), dtype=wp.float32) # type: ignore
):
    xij = 0.5 * (xij_)
    # gradi = torch.einsum('na, na -> n', torch.einsum('nab, nb -> na', DvDxi, xij), xij)
    gradi = wp.dot(matmul(DvDxi, xij), xij)
    # gradj = torch.einsum('na, na -> n', torch.einsum('nab, nb -> na', DvDxj, xij), xij)
    gradj = wp.dot(matmul(DvDxj, xij), xij)

    # rif = gradj.sgn() * gradj.abs().clamp(min = 1e-30)
    rif = wp.sign(gradj) * wp.max(wp.abs(gradj), 1e-30)
    # rjf = gradi.sgn() * gradi.abs().clamp(min = 1e-30)
    rjf = wp.sign(gradi) * wp.max(wp.abs(gradi), 1e-30)
    denom_i = wp.max(wp.abs(rif), 1e-30)
    denom_j = wp.max(wp.abs(rjf), 1e-30)
    if rif < 0.0:
        denom_i = -denom_i
    if rjf < 0.0:
        denom_j = -denom_j
    ri = gradi / denom_i
    rj = gradj / denom_j
    rij = wp.min(ri, rj)

    # rij = (gradi + 1e-30) / (gradj + 1e-30)
    phi = limiterVL(rij)
    return phi

@wp.func
def computeCrkSPHAccel_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = wp.float32, length=Any), hi: wp.float32, mi: wp.float32, rhoi: wp.float32, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any,
    # Domain and kernel parameters
    # periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    # Operation specific parameters
    gradientMode_int: wp.int32, # type: ignore
            
    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: wp.float32,   # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: wp.float32,  # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: wp.float32, Bi: vector(length=Any, dtype=wp.float32), gradAi: vector(length=Any, dtype=wp.float32), gradBi: matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
    
    vel_i: vector(length=Any, dtype=wp.float32), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    u_i: wp.float32, referenceEnergies: wp.array(dtype = wp.float32), # type: ignore

    cs_i: wp.float32, referenceCs: wp.array(dtype = wp.float32), # type: ignore
    viscositySwitch: wp.bool, alpha_i: wp.float32, referenceAlphas: wp.array(dtype = wp.float32), # type: ignore
    P_i: wp.float32, referencePressures: wp.array(dtype = wp.float32), # type: ignore
    viscosityParams: DiffusionParameters,

    gradV_i: matrix(shape=(Any, Any), dtype=wp.float32), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)),# type: ignore

    # Dummy value to allow allocation
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
):
    pressureTerm_i = P_i / (rhoi*rhoi )/ (omega_i if useGradHTerms else wp.float32(1.0))

    # Initialize the output value
    out = zero_like_warp(pressureAccel_ij[i])
    # # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        vel_j = referenceVelocities[j]
        u_j = referenceEnergies[j]

        _, Vj = getVolume_j(correctionData, j)
        P_j = referencePressures[j]
        cs_j = referenceCs[j]


        x_ij = computeDistanceVec(xi, xj, domainState.periodicity, domainState.domainMin, domainState.domainMax)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        gradV_j = referenceVelocityTensor[j]

        phi_ij = 1.0
        # we then have the eta terms that depends on the 'r'_ij terms which are not the distances!
        # vx_ij = (del_b v_i^a x_ij^a x_ij^b) / (del_b v_j^a x_ij^a x_ij^b)
        w_xi = sphKernel_xi(kernel_int, dim)
        # xi = Kernel_xi(config['kernel'], particles.positions.shape[0])
        # eta_max = getSetConfig(config, 'CRKSPH', 'eta_max', 4.0)
        eta_max = w_xi
        # eta_max = 1.0
        eta_crit = 1.0/4.0  * eta_max
        eta_fold = 0.2  * eta_max
        eta_i = x_ij/hi * eta_max
        eta_j = x_ij/hj * eta_max
            
        eta_i_norm = safe_sqrt(wp.dot(eta_i, eta_i))
        eta_j_norm = safe_sqrt(wp.dot(eta_j, eta_j))
        eta_ij = wp.min(eta_i_norm, eta_j_norm)
    
        factor = wp.float32(1.0)
        if eta_ij < eta_crit:
            factor = wp.exp(- ((eta_ij - eta_crit)/eta_fold)**2.0)
        # torch.where(eta_ij < eta_crit, torch.exp(- ((eta_ij - eta_crit)/eta_fold)**2), torch.ones_like(eta_ij))

        phi_ij = computeVanLeer(
            x_ij,
            gradV_i,
            gradV_j
        ) * factor

        phi_ij = wp.max(wp.min(phi_ij, 1.0), 0.0) # Ensure phi is between 0 and 1

        # the term in crk is 
        # v_hat_ij = v_i - v_j - phi_ij/2 * (gradV_i + gradV_j) * x_ij
        # we utilize the more common form of v_ij = v_j - v_i
        # so the correction termw ith flipped signs becomes
        # v_hat_ij = v_j - v_i + phi_ij/2 * (gradV_i + gradV_j) * x_ij
        # commuting the correction term to each velocity gives
        # v_hat_ij = (v_j + phi_ij/2 * gradV_j * x_ij) - (v_i - phi_ij/2 * gradV_i * x_ij)
        # or
        # v_dot_i = v_i - phi_ij/2 * gradV_i * x_ij
        # v_dot_j = v_j + phi_ij/2 * gradV_j * x_ij
        v_corr_i = phi_ij / 2.0 * matmul(gradV_i, x_ij)
        v_corr_j = phi_ij / 2.0 * matmul(gradV_j, x_ij)

        v_dot_i = vel_i #+ v_corr_i
        v_dot_j = vel_j #- v_corr_j

        pi_i = computePi_actual(
            xi, xj, 
            hi, hj,
            mi, mj,
            rhoi, rhoj,
            True, P_i, P_j,
            v_dot_i, v_dot_j,
            domainState,
            kernel_int,
            cs_i, cs_j,
            alpha_i, referenceAlphas[j] if viscositySwitch else wp.float32(1.0),
            viscosityParams, 
            False, False)
        pi_j = computePi_actual(
            xi, xj, 
            hi, hj,
            mi, mj,
            rhoi, rhoj,
            True, P_i, P_j,
            v_dot_i, v_dot_j,
            domainState,
            kernel_int,
            cs_i, cs_j,
            alpha_i, referenceAlphas[j] if viscositySwitch else wp.float32(1.0),
            viscosityParams, 
            True, False)
        
        gradw_i = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernel_int, wp.uint32(12), # scatter mode for gradW
            domainState.periodicity, domainState.domainMin, domainState.domainMax,
            True, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_i = matmul(Li, gradw_i)

        _, Aj, Bj, gradAj, gradBj = getCRK_j(correctionData, j)
        gradw_j = computeKernelGradientCRK(
            xj, xi,
            hj, hi,
            kernel_int, wp.uint32(12), # scatter mode for gradW
            domainState.periodicity, domainState.domainMin, domainState.domainMax,
            True, Aj, Bj, gradAj, gradBj
        )
        if useGradientRenormalization:
            gradw_j = matmul(Li, gradw_j)

        gradw_ij = 0.5 * (gradw_i - gradw_j)
        # gradw_ij = gradw_i
        # gradw_j = gradw_i # E.2 in crksph suggests using the super symmetric form


        # omegaj = referenceOmegas[j] if useGradHTerms else wp.float32(1.0)
        # pressureTerm_j = Pj / (rhoj*rhoj) / omegaj
        
        u_ij = v_dot_j - v_dot_i
        mu_ij = wp.dot(u_ij, x_ij) / (r_ij + 1e-14 * hi)
        # mu_ij = ux_ij #/ (r_ij + 1e-14 * hi)

        # note that the term here should be multiplied with rho_i if it was a gradient operation
        # however, because we are computing the pressure force this cancel out with the division by rho_i in the pressure term, so we do not include it here
        # we do need to include the minus sign because the pressure force is -P gradW
        # Vj = referenceVolumes[j]
        # Vi = mi/rhoi
        # Vj = mj/rhoj
        pressureTerm_ij = -(P_i + P_j) * gradw_ij * Vi * Vj / mi

        # pressureTerms_i = - P_i * gradw_i * Vi * Vj / mi
        # pressureTerms_j = - P_j * gradw_j * Vi * Vj / mi
        # pressureTerm_ij = 0.5 * (pressureTerms_i + pressureTerms_j)

        rho_bar = 0.5 * (rhoi + rhoj)
        Q_i = pi_i * rho_bar / rhoj * rhoi
        Q_j = pi_j * rho_bar

        viscosityTerm_ij = -(Q_i + Q_j) * Vj * mu_ij * gradw_ij * Vi / mi# *0.0
        # viscosityTerms_i = - Q_i * Vj * mu_ij * gradw_i * Vi / mi
        # viscosityTerms_j = - Q_j * Vj * mu_ij * gradw_j * Vi / mi
        # viscosityTerm_ij = 0.5 * (viscosityTerms_i + viscosityTerms_j)
        # viscosityTerm_ij = -(pi_i + pi_j) * gradw_ij * Vi * Vj / mi * mu_ij

        viscosityAccel_ij[jj] = viscosityTerm_ij
        pressureAccel_ij[jj] = pressureTerm_ij
        out += pressureTerm_ij + viscosityTerm_ij
    return out

from sphWarpCore.operations_grid.grid_util import checkOffset

@wp.func
def computeCrkSPHAccel_Func_Adjacency(
    i : wp.int32, dim: wp.int32, 

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, opInt: wp.int32, 
    
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=wp.float32)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    queryEnergies: wp.array(dtype = wp.float32), referenceEnergies: wp.array(dtype = wp.float32), # type: ignore
    queryCs: wp.array(dtype = wp.float32), referenceCs: wp.array(dtype = wp.float32), # type: ignore
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = wp.float32), referenceAlphas: wp.array(dtype = wp.float32), # type: ignore
    queryPressures: wp.array(dtype = wp.float32), referencePressures: wp.array(dtype = wp.float32), # type: ignore
    viscosityParams: DiffusionParameters,
    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)),# type: ignore
    accel: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=wp.float32)) # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    vel_i = queryVelocities[i]

    cs_i = queryCs[i] 
    alpha_i = queryAlphas[i] if viscositySwitch else wp.float32(1.0)
    u_i = queryEnergies[i]
    P_i = queryPressures[i] 
    gradV_i = queryVelocityTensor[i]

    out = zero_like_warp(accel[i])
    for o in range(numOffsets):
        beginIndex = wp.int32(0)
        numIndices = wp.int32(0)
        if useAdjacency:    
            beginIndex = adjacencyState.neighborOffsets[i]
            numIndices = adjacencyState.numNeighbors[i]
        else:
            beginIndex, numIndices = checkOffset(
                i, queryState.positions, gridState.numCells, gridState.D, 
                o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
            )
            if beginIndex < 0:
                continue
        
        out += computeCrkSPHAccel_Func_i(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState,  correctionData, domainState,
            mode_uint, kernel_int, gradientMode_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i,
            useVolume, Vi , 
            useCRK, Ai, Bi, gradA_i, gradB_i,
            vel_i, referenceVelocities,
            u_i, referenceEnergies,
            cs_i, referenceCs,
            viscositySwitch, alpha_i, referenceAlphas,
            P_i, referencePressures,
            viscosityParams,
            gradV_i, referenceVelocityTensor,

            pressureAccel_ij, viscosityAccel_ij
        )
    return out



@wp.kernel
def computeCrkSPHAccel_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=wp.float32)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    queryEnergies: wp.array(dtype = wp.float32), referenceEnergies: wp.array(dtype = wp.float32), # type: ignore
    queryCs: wp.array(dtype = wp.float32), referenceCs: wp.array(dtype = wp.float32), # type: ignore
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = wp.float32), referenceAlphas: wp.array(dtype = wp.float32), # type: ignore
    queryPressures: wp.array(dtype = wp.float32), referencePressures: wp.array(dtype = wp.float32), # type: ignore
    viscosityParams: DiffusionParameters,

    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)),# type: ignore
    # The last parameter is always the output array and should not be changed
    accel : wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    accel[i] = computeCrkSPHAccel_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, gradientMode_int,  opInt, #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        queryEnergies, referenceEnergies,
        queryCs, referenceCs,
        viscositySwitch, queryAlphas, referenceAlphas,
        queryPressures, referencePressures,
        viscosityParams,
        queryVelocityTensor, referenceVelocityTensor,
        accel, pressureAccel_ij, viscosityAccel_ij
    )

def computeCrkSPHAccelWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    conductivityParams: DiffusionParameters,
    queryVelocityTensor: torch.Tensor, referenceVelocityTensor: Optional[torch.Tensor] = None,
    queryEnergies: Optional[torch.Tensor] = None, referenceEnergies: Optional[torch.Tensor] = None,
    queryVelocities : Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryCs: Optional[torch.Tensor] = None, referenceCs: Optional[torch.Tensor] = None,
    queryAlphas: Optional[torch.Tensor] = None, referenceAlphas: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None, referencePressures: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if adjacency is None or isinstance(adjacency, CompactHashMap):
        raise ValueError("This module requires an adjacency structure.")
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    if referenceVelocityTensor is None:
        referenceVelocityTensor = queryVelocityTensor
    with record_function("warpSPH[computeCrkSPHAccel]"):
        with record_function("warpSPH[computeCrkSPHAccel] - Preprocessing"):
            # Preprocessing and input validation
            device = queryParticles.positions.device
            # args, device, dim = parseArguments(
            #     queryParticles, operationProperties, domain,
            #     queryVolumes, referenceVolumes,
            #     adjacency,
            #     referenceParticles,
            #     crkState,
            #     gradHState,
            #     renormalizationState,
            # )

            outputSize = queryParticles.positions.shape[0]
            outputDtype = castTorchToWarpAsBuiltins(queryParticles.positions).dtype
            outputSizes = (
                queryParticles.positions.shape[0],
                adjacency.i.shape[0],
                adjacency.i.shape[0]
            )
            outputDtypes = (
                outputDtype,
                outputDtype,
                outputDtype
            )

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
            
            queryEnergies_ = queryEnergies if queryEnergies is not None else (queryParticles.internalEnergies if hasattr(queryParticles, 'internalEnergies') else None)
            queryVelocities_ = queryVelocities if queryVelocities is not None else (queryParticles.velocities if hasattr(queryParticles, 'velocities') else None)
            queryCs_ = queryCs if queryCs is not None else (queryParticles.soundspeeds if hasattr(queryParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=torch.float32, device=device))
            queryAlphas_ = queryAlphas if queryAlphas is not None else (queryParticles.alphas if hasattr(queryParticles, 'alphas') else getCachedDummyTensor((1,), dtype=torch.float32, device=device))
            queryPressures_ = queryPressures if queryPressures is not None else (queryParticles.pressures if hasattr(queryParticles, 'pressures') else getCachedDummyTensor((1,), dtype=torch.float32, device=device))

            referenceEnergies_ = referenceEnergies if referenceEnergies is not None else (referenceParticles.internalEnergies if hasattr(referenceParticles, 'internalEnergies') else None)
            referenceVelocities_ = referenceVelocities if referenceVelocities is not None else (referenceParticles.velocities if hasattr(referenceParticles, 'velocities') else None)
            referenceCs_ = referenceCs if referenceCs is not None else (referenceParticles.soundspeeds if hasattr(referenceParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=torch.float32, device=device))
            referenceAlphas_ = referenceAlphas if referenceAlphas is not None else (referenceParticles.alphas if hasattr(referenceParticles, 'alphas') else getCachedDummyTensor((1,), dtype=torch.float32, device=device))
            referencePressures_ = referencePressures if referencePressures is not None else (referenceParticles.pressures if hasattr(referenceParticles, 'pressures') else getCachedDummyTensor((1,), dtype=torch.float32, device=device))

            if queryAlphas is not None or (hasattr(queryParticles, 'alphas') and queryParticles.alphas is not None):
                viscositySwitch = True
            else:
                viscositySwitch = False
            if queryCs_ is None:
                raise ValueError("Sound speeds must be provided either through queryCs or as a property of queryParticles.")
            if queryPressures_ is None:
                raise ValueError("Pressures must be provided either through queryPressures or as a property of queryParticles.")
            if queryVolumes is None:
                raise ValueError("Volumes must be provided either through queryVolumes or as a property of queryParticles.")
            if crkState is None:
                raise ValueError("CRKState must be provided for CRKSPH computations.")
            if queryVelocities_ is None:
                raise ValueError("Velocities must be provided either through queryVelocities or as a property of queryParticles.")
            if queryEnergies_ is None:
                raise ValueError("Energies must be provided either through queryEnergies or as a property of queryParticles.")

        with record_function("warpSPH[computeCrkSPHAccel] - Kernel Execution"):
            return warpWrapper2(
                launcher = launch_kernel,
                kernel   = computeCrkSPHAccel_Kernel,
                numThreads = queryParticles.positions.shape[0],
                outputSizes  = outputSizes,
                outputDtypes = outputDtypes,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    queryVolumes, referenceVolumes,
                    adjacency,
                    referenceParticles,
                    crkState,
                    gradHState,
                    renormalizationState,
                ),
                additionalArguments=(
                    queryVelocities_, referenceVelocities_,
                    queryEnergies_, referenceEnergies_,
                    queryCs_, referenceCs_,
                    viscositySwitch, queryAlphas_, referenceAlphas_,
                    queryPressures_, referencePressures_,
                    conductivityParams,
                    queryVelocityTensor, referenceVelocityTensor,
                ),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeCrkSPHAccel_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
