import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi, sphKernelScale
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters, getCRK_j

from ...configurations.crkSPH import CRKViscosity

from .limiter import computeVanLeer, crkLimiter

@wp.func
def computeCrkSPHdudt_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    # Domain and kernel parameters
    # periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
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
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t, # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    u_i: scalar_t, referenceEnergies: wp.array(dtype = scalar_t), # type: ignore

    cs_i: scalar_t, referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, alpha_i: scalar_t, referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    P_i: scalar_t, referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    crkViscosityParams: CRKViscosity,

    gradV_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore


    # Dummy value to allow allocation
):
    pressureTerm_i = P_i / (rhoi*rhoi )/ (omega_i if useGradHTerms else scalar_t(1.0))

    # Initialize the output value
    out = scalar_t(0.0)
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

        gradV_j = referenceVelocityTensor[j]
        x_ij = computeDistanceVec(xi, xj, domainState.periodicity, domainState.domainMin, domainState.domainMax)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        phi_ij = scalar_t(0.0)
        # we then have the eta terms that depends on the 'r'_ij terms which are not the distances!
        # vx_ij = (del_b v_i^a x_ij^a x_ij^b) / (del_b v_j^a x_ij^a x_ij^b)
        factor = scalar_t(1.0)
        
        if crkViscosityParams.enableCRKLimiter:
            factor = crkLimiter(
                x_ij,
                hi,
                hj,
                kernel_int,
                dim,
                crkViscosityParams.eta_crit,
                crkViscosityParams.eta_fold
            )

        if crkViscosityParams.enableVanLeerLimiter:
            phi_ij = computeVanLeer(
                x_ij,
                vel_i,
                vel_j,
                gradV_i,
                gradV_j
            ) * factor

        if crkViscosityParams.forceVanLeerOff:
            phi_ij = scalar_t(0.0)
        if crkViscosityParams.forceVanLeerOn:
            phi_ij = scalar_t(1.0)
        phi_ij = wp.max(wp.min(phi_ij, scalar_t(1.0)), scalar_t(0.0)) # Ensure phi is between 0 and 1
        v_corr_i = phi_ij / scalar_t(2.0) * matmul(gradV_i, x_ij)
        v_corr_j = phi_ij / scalar_t(2.0) * matmul(gradV_j, -x_ij)

        v_dot_i = vel_i - v_corr_i
        v_dot_j = vel_j + v_corr_j

        pi_i = computePi_actual(
            xi, xj, 
            hi, hj,
            mi, mj,
            rhoi, rhoj,
            True, P_i, P_j,
            v_dot_i, v_dot_j,
            domainState,
            kernel_int,
            cs_i, cs_i,
            alpha_i, referenceAlphas[j] if viscositySwitch else scalar_t(1.0),
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
            cs_j, cs_j,
            alpha_i, referenceAlphas[j] if viscositySwitch else scalar_t(1.0),
            viscosityParams, 
            True, False)
        
        gradw_i = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernel_int, wp.uint32(wp.static(SupportScheme.Scatter.value)), # match accel.py
            domainState.periodicity, domainState.domainMin, domainState.domainMax,
            True, Ai, Bi, gradAi, gradBi  # useCRK=True: must match accel.py gradient
        )
        if useGradientRenormalization:
            gradw_i = matmul(Li, gradw_i)

        _, Aj, Bj, gradAj, gradBj = getCRK_j(correctionData, j)
        gradw_j = computeKernelGradientCRK(
            xj, xi,
            hj, hi,
            kernel_int, wp.uint32(wp.static(SupportScheme.Scatter.value)), # match accel.py
            domainState.periodicity, domainState.domainMin, domainState.domainMax,
            True, Aj, Bj, gradAj, gradBj  # useCRK=True: must match accel.py gradient
        )
        if useGradientRenormalization:
            gradw_j = matmul(Li, gradw_j)

        smooth_i = hi / sphKernelScale(kernel_int, dim)
        smooth_j = hj / sphKernelScale(kernel_int, dim)

        eta_i = x_ij / smooth_i
        eta_j = x_ij / smooth_j

        vij_dot = v_dot_i - v_dot_j
        mu_ij = (wp.dot(vij_dot, eta_i)) / (wp.dot(eta_i, eta_i) + scalar_t(1.0e-7) * smooth_i * smooth_i)
        mu_ji = (wp.dot(vij_dot, eta_j)) / (wp.dot(eta_j, eta_j) + scalar_t(1.0e-7) * smooth_j * smooth_j)

        mu_ij = wp.min(scalar_t(0.0), mu_ij)
        mu_ji = wp.min(scalar_t(0.0), mu_ji)

        Cl = viscosityParams.C_l
        Cq = viscosityParams.C_q        
    
        Q_i = rhoi * (-Cl * cs_i * mu_ij + Cq * mu_ij * mu_ij)
        Q_j = rhoj * (-Cl * cs_j * mu_ji + Cq * mu_ji * mu_ji)

        # gradw_ij = 0.5*(gradw_i - gradw_j) mirrors accel.py's 0.5*(gradw_i + (-gradw_j))
        # same deltagrad as the force — required for energy-momentum consistency
        gradw_ij = scalar_t(0.5) * (gradw_i - gradw_j)

        Pj = referencePressures[j]
        # omegaj = referenceOmegas[j] if useGradHTerms else scalar_t(1.0)
        # pressureTerm_j = Pj / (rhoj*rhoj) / omegaj
        
        u_ij = v_dot_j - v_dot_i
        ux_ij = wp.dot(u_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi)
        mu_ij = ux_ij #/ (r_ij + scalar_t(1.0e-14) * hi)
        mu_ij = scalar_t(1.0)

        # note that the term here should be multiplied with rho_i if it was a gradient operation
        # however, because we are computing the pressure force this cancel out with the division by rho_i in the pressure term, so we do not include it here
        # we do need to include the minus sign because the pressure force is -P gradW


        u_ij = vel_i - vel_j
        dot = wp.dot(u_ij, gradw_ij)
        dot_ij = wp.dot(u_ij, gradw_i)
        dot_ji = wp.dot(u_ij, gradw_j)

        rho_bar = scalar_t(0.5) * (rhoi + rhoj)
        Pi = P_i
        Pj = P_j
        # Q_i = pi_i * rho_bar / rhoj * rhoi
        # Q_j = pi_j * rho_bar

        pTerm = Pj * dot * Vi * Vj / mi
        vTerm = scalar_t(1.0/2.0) * (Q_i + Q_j) * mu_ij * dot * Vi * Vj / mi

        # apparentVolume = mj/rhoj
        # pTerm = - apparentVolume * pressureTerm_i * rhoj * dot
        # vTerm = -0.5 * apparentVolume * pi_i * mu_ij * dot
        out += pTerm + vTerm
    return out

from sphWarpCore.operations_grid.grid_util import checkOffset

@wp.func
def computeCrkSPHdudt_Func_Adjacency(
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
    
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryEnergies: wp.array(dtype = scalar_t), referenceEnergies: wp.array(dtype = scalar_t), # type: ignore
    queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = scalar_t), referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    crkViscosityParams: CRKViscosity,

    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore
    dudt: wp.array(dtype = Any), # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(dudt[i])
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    vel_i = queryVelocities[i]

    cs_i = queryCs[i]
    alpha_i = queryAlphas[i] if viscositySwitch else scalar_t(1.0)
    u_i = queryEnergies[i]
    P_i = queryPressures[i]
    gradV_i = queryVelocityTensor[i]

    out = zero_like_warp(dudt)
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
        
        out += computeCrkSPHdudt_Func_i(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState, correctionData, domainState,
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
            crkViscosityParams,
            gradV_i, referenceVelocityTensor,
        )
    return out



@wp.kernel
def computeCrkSPHdudt_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryEnergies: wp.array(dtype = scalar_t), referenceEnergies: wp.array(dtype = scalar_t), # type: ignore
    queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = scalar_t), referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    crkViscosityParams: CRKViscosity,

    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore
    # The last parameter is always the output array and should not be changed
    out_dudt : wp.array(dtype = Any), # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    out_dudt[i] = computeCrkSPHdudt_Func_Adjacency(
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
        crkViscosityParams,
        queryVelocityTensor, referenceVelocityTensor,
        out_dudt
    )

def computeCrkSPHdudtWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    conductivityParams: DiffusionParameters,
    crkViscosityParams: CRKViscosity,
    
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
    with record_function("warpSPH[computeCrkSPHdudt]"):
        with record_function("warpSPH[computeCrkSPHdudt] - Preprocessing"):
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
            outputDtype = castTorchToWarpAsBuiltins(queryParticles.densities).dtype
            outputSizes = queryParticles.positions.shape[0],
            outputDtypes = outputDtype

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
            
            queryEnergies_ = queryEnergies if queryEnergies is not None else (queryParticles.internalEnergies if hasattr(queryParticles, 'internalEnergies') else None)
            queryVelocities_ = queryVelocities if queryVelocities is not None else (queryParticles.velocities if hasattr(queryParticles, 'velocities') else None)
            queryCs_ = queryCs if queryCs is not None else (queryParticles.soundspeeds if hasattr(queryParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))
            queryAlphas_ = queryAlphas if queryAlphas is not None else (queryParticles.alphas if hasattr(queryParticles, 'alphas') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))
            queryPressures_ = queryPressures if queryPressures is not None else (queryParticles.pressures if hasattr(queryParticles, 'pressures') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))

            referenceEnergies_ = referenceEnergies if referenceEnergies is not None else (referenceParticles.internalEnergies if hasattr(referenceParticles, 'internalEnergies') else None)
            referenceVelocities_ = referenceVelocities if referenceVelocities is not None else (referenceParticles.velocities if hasattr(referenceParticles, 'velocities') else None)
            referenceCs_ = referenceCs if referenceCs is not None else (referenceParticles.soundspeeds if hasattr(referenceParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))
            referenceAlphas_ = referenceAlphas if referenceAlphas is not None else (referenceParticles.alphas if hasattr(referenceParticles, 'alphas') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))
            referencePressures_ = referencePressures if referencePressures is not None else (referenceParticles.pressures if hasattr(referenceParticles, 'pressures') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))

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

        with record_function("warpSPH[computeCrkSPHdudt] - Kernel Execution"):
            return warpWrapper2(
                launcher = launch_kernel,
                kernel   = computeCrkSPHdudt_Kernel,
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
                    crkViscosityParams,
                    queryVelocityTensor, referenceVelocityTensor,
                ),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeCrkSPHdudt_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
