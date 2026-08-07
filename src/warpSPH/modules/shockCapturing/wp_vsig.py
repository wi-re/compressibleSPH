import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *


@wp.func
def computeVsig_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.

    # Domain and kernel parameters
    # periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    domainState: domainData,
    kernelProperties: kernelState,
    
    # Operation specific parameters
     # type: ignore
            
    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, Viscosity_i: scalar_t, referenceViscositys: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore

    individual_cs: wp.bool, cs_i: scalar_t, referenceCs: wp.array(dtype = scalar_t), # type: ignore
    
    # Dummy value to allow allocation
    outputValue: Any, # type: ignore
):
    # Initialize the output value
    out     = zero_like_warp(outputValue)
    
    # # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        vel_j = referenceVelocities[j]

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]
        cs_j = referenceCs[j] if individual_cs else scalar_t(1.0)

        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        

        v_ij = vel_i - vel_j
        mu_ij = wp.dot(v_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi)

        c_bar = scalar_t(0.5) * (cs_i + cs_j)
        vsigs           = c_bar - mu_ij
        # vsigs[mu_ij > 0]= 0
        if mu_ij > 0:
            vsigs = scalar_t(0.0)

        out = wp.max(out, vsigs)
        
    return out



@wp.func
def computeVsig_Func_Adjacency(
    i : wp.int32, dim: wp.int32, 

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState, 
    
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    individual_cs: wp.bool, queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    
    outputValue : Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(outputValue)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    vel_i = queryVelocities[i]

    cs_i = queryCs[i] if individual_cs else scalar_t(1.0)

    out = zero_like_warp(outputValue)
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
        
        out += computeVsig_Func_i(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi , correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,
            vel_i, referenceVelocities,
            individual_cs, cs_i, referenceCs,

            outputValue,

            # Viscosity function parameters
        )
    return out



@wp.kernel
def computeVsig_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    individual_cs: wp.bool, queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeVsig_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        individual_cs, queryCs, referenceCs,

        zero_like_warp(outputValues)
    )

def computeVsigWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    queryVelocities : Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryCs: Optional[torch.Tensor] = None, referenceCs: Optional[torch.Tensor] = None,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    with record_function("warpSPH[computeVsig]"):
        with record_function("warpSPH[computeVsig] - Preprocessing"):
            # Preprocessing and input validation
            # args, device, dim = parseArguments(
            #     queryParticles, operationProperties, domain,
            #     queryVolumes, referenceVolumes,
            #     adjacency,
            #     referenceParticles,
            #     crkState,
            #     gradHState,
            #     renormalizationState,
            # )
            device = queryParticles.positions.device
            outputSize = queryParticles.positions.shape[0]
            outputDtype = castTorchToWarpAsBuiltins(queryParticles.densities).dtype

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
            queryVelocities_ = queryVelocities if queryVelocities is not None else (queryParticles.velocities if hasattr(queryParticles, 'velocities') else None)
            queryCs_ = queryCs if queryCs is not None else (queryParticles.soundspeeds if hasattr(queryParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))
            
            referenceVelocities_ = referenceVelocities if referenceVelocities is not None else (referenceParticles.velocities if hasattr(referenceParticles, 'velocities') else None)
            referenceCs_ = referenceCs if referenceCs is not None else (referenceParticles.soundspeeds if hasattr(referenceParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=torch.scalar_t32, device=device))
            
            if queryCs is not None or (hasattr(queryParticles, 'soundspeeds') and queryParticles.soundspeeds is not None):
                individual_cs = True
            else:            
                individual_cs = False
            
            if queryVelocities_ is None:
                raise ValueError("Velocities must be provided either through queryVelocities or as a property of queryParticles.")

        with record_function("warpSPH[computeVsig] - Kernel Execution"):
            return warpWrapper2(
                launcher = launch_kernel,
                kernel   = computeVsig_Kernel,
                outputSizes  = outputSize,
                outputDtypes = outputDtype,
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
                    individual_cs, queryCs_, referenceCs_,
                ),
            )

        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeVsig_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         viscosityParams
        #     )

    return warp_result
