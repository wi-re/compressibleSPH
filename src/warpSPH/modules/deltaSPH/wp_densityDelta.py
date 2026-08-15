"""Per-particle delta-SPH density-diffusion kernel: for each neighbor pair,
builds a Fick's-law-style flux `psi_ij` from the inter-particle density
difference and (depending on `densityScheme`) a renormalized or
unrenormalized density gradient, then accumulates `apparentVolume *
dot(psi_ij, gradW_ij)` as an unscaled divergence — the delta/c_s/h prefactor
is applied by the caller (`densityDiffusion.py`).

`densityScheme` (a `DensityDiffusionScheme` int) selects the flux:
- `deltaSPH`: `psi_ij = (gradRhoL_i + gradRhoL_j) - 2*(rho_j-rho_i)*n_ij/r_ij`
  — renormalized gradients combined with the density-difference term.
- `denormalized`: the same combination but with the unrenormalized `gradRho`.
- `densityOnly`: just `-2*(rho_j-rho_i)*n_ij/r_ij` (no gradient term).
- `deltaOnly` / `denormalizedOnly`: just the renormalized/unrenormalized
  gradient sum (no density-difference term).
All branches guard the `x_ij` normalization and the density-difference
term's `1/r_ij` with a `1e-14 * h_i` epsilon to avoid division by zero at
zero separation. `computeDensityDiffusionDeltaSPH` is the public torch/warp
bridge; `_Func_i`/`_Func_Adjacency`/`_Kernel` are its warp-side
implementation and are not meant to be called directly.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeDensityDiffusionDeltaSPH']


from ...enumTypes import *

@wp.func
def computeDensityDiffusionDeltaSPH_Func_i(
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
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    gradRho_i: vector(length=Any, dtype=scalar_t), referenceGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    gradRhoL_i: vector(length=Any, dtype=scalar_t), referenceGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    densityScheme: wp.int32,
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

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        n_ij = x_ij / (r_ij + scalar_t(1.0e-14) * hi)


        # grad_ij = zero_like_warp(gradw_ij)
        # rho_ij = scalar_t(0.0)
        psi_ij = zero_like_warp(gradw_ij)
        if densityScheme == wp.int32(DensityDiffusionScheme.deltaSPH.value):
            grad_ij = gradRhoL_i + referenceGradRhoL[j]
            rho_ij = scalar_t(2.0) * (rhoj - rhoi) * n_ij / (r_ij + scalar_t(1.0e-14) * hi)
            psi_ij = grad_ij - rho_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.denormalized.value):
            grad_ij = gradRho_i + referenceGradRho[j]
            rho_ij = scalar_t(2.0) * (rhoj - rhoi) * n_ij / (r_ij + scalar_t(1.0e-14) * hi)
            psi_ij = grad_ij - rho_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.densityOnly.value):
            grad_ij = zero_like_warp(gradw_ij)
            rho_ij = scalar_t(2.0) * (rhoj - rhoi) * n_ij / (r_ij + scalar_t(1.0e-14) * hi)
            psi_ij = - rho_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.deltaOnly.value):
            grad_ij = gradRhoL_i + referenceGradRhoL[j]
            rho_ij = zero_like_warp(gradw_ij)
            psi_ij = grad_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.denormalizedOnly.value):
            grad_ij = gradRho_i + referenceGradRho[j]
            rho_ij = zero_like_warp(gradw_ij)
            psi_ij = grad_ij
        
        prod = wp.dot(psi_ij, gradw_ij)



        out += apparentVolume * prod
        
    return out



@wp.func
def computeDensityDiffusionDeltaSPH_Func_Adjacency(
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
    
    queryGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    densityScheme: wp.int32,

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
    
    gradRho_i = queryGradRho[i]
    gradRhoL_i = queryGradRhoL[i]

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
        
        out += computeDensityDiffusionDeltaSPH_Func_i(
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
            gradRho_i, referenceGradRho,
            gradRhoL_i, referenceGradRhoL,
            densityScheme,


            outputValue,

            # Viscosity function parameters
        )
    return out



@wp.kernel
def computeDensityDiffusionDeltaSPH_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above
    queryGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    densityScheme: wp.int32,

    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeDensityDiffusionDeltaSPH_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        # The parameters above are default parameters and shold not be changed
        queryGradRho, referenceGradRho,
        queryGradRhoL, referenceGradRhoL,
        densityScheme,


        zero_like_warp(outputValues)
    )


def computeDensityDiffusionDeltaSPH(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    densityScheme: DensityDiffusionScheme,

    queryGradRho: Optional[torch.Tensor] = None, referenceGradRho: Optional[torch.Tensor] = None,
    queryGradRhoL: Optional[torch.Tensor] = None, referenceGradRhoL: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if referenceGradRho is None:
        referenceGradRho = queryGradRho
    if referenceGradRhoL is None:
        referenceGradRhoL = queryGradRhoL

    with record_function("warpSPH[computeDensityDiffusion]"):
        with record_function("warpSPH[computeDensityDiffusion] - Preprocessing"):
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

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
            
            queryGradRho_ = queryGradRho if queryGradRho is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)
            queryGradRhoL_ = queryGradRhoL if queryGradRhoL is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)
            referenceGradRho_ = referenceGradRho if referenceGradRho is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)
            referenceGradRhoL_ = referenceGradRhoL if referenceGradRhoL is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)

        with record_function("warpSPH[computeDensityDiffusionDeltaSPH] - Kernel Execution"):
            return warpWrapper2(
                launcher = launch_kernel,
                kernel   = computeDensityDiffusionDeltaSPH_Kernel,
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
                    queryGradRho_, referenceGradRho_,
                    queryGradRhoL_, referenceGradRhoL_,
                    wp.int32(densityScheme.value)
                ),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeDensityDiffusionDeltaSPH_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
