from .common import *
from .switchState import *

from sphWarpCore import *



# def computeSecondOrderV(
#         dvdt : torch.Tensor, 
#         correctionMatrix : torch.Tensor,
#         particleState: CompressibleState,
#         simulationConfig: SimulationConfig,
#         schemeConfig: CompressibleSPHConfig,
#         supportScheme: Optional[SupportScheme] = None,
#         adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    
#     # The signs here should have been wrong, double check!
#     V = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
#     Vdot = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    
#     # i, j        = neighborhood.row, neighborhood.col
#     # v_i, v_j    = particles_a.velocities[i], particles_b.velocities[j]
#     # dv_i, dv_j  = dvdt_a[i], dvdt_b[j]
#     # h_i         = particles_a.supports[i]
#     # m_j         = particles_b.masses[j]
#     # rho_j       = particles_b.densities[j]
#     # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)

#     # gradW_i = kernel.jacobian(x_ij, h_i)
#     # v_ij = v_i - v_j
#     # dv_ij = dv_i - dv_j
    
#     # dyadicV = (m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', v_ij, gradW_i)
#     # dyadicVdot = (m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', dv_ij, gradW_i)
    
#     # V = scatter_sum(dyadicV, i, dim = 0, dim_size = particles_a.positions.shape[0])
#     correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
#     if correctVelocityGradient:
#         if correctionMatrix is not None:
#             V = torch.einsum('ijk, ikl -> ijl', correctionMatrix, V)
#         else:
#             raise ValueError('Correction matrix is None, but correctVelocityGradient is True')
#     V2 = torch.einsum('ijk, ikl -> ijl', V, V)
#     # Vdot_ = scatter_sum(dyadicVdot, i, dim = 0, dim_size = particles_a.positions.shape[0])
#     # From Cullen & Dehnen 2010 : We can estimate ∇˙ ·υ either from the change in the estimated ∇·υ over 
#     # the last time step or as the trace of V [This is not implemented here as this would require 
#     # storing the previous divergence estimate. Spheral (the code implementing CRKSPH by LLNL) does this.]    
#     # Note that, by virtue of equation (B11), we could estimate ∇˙·υ also as ∇· υ˙ - tr(V^2) with the 
#     # acceleration divergence ∇·υ˙ estimated using the standard divergence estimator, in the hope that 
#     # its O(h0) error term is small since the acceleration is hardly sheared.
#     divdotdvdt = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Divergence, supportScheme=supportScheme, gradientMode=GradientMode.Difference, consistentDivergence=False)
    
#     # divdotdvdt = sph_op(particles_a, particles_b, domain, kernel, neighborhood, 'gather', 'divergence', gradientMode = 'difference' , consistentDivergence = False, quantity=(dvdt_a, dvdt_b))
#     return divdotdvdt - torch.einsum('...ii', V2)

#     # Naïve alternative solution as described above
#     A = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
#     # A = sph_op(particles_a, particles_b, domain, kernel, neighborhood, 'gather', 'gradient', gradientMode = 'difference' , consistentDivergence = False, quantity=(dvdt_a, dvdt_b))
#     correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
#     if correctVelocityGradient:
#         if correctionMatrix is not None: # This is an assumption, not described in the paper
#             A = torch.einsum('ijk, ikl -> ijl', correctionMatrix, A)
#         else:
#             raise ValueError('Correction matrix is None, but correctVelocityGradient is True')
#     return torch.einsum('...ii', A + V2)

    
# from diffSPH.util import scatter_max
# from diffSPH.neighborhood import evalKernel, evalKernelGradient
def computeR(
        div:torch.Tensor,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None
        ): # Referred to as R in eq 17 in Cullen and Dehnen 2010, referred to as Xi in CRKSPH and Hopkins

    # i, j        = neighborhood.row, neighborhood.col
    # h_i         = particles_a.supports[i]
    # m_j         = particles_b.masses[j]
    # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    
    # W_i         = kernel.eval(x_ij, h_i)
    # term        = m_j * torch.sign(div[j]) * W_i

    term = particleState.densities * torch.sign(div) #* evalKernel(kernelValues, supportScheme, True)
    # The term is m[j] * |div[j]|, interpolate multiplies the given term with m[j]/rho[j] so need to premultiply by rho[j]
    summedTerm = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues=term
    )

    # summedTerm = SPHOperation(
    #     particles,
    #     quantity = term,
    #     kernel = kernel,
    #     neighborhood = neighborhood[0],
    #     kernelValues = neighborhood[1],
    #     operation= Operation.Interpolate        
    # )

    # CRKSPH and Hopkins compute 1-R as Xi, we follow the convention of Cullen and Dehnen 2010
    # This also means that our 'Xi' term is the limiter term that is not given an explicit name
    # in CRKSPH and Hopkins notations. Also mind the minus sign.
    return 1 / particleState.densities * summedTerm
    
def computeXi(
        div, 
        S, 
        R,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None): # See Cullen and Dehnen 2010, eq 18
    switchParams = schemeConfig.viscositySwitchParams
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode
    beta_xi = switchParams.beta_xi
    limitXi = switchParams.limitXi

    # beta_xi = getSetConfig(config, 'diffusionSwitch', 'beta_xi', 2) # from the original Cullen and Dehnen paper Eq. 18
    # limitXi = getSetConfig(config, 'diffusionSwitch', 'limitXi', False) # Workaround for testing for cases with 0 divergence
    
    nominator       = beta_xi * (1 - R)**4 * div
    denominatorLeft = nominator
    # This agrees with Spheral and Cullen and Dehnen. F.3 in the CRKSPH code is inconclusive
    # trace           = torch.einsum('...ii', torch.einsum("...ij, ...kj -> ...ki", S, S))
    trace = 0.0 if S is None else torch.einsum('...ii', torch.einsum("...ij, ...kj -> ...ki", S, S))
    
    if limitXi:
        return (nominator**2 + 1e-14 * particleState.supports) / (denominatorLeft**2 + trace + 1e-14 * particleState.supports)
    return (nominator**2) / (denominatorLeft**2 + trace + 1e-14 * particleState.supports)
    

from .wp_vsig import computeVsigWarp

def compute_vsig(
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    
    return computeVsigWarp(
        particleState,
        operationProperties=OperationProperties(
            kernel = simulationConfig.kernel,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryVelocities=particleState.velocities,
        queryCs = particleState.soundspeeds,
    )