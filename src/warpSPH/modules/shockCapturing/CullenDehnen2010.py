"""Cullen & Dehnen (2010) time-dependent artificial-viscosity switch.

Evaluates the per-particle limiter ``R``/``Xi`` (eq. 17-18) from the velocity
divergence and its sign-weighted SPH interpolation, derives a shear-based
target alpha via the CRKSPH-style ``xi`` kernel-support normalization, and
integrates it toward that target with a fixed decay length ``l = 0.05`` (eq.
16). ``computeCullenTerms`` performs both the target-alpha computation and its
time integration in one call (the original paper's convention); the trailing
``correctVelocityGradient`` branches of ``computeSecondOrderV`` apply the
CRKSPH gradient-renormalization matrix ``M_inv`` when configured. Several
alternate/commented-out formulations (a naive second-order divergence
estimate, an unused ``computeXi`` limiting workaround) are left in place from
prior experimentation; only the active code paths are wired into the schemes.
"""

from .common import *
from .switchState import *

from warpSPHCore import *
from ...systems.compressibleMonaghan import CompressibleState
from ...configurations import SimulationConfig
from ...configurations.compressibleConfig import CompressibleSPHConfig
from typing import Optional, Union
import torch



__all__ = ['computeSecondOrderV', 'computeR', 'computeXi', 'compute_vsig', 'computeCullenTerms', 'computeCullenUpdate']


def computeSecondOrderV(
        dvdt : torch.Tensor, 
        correctionMatrix : torch.Tensor,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    
    # The signs here should have been wrong, double check!
    V = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Gradient,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
            gradientMode = GradientScheme.Difference
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues=particleState.velocities
    )
    Vdot = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Gradient,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
            gradientMode = GradientScheme.Difference
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues=dvdt
    )

    # V = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    # Vdot = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    
    # i, j        = neighborhood.row, neighborhood.col
    # v_i, v_j    = particles_a.velocities[i], particles_b.velocities[j]
    # dv_i, dv_j  = dvdt_a[i], dvdt_b[j]
    # h_i         = particles_a.supports[i]
    # m_j         = particles_b.masses[j]
    # rho_j       = particles_b.densities[j]
    # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)

    # gradW_i = kernel.jacobian(x_ij, h_i)
    # v_ij = v_i - v_j
    # dv_ij = dv_i - dv_j
    
    # dyadicV = (m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', v_ij, gradW_i)
    # dyadicVdot = (m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', dv_ij, gradW_i)
    
    # V = scatter_sum(dyadicV, i, dim = 0, dim_size = particles_a.positions.shape[0])
    correctVelocityGradient = schemeConfig.viscositySwitchParams.correctVelocityGradient
    # correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
    if correctVelocityGradient:
        if correctionMatrix is not None:
            V = torch.einsum('ijk, ikl -> ijl', correctionMatrix, V)
        else:
            raise ValueError('Correction matrix is None, but correctVelocityGradient is True')
    V2 = torch.einsum('ijk, ikl -> ijl', V, V)
    # Vdot_ = scatter_sum(dyadicVdot, i, dim = 0, dim_size = particles_a.positions.shape[0])
    # From Cullen & Dehnen 2010 : We can estimate ∇˙ ·υ either from the change in the estimated ∇·υ over 
    # the last time step or as the trace of V [This is not implemented here as this would require 
    # storing the previous divergence estimate. Spheral (the code implementing CRKSPH by LLNL) does this.]    
    # Note that, by virtue of equation (B11), we could estimate ∇˙·υ also as ∇· υ˙ - tr(V^2) with the 
    # acceleration divergence ∇·υ˙ estimated using the standard divergence estimator, in the hope that 
    # its O(h0) error term is small since the acceleration is hardly sheared.
    divdotdvdt = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Divergence,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
            gradientMode = GradientScheme.Difference,
        ),
        consistentDivergence = False,
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues=dvdt
    )

    # divdotdvdt = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Divergence, supportScheme=supportScheme, gradientMode=GradientMode.Difference, consistentDivergence=False)
    
    # divdotdvdt = sph_op(particles_a, particles_b, domain, kernel, neighborhood, 'gather', 'divergence', gradientMode = 'difference' , consistentDivergence = False, quantity=(dvdt_a, dvdt_b))
    return divdotdvdt - torch.einsum('...ii', V2)

    # Naïve alternative solution as described above
    A = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Gradient,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
            gradientMode = GradientScheme.Difference
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues=dvdt
    )

    # A = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    # A = sph_op(particles_a, particles_b, domain, kernel, neighborhood, 'gather', 'gradient', gradientMode = 'difference' , consistentDivergence = False, quantity=(dvdt_a, dvdt_b))
    # correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
    if correctVelocityGradient:
        if correctionMatrix is not None: # This is an assumption, not described in the paper
            A = torch.einsum('ijk, ikl -> ijl', correctionMatrix, A)
        else:
            raise ValueError('Correction matrix is None, but correctVelocityGradient is True')
    return torch.einsum('...ii', A + V2)

    
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



from warpSPHCore import *
def computeCullenTerms(
        dt: float,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    
    switchConfig = schemeConfig.viscositySwitchParams
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode

    alpha_min = switchConfig.alpha_min

    # verbosePrint(verbose, '[Cullen]\t\tComputing Cullen Terms')
    correctVelocityGradient = switchConfig.correctVelocityGradient
    # correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
    if correctVelocityGradient:
    # verbosePrint(verbose, '[Cullen]\t\tComputing M')
        M = computeM(particleState, simulationConfig, schemeConfig, SupportScheme.Gather, adjacency) # E.6 in CRKSPH
        M_inv = torch.linalg.pinv(M)        
    else:
        M = None
        M_inv = None

    # verbosePrint(verbose, '[Cullen]\t\tComputing Shear Tensor')
    # div, S, Rot = computeShearTensor(psphState, psphState, domain, wrappedKernel, actualNeighbors, solverConfig, M_inv)
    # div = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Divergence, supportScheme=supportScheme, divergenceMode=DivergenceMode.div, gradientMode = GradientMode.Difference)

    div = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Divergence,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
            gradientMode = GradientScheme.Difference,
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues=particleState.velocities
    )

    # div = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(psphState.velocities, psphState.velocities))
    # verbosePrint(verbose, '[Cullen]\t\tComputing Limiter')
    R = computeR(div, particleState, simulationConfig, schemeConfig, SupportScheme.Gather, adjacency) # F.4
    # verbosePrint(verbose, '[Cullen]\t\tComputing Xi')
    S = None
    Rot = None
    Xi = computeXi(div, S, R, particleState, simulationConfig, schemeConfig, SupportScheme.Gather, adjacency ) 
    
    # The original Cullen and Dehnen paper already uses ddiv_dt for the local alpha
    # The hopkins modified form only uses ddiv_dt for the temporal evolution of alpha
    # verbosePrint(verbose, '[Cullen]\t\tSecond order Divergence')
    # ddiv_dt = computeSecondOrderV(psphState, psphState, domain, wrappedKernel, actualNeighbors, solverConfig, dvdt, dvdt, correctionMatrix=CDState.M_inv)
    # ddiv_dt = computeSecondOrderV(
    #     dvdt,
    #     M_inv,
    #     particleState,
    #     simulationConfig,
    #     schemeConfig,
    #     supportScheme,
    #     adjacency)

    # ddiv_dt = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(dvdt, dvdt))
    ddiv_dt = (div - particleState.divergence) / dt
    
    alpha_max = switchConfig.alpha_max
    # alpha_max = getSetConfig(config, 'diffusionSwitch', 'alpha_max', 2)# from CRKSPH
    # The 1/xi is based on Hopkins' ATHENA paper after eq. F17
    f_kern      = 1/sphKernel_xi(simulationConfig.kernel.value, particleState.positions.shape[1])
    # f_kern      = 1/Kernel_xi(kernel, particles.positions.shape[1])# wrappedKernel.xi(domain.dim)
    
    # Eq. 13 in Cullen and Dehnen 2010
    A_i = Xi * (-ddiv_dt).clamp(min = 0)
    v_sig = torch.abs(compute_vsig(particleState, simulationConfig, schemeConfig, supportScheme, adjacency))
    # v_sig = compute_vsig(particles, kernel, neighborhood, supportScheme, config)
    
    h_i = particleState.supports #* f_kern
    scaling = h_i**2 * A_i / (v_sig**2 + h_i **2 * A_i + 1e-14 * h_i)
    # if switchConfig.limitXi:
        # scaling = torch.where(torch.abs(A_i) < 1e-5, torch.ones_like(scaling), scaling) # This is a workaround to prevent extremely large alphas when A_i is close to 0, which can happen in low Mach number flows. This is not described in the original paper but seems necessary for stability in some cases.
    alphas = alpha_max * scaling
    alphas = alphas.clamp(min = alpha_min, max = alpha_max)
    
    alpha0s = particleState.alpha0s.clone()
    l = 0.05 # See Cullen and Dehnen 2010, eq 16
    tau_i = h_i * f_kern / (2 * l * v_sig)
    alpha_dot = (alphas - alpha0s) / tau_i
    
    # print(f'alphas: {alphas.min()}, {alphas.max()}, {alphas.mean()}')
    # print(f'alpha0s: {alpha0s.min()}, {alpha0s.max()}, {alpha0s.mean()}')
    # print(f'alpha_dot: {alpha_dot.min()}, {alpha_dot.max()}, {alpha_dot.mean()}')
    
    alpha0s = torch.where(alphas > alpha0s, alphas, alpha0s)
    
    alpha0s = torch.where(alphas < alpha0s, alpha0s + alpha_dot * dt, alpha0s)
    
    alpha0s = alpha0s.clamp(min = alpha_min, max = alpha_max)
    
    
    # verbosePrint(verbose, '[Cullen]\t\tComputing Alphas')
    # alphas = (Xi * psphState.alpha0s).clamp(min = alpha_min)
    return alpha0s, ViscositySwitchState(
        alpha0s = alpha0s,
        alphas = alphas,
        M = M,
        M_inv = M_inv,
        div = div,
        ddivdt = ddiv_dt,
        Shear = S,
        Rot = Rot,
        R = R,
        Xi = Xi,
        v_sig = v_sig
    )
    
def computeCullenUpdate(
        switchState: ViscositySwitchState,
        dt: float,
        dvdt: torch.Tensor,

        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):#psphState, domain, wrappedKernel, actualNeighbors, solverConfig, dvdt, CDState, dt, verbose = False):
    # The original Cullen and Dehnen paper already performs the update in the computeCullenTerms function
    # So we simply return the state here (for now)

    
    return switchState.alpha0s, ViscositySwitchState(
        alpha0s = switchState.alpha0s,
        alphas = switchState.alphas,
        M = switchState.M,
        M_inv = switchState.M_inv,
        div = switchState.div,
        ddivdt = switchState.ddivdt,
        Shear = switchState.Shear,
        Rot = switchState.Rot,
        R = switchState.R,
        Xi = switchState.Xi,
        v_sig = switchState.v_sig
    )

    