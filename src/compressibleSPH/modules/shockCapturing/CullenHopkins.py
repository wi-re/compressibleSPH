from .common import *
from .switchState import *

from sphWarpCore import *
from .CullenDehnen2010 import *

def computeHopkinsTerms(
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None
):
    switchConfig = schemeConfig.viscositySwitchParams
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode

    alpha_min = switchConfig.alpha_min
    correctVelocityGradient = switchConfig.correctVelocityGradient
    # alpha_min = getSetConfig(config, 'diffusionSwitch', 'alpha_min', 0.02)

    # verbosePrint(verbose, '[Cullen]\t\tComputing Cullen Terms')
    # correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
    if correctVelocityGradient:
        # verbosePrint(verbose, '[Cullen]\t\tComputing M')
        M = computeM(particleState, simulationConfig, schemeConfig, supportMode, adjacency)
        # M = computeM(particles, kernel, neighborhood, supportScheme, config)
        M_inv = torch.linalg.pinv(M)        
    else:
        M = None
        M_inv = None

    # verbosePrint(verbose, '[Cullen]\t\tComputing Shear Tensor')
    # div, S, Rot = computeShearTensor(particles, kernel, neighborhood, supportScheme, config, M_inv)
    div, S, Rot = computeShearTensor(M_inv, particleState, simulationConfig, schemeConfig, supportMode, adjacency)

    # div = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(psphState.velocities, psphState.velocities))
    # verbosePrint(verbose, '[Cullen]\t\tComputing Limiter')
    R = computeR(div, particleState, simulationConfig, schemeConfig, supportMode, adjacency)
    # verbosePrint(verbose, '[Cullen]\t\tComputing Xi')
    # S = None
    # Rot = None
    Xi = computeXi(div, S, R, particleState, simulationConfig, schemeConfig, supportMode, adjacency )
    
    # verbosePrint(verbose, '[Cullen]\t\tComputing Alphas')
    alphas = (Xi * particleState.alpha0s).clamp(min = alpha_min)
    return alphas, ViscositySwitchState(
        alpha0s = particleState.alpha0s,
        alphas = alphas,
        M = M,
        M_inv = M_inv,
        div = div,
        ddivdt = None,
        Shear = S,
        Rot = Rot,
        R = R,
        Xi = Xi,
        v_sig = None
    )
    
# from diffSPH.kernels import Kernel_xi
from sphWarpCore.kernels.wp_kernel import sphKernel_xi
def computeHopkinsUpdate(
        switchState: ViscositySwitchState,
        dt: float,
        dvdt: torch.Tensor,

        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None
):
    # This implements the Hopkins modified form
    
    switchConfig = schemeConfig.viscositySwitchParams
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode

    alpha_max = switchConfig.alpha_max
    # alpha_max = getSetConfig(config, 'diffusionSwitch', 'alpha_max', 2) # from CRKSPH
    # The 1/xi is based on Hopkins' ATHENA paper after eq. F17
    f_kern      = 1/sphKernel_xi(simulationConfig.kernel.value, particleState.positions.shape[1])
    # f_kern = 1/3 # hard coded result for cubic spline from Hopkins 2015, Hopkins also uses h=1 as the cutoff so these terms _should_ be equivalent?
    # f_kern = 1
    beta_c = switchConfig.beta_c
    beta_d = switchConfig.beta_d
    # beta_c = getSetConfig(config, 'diffusionSwitch', 'beta_c', 0.7)
    # beta_d = getSetConfig(config, 'diffusionSwitch', 'beta_d', 0.05)

    # verbosePrint(verbose, '[Cullen]\t\tComputing Cullen Update')
    # verbosePrint(verbose, '[Cullen]\t\tSecond order Divergence')
    # ddiv_dt = computeSecondOrderV(psphState, psphState, domain, wrappedKernel, actualNeighbors, solverConfig, dvdt, dvdt, correctionMatrix=CDState.M_inv)

    # ddiv_dt = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(dvdt, dvdt))
    ddiv_dt = (switchState.div - particleState.divergence) / dt

    # verbosePrint(verbose, '[Cullen]\t\tComputing Vsig')
    # There is an issue here!
    # Hopkins and CRKSPH both use the same offset term offset = beta_c * c_s^2 / (f_kern * h)^2
    # In CRKSPH F.5 for some reason the equation reads as alpha_max * |ddiv_dt| / (alpha_max * |ddiv_dt| + offset)
    # Whereas Hopkins uses alpha_max * |ddiv_dt| / (|ddiv_dt| + offset)
    
    # CRKSPH form
    # alpha_div = alpha_max * ddiv_dt.abs()
    # alpha_tmp = alpha_div / (alpha_div + beta_c * psphState.soundspeeds**2 / (f_kern * psphState.supports)**2)
    # Hopkins form
    alpha_div = ddiv_dt.abs()
    alpha_tmp = alpha_max * alpha_div / (alpha_div + beta_c * particleState.soundspeeds**2 / (f_kern * particleState.supports)**2 + 1e-14)
    
    # print(f'alpha_max: {alpha_max}')
    # print(f'alpha_div: {ddiv_dt.min()}, {ddiv_dt.max()}, {ddiv_dt.mean()}')
    # print(f'alpha_tmp: {alpha_tmp.min()}, {alpha_tmp.max()}, {alpha_tmp.mean()}')
    term = beta_c * particleState.soundspeeds**2 / (f_kern * particleState.supports)**2
    # print(f'term: {term.min()}, {term.max()}, {term.mean()}')
    # print(f'c_s: {particleState.soundspeeds.min()}, {particleState.soundspeeds.max()}, {particleState.soundspeeds.mean()}')
    
    
    alpha_tmp = torch.where(torch.logical_or(ddiv_dt > 0, switchState.div > 0), 0, alpha_tmp)
    # print(f'alpha_tmp(c): {alpha_tmp.min()}, {alpha_tmp.max()}, {alpha_tmp.mean()}')

    v_sig = compute_vsig(particleState, simulationConfig, schemeConfig, supportScheme, adjacency)

    # verbosePrint(verbose, '[Cullen]\t\tComputing Alpha0s')
    alpha_0 = particleState.alpha0s
    alpha_0_next = alpha_tmp + (alpha_0 - alpha_tmp) * torch.exp(- beta_d * dt * v_sig / (2 * f_kern * particleState.supports))
    alpha_0_next = torch.where(alpha_tmp >= alpha_0, alpha_tmp, alpha_0_next)
    alpha0s = alpha_0_next
    
    return alpha0s, ViscositySwitchState(
        alpha0s = alpha0s,
        alphas = switchState.alphas,
        M = switchState.M,
        M_inv = switchState.M_inv,
        div = switchState.div,
        ddivdt = ddiv_dt,
        Shear = switchState.Shear,
        Rot = switchState.Rot,
        R = switchState.R,
        Xi = switchState.Xi,
        v_sig = v_sig
    )

    