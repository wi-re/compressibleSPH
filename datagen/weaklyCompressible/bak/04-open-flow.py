# %matplotlib widget
# Boilerplate import code for all libraries
# Changes to the precision require re-loading the kernel and need to be done before any op uses them.
import argparse

import warpSPHCore_config as swc
from typing import Any
swc.configure(precision="float32", dim=Any) # precision: float16|half|float32|single|float64|double

import warpSPHCore as sph
from warpSPHCore.type_config import *
print(get_type_config()) # confirms active settings

# Initialize warp at this point
import warp as wp; wp.init()

import os
import torch
if torch.cuda.is_available(): # set the TORCH_CUDA_ARCH_LIST environment variable to the compute capability of the GPU for faster compiles
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# final import blocks that are generic
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
import shlex    
import subprocess
import shutil

# custom SPH libraries
from warpSPHIntegrators.integration import *
from warpSPHCore import *
from warpSPHPlotting import *

# This library
from warpSPH import *

# The case utilities that contain all the case setup functions for the various test cases
from warpSPH.caseUtils import *
import argparse
parser = argparse.ArgumentParser(description='Run the dam break simulation with obstacle.')

parser.add_argument('--nx', type=int, default=128, help='Number of particles along the x-axis')
parser.add_argument('--markerSize', type=int, default=8, help='Size of the markers in the plot')
parser.add_argument('--plotWidth', type=int, default=28, help='Width of the plot in inches')
parser.add_argument('--n_h', type=int, default=4, help='Target number of neighbors')
parser.add_argument('--L', type=float, default=2.0, help='Length of the domain')
parser.add_argument('--W', type=float, default=4.0, help='Width of the domain')
parser.add_argument('--fillRatio', type=float, default=0.25, help='Fill ratio for the domain')

parser.add_argument('--timeLimit', type=float, default=4.0, help='Time limit for the simulation')
parser.add_argument('--enableFreestream', action='store_true', help='Enable freestream boundary conditions')
parser.add_argument('--forcingWidth', type=float, default=2.0/16.0, help='Width of the forcing region')
parser.add_argument('--freeStreamVelocity', type=float, default=1.0, help='Velocity of the free stream')
parser.add_argument('--band', type=int, default=5, help='Number of particle bands around the domain for boundary conditions')

parser.add_argument('--targetDt', type=float, default=0.0005, help='Target timestep for the simulation')

parser.add_argument('--obstacleActive', action='store_true', help='If set, an obstacle will be included in the simulation')
parser.add_argument('--obstacleType', type=str, default='circle', help='Type of obstacle to include (none, circle, ellipse, box, roundedBox, equilateralTriangle, hexagon, horseshoe, star, nacaXXXX)')
parser.add_argument('--maxExtent', type=float, default=0.25, help='Maximum extent of the obstacle')
parser.add_argument('--aoa', type=float, default=0.0, help='Angle of attack of the obstacle in degrees')
parser.add_argument('--aspectRatio', type=float, default=1.0, help='Aspect ratio of the obstacle (for ellipse)')
parser.add_argument('--offsetX', type=float, default=-0.0, help='X offset of the obstacle')
parser.add_argument('--offsetY', type=float, default=0.0, help='Y offset of the obstacle')
parser.add_argument('--mergeBoundaries', action='store_true', help='If set, the boundaries will be merged into a single boundary region')

parser.add_argument('--linearMotion', action='store_true', help='Enable linear motion of the obstacle')
parser.add_argument('--angularMotion', action='store_true', help='Enable angular motion of the obstacle')
# The motion can either be fixed, i.e., constantly spinning, or it can be a function of time, e.g., sinusoidal motion. The user can specify the type of motion using the --motionType argument.
parser.add_argument('--motionType', type=str, default='fixed', help='Type of motion for the obstacle (fixed, sinusoidal for now)')
parser.add_argument('--motionFrequency', type=float, default=1.0, help='Frequency of the motion for the obstacle')

parser.add_argument('--linearVelocityDirection', type=float, nargs=2, default=[0.0, 1.0], help='Direction of the linear motion (as a 2D vector)')
parser.add_argument('--linearVelocityMagnitude', type=float, default=0.5, help='Magnitude of the linear motion')
parser.add_argument('--angularVelocityMagnitude', type=float, default=1.0, help='Magnitude of the angular motion')

parser.add_argument('--disableGravity', action='store_true', help='Disable gravity in the simulation')
parser.add_argument('--gravityDirection', type=float, nargs=2, default=[0.0, -1.0], help='Direction of gravity (default: [0.0, -1.0])')
parser.add_argument('--gravityMagnitude', type=float, default=9.81, help='Magnitude of gravity (default: 9.81)')

parser.add_argument('--enableSloshing', action='store_true', help='Enable sloshing motion in the simulation')
parser.add_argument('--sloshingAmplitude', type=float, default=0.1, help='Amplitude of sloshing motion (default: 0.1)')
parser.add_argument('--sloshingFrequency', type=float, default=1.0, help='Frequency of sloshing motion (default: 1.0)')

parser.add_argument('--caseName', type=str, default='4-open-flow', help='Name of the case to run (default: 12-dambreak)')
parser.add_argument('--plot', action='store_true', help='Enable plotting of the simulation results')
parser.add_argument('--plotInterval', type=int, default=10, help='Interval for plotting (default: 10)')

parser.add_argument('--bandWidth', type=float, default=16.0, help='Width of the band for the noise function')


args = parser.parse_args()


nx = args.nx
dim = 2
L = args.L
dx = L / nx
band = args.band
W = args.W
n_h = args.n_h
targetDt = args.targetDt


gamma = 5/3
rho0 = 1
nu_visc = 0.0005
freeSurface = True

timestamp = getCurrentTimestamp()
obstacleText = f'{args.obstacleType}_maxExtent{args.maxExtent}_aspectRatio{args.aspectRatio}_offsetX{args.offsetX}_offsetY{args.offsetY}'
motionText = f'linearMotion{args.linearMotion}_angularMotion{args.angularMotion}_motionType{args.motionType}_motionFrequency{args.motionFrequency}_linearVelocityDirection{args.linearVelocityDirection[0]}_{args.linearVelocityDirection[1]}_linearVelocityMagnitude{args.linearVelocityMagnitude}_angularVelocityMagnitude{args.angularVelocityMagnitude}'
caseName = f'{args.caseName}_{timestamp}_{nx}_{n_h}_{L}_{W}_{obstacleText}'

extraData = {
    'nx': nx,
    'dim': dim,
    'L': L,
    'n_h': n_h,

    'gamma': gamma,
    'rho0': rho0,
    'nu_visc': nu_visc,
}

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = get_torch_precision()


domain = buildDomainDescription(L + dx * (band) * 2, dim, True, device, dtype)
domain.min = torch.tensor([-W/2, -L/2 - dx * (band)], device = device, dtype = dtype)
domain.max = torch.tensor([W/2, L/2 + dx * (band)], device = device, dtype = dtype)
interiorDomain = buildDomainDescription(L, dim, False, device, dtype)
interiorDomain.min = torch.tensor([-W/2, -L/2], device = device, dtype = dtype)
interiorDomain.max = torch.tensor([W/2, L/2], device = device, dtype = dtype)


config, integrator = buildConfig(
    domain = domain,
    dim = dim,
    kernel = KernelFunctions.Wendland4,
    targetNeighbors = n_h_to_nH(4, dim),
    supportMode = SupportScheme.KernelMeanSymmetric,
    gradientMode = GradientScheme.Difference,
    laplacianMode = LaplacianScheme.Brookshaw,
    integrationScheme = IntegrationSchemeType.rungeKutta2,
    samplingScheme = SamplingScheme.regular,
    device = device,
    dtype = dtype,
    dt = None,
    adaptiveDt = True,
    cflFactor=0.3,
)
config.nx = nx + band * 2
config.dx = dx

config.minDt = 1e-8
# config.dx = L / (nx * 2)

scheme = WeaklyCompressibleSPHScheme.deltaSPH
SimulationSystem, SimulationState, SimulationConfig, SimulationUpdate, fn, export_fn, import_fn = buildScheme(scheme)


schemeConfig = SimulationConfig()
schemeConfig.surfaceDetectionConfig.active = freeSurface
schemeConfig.bandwith = L / args.bandWidth / config.dx


schemeConfig.gravityConfig.active = not args.disableGravity
schemeConfig.gravityConfig.type = GravityType.Directional
schemeConfig.gravityConfig.magnitude = args.gravityMagnitude
schemeConfig.gravityConfig.origin = args.gravityDirection   


fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)


import copy
interiorDomain2 = copy.deepcopy(interiorDomain)
interiorDomain2.min[0] -= 20 * (band)
interiorDomain2.max[0] += 20 * (band)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain2, invert = False)

# obstacleWidth = L/16
# obstacleHeight = L/4

fluidW = args.W
fluidH = args.L * args.fillRatio
box_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([fluidW/2,fluidH/2]).to(points.device)), torch.tensor([0.0,interiorDomain.min[1] + fluidH/2]).to(points.device)), invert = False)


from warpSPH.utils.naca import *


maxExtent = args.maxExtent
aspectRatio = args.aspectRatio
offsetX = args.offsetX
offsetY = args.offsetY
aoa = args.aoa

from utils import buildObstacleSDF

obstacle_sdf = buildObstacleSDF(args.obstacleType, args.offsetX, args.offsetY, args.maxExtent, args.aspectRatio, aoa, config, schemeConfig, L, W)

# obstacle_sdf = 

regions = []

regions.append(buildRegion(config, schemeConfig, box_sdf, RegionType.Fluid, initialConditions = {}, shortEdge = W > L))
# regions.append(buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip))
# if obstacle:

# bcType = BCType.noSlip
# if args.linearMotion or args.angularMotion:
bcType = BCType.constant

if args.obstacleActive:
    regions.append(buildRegion(config, schemeConfig, obstacle_sdf, RegionType.Boundary, initialConditions = {}, kind = bcType, shortEdge = W > L))
if args.band > 0:
    regions.append(buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip, shortEdge = W > L))


# fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
# regions = []

# regions.append(buildRegion(config, schemeConfig, fluid_sdf, RegionType.Fluid, initialConditions = {}))

for region in regions:
    region = filterRegion(region, regions)

# compressibleSystem = setupBasicWeaklyCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
compressibleSystem = initializeWeaklyCompressibleSimulation(regions, config, schemeConfig, SimulationSystem, SimulationState, verbose = True)
# compressibleSystem.state.positions = shuffleParticles(compressibleSystem.state, config, schemeConfig, 32, jitterAmount = 0.1)

if args.enableFreestream:
    u_freestream = args.freeStreamVelocity
    forcingWidth = args.forcingWidth

    forcingSDF = lambda points: sampleSDF(points, lambda x: getSDF('box')['function'](x, torch.tensor([W/2 - forcingWidth, L/2]).to(points.device)), invert = True)

    def ldcDirichlet(state, cfg, schemeCfg, positions, d, n, t, dt):
        velocities = state.velocities.clone()
        # mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
        velocities[:,0] = u_freestream * 2
        # v_diff = u_freestream - state.velocities[:,0]
        # # slowly ramp the velocities to the target velocity over time
        # velocities[:,0] = state.velocities[:,0] + v_diff * dt / 0.1
        return velocities

    def ldcDirichletDensity(state, cfg, schemeCfg, positions, d, n, t, dt):
        densities = state.densities.clone()
        densities[:] = rho0
        # ramp the densities to the target density over time
        densities[:] = state.densities[:] + (rho0 - state.densities[:]) * dt / 0.1
        return densities

    def ldcDirichletUpdate(state, cfg, schemeCfg, positions, d, n, t, dt):
        velocities = torch.zeros_like(state.velocities)
        mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
        velocities[:,0] = torch.where(mask, 0.0, velocities[:,0])
        return velocities

    def ldcDirichletUpdateDensity(state, cfg, schemeCfg, positions, d, n, t, dt):
        densities = torch.zeros_like(state.densities)
        mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
        densities[:] = torch.where(mask, 0.0, densities[:])
        return densities

    def ldcForcing(state, cfg, schemeCfg, positions, d, n, t, dt):
        forcing = torch.zeros_like(state.velocities)
        mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
        velocities = state.velocities.clone()
        velocities[:,0] = u_freestream
        v_diff = u_freestream - state.velocities[:,0]
        # slowly ramp the velocities to the target velocity over time
        velocities[:,0] = state.velocities[:,0] + v_diff * dt / 0.1

        forcing[:,0] = v_diff * dt / 0.1




        return forcing

    ldcBC = BoundaryCondition(
        type = BoundaryConditionType.dynamic,
        sdf = forcingSDF,
        dirichletFunctions = {
            # 'velocities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichlet(state, cfg, schemeCfg, positions, d, n, t, dt),
            # 'densities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichletDensity(state, cfg, schemeCfg, positions, d, n, t, dt)
        },
        updateFunctions = {
            # 'dvdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichletUpdate(state, cfg, schemeCfg, positions, d, n, t, dt),
            # 'drhodt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichletUpdateDensity(state, cfg, schemeCfg, positions, d, n, t, dt)
        },
        forcingFunctions = [ldcForcing]

    )
    schemeConfig.boundaryConditions = [ldcBC]
    minBoundaryDistance = torch.ones_like(compressibleSystem.state.positions[:,0]) * np.inf
    for region in regions:
        if region.type == RegionType.Boundary:
            distances = region.sdf(compressibleSystem.state.positions)[0]
            minBoundaryDistance = torch.min(minBoundaryDistance, distances)
    maxDistance = compressibleSystem.state.supports.max() * 2.0
    minBoundaryDistance = torch.clamp(minBoundaryDistance, min=0.0, max = maxDistance)
    ramp = (minBoundaryDistance) / maxDistance
    def rampFn(r):
        ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
        return torch.clamp(ramped, min=0.0, max = 1.0)
    ramped = rampFn(ramp)

    compressibleSystem.state.velocities[compressibleSystem.state.kinds == 0,0] += u_freestream * ramped[compressibleSystem.state.kinds == 0]




# k = 2
# u_mag = 1
# k_ = k

# ktgv = k_ / 2
# if k_ % 2 == 0:
#     phaseShift_x = np.pi / 2# / k
#     phaseShift_y = np.pi / 2# / k
# else:
#     phaseShift_x = 0
#     phaseShift_y = 0

# compressibleSystem.state.velocities[:,0] =  u_mag * torch.cos(ktgv * compressibleSystem.state.positions[:,0] * np.pi + phaseShift_x) * torch.sin(ktgv * np.pi * compressibleSystem.state.positions[:,1] + phaseShift_y)
# compressibleSystem.state.velocities[:,1] = -u_mag * torch.sin(ktgv * compressibleSystem.state.positions[:,0] * np.pi + phaseShift_x) * torch.cos(ktgv * np.pi * compressibleSystem.state.positions[:,1] + phaseShift_y)



schemeConfig.fluid.fixedSoundSpeed, config.dt = setupWeaklyCompressibleTimestep(config, schemeConfig, compressibleSystem, targetDt, verbose = True)
# print(f"Computed timestep: {config.dt:.6g}, target timestep: {targetDt:.6g}, diff: {abs(config.dt - targetDt):.6g}")

# ke0 = compressibleSystem.state.masses * torch.linalg.norm(compressibleSystem.state.velocities, dim=1)**2 * 0.5
# m_total = compressibleSystem.state.masses.sum().cpu().item()
# # kineticEnergy = np.array([ke.cpu().item() for ke in kes]).sum()

# domainL = config.domain.max[0].cpu().item() - config.domain.min[0].cpu().item()
# Ek0_theoretical = 0.25 * u_mag**2 * schemeConfig.fluid.restDensity * domainL**2
# print(f"Theoretical initial kinetic energy: {Ek0_theoretical:.6g}")
# print(f"Initial kinetic energy from simulation: {ke0.sum().cpu().item():.6g}")
# print(f'Difference: {abs(Ek0_theoretical - ke0.sum().cpu().item()):.6g}')

t = torch.tensor(0, device = device, dtype = dtype)

obstacleLinearVelocity = torch.tensor(args.linearVelocityDirection, device = device, dtype = dtype) * args.linearVelocityMagnitude
obstacleAngularVelocity = args.angularVelocityMagnitude
obstacleMotionType = args.motionType
obstacleMotionFrequency = args.motionFrequency
print(f"Obstacle motion type: {obstacleMotionType}, linear velocity: {obstacleLinearVelocity}, angular velocity: {obstacleAngularVelocity}, motion frequency: {obstacleMotionFrequency}")
if obstacleMotionType == 'fixed':
    linearVelocity = obstacleLinearVelocity#[None,:]
    angularVelocity = torch.tensor(obstacleAngularVelocity, device = device, dtype = dtype)
elif obstacleMotionType == 'sinusoidal':
    linearVelocity =  obstacleLinearVelocity * torch.cos(t * np.pi * obstacleMotionFrequency)#[:,None]
    angularVelocity = obstacleAngularVelocity * torch.cos(t * np.pi * obstacleMotionFrequency) 
else:
    raise ValueError(f"Unknown motion type: {obstacleMotionType}")

if args.linearMotion == True:
    config.rigidBodies[0].linearVelocity = linearVelocity
if args.angularMotion == True:
    config.rigidBodies[0].angularVelocity = angularVelocity
if hasattr(config, 'rigidBodies'):
    schemeConfig.rigidBodies = config.rigidBodies

runningState = compressibleSystem.initializeNewState()

# caseName = 'tgv'
exportPath = prepExport(f'{caseName}', config, schemeConfig, scheme, export_fn)
exportSimulationSystem(exportPath, 'initialState', scheme, compressibleSystem, exportAdjacency = False, stages = None, exportStagesAdjacency = False, extraData = dict({
    'frame_num': 0,
}, **extraData))

if args.plot:
    titleString = f'{args.caseName} | t = {runningState.t:.4g}/{args.timeLimit:.4g} | dt = {config.dt:.4g} | particles = {len(runningState.state.positions)} | nx = {nx} | n_h = {n_h} | L = {L} | W = {W} | obstacle: {args.obstacleActive}'
    markerSize = args.markerSize
    plotter = visualize(
        particleState = runningState.state,
        domain = config.domain,
        quantities = {
            "A": runningState.state.velocities,
            "B":runningState.state.densities,
            "C":runningState.state.UIDs
        },
        plotOptions = {
            "A": PlottingOptions(
                colorMap = UniformColorMap.viridis,
                markerSize = markerSize,
                midPoint = 0.0,
                quantityScaling = PlotScaling.Linear,
                mapping = Mapping.L2Norm,
                plotTitle = "velocities",
                plotTitleGap = 0.08,
                boundaryVisualization = VisualizeOptions.Visualize,
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
                # vMin=1e-10
                vMin = 0.0,
                vMax = schemeConfig.fluid.fixedSoundSpeed * 0.1
            ),
            "B": PlottingOptions(
                colorMap = DivergingColorMap.RdBu,
                flipColorMap=True,
                markerSize = markerSize,
                midPoint = 1.0,
                quantityScaling = PlotScaling.Linear,
                plotTitle = "densities",
                vMin = 0.95,
                vMax = 1.05,
                plotTitleGap = 0.08,
                boundaryVisualization = VisualizeOptions.Visualize,
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
            ),
            "C": PlottingOptions(
                colorMap = CyclicColorMap.twilight,
                # flipColorMap=True,
                markerSize = markerSize,
                # midPoint = 1.0,
                quantityScaling = PlotScaling.Linear,
                plotTitle = "UIDs",
                # vMin = 0.95,
                # vMax = 1.05
                plotTitleGap = 0.08,
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
            )
        },
        figTitle = titleString,
        mosaic = 'ABC',
        figsize= (args.plotWidth,5),
        backend='vispy',
        # backend='pyVista',
        # backendOptions = {
        #     # In notebooks, use trame for reliable live updates.
        #     'jupyter_backend': 'trame',
        # }
    )

    plotter.updateTitle(titleString)

    imagePath = f'{exportPath}/images'
    os.makedirs(imagePath, exist_ok = True)
    plotter.export(f'{imagePath}/frame_00000.png', dpi = 300)



schemeConfig.diffusionParams.inviscid = True
schemeConfig.diffusionParams.viscidNu = 0.01

nu = schemeConfig.diffusionParams.viscidNu if schemeConfig.diffusionParams.inviscid == False else alphaToNu(schemeConfig.diffusionParams.inviscidAlpha, schemeConfig.fluid.fixedSoundSpeed, compressibleSystem.state.supports.mean().cpu().item(), config.dim)
alpha = nuToAlpha(schemeConfig.diffusionParams.viscidNu, schemeConfig.fluid.fixedSoundSpeed, compressibleSystem.state.supports.mean().cpu().item(), config.dim) if schemeConfig.diffusionParams.inviscid == False else schemeConfig.diffusionParams.inviscidAlpha

print(f'Using inviscid: {schemeConfig.diffusionParams.inviscid}, nu: {nu:.6g}, alpha: {alpha:.6g}')


# Re = u_mag / nu * (domain.max[0].cpu().item() - domain.min[0].cpu().item()) / 2
# print(f"Reynolds number: {Re:.6g}\nnu: {nu:.6g} (alpha: {alpha:.6g})\nu_mag: {u_mag:.6g}, L: {(domain.max[0].cpu().item() - domain.min[0].cpu().item()) / 2:.6g}")
# if alpha < 0.01:
#     print(f'Running with a viscosity of alpha < 0.01 may result in unstable simulations.')

# nu_limit = alphaToNu(0.01, schemeConfig.fluid.fixedSoundSpeed, compressibleSystem.state.supports.mean().cpu().item(), config.dim)
# Re_limit = u_mag / nu_limit * (domain.max[0].cpu().item() - domain.min[0].cpu().item()) / 2
# print(f'Reynolds limit based on alpha = 0.01, nu = {nu_limit:.6g}, Re = {Re_limit:.6g}')



t_limit = args.timeLimit
nSteps = int(t_limit / config.dt)

runningState = compressibleSystem.initializeNewState()

kes = []
priorStep = None
for i in (tq := tqdm(range(nSteps), leave = False)):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    result = integrator.function(
        state = runningState,
        f = fn,
        dt = config.dt,  
        config = config,
        schemeConfig = schemeConfig,
        verbose = False,
        # priorStep = priorStep
    )
    kes.append(torch.sum(0.5 * result.state.state.masses * torch.sum(result.state.state.velocities**2, dim=1)))
    # print('max_vel:', torch.linalg.norm(result.state.state.velocities, dim = -1).max())
    end.record()
    torch.cuda.synchronize()
    priorStep = result.stages[-1]
    timing = begin.elapsed_time(end)

    runningState = result.state
    t = runningState.t
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device = device, dtype = dtype)
    if obstacleMotionType == 'fixed':
        linearVelocity = obstacleLinearVelocity
        angularVelocity = torch.tensor(obstacleAngularVelocity, device = device, dtype = dtype)
    elif obstacleMotionType == 'sinusoidal':
        linearVelocity =  obstacleLinearVelocity * torch.cos(t * np.pi * obstacleMotionFrequency)
        angularVelocity = obstacleAngularVelocity * torch.cos(t * np.pi * obstacleMotionFrequency) 

    if args.linearMotion == True:
        config.rigidBodies[0].linearVelocity = linearVelocity
        schemeConfig.rigidBodies[0].linearVelocity = linearVelocity
    if args.angularMotion == True:
        config.rigidBodies[0].angularVelocity = angularVelocity
        schemeConfig.rigidBodies[0].angularVelocity = angularVelocity

    if args.plot and plotter is not None:
        if i % 10 == 0 and i > 0:
            titleString = f'{args.caseName} | t = {runningState.t:.4g}/{args.timeLimit:.4g} | dt = {config.dt:.4g} | particles = {len(runningState.state.positions)} | nx = {nx} | n_h = {n_h} | L = {L} | W = {W} | obstacle: {args.obstacleActive} | max vel: {torch.linalg.norm(runningState.state.velocities, dim = -1).max():.3g} | iter time: {timing:.3f} ms'
            plotter.updateTitle(titleString)
            plotter.updateQuantities(
                {
                    "A": runningState.state.velocities,
                    "B": runningState.state.densities,
                    "C": runningState.state.UIDs
                },
                newParticleState = runningState.state,
            )
            plotter.export(f'{imagePath}/frame_{i:05d}.png', dpi = 300)
            
    maxVel = torch.linalg.norm(runningState.state.velocities, dim = -1).max()
    tq.set_description(f"Step {i+1}/{nSteps}, time: {(i+1)*config.dt:8.4g}/{t_limit:8.4g} | max vel: {maxVel:.3g} | iter time: {timing:.3f} ms")
    # t = {runningState.t:2f}, dt = {config.dt:.3g}, ptcls = {len(runningState.state.positions)}\nTotal Energy: {totalEnergy:.3g}, Kinetic Energy: {kineticEnergy:.3g}, Thermal Energy: {thermalEnergy:.3g}'
    # break
    if torch.any(torch.isnan(runningState.state.velocities)):
        print("NaN detected in velocities, stopping simulation.")
        break



# ts = np.arange(len(kes)) * config.dt.cpu().item()
# kineticEnergy = np.array([ke.cpu().item() for ke in kes])
# E_k0 = kineticEnergy[0]

# # Fit an effective viscosity from d/dt log(E_k) = -4 * (ktgv**2) * nu_eff
# mask = (ts > 0) & (kineticEnergy > 0)
# slope = np.polyfit(ts[mask], np.log(kineticEnergy[mask] / E_k0), 1)[0]
# nu_eff = -slope / (4 * ktgv**2)
# print(f"Estimated effective viscosity: {nu_eff:.6g}, actual viscosity: {schemeConfig.diffusionParams.viscidNu
# :.6g}, diff: {abs(nu_eff - schemeConfig.diffusionParams.viscidNu
# ):.6g}")


# fig, axis = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)

# ts = np.arange(len(kes)) * config.dt.cpu().item()
# kineticEnergy = np.array([ke.cpu().item() for ke in kes])

# nu = schemeConfig.diffusionParams.viscidNu

# Ek0_theoretical = 0.25 * u_mag**2 * schemeConfig.fluid.restDensity * domainL**2

# # For u ~ exp(-2*nu*k_mode^2*t), kinetic energy decays as exp(-4*nu*k_mode^2*t).
# E_kestimate = ke0.sum().cpu().item() * np.exp(-4 * ts * (ktgv**2) * nu_eff)
# E_ktarget = ke0.sum().cpu().item() * np.exp(-4 * ts * (ktgv**2) * nu)

# axis[0,0].plot(ts, kineticEnergy, label='Kinetic Energy')
# axis[0,0].plot(ts, E_ktarget, label=f'Kinetic Energy Target (nu = {nu:.4g})')
# axis[0,0].plot(ts, E_kestimate, label=f'Kinetic Energy Estimate (nu_eff = {nu_eff:.4g})')
# axis[0,0].set_xlabel('Time')
# axis[0,0].set_ylabel('Kinetic Energy')
# axis[0,0].legend()

# axis[0,0].axhline(Ek0_theoretical, color='black', linestyle='--', label='Theoretical Initial Kinetic Energy')
# axis[0,0].axhline(ke0.sum().cpu().item(), color='gray', linestyle='--', label='Initial Kinetic Energy from Simulation')

# axis[0,0].set_yscale('log')

# fig.tight_layout()
# fig.savefig(f'{exportPath}/kinetic_energy_decay.png', dpi=300)

# exportSimulationSystem(exportPath, f'finalState', scheme, runningState, exportAdjacency = False, stages = result.stages, exportStagesAdjacency = True, extraData = dict(**extraData, **{
#     'kineticEnergy': kineticEnergy,
#     # 'thermalEnergy': thermalEnergy,
#     # 'totalEnergy': totalEnergy,
#     'frame_num': i,
# }))

ffmpeg_cmd = "ffmpeg -y -loglevel error -hide_banner -framerate 50 -f image2 -pattern_type glob -i 'frame_*.png' -c:v libx264 -pix_fmt yuv420p -b:v 10M output.mp4"
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4  -vf "fps=50,scale=540:-1:flags=lanczos,palettegen" palette.png'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4 -i palette.png -filter_complex "fps=25,scale=540:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)

# now copy the output.mp4 and out.gif to the parent directory for easier access
shutil.copy(f'{imagePath}/output.mp4', f'{exportPath}/output.mp4')
shutil.copy(f'{imagePath}/out.gif', f'{exportPath}/out.gif');