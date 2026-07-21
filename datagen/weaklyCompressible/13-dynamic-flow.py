
# Boilerplate import code for all libraries
# Changes to the precision require re-loading the kernel and need to be done before any op uses them.
import sphWarpCore_config as swc
from typing import Any
swc.configure(precision="float32", dim=Any) # precision: float16|half|float32|single|float64|double

import sphWarpCore as sph
from sphWarpCore.type_config import *
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
from integrators.integration import *
from sphWarpCore import *
from warpPlot import *

# This library
from warpSPH import *

# The case utilities that contain all the case setup functions for the various test cases
from warpSPH.caseUtils import *

import argparse


parser = argparse.ArgumentParser(description='Run the dam break simulation with obstacle.')

parser.add_argument('--nx', type=int, default=128, help='Number of particles along the x-axis')
parser.add_argument('--n_h', type=int, default=4, help='Target number of neighbors')
parser.add_argument('--L', type=float, default=2.0, help='Length of the domain')
parser.add_argument('--W', type=float, default=4.0, help='Width of the domain')
parser.add_argument('--timeLimit', type=float, default=4.0, help='Time limit for the simulation')
parser.add_argument('--forcingWidth', type=float, default=2.0/16.0, help='Width of the forcing region')
parser.add_argument('--freeStreamVelocity', type=float, default=1.0, help='Velocity of the free stream')
parser.add_argument('--band', type=int, default=0, help='Number of particle bands around the domain for boundary conditions')

parser.add_argument('--targetDt', type=float, default=0.0005, help='Target timestep for the simulation')

parser.add_argument('--obstacleType', type=str, default='circle', help='Type of obstacle to include (none, circle, ellipse, box, roundedBox, equilateralTriangle, hexagon, horseshoe, star, nacaXXXX)')
parser.add_argument('--maxExtent', type=float, default=0.25, help='Maximum extent of the obstacle')
parser.add_argument('--aspectRatio', type=float, default=1.0, help='Aspect ratio of the obstacle (for ellipse)')
parser.add_argument('--offsetX', type=float, default=-1.0, help='X offset of the obstacle')
parser.add_argument('--offsetY', type=float, default=0.0, help='Y offset of the obstacle')

parser.add_argument('--linearMotion', action='store_true', help='Enable linear motion of the obstacle')
parser.add_argument('--angularMotion', action='store_true', help='Enable angular motion of the obstacle')
# The motion can either be fixed, i.e., constantly spinning, or it can be a function of time, e.g., sinusoidal motion. The user can specify the type of motion using the --motionType argument.
parser.add_argument('--motionType', type=str, default='fixed', help='Type of motion for the obstacle (fixed, sinusoidal for now)')
parser.add_argument('--motionFrequency', type=float, default=1.0, help='Frequency of the motion for the obstacle')

parser.add_argument('--linearVelocityDirection', type=float, nargs=2, default=[0.0, 1.0], help='Direction of the linear motion (as a 2D vector)')
parser.add_argument('--linearVelocityMagnitude', type=float, default=0.5, help='Magnitude of the linear motion')
parser.add_argument('--angularVelocityMagnitude', type=float, default=1.0, help='Magnitude of the angular motion')



parser.add_argument('--caseName', type=str, default='13-dynamic-flow', help='Name of the case to run (default: 12-dambreak)')
parser.add_argument('--plot', action='store_true', help='Enable plotting of the simulation results')
parser.add_argument('--plotInterval', type=int, default=10, help='Interval for plotting (default: 10)')

args = parser.parse_args()


nx = args.nx
dim = 2
L = args.L
W = args.W
n_h = args.n_h
targetDt = args.targetDt

gamma = 5/3
rho0 = 1
nu_visc = 0.0005
freeSurface = False

# obstacle = args.obstacle
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
    # 'obstacle': obstacle
}

import copy
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = get_torch_precision()


dx = L / (nx) * 3 / 2
band = args.band


domain = buildDomainDescription(L + dx * (band) * 2, dim, True, device, dtype)
domain.min = torch.tensor([-W/2 - dx * (band), -L/2 - dx * (band)], device = device, dtype = dtype)
domain.max = torch.tensor([W/2 + dx * (band), L/2 + dx * (band)], device = device, dtype = dtype)
interiorDomain = buildDomainDescription(L, dim, False, device, dtype)
interiorDomain.min = torch.tensor([-W/2, -L/2], device = device, dtype = dtype)
interiorDomain.max = torch.tensor([W/2, L/2], device = device, dtype = dtype)


# domain = buildDomainDescription(L + dx * (band) * 2, dim, True, device, dtype)
# interiorDomain = buildDomainDescription(L, dim, False, device, dtype)

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
config.dx = dx
config.nx = nx

config.minDt = 1e-8
# config.dx = L / (nx * 2)

scheme = WeaklyCompressibleSPHScheme.deltaSPH
SimulationSystem, SimulationState, SimulationConfig, SimulationUpdate, fn, export_fn, import_fn = buildScheme(scheme)


schemeConfig = SimulationConfig()
schemeConfig.surfaceDetectionConfig.active = freeSurface


fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert = False)

# obstacleWidth = L/16
# obstacleHeight = L/4

from warpSPH.utils.naca import *


maxExtent = args.maxExtent
aspectRatio = args.aspectRatio
offsetX = args.offsetX
offsetY = args.offsetY

def scaleFn(points, scaleX, scaleY):
    newPoints = points.clone()
    # return newPoints
    newPoints[:,0] = points[:,0] / scaleX
    newPoints[:,1] = points[:,1] / scaleY
    # print(f"Scaled points: {newPoints}")
    return newPoints

def translateFn(points, translateX, translateY):
    newPoints = points.clone()
    newPoints[:,0] = points[:,0] - translateX
    newPoints[:,1] = points[:,1] - translateY
    return newPoints

def rotateFn(points, angle):
    newPoints = points.clone()
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    newPoints[:,0] = points[:,0] * cos_angle - points[:,1] * sin_angle
    newPoints[:,1] = points[:,0] * sin_angle + points[:,1] * cos_angle
    return newPoints

obstacleType = args.obstacleType

# circle
if obstacleType == 'circle':
    obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('circle')['function'](translateFn(x, offsetX, offsetY), torch.tensor(maxExtent).to(points.device)), invert = False)
# ellipse (emulated as a scaled circle)
elif obstacleType == 'ellipse':
    obstacle_sdf = lambda points: sampleSDF(scaleFn(points, maxExtent, maxExtent / aspectRatio), lambda x: getSDF('circle')['function'](x, torch.tensor(1.0).to(points.device)), invert = False)
# box
elif obstacleType == 'box':
    obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('box')['function'](x, torch.tensor([maxExtent,maxExtent / aspectRatio]).to(points.device)))
# roundedBox
elif obstacleType == 'roundedBox':
    obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('roundedBox')['function'](x, torch.tensor([maxExtent,maxExtent / aspectRatio]).to(points.device), torch.tensor([maxExtent/5] * 4).to(points.device)), invert = False)
# equilateralTriangle
elif obstacleType == 'equilateralTriangle':
    obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('equilateralTriangle')['function'](x, maxExtent), invert = False)
# hexagon
elif obstacleType == 'hexagon':
    obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('hexagon')['function'](x, torch.tensor(1/4).to(points.device)), invert = False)
# horseshoe
elif obstacleType == 'horseshoe':
    aperture = np.pi / 4
    obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('horseshoe')['function'](x, torch.tensor([np.sin(aperture), np.cos(aperture)]).to(points.device), maxExtent*0.85, maxExtent/8), invert = False)
# star
elif obstacleType == 'star':
    obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('star5')['function'](x,maxExtent, maxExtent *1.25), invert = False)
# naca 4412
elif obstacleType.startswith('naca'):
    naca_id = obstacleType[4:]

    aoa = args.aoa
    aoa_rad = aoa / 180 * np.pi

    scale = 1.5

    obstacle_sdf = lambda points: eval_naca(naca_id, scaleFn(rotateFn(translateFn(points, 0.0, 0.0), torch.tensor(aoa_rad).to(points.device)), scale, scale))
    tempRegion = buildRegion(config, schemeConfig, obstacle_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip, shortEdge = W > L)
    aabb = (torch.min(tempRegion.particles.positions, dim=0).values, torch.max(tempRegion.particles.positions, dim=0).values)
    # shift the airfoil so the vertical center of the bounding box is at y=0 and the leading edge is at x=offsetX
    new_offsetY = -(aabb[0][1] + aabb[1][1]) / 2
    new_offsetX = offsetX - aabb[0][0]
    obstacle_sdf = lambda points: eval_naca(naca_id, scaleFn(rotateFn(translateFn(points, new_offsetX, new_offsetY), torch.tensor(aoa_rad).to(points.device)), scale, scale))



# obstacle_sdf = 

regions = []

regions.append(buildRegion(config, schemeConfig, fluid_sdf, RegionType.Fluid, initialConditions = {}, shortEdge = W > L))
# regions.append(buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip))
# if obstacle:

bcType = BCType.noSlip
if args.linearMotion or args.angularMotion:
    bcType = BCType.constant
regions.append(buildRegion(config, schemeConfig, obstacle_sdf, RegionType.Boundary, initialConditions = {}, kind = bcType, shortEdge = W > L))



for region in regions:
    region = filterRegion(region, regions)
config.regions = schemeConfig.regions = regions


bdyPtcls = regions[-1].particles.positions
print(f'AABB of boundary particles: {torch.min(bdyPtcls, dim=0).values} to {torch.max(bdyPtcls, dim=0).values}')

# compressibleSystem = initializeWeaklyCompressibleSimulation(regions, config, schemeConfig, SimulationSystem, SimulationState, verbose = True)

# compressibleSystem.state.positions = shuffleParticles(compressibleSystem.state, config, schemeConfig, 128, jitterAmount = 1.0)


compressibleSystem = initializeWeaklyCompressibleSimulation(regions, config, schemeConfig, SimulationSystem, SimulationState, verbose = True)

# compressibleSystem.state.positions = shuffleParticles(compressibleSystem.state, config, schemeConfig, 128, jitterAmount = 1.0)

u_freestream = args.freeStreamVelocity
minBoundaryDistance = torch.ones_like(compressibleSystem.state.positions[:,0]) * np.inf
for region in regions:
    if region.type == RegionType.Boundary:
        distances = region.sdf(compressibleSystem.state.positions)[0]
        minBoundaryDistance = torch.min(minBoundaryDistance, distances)
maxDistance = compressibleSystem.state.supports.max() * 4.0
minBoundaryDistance = torch.clamp(minBoundaryDistance, min=0.0, max = maxDistance)
ramp = (minBoundaryDistance) / maxDistance
def rampFn(r):
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
    return torch.clamp(ramped, min=0.0, max = 1.0)
ramped = rampFn(ramp)

compressibleSystem.state.velocities[compressibleSystem.state.kinds == 0,0] = u_freestream * ramped[compressibleSystem.state.kinds == 0]


schemeConfig.fluid.fixedSoundSpeed, config.dt = setupWeaklyCompressibleTimestep(config, schemeConfig, compressibleSystem, targetDt, verbose = True)
# config.dt = config.dt * 2
print(f"Computed timestep: {config.dt:.6g}, target timestep: {targetDt:.6g}, diff: {abs(config.dt - targetDt):.6g}, c0: {schemeConfig.fluid.fixedSoundSpeed:.6g}")

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


enforceDirichlet(compressibleSystem, compressibleSystem.t, config.dt, config, schemeConfig)

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
schemeConfig.rigidBodies = config.rigidBodies

runningState = compressibleSystem.initializeNewState()

exportPath = prepExport(f'{caseName}', config, schemeConfig, scheme, export_fn)
exportSimulationSystem(exportPath, 'initialState', scheme, compressibleSystem, exportAdjacency = False, stages = None, exportStagesAdjacency = False, extraData = dict({
    'frame_num': 0,
}, **extraData))

schemeConfig.diffusionParams.inviscid = True
schemeConfig.diffusionParams.viscidNu = 0.01

nu = schemeConfig.diffusionParams.viscidNu if schemeConfig.diffusionParams.inviscid == False else alphaToNu(schemeConfig.diffusionParams.inviscidAlpha, schemeConfig.fluid.fixedSoundSpeed, compressibleSystem.state.supports.mean().cpu().item(), config.dim)
alpha = nuToAlpha(schemeConfig.diffusionParams.viscidNu, schemeConfig.fluid.fixedSoundSpeed, compressibleSystem.state.supports.mean().cpu().item(), config.dim) if schemeConfig.diffusionParams.inviscid == False else schemeConfig.diffusionParams.inviscidAlpha

print(f'Using inviscid: {schemeConfig.diffusionParams.inviscid}, nu: {nu:.6g}, alpha: {alpha:.6g}')

u_mag = 1
Re = u_mag / nu * (domain.max[0].cpu().item() - domain.min[0].cpu().item()) / 2
print(f"Reynolds number: {Re:.6g}\nnu: {nu:.6g} (alpha: {alpha:.6g})\nu_mag: {u_mag:.6g}, L: {(domain.max[0].cpu().item() - domain.min[0].cpu().item()) / 2:.6g}")
if alpha < 0.01:
    print(f'Running with a viscosity of alpha < 0.01 may result in unstable simulations.')
nu_limit = alphaToNu(0.01, schemeConfig.fluid.fixedSoundSpeed, compressibleSystem.state.supports.mean().cpu().item(), config.dim)
Re_limit = u_mag / nu_limit * (domain.max[0].cpu().item() - domain.min[0].cpu().item()) / 2
print(f'Reynolds limit based on alpha = 0.01, nu = {nu_limit:.6g}, Re = {Re_limit:.6g}')

result = integrator.function(
    state = runningState,
    f = fn,
    dt = config.dt,  
    config = config,
    schemeConfig = schemeConfig,
    verbose = False,
    # priorStep = priorStep
)
if args.plot:
    markerSize = 6
    plotter = visualize(
        particleState = runningState.state,
        domain = config.domain,
        quantities = {
            "A": runningState.state.velocities,
            "B":result.state.state.UIDs,
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
                boundaryVisualization= VisualizeOptions.Visualize,
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
                # vMin=0,
                # vMax=1.0
            ),
            "B": PlottingOptions(
                colorMap = CyclicColorMap.twilight,
                # colorMap = UniformColorMap.viridis,
                # flipColorMap=True,
                markerSize = markerSize,
                # midPoint = 1.0,
                quantityScaling = PlotScaling.Linear,
                plotTitle = "UIDs",
                boundaryVisualization= VisualizeOptions.Hide,
                fluidVisualization= VisualizeOptions.Visualize,
                plotTitleGap = 0.08,
                # vMin = 0.99,
                # vMax = 1.01
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
            ),
        },
        figTitle = "Initial State",
        mosaic = 'AB',
        figsize= (16,5),
        backend='vispy',
        # backend='pyVista',
        # backendOptions = {
        #     # In notebooks, use trame for reliable live updates.
        #     'jupyter_backend': 'trame',
        # }
        )

    imagePath = f'{exportPath}/images'
    os.makedirs(imagePath, exist_ok = True)
    plotter.export(f'{imagePath}/frame_00000.png', dpi = 300)


nSteps = int(args.timeLimit / config.dt)

runningState = compressibleSystem.initializeNewState()
# schemeConfig.rigidBodies[0].linearVelocity = 0.0

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
    # schemeConfig.rigidBodies[0].linearVelocity = 0.5 * torch.cos(t * np.pi * 2)
    # linearVelocity = 0.5 * torch.cos(t * np.pi * 2)
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

    currentState = runningState.state
    # print(f'-' * 80)
    # print(f'Fluid density stats: min={currentState.densities[currentState.kinds == 0].min().item()}, max={currentState.densities[currentState.kinds == 0].max().item()}, mean={currentState.densities[currentState.kinds == 0].mean().item()}')
    # print(f'Boundary density stats: min={currentState.densities[currentState.kinds == 1].min().item()}, max={currentState.densities[currentState.kinds == 1].max().item()}, mean={currentState.densities[currentState.kinds == 1].mean().item()}')
    if args.plot:
        if i % args.plotInterval == 0 :
            densities = computeDensities(runningState.state, config, schemeConfig, None)

            plotter.updateQuantities(
                {
                    "A": runningState.state.velocities,
                    "B": runningState.state.UIDs,
                },
                newParticleState = runningState.state,
            )
            plotter.export(f'{imagePath}/frame_{i:05d}.png', dpi = 300)
            plotter.updateTitle(f"Step {i+1}/{nSteps}, time: {(i+1)*config.dt:8.4g}/{args.timeLimit:8.4g} | max vel: {torch.linalg.norm(runningState.state.velocities, dim = -1).max():.3g} | iter time: {timing:.3f} ms | linear velocity: {linearVelocity.cpu().numpy() if args.linearMotion else 'N/A'}, angular velocity: {angularVelocity if args.angularMotion else 'N/A'}")
        # break
            
    maxVel = torch.linalg.norm(runningState.state.velocities, dim = -1).max()
    tq.set_description(f"Step {i+1}/{nSteps}, time: {(i+1)*config.dt:8.4g}/{args.timeLimit:8.4g} | max vel: {maxVel:.3g} | iter time: {timing:.3f} ms")
    # t = {runningState.t:2f}, dt = {config.dt:.3g}, ptcls = {len(runningState.state.positions)}\nTotal Energy: {totalEnergy:.3g}, Kinetic Energy: {kineticEnergy:.3g}, Thermal Energy: {thermalEnergy:.3g}'
    # break
    if torch.any(torch.isnan(runningState.state.velocities)):
        print("NaN detected in velocities, stopping simulation.")
        break


ffmpeg_cmd = "ffmpeg -y -loglevel error -hide_banner -framerate 50 -f image2 -pattern_type glob -i 'frame_*.png' -c:v libx264 -pix_fmt yuv420p -b:v 10M output.mp4"
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4  -vf "fps=50,scale=540:-1:flags=lanczos,palettegen" palette.png'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4 -i palette.png -filter_complex "fps=25,scale=540:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)

# # now copy the output.mp4 and out.gif to the parent directory for easier access
shutil.copy(f'{imagePath}/output.mp4', f'{exportPath}/output.mp4')
shutil.copy(f'{imagePath}/out.gif', f'{exportPath}/out.gif');