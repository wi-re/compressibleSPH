
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
parser.add_argument('--timeLimit', type=float, default=4.0, help='Time limit for the simulation')

parser.add_argument('--targetDt', type=float, default=0.0005, help='Target timestep for the simulation')

parser.add_argument('--obstacle', action='store_true', help='Include an obstacle in the simulation')
parser.add_argument('--band', type=int, default=5, help='Number of extra particle layers around the domain')
parser.add_argument('--obstacleWidthRatio', type=float, default=16, help='Width of the obstacle (default: 16) -> Width = L/16')
parser.add_argument('--obstacleHeightRatio', type=float, default=4, help='Height of the obstacle (default: 4) -> Height = L/4')
parser.add_argument('--obstacleAngle', type=float, default=-math.pi/8, help='Angle of the obstacle in radians (default: -pi/8)')
parser.add_argument('--fluidWidthRatio', type=float, default=6/5, help='Width of the fluid region (default: L*5/6)')
parser.add_argument('--fluidHeightRatio', type=float, default=3, help='Height of the fluid region (default: L/3)')
parser.add_argument('--gravityMagnitude', type=float, default=9.81, help='Magnitude of gravity (default: 9.81)')
parser.add_argument('--gravityDirection', type=float, nargs=2, default=[0.0, -1.0], help='Direction of gravity as a 2D vector (default: [0.0, -1.0])')

parser.add_argument('--caseName', type=str, default='12-dambreak', help='Name of the case to run (default: 12-dambreak)')
parser.add_argument('--plot', action='store_true', help='Enable plotting of the simulation results')
parser.add_argument('--plotInterval', type=int, default=10, help='Interval for plotting (default: 10)')

args = parser.parse_args()

obstacleWidth = args.L/args.obstacleWidthRatio
obstacleHeight = args.L/args.obstacleHeightRatio
W = args.L/args.fluidWidthRatio
H = args.L/args.fluidHeightRatio

angle = args.obstacleAngle


nx = args.nx
dim = 2
L = args.L
n_h = args.n_h
targetDt = args.targetDt

gamma = 5/3
rho0 = 1
nu_visc = 0.0005
freeSurface = True

obstacle = args.obstacle
timestamp = getCurrentTimestamp()

obstacleText = f'wObstacle_{args.obstacleWidthRatio}_{args.obstacleHeightRatio}_{angle:.2f}' if obstacle else 'noObstacle'

caseName = f'{args.caseName}_{timestamp}_{nx}_{n_h}_{L}_{args.band}_{obstacleText}_fluid_{args.fluidWidthRatio}_{args.fluidHeightRatio}'

extraData = {
    'nx': nx,
    'dim': dim,
    'L': L,
    'n_h': n_h,

    'gamma': gamma,
    'rho0': rho0,
    'nu_visc': nu_visc,
    'obstacle': obstacle
}

import copy
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = get_torch_precision()


dx = L / (nx) * 3 / 2
band = args.band


domain = buildDomainDescription(l = L + dx * (band) * 2, dim = dim, periodic = False, device = device, dtype = dtype)
domain.min = torch.tensor([-L, -L/2], device = device, dtype = dtype)
domain.max = torch.tensor([L, L/2], device = device, dtype = dtype)

interiorDomain = copy.deepcopy(domain)

domain.min -= dx * band
domain.max += dx * band


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

schemeConfig.gravityConfig.active = True
schemeConfig.gravityConfig.type = GravityType.Directional
schemeConfig.gravityConfig.magnitude = args.gravityMagnitude
schemeConfig.gravityConfig.direction = torch.tensor(args.gravityDirection, device=device, dtype=dtype)

fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert = False)

# obstacleWidth = L/16
# obstacleHeight = L/4

obstacle_sdf = lambda x: getSDF('box')['function'](x, torch.tensor([obstacleWidth/2,obstacleHeight/2]).to(x.device))


translate = lambda sdf, offset: operatorDict['translate'](sdf, torch.tensor(offset).to(device))
rotate = lambda sdf, angle: operatorDict['rotate'](sdf, angle)
union = lambda sdf1, sdf2: operatorDict['union'](sdf1, sdf2)

# W = L*5/6
# H = L/3

# angle = -math.pi/8

downShift = L/2 - obstacleHeight/2 + obstacleWidth * math.sin(abs(angle)) *2
rightShift = W/4

obstacle_sdf = rotate(obstacle_sdf, angle)
obstacle_sdf = translate(obstacle_sdf, [rightShift, -downShift])



merged_sdf = lambda x: domainSDF(x, interiorDomain, invert = False)

merged_sdf = union(merged_sdf, obstacle_sdf)
# merged_sdf = translate(obstacle_sdf, [L/2, -L/4])
domain_sdf = lambda x: sampleSDF(x, merged_sdf, invert=False)

box_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([W/2,H/2]).to(points.device)), torch.tensor([interiorDomain.min[0]+W/2,interiorDomain.min[1] + H/2]).to(points.device)), invert = False)
# config['particle']['shortEdge'] = True

# inlet_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([L/16,L/2]).to(points.device)), torch.tensor([domain.min[0]+L/16,0]).to(points.device)), invert = False)
# outlet_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([L/12,L]).to(points.device)), torch.tensor([domain.max[0]-L/12,0]).to(points.device)), invert = False)
# outletBuffer_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([L/8,L]).to(points.device)), torch.tensor([domain.max[0]-L/8,0]).to(points.device)), invert = False)


regions = []

regions.append(buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.constant))
# regions.append(buildRegion(config, schemeConfig, obstacle_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.constant))
regions.append(buildRegion(config, schemeConfig, box_sdf, RegionType.Fluid, initialConditions = {}))
# regions.append(buildRegion(sdf = fluid_sdf, config = config, type = 'forcing', dirichletValues={'velocities': forcing}))

# regions.append(buildRegion(sdf = inlet_sdf, config = config, type = 'inlet', dirichletValues={'densities': config['fluid']['rho0'], 'velocities': torch.tensor([1,0], device = device, dtype = dtype)}, updateValues = {'densities': 0, 'velocities': torch.tensor([0,0], device = device, dtype = dtype)}))

# regions.append(buildRegion(sdf = outlet_sdf, config = config, type = 'outlet'))
# regions.append(buildRegion(sdf = outletBuffer_sdf, config = config, type = 'buffer', bufferValues = ['densities', 'velocities', 'pressures']))

# regions.append(buildRegion(sdf = box_sdf, config = config, type = 'dirichlet', dirichletValues={'densities': 2.0, 'velocities': torch.tensor([1,2], device = device, dtype = dtype), 'pressures': lambda x: torch.where(x[:,0] > 0, 0.0, 1.0)}, updateValues = {'densities': 2.0}))


for region in regions:
    region = filterRegion(region, regions)



config.regions = schemeConfig.regions = regions

compressibleSystem = initializeWeaklyCompressibleSimulation(regions, config, schemeConfig, SimulationSystem, SimulationState, verbose = True)

# compressibleSystem.state.positions = shuffleParticles(compressibleSystem.state, config, schemeConfig, 128, jitterAmount = 1.0)


schemeConfig.fluid.fixedSoundSpeed, config.dt = setupWeaklyCompressibleTimestep(config, schemeConfig, compressibleSystem, targetDt, verbose = True)
print(f"Computed timestep: {config.dt:.6g}, target timestep: {targetDt:.6g}, diff: {abs(config.dt - targetDt):.6g}")

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
    markerSize = 4
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
                plotTitle = "Densities",
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
        figTitle = "Wave Equation Example",
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