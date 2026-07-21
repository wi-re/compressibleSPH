
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
parser.add_argument('--aoa', type=float, default=0.0, help='Angle of attack of the obstacle in degrees')
parser.add_argument('--singleObstacle', action='store_true', help='If set, only a single obstacle will be placed in the domain')

parser.add_argument('--offsetLR', type=float, default=0.5, help='x offset of the obstacle')
parser.add_argument('--offsetTD', type=float, default=0.0, help='y offset of the obstacle')
parser.add_argument('--velocity', type=float, default=1.0, help='Velocity of the obstacle')

parser.add_argument('--initialAngularVelocity', type=float, default=0.0, help='Initial angular velocity of the fluid (set relative to the origin)')
parser.add_argument('--enablePotentialField', action='store_true', help='If set, a potential field will be applied to the fluid')
parser.add_argument('--potentialFieldStrength', type=float, default=1.0, help='Strength of the potential field applied to the fluid')
parser.add_argument('--potentialFieldCenter', type=float, nargs=2, default=[0.0, 0.0], help='Center of the potential field applied to the fluid (x, y)')

parser.add_argument('--caseName', type=str, default='01-impact', help='Name of the case to run (default: 12-dambreak)')
parser.add_argument('--plot', action='store_true', help='Enable plotting of the simulation results')
parser.add_argument('--plotInterval', type=int, default=10, help='Interval for plotting (default: 10)')

args = parser.parse_args()


nx = args.nx
dim = 2
L = args.L
W = args.L
n_h = args.n_h
targetDt = args.targetDt

gamma = 5/3
rho0 = 1
nu_visc = 0.0005
freeSurface = True

# obstacle = args.obstacle
timestamp = getCurrentTimestamp()

obstacleText = f"{args.obstacleType}_maxExtent{args.maxExtent}_aspectRatio{args.aspectRatio}_offsetLR{args.offsetLR}_offsetTD{args.offsetTD}_velocity{args.velocity}"
caseName = f'{args.caseName}_{timestamp}_{nx}_{n_h}_{L}_{W}_{obstacleText}'

extraData = {
    'nx': nx,
    'dim': dim,
    'L': L,
    'n_h': n_h,

    'gamma': gamma,
    'rho0': rho0,
    'nu_visc': nu_visc,
    'initialAngularVelocity': args.initialAngularVelocity,
    'enablePotentialField': args.enablePotentialField,
    'potentialFieldStrength': args.potentialFieldStrength,
    'potentialFieldCenter': args.potentialFieldCenter,
    # 'obstacle': obstacle
}
########################################################################################################################
# Generic initialization code #
########################################################################################################################
import copy
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = get_torch_precision()


dx = L / (nx)# * 3 / 2
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
    targetNeighbors = n_h_to_nH(args.n_h, dim),
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


from warpSPH.utils.naca import *

if args.enablePotentialField:
    B = args.potentialFieldStrength

    schemeConfig.gravityConfig.active = True
    schemeConfig.gravityConfig.type = GravityType.PotentialField
    schemeConfig.gravityConfig.magnitude = B
    schemeConfig.gravityConfig.origin =[args.potentialFieldCenter[0], args.potentialFieldCenter[1]]

########################################################################################################################
# Build the reigons per case #
########################################################################################################################
fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert = False)



maxExtent = args.maxExtent
aspectRatio = args.aspectRatio
offsetX = args.offsetLR
offsetY = args.offsetTD

from utils import buildObstacleSDF

left_fluid = buildObstacleSDF(args.obstacleType, -args.offsetLR, -args.offsetTD, args.maxExtent, args.aspectRatio, args.aoa, config, schemeConfig, L, L)
right_fluid = buildObstacleSDF(args.obstacleType, args.offsetLR, args.offsetTD, args.maxExtent, args.aspectRatio, args.aoa, config, schemeConfig, L, L)

regions = []

regions.append(buildRegion(config, schemeConfig, left_fluid, RegionType.Fluid, initialConditions = {}, shortEdge = W > L))
regions.append(buildRegion(config, schemeConfig, right_fluid, RegionType.Fluid, initialConditions = {}, shortEdge = W > L))

compressibleSystem = initializeWeaklyCompressibleSimulation(regions, config, schemeConfig, SimulationSystem, SimulationState, verbose = True)


compressibleSystem.state.velocities[compressibleSystem.state.positions[:,0] < 0,0] = args.velocity
compressibleSystem.state.velocities[compressibleSystem.state.positions[:,0] > 0,0] = -args.velocity

compressibleSystem.state.velocities[:,0] += args.initialAngularVelocity * (compressibleSystem.state.positions[:,1] - args.potentialFieldCenter[1])
compressibleSystem.state.velocities[:,1] += -args.initialAngularVelocity * (compressibleSystem.state.positions[:,0] - args.potentialFieldCenter[0])

########################################################################################################################
# Setup the system #
########################################################################################################################

schemeConfig.fluid.fixedSoundSpeed, config.dt = setupWeaklyCompressibleTimestep(config, schemeConfig, compressibleSystem, targetDt, verbose = True)
# config.dt = config.dt * 2
print(f"Computed timestep: {config.dt:.6g}, target timestep: {targetDt:.6g}, diff: {abs(config.dt - targetDt):.6g}, c0: {schemeConfig.fluid.fixedSoundSpeed:.6g}")

runningState = compressibleSystem.initializeNewState()

exportPath = prepExport(f'{caseName}', config, schemeConfig, scheme, export_fn)
exportSimulationSystem(exportPath, 'initialState', scheme, compressibleSystem, exportAdjacency = False, stages = None, exportStagesAdjacency = False, extraData = dict({
    'frame_num': 0,
}, **extraData))

schemeConfig.diffusionParams.inviscid = True
schemeConfig.diffusionParams.viscidNu = 0.01

########################################################################################################################
# Setup the plotting #
########################################################################################################################

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
        figsize= (16,10),
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
########################################################################################################################
# Run the simulation #
########################################################################################################################

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
            plotter.updateTitle(f"Step {i+1}/{nSteps}, time: {(i+1)*config.dt:8.4g}/{args.timeLimit:8.4g} | max vel: {torch.linalg.norm(runningState.state.velocities, dim = -1).max():.3g} | iter time: {timing:.3f} ms")
        # break
            
    maxVel = torch.linalg.norm(runningState.state.velocities, dim = -1).max()
    tq.set_description(f"Step {i+1}/{nSteps}, time: {(i+1)*config.dt:8.4g}/{args.timeLimit:8.4g} | ptcls: {len(runningState.state.positions)} | max vel: {maxVel:.3g} | iter time: {timing:.3f} ms")
    # t = {runningState.t:2f}, dt = {config.dt:.3g}, ptcls = {len(runningState.state.positions)}\nTotal Energy: {totalEnergy:.3g}, Kinetic Energy: {kineticEnergy:.3g}, Thermal Energy: {thermalEnergy:.3g}'
    # break
    if torch.any(torch.isnan(runningState.state.velocities)):
        print("NaN detected in velocities, stopping simulation.")
        break

########################################################################################################################
# Finalize the simulation #
########################################################################################################################

ffmpeg_cmd = "ffmpeg -y -loglevel error -hide_banner -framerate 50 -f image2 -pattern_type glob -i 'frame_*.png' -c:v libx264 -pix_fmt yuv420p -b:v 10M output.mp4"
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4  -vf "fps=50,scale=540:-1:flags=lanczos,palettegen" palette.png'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4 -i palette.png -filter_complex "fps=25,scale=540:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)

# # now copy the output.mp4 and out.gif to the parent directory for easier access
shutil.copy(f'{imagePath}/output.mp4', f'{exportPath}/output.mp4')
shutil.copy(f'{imagePath}/out.gif', f'{exportPath}/out.gif');