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
import copy
import argparse
import shlex    
import subprocess

# custom SPH libraries
from integrators.integration import *
from sphWarpCore import *

# This library
from compressibleSPH import *
# end of boilerplate imports


from compressibleSPH.caseUtils.sod import *


argumentParser = argparse.ArgumentParser(description="Run the Sod shock tube simulation.")
# General IO Parameters
argumentParser.add_argument('--exportPath', type=str, default='export/', help='Path to the export directory containing the saved state and config to resume from.')
argumentParser.add_argument('--plot', action='store_true', help='Whether to plot the results during the simulation.')
argumentParser.add_argument('--store', action='store_true', help='Whether to resume save the simulationstates.')
argumentParser.add_argument('--plotInterval', type=int, default=10, help='Interval (in steps) at which to plot the results.')
argumentParser.add_argument('--storeInterval', type=int, default=50, help='Interval (in steps) at which to save the simulation state.')
argumentParser.add_argument('--caseName', type=str, default='01-sodShockTube', help='Name of the case for export purposes.')

# Physical Parameters
argumentParser.add_argument('--t_limit', type=float, default=0.15, help='Time limit to run the simulation to.')
argumentParser.add_argument('--n_h', type=float, default=4.0, help='Smoothing length factor (h = n_h * dx).')
argumentParser.add_argument('--gamma', type=float, default=5/3, help='Adiabatic index of the gas.')
argumentParser.add_argument('--L', type=float, default=2.0, help='Length of the domain.')
argumentParser.add_argument('--nx', type=int, default=800, help='Number of particles in the domain.')
argumentParser.add_argument('--smoothIC', action='store_true', help='Whether to smooth the initial conditions.')
argumentParser.add_argument('--samplingRatio', type=int, default=4, help='Sampling ratio for the initial particle distribution.')

# SPH Scheme Parameters
argumentParser.add_argument('--kernel', type=str, default='B7', help='SPH kernel to use. Options: B7, WendlandC2, WendlandC4, WendlandC6.')
argumentParser.add_argument('--integrationScheme', type=str, default='rungeKutta2', help='Time integration scheme to use. Options: euler, symplecticEuler, velocityVerlet, rungeKutta2, rungeKutta4.')
argumentParser.add_argument('--viscositySwitch', type=str, default='NoneSwitch', help='Artificial viscosity switch to use. Options: NoneSwitch, Morris, CullenDehnen.')
argumentParser.add_argument('--adaptiveSupportScheme', type=str, default='Owen', help='Adaptive support scheme to use. Options: None, Owen.')
argumentParser.add_argument('--adaptiveSupportCorrections', action='store_true', help='Whether to apply corrections when using adaptive support.')
argumentParser.add_argument('--compressibleSPHScheme', type=str, default='CompSPH', help='Compressible SPH scheme to use. Options: Monaghan1997, Colagrossi2003, HuAdami2009, Grenier2009, Antuono2010.')

# Sod shock tube specific parameters
argumentParser.add_argument('--left_rho', type=float, default=1.0, help='Density of the left state.')
argumentParser.add_argument('--left_pressure', type=float, default=1.0, help='Pressure of the left state.')
argumentParser.add_argument('--left_velocity', type=float, default=0.0, help='Velocity of the left state.')
argumentParser.add_argument('--right_rho', type=float, default=0.25, help='Density of the right state.')
argumentParser.add_argument('--right_pressure', type=float, default=0.1795, help='Pressure of the right state.')
argumentParser.add_argument('--right_velocity', type=float, default=0.0, help='Velocity of the right state.')

args = argumentParser.parse_args()

############################################################################################################


nx = args.nx
gamma = args.gamma
leftState = sodInitialState(args.left_pressure, args.left_rho, args.left_velocity)
rightState = sodInitialState(args.right_pressure, args.right_rho, args.right_velocity)

samplingRatio = args.samplingRatio
smoothIC = args.smoothIC

initialStateDict = {
    'left_rho': leftState.rho,
    'left_pressure': leftState.p,
    'left_velocity': leftState.v,
    'right_rho': rightState.rho,
    'right_pressure': rightState.p,
    'right_velocity': rightState.v,
    'gamma': gamma,
    'nx': nx,
    'smoothIC': smoothIC,
    'samplingRatio': samplingRatio,
}

caseName = args.caseName

timeLimit = args.t_limit
L = args.L
dim = 1
# n_h = 4
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = get_torch_precision()

config, integrator = buildConfig(
    domain = buildDomainDescription(L, dim, True, device, dtype),
    dim = dim,
    kernel = parseKernelFunctions(args.kernel),
    targetNeighbors = n_h_to_nH(args.n_h, dim),
    
    supportMode = SupportScheme.Gather,
    gradientMode = GradientScheme.Difference,
    laplacianMode = LaplacianScheme.Brookshaw,
    integrationScheme = parseIntegrationScheme(args.integrationScheme),
    samplingScheme = SamplingScheme.regular,
    device = device,
    dtype = dtype,
    dt = 1e-3,
    adaptiveDt = True,
    cflFactor=0.3,
)
config.nx = nx

scheme = parseCompressibleSPHScheme(args.compressibleSPHScheme)

SimulationSystem, SimulationState, SimulationConfig, SimulationUpdate, fn, export_fn, import_fn = buildScheme(scheme)

schemeConfig = SimulationConfig()
schemeConfig.gamma = gamma
schemeConfig.rho0 = leftState.rho

schemeConfig.viscositySwitchParams.scheme = parseViscositySwitch(args.viscositySwitch)
schemeConfig.adaptiveSupportScheme = parseAdaptiveSupportScheme(args.adaptiveSupportScheme)
schemeConfig.adaptiveSupportCorrections = args.adaptiveSupportCorrections

### Configuration setup fully at this point, now build the system

compSystem = buildSod1D(
    SimulationSystem, SimulationState,
    samplingRatio,
    leftState,
    rightState,
    gamma, config,
    True,
    
    adaptiveSupportScheme = schemeConfig.adaptiveSupportScheme,
)
kineticEnergy_ = 0.5 * (torch.linalg.norm(compSystem.state.velocities, dim = -1) **2 * compSystem.state.masses).sum()
thermalEnergy_ = (compSystem.state.internalEnergies * compSystem.state.masses).sum()
totalEnergy = kineticEnergy_ + thermalEnergy_

runningState = compSystem.initializeNewState()

## Plot the initial state

if args.plot:
    fig, axis = plotSod(runningState.state, config, schemeConfig, config.domain, gamma, leftState, rightState, plotReference = True, plotLabels = False, scatter = False, t_ = runningState.t)

if args.store:
    exportPath = prepExport(f'{caseName}', config, schemeConfig, scheme, export_fn)
    exportSimulationSystem(exportPath, 'initialState', scheme, compSystem, exportAdjacency = False, stages = None, exportStagesAdjacency = False, extraData = dict({
        'kineticEnergy': kineticEnergy_,
        'thermalEnergy': thermalEnergy_,
        'totalEnergy': totalEnergy,
        'frame_num': 0,
    }, **initialStateDict))

if args.plot:
    imagePath = f'{exportPath}/images'
    os.makedirs(imagePath, exist_ok = True)
    fig.savefig(f'{imagePath}/frame_{0:05d}.png')

    
nSteps = int(timeLimit / config.dt)
print(f"Running with dt: {config.dt}, which gives nSteps: {nSteps}")


trajectory = []
runningState = compSystem.initializeNewState()
priorStep = None
for i in (tq := tqdm(range(nSteps), leave = True)):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    # print(f"Step {i+1}/{nSteps}, time: {runningState.t:.4f}")
    result = integrator.function(
        state = runningState,
        f = fn,
        dt = config.dt,
        config = config,
        compParams = schemeConfig,
        verbose = False,
        # priorStep = priorStep
    )
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = begin.elapsed_time(end)

    runningState = result.state
    priorStep = result.stages[-1]

    kineticEnergy_ = 0.5 * (torch.linalg.norm(runningState.state.velocities, dim = -1) **2 * runningState.state.masses).sum()
    thermalEnergy_ = (runningState.state.internalEnergies * runningState.state.masses).sum()
    totalEnergy = kineticEnergy_ + thermalEnergy_

    tq.set_description(f"t: {runningState.t:.4f}, KE: {kineticEnergy_:.4f}, TE: {thermalEnergy_:.4f}, TE+KE: {totalEnergy:.4f}")
        
    trajectory.append((
        i,
        runningState.t,
        elapsed_time_ms,
        kineticEnergy_.cpu().item(),
        thermalEnergy_.cpu().item(),
        totalEnergy.cpu().item(),
    ))
    if args.plot:
        if (i % args.plotInterval == 0 or i == nSteps - 1) and i > 0:
            for ax in axis.flatten():
                ax.clear()
            plotSod_(fig, axis, runningState.state, config, schemeConfig, config.domain, gamma, leftState, rightState, plotReference = True, plotLabels = False, scatter = True, t_ = runningState.t)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(f'{imagePath}/frame_{i:05d}.png')
    if args.store:
        if i % args.storeInterval == 0:
            exportSimulationSystem(exportPath, f'state_{i:04d}', scheme, runningState, exportAdjacency = False, stages = result.stages, exportStagesAdjacency = True, extraData = dict(**initialStateDict, **{
                'kineticEnergy': kineticEnergy_,
                'thermalEnergy': thermalEnergy_,
                'totalEnergy': totalEnergy,
                'frame_num': i,
            }))

if args.store:
    exportSimulationSystem(exportPath, f'finalState', scheme, runningState, exportAdjacency = False, stages = result.stages, exportStagesAdjacency = True, extraData = dict(**initialStateDict, **{
        'kineticEnergy': kineticEnergy_,
        'thermalEnergy': thermalEnergy_,
        'totalEnergy': totalEnergy,
        'frame_num': i,
    }))

if args.plot:# Now run these commands:
    # ffmpeg -framerate 50 -f image2 -pattern_type glob -i 'frame_*.png' -c:v libx264 -pix_fmt yuv420p -b:v 10M output.mp4
    # ffmpeg -i output.mp4  -vf "fps=50,scale=540:-1:flags=lanczos,palettegen" palette.png
    # ffmpeg -i output.mp4 -i palette.png -filter_complex "fps=25,scale=540:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif
    ffmpeg_cmd = "ffmpeg -y -loglevel error -hide_banner -framerate 50 -f image2 -pattern_type glob -i 'frame_*.png' -c:v libx264 -pix_fmt yuv420p -b:v 10M output.mp4"
    subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
    ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4  -vf "fps=50,scale=540:-1:flags=lanczos,palettegen" palette.png'
    subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
    ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4 -i palette.png -filter_complex "fps=25,scale=540:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif'
    subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)

    # now copy the output.mp4 and out.gif to the parent directory for easier access
    import shutil
    shutil.copy(f'{imagePath}/output.mp4', f'{exportPath}/output.mp4')
    shutil.copy(f'{imagePath}/out.gif', f'{exportPath}/out.gif');