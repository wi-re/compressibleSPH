# %matplotlib widget
# Boilerplate import code for all libraries
# Changes to the precision require re-loading the kernel and need to be done before any op uses them.
import argparse
import json

import h5py
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

from parser import *
from warpSPH.utils.naca import *
from plot import *

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = get_torch_precision()

args = parser.parse_args()

freeSurface = True if (not args.semiPeriodic and not args.fullyPeriodic) else False
freeSurface = True if args.fillRatio < 1.0 else freeSurface

simSetup = SimulationProperties(
    device = device,
    dtype = dtype,
    nx=args.nx,
    dim = 2,
    L = args.L,
    W = args.W,
    dx = args.L / args.nx,
    band = args.band,
    n_h = args.n_h,
    targetDt = args.targetDt,
    freeSurface = freeSurface,
    semiPeriodic = args.semiPeriodic,
    fullyPeriodic = args.fullyPeriodic,
)

timestamp = getCurrentTimestamp()
obstacleText = f'obstacle_{args.maxExtent:.4g}_{args.aoa:.4g}_{args.offsetX:.4g}' if args.obstacleActive else 'no_obstacle'
caseName = f'{args.caseName}/{timestamp}_{simSetup.nx}_{simSetup.n_h}_{simSetup.L}_{simSetup.W}_{obstacleText}'


domain, interiorDomain = buildDomain(simSetup)

config, integrator = buildConfig(
    domain = domain,
    dim = simSetup.dim,
    kernel = KernelFunctions.Wendland4,
    targetNeighbors = n_h_to_nH(simSetup.n_h, simSetup.dim),
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
config.nx = simSetup.nx + 2 * simSetup.band
config.dx = simSetup.dx

config.minDt = 1e-8

extraData = buildExtraData(args, config, freeSurface, timestamp, obstacleText, caseName, simSetup)

scheme = WeaklyCompressibleSPHScheme.deltaSPH
SimulationSystem, SimulationState, SimulationConfig, SimulationUpdate, fn, export_fn, import_fn = buildScheme(scheme)

schemeConfig = SimulationConfig()
schemeConfig.surfaceDetectionConfig.active = freeSurface
schemeConfig.gravityConfig.active = not args.disableGravity
schemeConfig.gravityConfig.type = GravityType.Directional
schemeConfig.gravityConfig.magnitude = args.gravityMagnitude
schemeConfig.gravityConfig.origin = args.gravityDirection   
schemeConfig.bandwith = simSetup.L / args.bandWidth / config.dx

# schemeConfig.fluid.eosType = EquationOfState.stiffTait

# round maxExtent to nearest multiple of dx
maxExtent = round(args.maxExtent / config.dx) * config.dx
# round the offsets to nearest multiple of dx
offsetX = round(args.offsetX / config.dx) * config.dx
# offsetY = round(args.offsetY / config.dx) * config.dx

presets = buildPresetObstacles(maxExtent, offsetX, args.L, args.fillRatio, args.aoa)
obstacle = presets.get(args.obstacleType)
obstacle['offsetY'] = round(obstacle['offsetY'] / config.dx) * config.dx
schemeConfig.regions = buildRegions(config, schemeConfig, simSetup, args, domain, interiorDomain, obstacle)
schemeConfig.boundaryConditions = []

compressibleSystem = initializeWeaklyCompressibleSimulation(schemeConfig.regions, config, schemeConfig, SimulationSystem, SimulationState, verbose = True)


sampleNoise(compressibleSystem, config, schemeConfig, simSetup, args)
setupFreestream(compressibleSystem, config, schemeConfig, simSetup, args)
setupKolmogorov(compressibleSystem, config, schemeConfig, simSetup, args)


schemeConfig.fluid.fixedSoundSpeed, config.dt = setupWeaklyCompressibleTimestep(config, schemeConfig, compressibleSystem, args.targetDt, verbose = True)
# Make the problem slightly stiffer
# schemeConfig.fluid.fixedSoundSpeed *= 1.25
# t = torch.tensor(0, device = device, dtype = dtype)

runningState = compressibleSystem.initializeNewState()

exportPath = prepExport(f'{caseName}', config, schemeConfig, scheme, export_fn)

if args.plot:
    plotter = setupPlotter(runningState, args, simSetup, config, schemeConfig)
    imagePath = f'{exportPath}/images'
    os.makedirs(imagePath, exist_ok = True)
    plotter.export(f'{imagePath}/frame_00000.png', dpi = 300)


nSteps = int(args.timeLimit / config.dt)

outFile = createOutFile(exportPath)


groups = writeInitialData(exportPath, outFile, scheme, config, schemeConfig, args, runningState, extraData = extraData)

lastExport = -np.inf

exportSteps = args.exportInterval / config.dt
exportSteps = int(exportSteps)
print(f'Exporting every {exportSteps} steps (dt = {config.dt:.6g}, export interval = {args.exportInterval:.6g})')

priorStep = None
plotTiming = 0.0
exportTiming = 0.0
import time
for i in (tq := tqdm(range(nSteps), leave = False)):
    if torch.cuda.is_available():
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
    else:
        begin = time.time()

    result = integrator.function(
        state = runningState,
        f = fn,
        dt = config.dt,  
        config = config,
        schemeConfig = schemeConfig,
        verbose = False,
        # priorStep = priorStep
    )
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        timing = begin.elapsed_time(end)
    else:
        end = time.time()
        timing = (end - begin) * 1000.0


    priorStep = result.stages[-1]

    cpuBeginExport = time.time()
    if i % exportSteps == 0:
        writeFrame(groups, i, result.state, result.stages, config = config, schemeConfig = schemeConfig, uniqueParticles = True, writeStages = False)
        lastExport = result.state.t
        cpuEndExport = time.time()
        exportTiming = (cpuEndExport - cpuBeginExport) * 1000.0

    runningState = result.state
    t = runningState.t

    plotBegin = time.time()
    if args.plot and plotter is not None:
        if i % 10 == 0 and i > 0:
            updatePlot(plotter, runningState, args, simSetup, config, schemeConfig, timing)        
            plotter.export(f'{imagePath}/frame_{i:05d}.png', dpi = 300)
            plotEnd = time.time()
            plotTiming = (plotEnd - plotBegin) * 1000.0
            
    maxVel = torch.linalg.norm(runningState.state.velocities, dim = -1).max()
    tq.set_description(f"Step {i+1}/{nSteps}, time: {(i+1)*config.dt:8.4g}/{args.timeLimit:8.4g} | max vel: {maxVel:.3g} | iter time: {timing:.3f} ms | export time: {exportTiming:.3f} ms | plot time: {plotTiming:.3f} ms")

    if torch.any(torch.isnan(runningState.state.velocities)):
        print("NaN detected in velocities, stopping simulation.")
        break


outFile.close()


ffmpeg_cmd = "ffmpeg -y -loglevel error -hide_banner -framerate 100 -f image2 -pattern_type glob -i 'frame_*.png' -c:v libx264 -pix_fmt yuv420p -b:v 10M output.mp4"
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4  -vf "fps=50,scale=540:-1:flags=lanczos,palettegen" palette.png'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)
ffmpeg_cmd = 'ffmpeg -y -loglevel error -hide_banner -i output.mp4 -i palette.png -filter_complex "fps=25,scale=540:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif'
subprocess.run(shlex.split(ffmpeg_cmd), check=True, cwd = imagePath)

# now copy the output.mp4 and out.gif to the parent directory for easier access
shutil.copy(f'{imagePath}/output.mp4', f'{exportPath}/output.mp4')
shutil.copy(f'{imagePath}/out.gif', f'{exportPath}/out.gif');


os.makedirs(f'compressed', exist_ok = True)

exportFolder = exportPath.split('/')[-1]
shutil.move(f'{exportPath}/trajectory.h5', f'compressed/trajectory_{args.caseName}_{exportFolder}.hdf5')
shutil.copy(f'{exportPath}/output.mp4', f'compressed/video_{args.caseName}_{exportFolder}.mp4')


frameFiles = os.listdir(imagePath)
frameFiles = [f for f in frameFiles if f.endswith('.png') and f.startswith('frame_')]
frameFiles = sorted(frameFiles, key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by frame number

# for frameFile in frameFiles:

shutil.copy(f'{imagePath}/{frameFiles[0]}', f'compressed/first_frame_{args.caseName}_{exportFolder}.png')
shutil.copy(f'{imagePath}/{frameFiles[-1]}', f'compressed/last_frame_{args.caseName}_{exportFolder}.png')