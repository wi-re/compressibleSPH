# Boilerplate import code for all libraries
# Changes to the precision require re-loading the kernel and need to be done before any op uses them.
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
import copy
import argparse
import shlex    
import subprocess

# custom SPH libraries
from warpSPHIntegrators.integration import *
from warpSPHCore import *

# This library
from warpSPH import *
# end of boilerplate imports

from warpSPH.caseUtils import *



argparser = argparse.ArgumentParser(description='Resume a Sod Shock Tube 1D simulation from a saved state.')
argparser.add_argument('--exportPath', type=str, default='export/01-sodShockTube', help='Path to the export directory containing the saved state and config to resume from.')
argparser.add_argument('--fileName', type=str, default='trajectory/finalState.h5', help='Name of the saved state file to import from the exportPath/trajectory directory.')
argparser.add_argument('--plot', action='store_true', help='Whether to plot the results during the simulation.')
argparser.add_argument('--store', action='store_true', help='Whether to resume save the simulationstates.')
argparser.add_argument('--plotInterval', type=int, default=10, help='Interval (in steps) at which to plot the results.')
argparser.add_argument('--storeInterval', type=int, default=50, help='Interval (in steps) at which to save the simulation state.')

argparser.add_argument('--t_limit', type=float, default=0.3, help='Time limit to run the simulation to.')

args = argparser.parse_args()
############################################################################################################


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = get_torch_precision()

exportPath = args.exportPath
fileName = args.fileName

importedSystem, importedStages, importedScheme, extraData = importSimulationSystem(f'{exportPath}/{fileName}', device, dtype)
bundle = buildScheme(importedScheme)
SimulationSystem, SimulationState = bundle.SimulationSystem, bundle.SimulationState
SimulationUpdate = bundle.SimulationUpdate
fn, export_fn, import_fn = bundle.stepFunction, bundle.exportFunction, bundle.importFunction

config, schemeConfig = importConfigs(f'{exportPath}/config.json', import_fn)

integrator = getIntegrator(config.integrationScheme)

leftState = sodInitialState(extraData['left_pressure'], extraData['left_rho'], extraData['left_velocity'])
rightState = sodInitialState(extraData['right_pressure'], extraData['right_rho'], extraData['right_velocity'])

initialStateDict = {
    'left_rho': leftState.rho,
    'left_pressure': leftState.p,
    'left_velocity': leftState.v,
    'right_rho': rightState.rho,
    'right_pressure': rightState.p,
    'right_velocity': rightState.v,
    'gamma': schemeConfig.gamma,
    'nx': extraData['nx'],
    'smoothIC': extraData['smoothIC'],
    'samplingRatio': extraData['samplingRatio'],
}


### Configuration setup fully at this point, now build the system

runningState = importedSystem.initializeNewState()
startIndex = extraData['frame_num']
t_limit = args.t_limit
delta_t = t_limit - runningState.t
nSteps = int(delta_t / config.dt)
print(f"Resuming from frame {startIndex} at time {runningState.t:.5f}. Running {nSteps} steps to reach {t_limit:.5f} seconds.")

if args.plot:
    imagePath = f'{exportPath}/images'
    os.makedirs(imagePath, exist_ok=True)

fig, axis = plotSod(runningState.state, config, schemeConfig, config.domain, schemeConfig.gamma, leftState, rightState, plotReference = True, plotLabels = False, scatter = False, t_ = runningState.t)


trajectory = []
runningState = importedSystem.initializeNewState()
priorStep = copy.deepcopy(importedStages[-1]) if importedStages is not None else None

for i in (tq := tqdm(range(startIndex, startIndex + nSteps), leave = True)):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    # print(f"Step {i+1}/{nSteps}, time: {runningState.t:.4f}")
    result = integrator.function(
        state = runningState,
        f = fn,
        dt = config.dt,
        config = config,
        schemeConfig = schemeConfig,
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
            plotSod_(fig, axis, runningState.state, config, schemeConfig, config.domain, schemeConfig.gamma, leftState, rightState, plotReference = True, plotLabels = False, scatter = True, t_ = runningState.t)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(f'{imagePath}/frame_{i:05d}.png')
    if args.store:
        if i % args.storeInterval == 0:
            exportSimulationSystem(exportPath, f'state_{i:04d}', importedScheme, runningState, exportAdjacency = False, stages = result.stages, exportStagesAdjacency = True, extraData = dict(**initialStateDict, **{
                'kineticEnergy': kineticEnergy_,
                'thermalEnergy': thermalEnergy_,
                'totalEnergy': totalEnergy,
                'frame_num': i,
            }))

if args.store:
    exportSimulationSystem(exportPath, f'finalState', importedScheme, runningState, exportAdjacency = False, stages = result.stages, exportStagesAdjacency = True, extraData = dict(**initialStateDict, **{
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