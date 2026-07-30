
import argparse
parser = argparse.ArgumentParser(description='Run the dam break simulation with obstacle.')

parser.add_argument('--nx', type=int, default=128, help='Number of particles along the x-axis')
parser.add_argument('--markerSize', type=int, default=4, help='Size of the markers in the plot')
parser.add_argument('--plotWidth', type=int, default=28, help='Width of the plot in inches')
parser.add_argument('--plotHeight', type=int, default=8, help='Height of the plot in inches')
parser.add_argument('--plotDensity', action='store_true', help='Plot density in the visualization')
parser.add_argument('--n_h', type=int, default=4, help='Target number of neighbors')
parser.add_argument('--L', type=float, default=2.0, help='Length of the domain')
parser.add_argument('--W', type=float, default=4.0, help='Width of the domain')

# parser.add_argument('--fluidWidth', type=float, default=2.0 * 5.0 / 6.0, help='Width of the fluid region') # L * 5/6
# parser.add_argument('--fluidHeight', type=float, default=2.0 / 3.0, help='Height of the fluid region') # L / 3

parser.add_argument('--timeLimit', type=float, default=4.0, help='Time limit for the simulation')
parser.add_argument('--enableFreestream', action='store_true', help='Enable freestream boundary conditions')
parser.add_argument('--forcingWidth', type=float, default=2.0/16.0, help='Width of the forcing region')
parser.add_argument('--freeStreamVelocity', type=float, default=1.0, help='Velocity of the free stream')
parser.add_argument('--band', type=int, default=5, help='Number of particle bands around the domain for boundary conditions')

parser.add_argument('--targetDt', type=float, default=0.0005, help='Target timestep for the simulation')

parser.add_argument('--caseName', type=str, default='3-dambreak', help='Name of the case to run (default: 12-dambreak)')
parser.add_argument('--plot', action='store_true', help='Enable plotting of the simulation results')
parser.add_argument('--plotInterval', type=int, default=10, help='Interval for plotting (default: 10)')

parser.add_argument('--disableGravity', action='store_true', help='Disable gravity in the simulation')
parser.add_argument('--gravityDirection', type=float, nargs=2, default=[0.0, -1.0], help='Direction of gravity (default: [0.0, -1.0])')
parser.add_argument('--gravityMagnitude', type=float, default=9.81, help='Magnitude of gravity (default: 9.81)')

parser.add_argument('--enableSloshing', action='store_true', help='Enable sloshing motion in the simulation')
parser.add_argument('--sloshingAmplitude', type=float, default=0.1, help='Amplitude of sloshing motion (default: 0.1)')
parser.add_argument('--sloshingFrequency', type=float, default=1.0, help='Frequency of sloshing motion (default: 1.0)')

parser.add_argument('--obstacleActive', action='store_true', help='Enable obstacle in the simulation')
parser.add_argument('--obstacleType', type=str, default='circle', help='Type of obstacle to include (none, circle, ellipse, box, roundedBox, equilateralTriangle, hexagon, horseshoe, star, nacaXXXX)')

parser.add_argument('--offsetX', type=float, default=3.0/4.0, help='X offset of the obstacle (default: 0.0)') # W/4
# parser.add_argument('--obstacleOffsetY', type=float, default=0.0, help='Y offset of the obstacle (default: 0.0)')
parser.add_argument('--aoa', type=float, default=0.0, help='Angle of the obstacle (default: 0.0)')
parser.add_argument('--maxExtent', type=float, default=1.0/16.0, help='Width of the obstacle (default: 1.0/16.0)') # L/16
# parser.add_argument('--obstacleHeight', type=float, default=1.0/4.0, help='Height of the obstacle (default: 1.0/4.0)') # L/4

parser.add_argument('--fillRatio', type=float, default=1.0/3.0, help='Fill ratio for the domain')
parser.add_argument('--semiPeriodic', action='store_true', help='Enable semi-periodic boundary conditions')
parser.add_argument('--fullyPeriodic', action='store_true', help='Enable fully periodic boundary conditions')
parser.add_argument('--fluidWidth', type=float, default=5/2 * 1/3, help='Width of the fluid region (default: 4.0)')

parser.add_argument('--enableNoise', action='store_true', help='Enable noise in the initial conditions')
parser.add_argument('--octaves', type=int, default=3, help='Number of octaves for the noise function')
parser.add_argument('--lacunarity', type=int, default=2, help='Lacunarity for the noise function')
parser.add_argument('--persistence', type=float, default=0.5, help='Persistence for the noise function')
parser.add_argument('--baseFrequency', type=int, default=2, help='Base frequency for the noise function')
parser.add_argument('--kind', type=str, default='perlin', help='Kind of noise function (perlin, simplex, etc.)')
parser.add_argument('--seed', type=int, default=45906734, help='Seed for the noise function')
parser.add_argument('--noiseAmplitude', type=float, default=1.0, help='Amplitude of the noise in the initial conditions')
parser.add_argument('--bandWidth', type=float, default=16.0, help='Width of the band for the noise function')

parser.add_argument('--enableKolmogorovForcing', action='store_true', help='Enable Kolmogorov forcing')
parser.add_argument('--kolmogorovForcingAmplitude', type=float, default=1/3, help='Amplitude of the Kolmogorov forcing')
parser.add_argument('--kolmogorovForcingWavenumber', type=int, default=2, help='Wavenumber of the Kolmogorov forcing')

parser.add_argument('--exportInterval', type=float, default=0.002, help='Interval for exporting simulation data')

from utils import *

def buildExtraData(args, config, freeSurface, timestamp, obstacleText, caseNamem, simSetup):        
    extraData = {
        'nx': args.nx,
        'markerSize': args.markerSize,
        'plotWidth': args.plotWidth,
        'plotHeight': args.plotHeight,
        'plotDensity': args.plotDensity,
        'n_h': args.n_h,
        'L': args.L,
        'W': args.W,

        'timeLimit': args.timeLimit,
        'enableFreestream': args.enableFreestream,
        'forcingWidth': args.forcingWidth,
        'freeStreamVelocity': args.freeStreamVelocity,
        'band': args.band,

        'targetDt': args.targetDt,

        'caseName': args.caseName,
        'plot': args.plot,
        'plotInterval': args.plotInterval,

        'disableGravity': args.disableGravity,
        'gravityDirection': args.gravityDirection,
        'gravityMagnitude': args.gravityMagnitude,

        'enableSloshing': args.enableSloshing,
        'sloshingAmplitude': args.sloshingAmplitude,
        'sloshingFrequency': args.sloshingFrequency,

        'obstacleActive': args.obstacleActive,
        'obstacleType': args.obstacleType,

        'offsetX': args.offsetX,
        'aoa': args.aoa,
        'maxExtent': args.maxExtent,

        'fillRatio': args.fillRatio,
        'semiPeriodic': args.semiPeriodic,
        'fullyPeriodic': args.fullyPeriodic,
        'fluidWidth': args.fluidWidth,

        'freeSurface': freeSurface,
        'timestamp': timestamp,
        'obstacleText': obstacleText,
        # 'caseNameFull': caseName,

        "enableNoise": args.enableNoise,
        "octaves": args.octaves,
        "lacunarity": args.lacunarity,
        "persistence": args.persistence,
        "baseFrequency": args.baseFrequency,
        "kind": args.kind,
        "seed": args.seed,
        "noiseAmplitude": args.noiseAmplitude,
        "bandWidth": args.bandWidth,

        "enableKolmogorovForcing": args.enableKolmogorovForcing,
        "kolmogorovForcingAmplitude": args.kolmogorovForcingAmplitude,
        "kolmogorovForcingWavenumber": args.kolmogorovForcingWavenumber,
        "exportInterval": args.exportInterval,
    }

    extraData['config'] = {
        'domain_min': config.domain.min.cpu().numpy().tolist(),
        'domain_max': config.domain.max.cpu().numpy().tolist(),
        'dim': config.dim,
        'kernel': config.kernel.name,
        'targetNeighbors': config.targetNeighbors,
        'supportMode': config.supportMode.name,
        'gradientMode': config.gradientMode.name,
        'laplacianMode': config.laplacianMode.name,
        'integrationScheme': config.integrationScheme.name,
        'samplingScheme': config.samplingScheme.name,
        'device': str(config.device),
        'dtype': str(config.dtype),
        # 'dt': config.dt,
        'adaptiveDt': config.adaptiveDt,
        'cflFactor': config.cflFactor,
    }

    extraData['simSetup'] = {
        'nx': simSetup.nx,
        'dim': simSetup.dim,
        'L': simSetup.L,
        'W': simSetup.W,
        'dx': simSetup.dx,
        'band': simSetup.band,
        'n_h': simSetup.n_h,
        'targetDt': simSetup.targetDt,
        'freeSurface': simSetup.freeSurface,
        'semiPeriodic': simSetup.semiPeriodic,
        'fullyPeriodic': simSetup.fullyPeriodic,
    }

    return extraData