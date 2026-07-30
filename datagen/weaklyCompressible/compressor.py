import h5py
import json

import numpy as np
import os
import argparse

from tqdm.autonotebook import tqdm

parser = argparse.ArgumentParser(description='Compress a weakly compressible SPH simulation trajectory.')

parser.add_argument('--directory', type=str, required=True, help='Directory containing the trajectory.h5 and config.json files.')
parser.add_argument('--exportInterval', type=float, default=-1.0, help='Export interval for compression (default: 0.002).')

args = parser.parse_args()
directory = args.directory
exportInterval = args.exportInterval

print(f'Compressing trajectory in directory: {directory} with export interval: {exportInterval}')

trajectoryFile = f'{directory}/trajectory.h5'
configFile = f'{directory}/config.json'

loadedConfig = json.load(open(configFile, 'r'))

trajectory = h5py.File(trajectoryFile, 'r')
numStates = len(trajectory['states'])
print(f'Loaded trajectory with {numStates} states from {trajectoryFile}')

# exportInterval = 0.002
# dt = loadedConfig['config']['dt']
state_keys = list(trajectory['states'].keys())
state_keys = sorted(state_keys, key=lambda x: int(x.split('_')[1]))
dt = trajectory['states'][state_keys[1]].attrs['time'] - trajectory['states'][state_keys[0]].attrs['time']

exportRatio = exportInterval / dt
exportRatio_i = int(exportRatio)
if exportInterval <= 0:
    exportRatio = 1
    exportRatio_i = 1
print(f'Export ratio: {exportRatio} [{exportRatio_i}] (dt={dt}, exportInterval={exportInterval})')

folderName = os.path.basename(trajectoryFile.split('/')[-2] if os.path.isdir(trajectoryFile) else os.path.dirname(trajectoryFile).split('/')[-1])
print(f'Loaded simulation from {folderName} with {numStates} states, export ratio: {exportRatio} [{exportRatio_i}] (dt={dt}, exportInterval={exportInterval})')

outFilePath = f'compressed/trajectory_{trajectory.attrs["caseName"]}_{folderName}.h5'
os.makedirs(os.path.dirname(outFilePath), exist_ok=True)

import shutil
if os.path.exists(f'{directory}/output.mp4'):
    print(f'Found output.mp4 in {directory}, copying to compressed directory')
    shutil.copy(f'{directory}/output.mp4', f'compressed/video_{trajectory.attrs["caseName"]}_{folderName}.mp4')

if os.path.exists(f'{directory}/images/'):
    imageFiles = os.listdir(f'{directory}/images/')
    imageFiles = [f for f in imageFiles if f.endswith('.png') and f.startswith('frame_')]
    # print(f'Found {len(imageFiles)} images in {directory}/images/')
    sortedImageFiles = sorted(imageFiles, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f'Found {len(sortedImageFiles)} images in {directory}/images/')

    shutil.copy(f'{directory}/images/{sortedImageFiles[0]}', f'compressed/initial_{trajectory.attrs["caseName"]}_{folderName}.png')
    shutil.copy(f'{directory}/images/{sortedImageFiles[-1]}', f'compressed/final_{trajectory.attrs["caseName"]}_{folderName}.png')

print(f'Writing compressed trajectory to {outFilePath}')
outFile = h5py.File(outFilePath, 'w')

for attr in trajectory.attrs:
    outFile.attrs[attr] = trajectory.attrs[attr]
print(f'Copied attributes from original trajectory to compressed trajectory ({len(list(trajectory.attrs.keys()))} attributes)')

outFile.attrs['exportInterval'] = exportInterval if exportInterval > 0 else dt
outFile.attrs['original_dt'] = dt
outFile.attrs['exportRatio'] = exportRatio

outFile.create_group('config')

from utils import *

copy_dict_to_h5(outFile['config'], loadedConfig)


restoredConfig = restoreConfig_from_h5(outFile['config'])

def check_equal(d1, d2):
    if sorted(d1.keys()) != sorted(d2.keys()):
        print(f'Keys do not match: {sorted(d1.keys())} vs {sorted(d2.keys())}')
        if len(d1.keys()) != len(d2.keys()):
            print(f'Number of keys do not match: {len(d1.keys())} vs {len(d2.keys())}')

        # else:
        for key in d1.keys():
            if key not in d2:
                # pass
                print(f'Key {key} not found in second dictionary')
            else:
                pass
                # print(f'Key {key} found in both dictionaries')
        for key in d2.keys():
            if key not in d1:
                # pass
                print(f'Key {key} not found in first dictionary')
            else:
                pass
                # print(f'Key {key} found in both dictionaries')
        return False
    for key in d1.keys():
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            if not check_equal(d1[key], d2[key]):
                print(f'Dictionaries do not match for key {key}: {d1[key]} vs {d2[key]}')
                return False
        elif isinstance(d1[key], list) and isinstance(d2[key], list):
            if len(d1[key]) != len(d2[key]):
                print(f'List lengths do not match for key {key}: {len(d1[key])} vs {len(d2[key])}')
                return False
            for i in range(len(d1[key])):
                if isinstance(d1[key][i], dict) and isinstance(d2[key][i], dict):
                    if not check_equal(d1[key][i], d2[key][i]):
                        print(f'Dictionaries do not match for key {key}: {d1[key][i]} vs {d2[key][i]}')
                        return False
                elif d1[key][i] != d2[key][i]:
                    print(f'Values do not match for key {key}: {d1[key][i]} vs {d2[key][i]}')
                    return False
        elif d1[key] != d2[key]:
            print(f'Values do not match for key {key}: {d1[key]} vs {d2[key]}')
            return False
    return True
print(f'Config restored from compressed trajectory is equal to original config: {check_equal(loadedConfig, restoredConfig)}')

removeGhost = True
kinds = trajectory['initialState']['kinds'][:]
mask = kinds != 2 if removeGhost else np.ones_like(kinds, dtype=bool)
mask = kinds == 0
# print(f'Initial state: {np.sum(mask)} particles after removing ghost particles (removeGhost={removeGhost})')

boundaryMask = kinds == 1
boundaryPositions = trajectory['initialState']['positions'][boundaryMask]
boundaryMasses = trajectory['initialState']['masses'][boundaryMask]
boundarySupports = trajectory['initialState']['supports'][boundaryMask]
boundaryUIDs = trajectory['initialState']['UIDs'][boundaryMask]
boundaryKinds = trajectory['initialState']['kinds'][boundaryMask]

outFile.create_dataset('boundaryPositions', data=boundaryPositions, dtype=np.float32)
outFile.create_dataset('boundaryMasses', data=boundaryMasses, dtype=np.float32)
outFile.create_dataset('boundarySupports', data=boundarySupports, dtype=np.float32)
outFile.create_dataset('boundaryUIDs', data=boundaryUIDs, dtype=np.int32)
outFile.create_dataset('boundaryKinds', data=boundaryKinds, dtype=np.int32)
print(f'Boundary particles written to compressed trajectory ({len(boundaryPositions)} positions)')


if hasattr(trajectory['initialState'], 'ghostOffsets'):
    boundaryGhostOffsets = trajectory['initialState']['ghostOffsets'][boundaryMask]
    outFile.create_dataset('boundaryGhostOffsets', data=boundaryGhostOffsets, dtype=np.int32)
else:
    print('No ghostOffsets found in initialState, skipping boundaryGhostOffsets')

fluidMask = kinds == 0
fluidPositions = trajectory['initialState']['positions'][fluidMask]
fluidMasses = trajectory['initialState']['masses'][fluidMask]
fluidSupports = trajectory['initialState']['supports'][fluidMask]
fluidUIDs = trajectory['initialState']['UIDs'][fluidMask]
fluidKinds = trajectory['initialState']['kinds'][fluidMask]

outFile.create_dataset('fluidPositions', data=fluidPositions, dtype=np.float32)
outFile.create_dataset('fluidMasses', data=fluidMasses, dtype=np.float32)
outFile.create_dataset('fluidSupports', data=fluidSupports, dtype=np.float32)
outFile.create_dataset('fluidUIDs', data=fluidUIDs, dtype=np.int32)
outFile.create_dataset('fluidKinds', data=fluidKinds, dtype=np.int32)
print(f'Initial fluid particles written to compressed trajectory ({len(fluidPositions)} positions)')

ghostMask = kinds == 2
ghostPositions = trajectory['initialState']['positions'][ghostMask]
ghostMasses = trajectory['initialState']['masses'][ghostMask]
ghostSupports = trajectory['initialState']['supports'][ghostMask]
ghostUIDs = trajectory['initialState']['UIDs'][ghostMask]
ghostKinds = trajectory['initialState']['kinds'][ghostMask]
outFile.create_dataset('ghostPositions', data=ghostPositions, dtype=np.float32)
outFile.create_dataset('ghostMasses', data=ghostMasses, dtype=np.float32)
outFile.create_dataset('ghostSupports', data=ghostSupports, dtype=np.float32)
outFile.create_dataset('ghostUIDs', data=ghostUIDs, dtype=np.int32)
outFile.create_dataset('ghostKinds', data=ghostKinds, dtype=np.int32)


outFile.create_dataset('combinedPositions', data=trajectory['initialState']['positions'][:], dtype=np.float32)
outFile.create_dataset('combinedMasses', data=trajectory['initialState']['masses'][:], dtype=np.float32)
outFile.create_dataset('combinedSupports', data=trajectory['initialState']['supports'][:], dtype=np.float32)
outFile.create_dataset('combinedUIDs', data=trajectory['initialState']['UIDs'][:], dtype=np.int32)
outFile.create_dataset('combinedKinds', data=trajectory['initialState']['kinds'][:], dtype=np.int32)
outFile.create_dataset('combinedDensities', data=trajectory['initialState']['densities'][:], dtype=np.float32)
outFile.create_dataset('combinedVelocities', data=trajectory['initialState']['velocities'][:], dtype=np.float32)
outFile.create_dataset('combinedMaterials', data=trajectory['initialState']['materials'][:], dtype=np.float32)
if 'ghostOffsets' in trajectory['initialState']:
    outFile.create_dataset('combinedGhostOffsets', data=trajectory['initialState']['ghostOffsets'][:], dtype=np.float32)
    outFile.create_dataset('combinedGhostIndices', data=trajectory['initialState']['ghostIndices'][:], dtype=np.int32)

times = []
times.append(trajectory['initialState'].attrs['time'])

for k, key in tqdm(enumerate(state_keys)):
    if (k+1) % exportRatio_i == 0 and k > 0:
        # compressedPositions.append(trajectory['states'][key]['positions'][:])
        # compressedVelocities.append(trajectory['states'][key]['velocities'][:])
        # compressedDensities.append(trajectory['states'][key]['densities'][:])
        times.append(trajectory['states'][key].attrs['time'])
times = np.array(times, dtype=np.float32)
print(f'Compressed {len(times)} states from {numStates} states, export ratio: {exportRatio} [{exportRatio_i}] (dt={dt}, exportInterval={exportInterval})')
print(f'Times shape: {times.shape}, dtype: {times.dtype}, min: {np.min(times)}, max: {np.max(times)}')
if 'times' not in outFile:
    outFile.create_dataset('times', data=times, dtype=np.float32)
else:
    del outFile['times']
    outFile.create_dataset('times', data=times, dtype=np.float32)
    # outFile['times'][:] = times
# del(times)


totalPositionsShape = (len(times),) + fluidPositions.shape
print(f'Total positions shape: {totalPositionsShape}, dtype: {fluidPositions.dtype}')
totalPositionBytes = np.prod(totalPositionsShape) * fluidPositions.dtype.itemsize
print(f'Total positions bytes: {totalPositionBytes} ({totalPositionBytes/1024**2:.2f} MB)')


compressedPositions = []
compressedPositions.append(trajectory['initialState']['positions'][:])

for k, key in tqdm(enumerate(state_keys)):
    if (k+1) % exportRatio_i == 0 and k > 0:
        compressedPositions.append(trajectory['states'][key]['positions'][:])
        # compressedVelocities.append(trajectory['states'][key]['velocities'][:])
        # compressedDensities.append(trajectory['states'][key]['densities'][:])
        # times.append(trajectory['states'][key].attrs['time'])
pos = np.array(compressedPositions, dtype=np.float32)
print(f'Compressed {len(compressedPositions)} states from {numStates} states, export ratio: {exportRatio} [{exportRatio_i}] (dt={dt}, exportInterval={exportInterval})')
print(f'Positions shape: {pos.shape}, dtype: {pos.dtype}, min: {np.min(pos)}, max: {np.max(pos)}')
if 'positions' not in outFile:
    outFile.create_dataset('positions', data=pos, dtype=np.float32)
else:
    outFile['positions'][:] = pos
del(pos)

compressedVelocities = []
compressedVelocities.append(trajectory['initialState']['velocities'][:])

for k, key in tqdm(enumerate(state_keys)):
    if (k+1) % exportRatio_i == 0 and k > 0:
        compressedVelocities.append(trajectory['states'][key]['velocities'][:])
        # compressedDensities.append(trajectory['states'][key]['densities'][:])
        # times.append(trajectory['states'][key].attrs['time'])
vel = np.array(compressedVelocities, dtype=np.float32)
print(f'Compressed {len(compressedVelocities)} states from {numStates} states, export ratio: {exportRatio} [{exportRatio_i}] (dt={dt}, exportInterval={exportInterval})')
print(f'Velocities shape: {vel.shape}, dtype: {vel.dtype}, min: {np.min(vel)}, max: {np.max(vel)}')
if 'velocities' not in outFile:
    outFile.create_dataset('velocities', data=vel, dtype=np.float32)
else:
    outFile['velocities'][:] = vel
del(vel)

compressedDensities = []
compressedDensities.append(trajectory['initialState']['densities'][:])

for k, key in tqdm(enumerate(state_keys)):
    if (k+1) % exportRatio_i == 0 and k > 0:
        compressedDensities.append(trajectory['states'][key]['densities'][:])

dens = np.array(compressedDensities, dtype=np.float32)
print(f'Compressed {len(compressedDensities)} states from {numStates} states, export ratio: {exportRatio} [{exportRatio_i}] (dt={dt}, exportInterval={exportInterval})')
print(f'Densities shape: {dens.shape}, dtype: {dens.dtype}, min: {np.min(dens)}, max: {np.max(dens)}')
if 'densities' not in outFile:
    outFile.create_dataset('densities', data=dens, dtype=np.float32)
else:
    outFile['densities'][:] = dens
del(dens)

# compressedOffsets = []
# if 'ghostOffsets' in trajectory['initialState']:
#     compressedOffsets.append(trajectory['initialState']['ghostOffsets'][:])
#     for k, key in tqdm(enumerate(state_keys)):
#         if (k+1) % exportRatio_i == 0 and k > 0:
#             compressedOffsets.append(trajectory['states'][key]['ghostOffsets'][:])
#     offsets = np.array(compressedOffsets, dtype=np.int32)
#     print(f'Compressed {len(compressedOffsets)} states from {numStates} states, export ratio: {exportRatio} [{exportRatio_i}] (dt={dt}, exportInterval={exportInterval})')
#     print(f'Ghost offsets shape: {offsets.shape}, dtype: {offsets.dtype}, min: {np.min(offsets)}, max: {np.max(offsets)}')
#     if 'ghostOffsets' not in outFile:
#         outFile.create_dataset('ghostOffsets', data=offsets, dtype=np.int32)
#     else:
#         outFile['ghostOffsets'][:] = offsets
#     del(offsets)

outFile.close()