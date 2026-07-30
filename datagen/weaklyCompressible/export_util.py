import os
import h5py
import json

from utils import *
from typing import Any
import os, pickle
import dill
import codecs


from typing import Callable, List, Tuple, Dict, Any

def _encode_callable(fn: Callable) -> str:
    # dill can serialize local lambdas/closures used in case builders.
    return codecs.encode(dill.dumps(fn), 'base64').decode()


def _decode_callable(encoded_fn: str) -> Callable:
    raw = codecs.decode(encoded_fn.encode(), 'base64')
    try:
        return dill.loads(raw)
    except Exception:
        # Backward compatibility for configs written with pickle.
        return pickle.loads(raw)

def createOutFile(
        exportPath: str
):
    os.makedirs(exportPath, exist_ok=True)
    outFile = h5py.File(f'{exportPath}/trajectory.h5', 'w')

    return outFile


def writeInitialData(
        exportPath: str,
        outFile: h5py.File,
        scheme: Any,
        config: SimulationConfig,
        schemeConfig: Any,
        args: Any,
        runningState: Any,
        extraData: dict = {}
):
    outFile.attrs['scheme'] = scheme.name if isinstance(scheme, CompressibleSPHScheme) or isinstance(scheme, WeaklyCompressibleSPHScheme) else scheme
    outFile.attrs['time'] = runningState.t if isinstance(runningState.t, float) else runningState.t.cpu().item()
    outFile.create_group('states')
    outFile.create_group('stages')


    uniqueParticles = True
    writeStages = False

    for key, value in extraData.items():
        if not isinstance(value, dict):
            # print(f'Writing attribute: {key} = {value}')
            outFile.attrs[key] = value
        else:
            # print(f'Writing nested attributes for: {key}')
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (list, np.ndarray)):
                    # print(f'Writing attribute: {key}_{subkey} = {subvalue}')
                    outFile.attrs[f'{key}_{subkey}'] = np.array(subvalue)
                else:
                    # print(f'Writing attribute: {key}_{subkey} = {subvalue}')
                    outFile.attrs[f'{key}_{subkey}'] = subvalue

    outFile.attrs['uniqueParticles'] = uniqueParticles
    outFile.attrs['exportInterval'] = args.exportInterval
    outFile.attrs['original_dt'] = config.dt if isinstance(config.dt, float) else config.dt.cpu().item()
    outFile.attrs['exportRatio'] = args.exportInterval / (config.dt if isinstance(config.dt, float) else config.dt.cpu().item())

    outFile.create_group('config')
    loadedConfig = json.load(open(f'{exportPath}/config.json', 'r'))
    copy_dict_to_h5(outFile['config'], loadedConfig)

    # Export Rigid Body
        
    rigidBodyGroup = outFile.create_group('rigidBodies')
    for r, rigidBody in enumerate(schemeConfig.rigidBodies):
        cGroup = rigidBodyGroup.create_group(f'rigidBody_{r:02d}')
        cGroup.attrs['centerOfMass'] = rigidBody.centerOfMass.cpu().numpy()
        cGroup.attrs['orientation'] = rigidBody.orientation.cpu().numpy()
        cGroup.attrs['angularVelocity'] = rigidBody.angularVelocity.cpu().numpy()
        cGroup.attrs['linearVelocity'] = rigidBody.linearVelocity.cpu().numpy()
        cGroup.attrs['mass'] = rigidBody.mass.cpu().numpy()
        cGroup.attrs['inertia'] = rigidBody.inertia.cpu().numpy()

        cGroup.create_dataset('particlePositions', data=rigidBody.particlePositions.cpu().numpy())
        cGroup.create_dataset('particleVelocities', data=rigidBody.particleVelocities.cpu().numpy())
        cGroup.create_dataset('particleMasses', data=rigidBody.particleMasses.cpu().numpy())
        cGroup.create_dataset('particleUIDs', data=rigidBody.particleUIDs.cpu().numpy())
        cGroup.create_dataset('particleIndices', data=rigidBody.particleIndices.cpu().numpy())
        cGroup.create_dataset('particleBoundaryDistances', data=rigidBody.particleBoundaryDistances.cpu().numpy())
        cGroup.create_dataset('particleBoundaryNormals', data=rigidBody.particleBoundaryNormals.cpu().numpy())

        cGroup.create_dataset('particlePositions', data=rigidBody.ghostParticlePositions.cpu().numpy())
        cGroup.create_dataset('particleIndices', data=rigidBody.ghostParticleIndices.cpu().numpy())
        cGroup.create_dataset('particleUIDs', data=rigidBody.ghostParticleUIDs.cpu().numpy())
        cGroup.create_dataset('particleBoundaryDistances', data=rigidBody.ghostParticleBoundaryDistances.cpu().numpy())
        cGroup.create_dataset('particleBoundaryNormals', data=rigidBody.ghostParticleBoundaryNormals.cpu().numpy())

        cGroup.attrs['bodyID'] = rigidBody.bodyID
        cGroup.attrs['kind'] = rigidBody.kind.value

        cGroup.attrs['sdf'] = _encode_callable(rigidBody.sdf)


    # Export Boundary Particles
    initialState = runningState.state
    kinds = initialState.kinds
    boundaryMask = kinds == 1
    boundaryPositions = initialState.positions[boundaryMask]
    boundaryMasses = initialState.masses[boundaryMask]
    boundarySupports = initialState.supports[boundaryMask]
    boundaryUIDs = initialState.UIDs[boundaryMask]
    boundaryKinds = initialState.kinds[boundaryMask]
    boundaryOffsets = initialState.ghostOffsets[boundaryMask] if initialState.ghostOffsets is not None else None

    outFile.create_dataset('boundaryPositions', data=boundaryPositions.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('boundaryMasses', data=boundaryMasses.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('boundarySupports', data=boundarySupports.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('boundaryUIDs', data=boundaryUIDs.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('boundaryKinds', data=boundaryKinds.detach().cpu().numpy(), dtype=np.int32)
    if boundaryOffsets is not None:
        outFile.create_dataset('boundaryOffsets', data=boundaryOffsets.detach().cpu().numpy(), dtype=np.int32)
    print(f'Boundary particles written to compressed trajectory ({len(boundaryPositions)} positions)')

    # Export Fluid Particles
    fluidMask = kinds == 0
    fluidPositions = initialState.positions[fluidMask]
    fluidMasses = initialState.masses[fluidMask]
    fluidSupports = initialState.supports[fluidMask]
    fluidUIDs = initialState.UIDs[fluidMask]
    fluidKinds = initialState.kinds[fluidMask]

    outFile.create_dataset('fluidPositions', data=fluidPositions.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('fluidMasses', data=fluidMasses.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('fluidSupports', data=fluidSupports.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('fluidUIDs', data=fluidUIDs.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('fluidKinds', data=fluidKinds.detach().cpu().numpy(), dtype=np.int32)
    print(f'Initial fluid particles written to compressed trajectory ({len(fluidPositions)} positions)')

    # Export Ghost Particles
    ghostMask = kinds == 2
    ghostPositions = initialState.positions[ghostMask]
    ghostMasses = initialState.masses[ghostMask]
    ghostSupports = initialState.supports[ghostMask]
    ghostUIDs = initialState.UIDs[ghostMask]
    ghostKinds = initialState.kinds[ghostMask]
    outFile.create_dataset('ghostPositions', data=ghostPositions.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('ghostMasses', data=ghostMasses.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('ghostSupports', data=ghostSupports.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('ghostUIDs', data=ghostUIDs.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('ghostKinds', data=ghostKinds.detach().cpu().numpy(), dtype=np.int32)

    # Export Overall Particle State
    outFile.create_dataset('combinedPositions', data=initialState.positions.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedMasses', data=initialState.masses.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedSupports', data=initialState.supports.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedUIDs', data=initialState.UIDs.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('combinedKinds', data=initialState.kinds.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('combinedDensities', data=initialState.densities.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedVelocities', data=initialState.velocities.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedMaterials', data=initialState.materials.detach().cpu().numpy(), dtype=np.float32)
    if  initialState.ghostOffsets is not None:
        outFile.create_dataset('combinedGhostOffsets', data=initialState.ghostOffsets.detach().cpu().numpy(), dtype=np.float32)
        outFile.create_dataset('combinedGhostIndices', data=initialState.ghostIndices.detach().cpu().numpy(), dtype=np.int32)

    # Prepare Groups
    positionGroup = outFile.create_group('positions')
    velocityGroup = outFile.create_group('velocities')
    densityGroup = outFile.create_group('densities')
    timeGroup = outFile.create_group('times')
    rigidBodyGroup = outFile.create_group('rigidBodyTrajectories')


    return (positionGroup, velocityGroup, densityGroup, timeGroup, rigidBodyGroup)



def writeFrame(groups, i, state, stages, config, schemeConfig, uniqueParticles = True, writeStages = False):
    positionGroup, velocityGroup, densityGroup, timeGroup, rigidBodyGroup = groups
    # frameGroup = outFile['states'].create_group(f'frame_{i:05d}')
    # stageGroup = outFile['stages'].create_group(f'frame_{i:05d}')

    # frameGroup.attrs['time'] = state.t if isinstance(state.t, float) else state.t.cpu().item()
    # frameGroup.attrs['num_particles'] = len(state.state.positions)
    # frameGroup.attrs['num_fluid_particles'] = len(state.state.positions[state.state.kinds == 0])
    # frameGroup.attrs['num_boundary_particles'] = len(state.state.positions[state.state.kinds == 1])
    # frameGroup.attrs['dt'] = config.dt if isinstance(config.dt, float) else config.dt.cpu().item()

    positionGroup.create_dataset(f'frame_{i:05d}', data=state.state.positions.cpu().numpy())
    velocityGroup.create_dataset(f'frame_{i:05d}', data=state.state.velocities.cpu().numpy())
    densityGroup.create_dataset(f'frame_{i:05d}', data=state.state.densities.cpu().numpy())
    timeGroup.create_dataset(f'frame_{i:05d}', data=np.array([state.t.cpu().item() if isinstance(state.t, torch.Tensor) else state.t], dtype=np.float32))

    # if not(uniqueParticles):
    #     frameGroup.create_dataset('masses', data=state.state.masses.cpu().numpy())
    #     frameGroup.create_dataset('supports', data=state.state.supports.cpu().numpy())
    #     frameGroup.create_dataset('kinds', data=state.state.kinds.cpu().numpy())

    #     frameGroup.create_dataset('UIDs', data=state.state.UIDs.cpu().numpy())

    rbg = rigidBodyGroup.create_group(f'frame_{i:05d}')

    for r, rigidBody in enumerate(schemeConfig.rigidBodies):
        rbg.attrs[f'rigidBody_{r:02d}_centerOfMass'] = rigidBody.centerOfMass.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_orientation'] = rigidBody.orientation.cpu().numpy()
        
        rbg.attrs[f'rigidBody_{r:02d}_angularVelocity'] = rigidBody.angularVelocity.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_linearVelocity'] = rigidBody.linearVelocity.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_mass'] = rigidBody.mass.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_inertia'] = rigidBody.inertia.cpu().numpy()

    # if writeStages:
    #     for j, stage in enumerate(stages):
    #         stageGroup.create_dataset(f'stage_{j:02d}_positions', data=stage.update.dxdt.cpu().numpy())
    #         stageGroup.create_dataset(f'stage_{j:02d}_velocities', data=stage.update.dvdt.cpu().numpy())
    #         stageGroup.create_dataset(f'stage_{j:02d}_densities', data=stage.update.drhodt.cpu().numpy())