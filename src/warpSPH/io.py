import h5py as h5

from warpSPHCore import *
import torch
from typing import Optional, Any
import numpy as np

def dumpAdjacency(adjacency, adjacencyGroup):
    adjacencyGroup.create_dataset('i', data=adjacency.i.cpu().numpy())
    adjacencyGroup.create_dataset('j', data=adjacency.j.cpu().numpy())
    adjacencyGroup.create_dataset('numNeighbors', data=adjacency.numNeighbors.cpu().numpy())
    adjacencyGroup.create_dataset('edgeOffsets', data=adjacency.edgeOffsets.cpu().numpy())

    adjacencyGroup.create_dataset('queryPositions', data=adjacency.queryPositions.cpu().numpy())
    adjacencyGroup.create_dataset('querySupports', data=adjacency.querySupports.cpu().numpy())
    adjacencyGroup.create_dataset('referencePositions', data=adjacency.referencePositions.cpu().numpy())
    adjacencyGroup.create_dataset('referenceSupports', data=adjacency.referenceSupports.cpu().numpy())

    adjacencyGroup.attrs['numRows'] = adjacency.numRows
    adjacencyGroup.attrs['numCols'] = adjacency.numCols

def dumpState(state, stateGroup):
    for fieldName in state.__dict__.keys():
        if fieldName.startswith('_'):
            continue
        fieldValue = getattr(state, fieldName)
        # print(f'Dumping field {fieldName}... -> {type(fieldValue)}')
        if isinstance(fieldValue, torch.Tensor):
            stateGroup.create_dataset(fieldName, data=fieldValue.cpu().numpy())
        elif fieldValue is None:
            stateGroup.attrs[fieldName] = 'None'
        else:
            stateGroup.attrs[fieldName] = fieldValue

def dumpStage(stage, stageGroups, index, SimulationState, exportStagesAdjacency):
    stageGroups[index].attrs['index'] = index

    auxGroup = stageGroups[index].create_group('aux')
    for auxIndex, aux in enumerate(stage.aux):
        if isinstance(aux, AdjacencyList):
            if exportStagesAdjacency:
                auxAdjacencyGroup = auxGroup.create_group(f'{auxIndex}_adjacency')
                dumpAdjacency(aux, auxAdjacencyGroup)
        elif isinstance(aux, SimulationState):
            auxStateGroup = auxGroup.create_group(f'{auxIndex}_state')
            dumpState(aux, auxStateGroup)
        else: 
            print(f'Unsupported aux type: {type(aux)}')

    for updateKey, updateValue in stage.update.__dict__.items():
        if isinstance(updateValue, torch.Tensor):
            stageGroups[index].create_dataset(updateKey, data=updateValue.cpu().numpy())
        else:
            continue


from typing import Any
from .enumTypes import CompressibleSPHScheme
import os
import h5py
import torch

def exportSimulationSystem(
    exportPath,
    tag,
    scheme: CompressibleSPHScheme,
    system: Any,
    exportAdjacency = False,
    stages = None,
    exportStagesAdjacency = True,
    extraData = None,
):
    outFolderPath = f'{exportPath}/trajectory/'
    os.makedirs(outFolderPath, exist_ok=True)
    outFile = h5py.File(f'{exportPath}/trajectory/{tag}.h5', 'w')

    outFile.attrs['scheme'] = scheme.name if isinstance(scheme, CompressibleSPHScheme) or isinstance(scheme, WeaklyCompressibleSPHScheme) else scheme
    outFile.attrs['time'] = system.t if isinstance(system.t, float) else system.t.cpu().item()

    if system.adjacency is not None:
        adjacencyGroup = outFile.create_group('adjacency')
        dumpAdjacency(system.adjacency, adjacencyGroup)

    stateGroup = outFile.create_group('state')
    dumpState(system.state, stateGroup)

    if stages is not None:
        stageGroups = outFile.create_group('stages')
        stageGroupList = []
        for stageIndex, stage in enumerate(stages):
            stageGroupList.append(stageGroups.create_group(f'stage_{stageIndex}'))
            # dumpState(stage, stageGroup)
            dumpStage(stages[stageIndex], stageGroupList, stageIndex, type(system.state), exportStagesAdjacency)

    for key, value in extraData.items() if extraData is not None else {}:
        if isinstance(value, dict):
            extraGroup = outFile.create_group(f'dict_{key}')
            for subKey, subValue in value.items():
                # print(f'Exporting extra data key {key}[{subKey}] with value {subValue} of type {type(subValue)}')
                if isinstance(subValue, torch.Tensor):
                    extraGroup.create_dataset(subKey, data=subValue.cpu().numpy())
                else:
                    extraGroup.attrs[subKey] = subValue
            outFile.attrs[key] = 'dict'
        elif isinstance(value, list):
            extraGroup = outFile.create_group(f'dict_{key}')
            for subIndex, subValue in enumerate(value):
                if isinstance(subValue, torch.Tensor):
                    extraGroup.create_dataset(f'{subIndex}', data=subValue.cpu().numpy())
                elif isinstance(subValue, dict):
                    subGroup = extraGroup.create_group(f'{subIndex}')
                    for subSubKey, subSubValue in subValue.items():
                        if isinstance(subSubValue, torch.Tensor):
                            subGroup.create_dataset(subSubKey, data=subSubValue.cpu().numpy())
                        else:
                            # print(f'Exporting extra data key {key}[{subIndex}][{subSubKey}] with value {subSubValue} of type {type(subSubValue)}')
                            subGroup.attrs[subSubKey] = subSubValue
                else:
                    # print(f'Exporting extra data key {key}[{subIndex}] with value {subValue} of type {type(subValue)}')
                    extraGroup.attrs[f'{subIndex}'] = subValue
            outFile.attrs[key] = 'list'
        elif isinstance(value, torch.Tensor):
            outFile.create_dataset(key, data=value.cpu().numpy())
        else:
            # print(f'Exporting extra data key {key} with value {value} of type {type(value)}')
            outFile.attrs[key] = value

    outFile.close()

def hdfDtypeToTorchDtype(hdfDtype):
    if hdfDtype == "<i4":
        return torch.int32
    elif hdfDtype == "<i8":
        return torch.int64
    elif hdfDtype == "<f4":
        return torch.float32
    elif hdfDtype == "<f8":
        return torch.float64
    elif hdfDtype == 'bool':
        return torch.bool
    else:
        raise ValueError(f'Unsupported HDF dtype: {hdfDtype}')



def loadAdjacency(adjacencyGroup, device):

    i = torch.from_numpy(adjacencyGroup['i'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['i'].dtype))
    j = torch.from_numpy(adjacencyGroup['j'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['j'].dtype))
    numNeighbors = torch.from_numpy(adjacencyGroup['numNeighbors'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['numNeighbors'].dtype))
    edgeOffsets = torch.from_numpy(adjacencyGroup['edgeOffsets'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['edgeOffsets'].dtype))

    queryPositions = torch.from_numpy(adjacencyGroup['queryPositions'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['queryPositions'].dtype))
    querySupports = torch.from_numpy(adjacencyGroup['querySupports'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['querySupports'].dtype))
    referencePositions = torch.from_numpy(adjacencyGroup['referencePositions'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['referencePositions'].dtype))
    referenceSupports = torch.from_numpy(adjacencyGroup['referenceSupports'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['referenceSupports'].dtype))

    numRows = adjacencyGroup.attrs['numRows']
    numCols = adjacencyGroup.attrs['numCols']

    return AdjacencyList(i, j, numNeighbors, edgeOffsets, numRows, numCols, queryPositions, referencePositions, querySupports, referenceSupports)

def loadState(stateGroup, device, SimulationState):
    stateDict = {}
    for fieldName, fieldValue in stateGroup.items():
        # print(f'Loading field {fieldName}... -> {type(fieldValue)}', fieldValue)
        if isinstance(fieldValue, h5py.Dataset):
            stateDict[fieldName] = torch.from_numpy(fieldValue[:]).to(device).to(hdfDtypeToTorchDtype(fieldValue.dtype)) if fieldValue.shape else torch.tensor(fieldValue[()], device=device, dtype=hdfDtypeToTorchDtype(fieldValue.dtype))
        else:
            stateDict[fieldName] = fieldValue
    return SimulationState(**stateDict)

from warpSPHIntegrators.specs import StageResult
def loadStage(stageGroup, device, SimulationState, SimulationUpdate):
    index = stageGroup.attrs['index']

    aux = []
    for auxKey, auxValue in stageGroup['aux'].items():
        if '_adjacency' in auxKey:
            aux.append(loadAdjacency(auxValue, device))
        elif '_state' in auxKey:
            aux.append(loadState(auxValue, device, SimulationState))
        else:
            print(f'Unsupported aux type for key {auxKey}')

    updateDict = {}
    for updateKey, updateValue in stageGroup.items():
        if updateKey == 'aux':
            continue
        updateDict[updateKey] = torch.from_numpy(updateValue[:]).to(device).to(hdfDtypeToTorchDtype(updateValue.dtype)) if updateValue.shape else torch.tensor(updateValue[()], device=device, dtype=hdfDtypeToTorchDtype(updateValue.dtype))

    return StageResult(aux=aux, update=SimulationUpdate(**updateDict))


def schemeNameToSimulationScheme(name: str) -> CompressibleSPHScheme:
    for scheme in CompressibleSPHScheme:
        # print(f'Comparing {scheme.name.lower()} to {name.lower()}')
        if scheme.name.lower() == name.lower():
            return scheme
    for scheme in WeaklyCompressibleSPHScheme:
        if scheme.name.lower() == name.lower():
            return scheme
    raise ValueError(f'Unsupported scheme name: {name}')


from .schemes import buildScheme

def importSimulationSystem(
    importPath,
    device,
    dtype,
    SimulationSystem : Optional[Any] = None,
    SimulationState: Optional[Any] = None,
    SimulationUpdate: Optional[Any] = None,
):
    inFile = h5py.File(importPath, 'r')

    t = inFile.attrs['time']   
    scheme = inFile.attrs['scheme']
    schemeEnum = schemeNameToSimulationScheme(scheme)
    if SimulationSystem is None or SimulationState is None or SimulationUpdate is None:
        bundle = buildScheme(schemeEnum)
        SimulationSystem = bundle.SimulationSystem
        SimulationState = bundle.SimulationState
        SimulationUpdate = bundle.SimulationUpdate

    adjacency = loadAdjacency(inFile['adjacency'], device) if 'adjacency' in inFile else None
    state = loadState(inFile['state'], device, SimulationState)

    stageGroups = inFile['stages'] if 'stages' in inFile else None
    stages = []
    if stageGroups is not None:
        for stageKey in stageGroups.keys():
            stages.append(loadStage(stageGroups[stageKey], device, SimulationState, SimulationUpdate))

    extraData = {}
    for key, value in inFile.items():
        if key in ['adjacency', 'state', 'stages']:
            continue
        if key.startswith('dict_'):
            continue
        else:
            extraData[key] = torch.from_numpy(value[:]).to(device).to(hdfDtypeToTorchDtype(value.dtype)) if value.shape else torch.tensor(value[()], device=device, dtype=hdfDtypeToTorchDtype(value.dtype))

    for key, value in inFile.attrs.items():
        if key in ['scheme', 'time']:
            continue
        if value == 'dict':
            dictKey = key[len('dict_'):] if key.startswith('dict_') else key
            extraData[dictKey] = {}
            dictGroup = inFile[f'dict_{dictKey}'] if f'dict_{dictKey}' in inFile else inFile[dictKey]
            for subKey, subValue in dictGroup.items():
                if isinstance(subValue, h5py.Dataset):
                    extraData[dictKey][subKey] = torch.from_numpy(subValue[:]).to(device).to(hdfDtypeToTorchDtype(subValue.dtype)) if subValue.shape else torch.tensor(subValue[()], device=device, dtype=hdfDtypeToTorchDtype(subValue.dtype))
                else:
                    extraData[dictKey][subKey] = subValue
        elif value == 'list':
            listKey = key[len('dict_'):] if key.startswith('dict_') else key
            extraData[listKey] = []
            listGroup = inFile[f'dict_{listKey}'] if f'dict_{listKey}' in inFile else inFile[listKey]
            for subIndex, subValue in listGroup.items():
                if isinstance(subValue, h5py.Dataset):
                    extraData[listKey].append(torch.from_numpy(subValue[:]).to(device).to(hdfDtypeToTorchDtype(subValue.dtype)) if subValue.shape else torch.tensor(subValue[()], device=device, dtype=hdfDtypeToTorchDtype(subValue.dtype)))
                elif isinstance(subValue, h5py.Group):
                    subDict = {}
                    for subSubKey, subSubValue in subValue.items():
                        if isinstance(subSubValue, h5py.Dataset):
                            subDict[subSubKey] = torch.from_numpy(subSubValue[:]).to(device).to(hdfDtypeToTorchDtype(subSubValue.dtype)) if subSubValue.shape else torch.tensor(subSubValue[()], device=device, dtype=hdfDtypeToTorchDtype(subSubValue.dtype))
                        else:
                            subDict[subSubKey] = subSubValue
                    extraData[listKey].append(subDict)
                else:
                    extraData[listKey].append(subValue)
        else:
            extraData[key] = value

    inFile.close()

    return SimulationSystem(state=state, adjacency=adjacency, t = t), stages, schemeEnum, extraData

import json
import pickle
import dill
import codecs

from .utils import getCurrentTimestamp
from .configurations import *

def prepExport(caseName, config, schemeConfig, scheme, export_fn, exportRoot=None):
    """Write ``config.json`` for a run and return its output directory.

    ``exportRoot`` selects the parent directory for ``caseName``. It defaults to
    the ``WARPSPH_EXPORT_ROOT`` environment variable, falling back to ``export``
    relative to the CWD -- the historical behaviour. Overriding it lets parallel
    sweeps write to separate trees instead of colliding on ``export/{caseName}``.
    """
    currentTime = getCurrentTimestamp()

    cfg = configurationToDict(config)
    schemeCfg = export_fn(schemeConfig)

    exportDict = {
        'scheme': scheme.name if isinstance(scheme, CompressibleSPHScheme) or isinstance(scheme, WeaklyCompressibleSPHScheme) or isinstance(scheme, IncompressibleSPHScheme) else scheme,
        'config': cfg,
        'schemeConfig': schemeCfg,
        'timestamp': currentTime,
    }

    if exportRoot is None:
        exportRoot = os.environ.get('WARPSPH_EXPORT_ROOT', 'export')

    exportPath = os.path.join(exportRoot, caseName)

    os.makedirs(exportPath, exist_ok=True)
    configPath = os.path.join(exportPath, 'config.json')

    with open(configPath, 'w') as f:
        json.dump(exportDict, f, indent=4)

    return exportPath



def importConfigs(configPath, import_fn):
    # configPath = f'{path}/config.json'
    with open(configPath, 'r') as f:
        configDict = json.load(f)
        
    return dictToConfig(configDict['config']), import_fn(configDict['schemeConfig'])


def _encode_callable(fn):
    return codecs.encode(dill.dumps(fn), 'base64').decode()


def _decode_callable(encoded_fn):
    raw = codecs.decode(encoded_fn.encode(), 'base64')
    try:
        return dill.loads(raw)
    except Exception:
        return pickle.loads(raw)


def copy_dict_to_h5(group, d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            subgroup.attrs['taggedType'] = 'dict'
            copy_dict_to_h5(subgroup, value, indent + 1)
        elif isinstance(value, list):
            subgroup = group.create_group(key)
            subgroup.attrs['taggedType'] = 'list'
            copy_dict_to_h5(subgroup, {f'item_{i}': v for i, v in enumerate(value)}, indent + 1)
        else:
            if value is None:
                continue
            group.attrs[key] = value


def restore_config_from_h5(group, indent=0):
    config = {}
    for key, value in group.attrs.items():
        if key != 'taggedType':
            config[key] = value
    for key, subgroup in group.items():
        if subgroup.attrs['taggedType'] == 'dict':
            config[key] = restore_config_from_h5(subgroup, indent + 1)
        elif subgroup.attrs['taggedType'] == 'list':
            config[key] = [restore_config_from_h5(subgroup[f'item_{i}'], indent + 1) for i in range(len(subgroup))]
            subkeys = [k for k in list(subgroup.attrs.keys()) if k.startswith('item_')]
            for i in range(len(subkeys)):
                item_key = f'item_{i}'
                if item_key in subgroup.attrs:
                    config[key].append(subgroup.attrs[item_key])
        else:
            raise ValueError(f'Unknown type for subgroup {key}: {subgroup.attrs["taggedType"]}')
    return config


# Backward-compatible alias used by existing notebooks/scripts.
def restoreConfig_from_h5(group, indent=0):
    return restore_config_from_h5(group, indent=indent)


def createOutFile(exportPath: str):
    os.makedirs(exportPath, exist_ok=True)
    return h5py.File(f'{exportPath}/trajectory.h5', 'w')


def writeInitialData(
    exportPath: str,
    outFile: h5py.File,
    scheme: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    args: Any,
    runningState: Any,
    extraData: dict | None = None,
):
    if extraData is None:
        extraData = {}

    outFile.attrs['scheme'] = scheme.name if isinstance(scheme, CompressibleSPHScheme) or isinstance(scheme, WeaklyCompressibleSPHScheme) else scheme
    outFile.attrs['time'] = runningState.t if isinstance(runningState.t, float) else runningState.t.cpu().item()
    outFile.create_group('states')
    outFile.create_group('stages')

    uniqueParticles = True

    for key, value in extraData.items():
        if not isinstance(value, dict):
            outFile.attrs[key] = value
        else:
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (list, np.ndarray)):
                    outFile.attrs[f'{key}_{subkey}'] = np.array(subvalue)
                else:
                    outFile.attrs[f'{key}_{subkey}'] = subvalue

    outFile.attrs['uniqueParticles'] = uniqueParticles
    outFile.attrs['exportInterval'] = args.exportInterval
    outFile.attrs['original_dt'] = config.dt if isinstance(config.dt, float) else config.dt.cpu().item()
    outFile.attrs['exportRatio'] = args.exportInterval / (config.dt if isinstance(config.dt, float) else config.dt.cpu().item())

    outFile.create_group('config')
    loadedConfig = json.load(open(f'{exportPath}/config.json', 'r'))
    copy_dict_to_h5(outFile['config'], loadedConfig)

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

    outFile.create_dataset('combinedPositions', data=initialState.positions.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedMasses', data=initialState.masses.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedSupports', data=initialState.supports.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedUIDs', data=initialState.UIDs.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('combinedKinds', data=initialState.kinds.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('combinedDensities', data=initialState.densities.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedVelocities', data=initialState.velocities.detach().cpu().numpy(), dtype=np.float32)
    outFile.create_dataset('combinedMaterials', data=initialState.materials.detach().cpu().numpy(), dtype=np.float32)
    if initialState.ghostOffsets is not None:
        outFile.create_dataset('combinedGhostOffsets', data=initialState.ghostOffsets.detach().cpu().numpy(), dtype=np.float32)
        outFile.create_dataset('combinedGhostIndices', data=initialState.ghostIndices.detach().cpu().numpy(), dtype=np.int32)

    positionGroup = outFile.create_group('positions')
    velocityGroup = outFile.create_group('velocities')
    densityGroup = outFile.create_group('densities')
    timeGroup = outFile.create_group('times')
    rigidBodyGroup = outFile.create_group('rigidBodyTrajectories')

    return (positionGroup, velocityGroup, densityGroup, timeGroup, rigidBodyGroup)


def writeFrame(groups, i, state, stages, config, schemeConfig, uniqueParticles=True, writeStages=False):
    positionGroup, velocityGroup, densityGroup, timeGroup, rigidBodyGroup = groups

    positionGroup.create_dataset(f'frame_{i:05d}', data=state.state.positions.cpu().numpy())
    velocityGroup.create_dataset(f'frame_{i:05d}', data=state.state.velocities.cpu().numpy())
    densityGroup.create_dataset(f'frame_{i:05d}', data=state.state.densities.cpu().numpy())
    timeGroup.create_dataset(
        f'frame_{i:05d}',
        data=np.array([state.t.cpu().item() if isinstance(state.t, torch.Tensor) else state.t], dtype=np.float32),
    )

    rbg = rigidBodyGroup.create_group(f'frame_{i:05d}')
    for r, rigidBody in enumerate(schemeConfig.rigidBodies):
        rbg.attrs[f'rigidBody_{r:02d}_centerOfMass'] = rigidBody.centerOfMass.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_orientation'] = rigidBody.orientation.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_angularVelocity'] = rigidBody.angularVelocity.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_linearVelocity'] = rigidBody.linearVelocity.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_mass'] = rigidBody.mass.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_inertia'] = rigidBody.inertia.cpu().numpy()


from .enumTypes import *
from warpSPHIntegrators.integration import IntegrationSchemeType
def parseKernelFunctions(kernelName):
    for kernel in KernelFunctions:
        if kernel.name.lower() == kernelName.lower():
            return kernel
    raise ValueError(f"Invalid kernel name: {kernelName}. Valid options are: {[k.name for k in KernelFunctions]}")
def parseIntegrationScheme(integrationSchemeName):
    for scheme in IntegrationSchemeType:
        if scheme.name.lower() == integrationSchemeName.lower():
            return scheme
    raise ValueError(f"Invalid integration scheme name: {integrationSchemeName}. Valid options are: {[s.name for s in IntegrationSchemeType]}")
def parseViscositySwitch(viscositySwitchName):
    for switch in ViscositySwitch:
        if switch.name.lower() == viscositySwitchName.lower():
            return switch
    raise ValueError(f"Invalid viscosity switch name: {viscositySwitchName}. Valid options are: {[s.name for s in ViscositySwitch]}")
def parseAdaptiveSupportScheme(adaptiveSupportSchemeName):
    for scheme in AdaptiveSupportScheme:
        if scheme.name.lower() == adaptiveSupportSchemeName.lower():
            return scheme
    raise ValueError(f"Invalid adaptive support scheme name: {adaptiveSupportSchemeName}. Valid options are: {[s.name for s in AdaptiveSupportScheme]}")
def parseCompressibleSPHScheme(schemeName):
    for scheme in CompressibleSPHScheme:
        if scheme.name.lower() == schemeName.lower():
            return scheme
    raise ValueError(f"Invalid compressible SPH scheme name: {schemeName}. Valid options are: {[s.name for s in CompressibleSPHScheme]}")
def parseIncompressibleSPHScheme(schemeName):
    for scheme in IncompressibleSPHScheme:
        if scheme.name.lower() == schemeName.lower():
            return scheme
    raise ValueError(f"Invalid incompressible SPH scheme name: {schemeName}. Valid options are: {[s.name for s in IncompressibleSPHScheme]}")
def parseWeaklyCompressibleSPHScheme(schemeName):
    for scheme in WeaklyCompressibleSPHScheme:
        if scheme.name.lower() == schemeName.lower():
            return scheme
    raise ValueError(f"Invalid weakly compressible SPH scheme name: {schemeName}. Valid options are: {[s.name for s in WeaklyCompressibleSPHScheme]}")