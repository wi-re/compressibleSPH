import h5py as h5

from sphWarpCore import *
import torch
from typing import Optional, Any

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

    outFile.attrs['scheme'] = scheme.name if isinstance(scheme, CompressibleSPHScheme) else scheme
    outFile.attrs['time'] = system.t

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
        if isinstance(value, torch.Tensor):
            outFile.create_dataset(key, data=value.cpu().numpy())
        else:
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

from integrators.specs import StageResult
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
        SimulationSystem, SimulationState, SimulationConfig, SimulationUpdate, fn, export_fn, import_fn = buildScheme(schemeEnum)

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
        extraData[key] = torch.from_numpy(value[:]).to(device).to(hdfDtypeToTorchDtype(value.dtype)) if value.shape else torch.tensor(value[()], device=device, dtype=hdfDtypeToTorchDtype(value.dtype))

    for key, value in inFile.attrs.items():
        if key in ['scheme', 'time']:
            continue
        extraData[key] = value

    inFile.close()

    return SimulationSystem(state=state, adjacency=adjacency, t = t), stages, schemeEnum, extraData

import json

from .utils import getCurrentTimestamp
from .configurations import *

def prepExport(caseName, config, schemeConfig, scheme, export_fn):
    currentTime = getCurrentTimestamp()

    cfg = configurationToDict(config)
    schemeCfg = export_fn(schemeConfig)

    exportDict = {
        'scheme': scheme.name if isinstance(scheme, CompressibleSPHScheme) else scheme,
        'config': cfg,
        'schemeConfig': schemeCfg,
        'timestamp': currentTime,
    }

    exportPath = f'export/{caseName}'

    os.makedirs(f'export/{caseName}', exist_ok=True)
    configPath = f'export/{caseName}/config.json'

    json.dump(exportDict, open(configPath, 'w'), indent=4)

    return exportPath



def importConfigs(configPath, import_fn):
    # configPath = f'{path}/config.json'
    with open(configPath, 'r') as f:
        configDict = json.load(f)
        
    return dictToConfig(configDict['config']), import_fn(configDict['schemeConfig'])


from .enumTypes import *
from integrators.integration import IntegrationSchemeType
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
