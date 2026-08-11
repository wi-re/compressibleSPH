"""Reading a run back from disk: the inverse of :mod:`.export`."""

import json
from typing import Any, Optional

import h5py
import torch

from ..configurations import dictToConfig
from ..enumTypes import CompressibleSPHScheme, WeaklyCompressibleSPHScheme, IncompressibleSPHScheme
from ..schemes import buildScheme
from .hdf5 import loadAdjacency, loadState, loadStage, hdfDtypeToTorchDtype


def schemeNameToSimulationScheme(name: str) -> CompressibleSPHScheme:
    for scheme in CompressibleSPHScheme:
        # print(f'Comparing {scheme.name.lower()} to {name.lower()}')
        if scheme.name.lower() == name.lower():
            return scheme
    for scheme in WeaklyCompressibleSPHScheme:
        if scheme.name.lower() == name.lower():
            return scheme
    # Mirrors `schemeAttribute`: whatever can be written must be readable back.
    for scheme in IncompressibleSPHScheme:
        if scheme.name.lower() == name.lower():
            return scheme
    raise ValueError(f'Unsupported scheme name: {name}')


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


def importConfigs(configPath, import_fn):
    # configPath = f'{path}/config.json'
    with open(configPath, 'r') as f:
        configDict = json.load(f)

    return dictToConfig(configDict['config']), import_fn(configDict['schemeConfig'])


__all__ = ['schemeNameToSimulationScheme', 'importSimulationSystem', 'importConfigs']
