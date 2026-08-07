from .optimalSupportOwen import evaluateOptimalSupportOwen
from .optimalSupportMonaghan import evaluateOptimalSupportMonaghan
from ...systems.baseState import BaseState
from ...configurations import SimulationConfig, CompressibleSPHConfig
import numpy as np
import torch
from warpSPHCore import *
from typing import Optional, Union
from ...enumTypes import AdaptiveSupportScheme
from torch.profiler import profile, record_function, ProfilerActivity

def evaluateOptimalSupport(
        particleState: BaseState,
        config: SimulationConfig,
        compParams: CompressibleSPHConfig,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        adjacency: Optional[AdjacencyList] = None,
):
    with record_function("[warpSPH] - evaluateOptimalSupport"):
        if compParams.adaptiveSupportScheme == AdaptiveSupportScheme.Owen:
            return evaluateOptimalSupportOwen(
                particles = particleState,
                config = config,
                compConfig = compParams,
                kernel_ = config.kernel,
                adjacency = adjacency,
                supportScheme = supportScheme,
                verbose = False
            )
        elif compParams.adaptiveSupportScheme == AdaptiveSupportScheme.Monaghan:
            return evaluateOptimalSupportMonaghan(
                particleState = particleState,
                config = config,
                compParams = compParams,
                supportScheme = supportScheme,
                adjacency = adjacency
            )
        elif compParams.adaptiveSupportScheme == AdaptiveSupportScheme.NoScheme:

            # print('Adaptive support scheme set to NoneSupport, skipping optimal support evaluation')
            return particleState.densities, particleState.supports, adjacency, None, None
        else:
            raise ValueError(f"Unsupported adaptive support scheme: {compParams.adaptiveSupportScheme}")