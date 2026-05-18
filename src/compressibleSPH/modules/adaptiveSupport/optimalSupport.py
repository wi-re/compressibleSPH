from .optimalSupportOwen import evaluateOptimalSupportOwen
from .optimalSupportMonaghan import evaluateOptimalSupportMonaghan
from ...systems.baseState import BaseState
from ...config import SimulationConfig, CompressibleSPHConfig
import numpy as np
import torch
from sphWarpCore import *
from typing import Optional, Union
from ...enumTypes import AdaptiveSupportScheme

def evaluateOptimalSupport(
        particleState: BaseState,
        config: SimulationConfig,
        compParams: CompressibleSPHConfig,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        adjacency: Optional[AdjacencyList] = None,
):
    if CompressibleSPHConfig.adaptiveSupportScheme == AdaptiveSupportScheme.Owen:
        return evaluateOptimalSupportOwen(
            particles = particleState,
            config = config,
            compConfig = compParams,
            kernel_ = config.kernel,
            adjacency = adjacency,
            supportScheme = supportScheme,
            verbose = False
        )
    elif CompressibleSPHConfig.adaptiveSupportScheme == AdaptiveSupportScheme.Monaghan:
        return evaluateOptimalSupportMonaghan(
            particleState = particleState,
            config = config,
            compParams = compParams,
            supportScheme = supportScheme,
            adjacency = adjacency
        )
    else:
        raise ValueError(f"Unsupported adaptive support scheme: {CompressibleSPHConfig.adaptiveSupportScheme}")