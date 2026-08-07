import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

from .maronneNormals import computeNormalsMaronne
from .wp_barecasco import computeBarecascoSurfaceDetectionWarp





def detectFreeSurfaceBarecasco(
        currentState: Any, 
        config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        
        returnNormals: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    normals, fsm = computeBarecascoSurfaceDetectionWarp(
        currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            operationMode = OperationDirection.AllToAll,
            supportMode = SupportScheme.SuperSymmetric
        ),
        domain = config.domain,
        adjacency = adjacency,
        barecascoThreshold = surfaceConfig.barecascoThreshold
    )

    fs = fsm < 0.5
    return fs if not returnNormals else (fs, normals)