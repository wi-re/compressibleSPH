from enum import Enum
import torch    

@torch.jit.script
class EnergyScheme(Enum):
    equalWork = 0
    PdV = 1
    diminishing = 2
    monotonic = 3
    hybrid = 4
    CRK = 5
