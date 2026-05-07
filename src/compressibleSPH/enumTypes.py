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

@torch.jit.script
class AdaptiveSupportScheme(Enum):
    NoScheme = 0
    Monaghan = 1
    Owen = 2

@torch.jit.script
class ViscositySwitch(Enum):
    Balsara1995 = 0
    Colagrossi2004 = 1
    CullenDehnen2010 = 2
    CullenHopkins = 3
    MorrisMonaghan1997 = 4
    Rosswog2000 = 5
    NoneSwitch = 6