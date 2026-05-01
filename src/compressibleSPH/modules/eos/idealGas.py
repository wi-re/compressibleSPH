import torch
from typing import Optional, Tuple
from torch.profiler import profile, record_function, ProfilerActivity

def idealGasEOS(A: Optional[torch.Tensor], u: Optional[torch.Tensor], P: Optional[torch.Tensor], 
             rho: torch.Tensor, gamma: float):
    with record_function("EOS[IdealGas]"):
        given = (1 if A is not None else 0) + (1 if u is not None else 0) + (1 if P is not None else 0)
        # if given < 2:
            # raise ValueError('At least two of the three parameters must be given')
        
        P_, u_, A_, rho_ = P, u, A, rho
        
        c_s = None
        if u is not None:
            c_s = torch.sqrt(u.abs() * gamma * (gamma - 1))
        elif P is not None:
            c_s = torch.sqrt(gamma * P.abs() / rho)
        elif A is not None:
            c_s = torch.sqrt(gamma * rho ** (gamma - 1) * A)

        if P is None and u is not None:
            P_ = (gamma - 1) * rho * u
        elif P is None and A is not None:
            P_ = A * rho**gamma
        
        if u is None and A is not None:
            u_ = A * rho**(gamma - 1) / (gamma - 1)
        elif u is None and P is not None:
            u_ = P / rho / (gamma - 1)

        if A is None and u is not None:
            A_ = u * (gamma - 1) * rho**(1 - gamma)
        elif A is None and P is not None:
            A_ = P / rho**gamma

        return A_, u_, P_, c_s

