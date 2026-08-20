"""`RelaxedJacobiSolverConfig` (min/max iterations, tolerance, relaxation factor)
and `IncompressibleSolverConfig`, which bundles two of the former
(`pressureSolver`, `divergenceFreeSolver`) plus `integrateRho`. Embedded as
`.solverConfig` on `IncompressibleSPHConfig` and read via
`schemeConfig.solverConfig.{pressureSolver,divergenceFreeSolver}.*` by
`modules/incompressible/{incompressible,divergenceFree}.py` and
`schemes/dfsph.py`. `buildDefaultPSConfig`/`buildDefaultDFConfig` give the
pressure and divergence-free solvers different tuned defaults (iteration caps,
tolerances, relaxation) rather than sharing one default.
"""

__all__ = ['PressureSolverType', 'RelaxedJacobiSolverConfig', 'buildDefaultPSConfig', 'buildDefaultDFConfig', 'IncompressibleSolverConfig', 'buildDefaultIncompressibleSolverConfig']

from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum

class PressureSolverType(Enum):
    """How the incompressible pressure Poisson equation ``A p = b`` is solved.

    The operator is the IISPH matrix-free pressure operator
    ``A = dt * (IISPH pressure shift o IISPH pressure accel)`` with source term
    ``b`` the IISPH divergence, preconditioned by the IISPH diagonal (``1/D``).
    ``relaxedJacobi`` (the default) keeps the historical matrix-free relaxed
    Jacobi iteration byte-for-byte; the Krylov options are opt-in alternatives
    that solve the same ``A p = b``. See ``INCOMPRESSIBLE_SOLVER_PLAN.md``.
    """
    relaxedJacobi = 0   # default: the existing relaxed-Jacobi path (unchanged)
    cg = 1              # (preconditioned) conjugate gradient -- gated on an SPD/symmetry probe
    bicg = 2            # bi-conjugate gradient -- needs the adjoint matvec A^T
    bicgStab = 3        # bi-conjugate gradient stabilized
    gmres = 4           # (restarted) generalized minimal residual
    minres = 5          # minimum residual -- for this symmetric (not necessarily definite) operator


@dataclass
class RelaxedJacobiSolverConfig:
    minIterations: int = field(default=1, metadata={"description": "Minimum number of iterations for the relaxed Jacobi solver"})
    maxIterations: int = field(default=10, metadata={"description": "Maximum number of iterations (used by both the relaxed-Jacobi and the Krylov paths)"})
    tolerance: float = field(default=1e-3, metadata={"description": "Tolerance for the relaxed Jacobi solver (mean |residual|; ignored by the Krylov paths)"})
    relaxationFactor: float = field(default=0.5, metadata={"description": "Relaxation factor for the relaxed Jacobi solver (ignored by the Krylov paths)"})
    solverType: PressureSolverType = field(default=PressureSolverType.relaxedJacobi, metadata={"description": "Pressure solver: relaxedJacobi (default) or a Krylov method (cg/bicg/bicgStab/gmres/minres)"})
    rtol: float = field(default=1e-5, metadata={"description": "Relative residual tolerance for the Krylov solvers (converge when ||r|| < atol + rtol*||b||)"})
    atol: float = field(default=0.0, metadata={"description": "Absolute residual floor for the Krylov solvers (0 = relative tolerance only)"})
    restart: int = field(default=30, metadata={"description": "GMRES restart length (ignored by the other solvers)"})
    krylovFp64: bool = field(default=False, metadata={"description": "Run the Krylov recurrence in float64 while the SPH matvec stays float32 (opt-in; improves the residual by roughly an order of magnitude on this ill-conditioned operator at negligible extra cost)"})


def buildDefaultPSConfig() -> RelaxedJacobiSolverConfig:
    return RelaxedJacobiSolverConfig(
        minIterations=2,
        maxIterations=64,
        tolerance=5e-4,
        relaxationFactor=0.3
    )
def buildDefaultDFConfig() -> RelaxedJacobiSolverConfig:
    return RelaxedJacobiSolverConfig(
        minIterations=2,
        maxIterations=32,
        tolerance=2.5e-3,
        relaxationFactor=0.3
    )


@dataclass 
class IncompressibleSolverConfig:
    pressureSolver: RelaxedJacobiSolverConfig = field(default_factory=buildDefaultPSConfig, metadata={"description": "Configuration for the pressure solver"})
    divergenceFreeSolver: RelaxedJacobiSolverConfig = field(default_factory=buildDefaultDFConfig, metadata={"description": "Configuration for the divergence-free solver"})
    integrateRho: bool = field(default=False, metadata={"description": "Whether to integrate density in the incompressible solver"})

def buildDefaultIncompressibleSolverConfig() -> IncompressibleSolverConfig:
    return IncompressibleSolverConfig(
        pressureSolver=buildDefaultPSConfig(),
        divergenceFreeSolver=buildDefaultDFConfig(),
        integrateRho=False
    )
