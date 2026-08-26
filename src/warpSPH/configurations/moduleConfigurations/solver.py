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

__all__ = ['PressureSolverType', 'JacobiRelaxationMode', 'BoundaryPressureMode', 'RelaxedJacobiSolverConfig', 'buildDefaultPSConfig', 'buildDefaultDFConfig', 'IncompressibleSolverConfig', 'buildDefaultIncompressibleSolverConfig']

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


class JacobiRelaxationMode(Enum):
    """How the relaxed-Jacobi path chooses its per-step relaxation size
    (only used when ``solverType`` is ``relaxedJacobi``).

    The update is ``p <- p + omega * D^-1 * r`` with ``D = diag(A)``. Because
    ``D^-1 A`` is similar to the symmetric ``|D|^-1/2 (-A) |D|^-1/2 >= 0``,
    ``fixed`` converges iff ``omega < 2/rho(D^-1 A)`` -- a state-dependent
    stability window (measured ~0.355 on the TGV operator family, so the
    historical omega=0.5 default diverges and 0.3 sits inside with ~15%
    margin). ``optimal`` removes the window entirely: each step uses the
    exact residual minimizer ``omega_k = (r . A D^-1 r)/||A D^-1 r||^2``,
    which costs the same single matvec as the fixed step and decreases the
    residual monotonically for any starting size. See
    ``docs/regression/incompressible_pressure_solver_choice.md``.
    """
    fixed = 0    # default: constant relaxationFactor (byte-identical history)
    optimal = 1  # per-step exact residual-minimizing size (IISPH solver only)


class BoundaryPressureMode(Enum):
    """How `kind==1` boundary particles are handled by the incompressible
    pressure solvers (`solveDivergenceFree`/`solveIncompressible`) and by
    `schemes/dfsph.py`'s mDBC wiring.

    In all three modes, boundary particles are excluded from the pressure
    *unknowns*: their pressure is held fixed (not driven by the
    Jacobi/Krylov update) for the duration of a solve, they are excluded
    from the gauge-fixing mean, and their pressure acceleration (`a_p`) is
    zeroed post-solve -- the one-way-coupling contract already enforced
    downstream by `nonFluidMask` in `dfsph_step`. What differs is the value
    their pressure is held at, and how their density is computed:

    - `plain`: no mDBC at all. Boundary density comes from plain SPH
      summation like a fluid particle; boundary pressure is held at 0.
    - `mdbcDensity`: boundary density is mDBC-extrapolated
      (`computeMdbcDensity`, Liu-Liu MLS from fluid neighbors); boundary
      pressure is still held at 0 (no pressure extrapolation).
    - `mdbcMlsPressure`: same density extrapolation as `mdbcDensity`, plus
      the fluid pressure field is itself Liu-Liu MLS-projected onto
      boundary particles after each `solveDivergenceFree` call
      (`computeMdbcPressure`), so boundary particles carry a physically
      consistent pressure for the *next* step's force computation on fluid
      neighbors, rather than an artificial zero-pressure wall.
    """
    plain = 0
    mdbcDensity = 1
    mdbcMlsPressure = 2


@dataclass
class RelaxedJacobiSolverConfig:
    minIterations: int = field(default=1, metadata={"description": "Minimum number of iterations for the relaxed Jacobi solver"})
    maxIterations: int = field(default=10, metadata={"description": "Maximum number of iterations (used by both the relaxed-Jacobi and the Krylov paths)"})
    tolerance: float = field(default=1e-3, metadata={"description": "Tolerance for the relaxed Jacobi solver (mean |residual|; ignored by the Krylov paths)"})
    relaxationFactor: float = field(default=0.5, metadata={"description": "Relaxation factor for the relaxed Jacobi solver (ignored by the Krylov paths and by relaxationMode='optimal')"})
    relaxationMode: JacobiRelaxationMode = field(default=JacobiRelaxationMode.fixed, metadata={"description": "Relaxation mode for the relaxed-Jacobi path: fixed (constant relaxationFactor, byte-identical default) or optimal (per-step exact residual-minimizing step; same matvec count, monotonically decreasing residual, no stability window; divergenceFree/IISPH solver only)"})
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
    boundaryPressureMode: BoundaryPressureMode = field(default=BoundaryPressureMode.mdbcDensity, metadata={"description": "How kind==1 boundary particles are handled by the pressure solvers: plain (no mDBC), mdbcDensity (mDBC density extrapolation only, matching this scheme's historical always-on behavior), or mdbcMlsPressure (mDBC density + MLS-projected boundary pressure)"})

def buildDefaultIncompressibleSolverConfig() -> IncompressibleSolverConfig:
    return IncompressibleSolverConfig(
        pressureSolver=buildDefaultPSConfig(),
        divergenceFreeSolver=buildDefaultDFConfig(),
        integrateRho=False
    )
