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

__all__ = ['PressureSolverType', 'JacobiRelaxationMode', 'BoundaryPressureMode', 'ShiftPressureGauge', 'RelaxedJacobiSolverConfig', 'buildDefaultPSConfig', 'buildDefaultDFConfig', 'IncompressibleSolverConfig', 'buildDefaultIncompressibleSolverConfig']

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


class ShiftPressureGauge(Enum):
    """How `solveIncompressible` pins the constant (null-space) component of
    its pressure field each iteration.

    Both of this library's callers use `solveIncompressible` as an *implicit
    particle-shifting* solve, not a momentum pressure solve
    (`systems/incompressible.py`'s `finalize`, which feeds its output straight
    into a position shift `dx = dt**2 * a_p`; and `cases/tgv.py`'s lattice
    relaxation). Its "pressure" is therefore a shifting potential, and its
    operator has the same (near-)constant null space every pure-Neumann PPE
    has -- but unlike `solveDivergenceFree`, whose source term (a divergence)
    is mean-zero by pair antisymmetry, this solver's source term
    (`rho0 - rhoStar`) carries a persistent negative mean that no pressure
    field can remove: the SPH summation density's particle average rises
    quadratically with disorder and is bounded below by its lattice value, so
    `mean_i rho_i == rho0` is unattainable for any disordered configuration
    (`scripts/probe_densityBiasVsDisorder.py`). The solver reacts by driving
    the only mode with a nonzero mean response -- the constant one, whose
    response is weak -- to ever larger amplitude, i.e. it winds up like an
    integral controller with an unreachable setpoint.

    - `nonNegativeClamp`: the historical behavior. `clamp(p, min=0)` each
      iteration. Physically motivated (a shifting potential that never pulls
      particles together) but it is a floor, not a gauge: nothing pins the
      constant mode, so it drifts upward without bound. Measured on
      `kolmogorovIncompressible` at nx=128: the mean climbs to 2.4e6 and the
      run NaNs at step 574.
    - `minShift`: subtract the fluid minimum instead, `p -= p[fluid].min()`.
      Non-negative by construction *and* gauge-fixed (the constant mode is
      pinned by the field's own shape rather than left free), and, unlike the
      clamp, it never discards the field's negative part -- it only
      translates it, so the shift forces the solve actually computed survive.
      Same case/resolution: bounded at ~29 over 1000 steps with no trend,
      and mean/max density error roughly halved.

    Mean-centering (`p -= p[fluid].mean()`, what `solveDivergenceFree` does)
    is *not* an option here: it was tested and diverges within ~150 steps,
    because it gives up the non-negativity that keeps the shift from pulling
    particles together, and because the resulting near-zero mean removes the
    background-pressure de-clumping force this solver relies on. See
    `DFSPH_IMPROVEMENT_PLAN.md` Part 4 for the full comparison.

    **`minShift` only applies where the constant mode is actually free.**
    `solveIncompressible` falls back to `nonNegativeClamp` for any solve that
    has pinned pressure rows (`kind != 0`) or free-surface particles, because
    in both cases the constant is neither free (Dirichlet rows fix it) nor
    forceless (where kernel support is truncated the gradients stop summing to
    zero, so a uniform pressure exerts a large real force). Setting `minShift`
    on a wall-bounded or free-surface case is therefore a no-op, not a
    silently different answer -- and deliberately so: forcing it through on
    the bounded `randomFlowIncompressible` at nx=128 diverges at t=0.69 where
    the clamp reaches t=5.5.
    """
    nonNegativeClamp = 0
    minShift = 1


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
    mdbcPressureRelaxation: float = field(default=0.3, metadata={"description": "Under-relaxation for BoundaryPressureMode.mdbcMlsPressure's boundary pressure update (new = old + factor*(projected - old), matching the divergence-free solver's own default relaxationFactor). Ignored by plain/mdbcDensity. The one-step-lagged MLS projection closes a positive feedback loop with the fluid pressure solve (a larger boundary pressure drives a larger nearby fluid pressure gradient, which projects to an even larger boundary pressure next step); without damping this diverges within single-digit steps even on a well-sampled boundary (see DFSPH_IMPROVEMENT_PLAN.md's mdbcMlsPressure instability finding)."})
    shiftPressureGauge: ShiftPressureGauge = field(default=ShiftPressureGauge.minShift, metadata={"description": "How solveIncompressible (the implicit particle-shifting solve) pins the constant null-space component of its pressure field: minShift (the default; subtract the fluid minimum -- non-negative and gauge-fixed) or nonNegativeClamp (the historical clamp(p, min=0); a floor, not a gauge, so the constant mode drifts up without bound and NaNs kolmogorovIncompressible at nx=128/step 574). Only differs on solves where the constant mode is genuinely free -- see ShiftPressureGauge's docstring and DFSPH_IMPROVEMENT_PLAN.md Part 4."})
    mdbcNoPenetrationShift: bool = field(default=True, metadata={"description": "Whether dfsph_step applies computeMdbcNoPenShift's soft per-particle velocity-damping correction near mDBC boundaries. Default True preserves the scheme's historical always-on behavior; the original DFSPH paper (Bender & Koschier) has no such term and relies on the pressure projection alone to prevent penetration, so this is an experimental A/B toggle (DFSPH_IMPROVEMENT_PLAN.md) to check whether it is actually helping or is a crutch that makes the near-wall density error worse -- not a permanent design decision."})

def buildDefaultIncompressibleSolverConfig() -> IncompressibleSolverConfig:
    return IncompressibleSolverConfig(
        pressureSolver=buildDefaultPSConfig(),
        divergenceFreeSolver=buildDefaultDFConfig(),
        integrateRho=False
    )
