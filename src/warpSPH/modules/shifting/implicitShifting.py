"""Implicit particle shifting (IPS): instead of `delta.computeDeltaShift`'s
explicit, per-step anti-clustering nudge, this solves directly for the
position shift that (locally) equalizes the SPH concentration field
`C_i = sum_j omega_j * W_ij`, `omega_j = m_j/rho_j` (or `m_j/rho0`), via one
Newton step on `grad C = 0`: `Hess(C) @ dx = -grad(C)`, solved with a
matrix-free, Jacobi-preconditioned BiCGStab (`bicgstab.bicgstabSolve`) over
the neighbor graph. Ported from `diffSPH.modules.shifting.implicitShifting`;
the per-pair kernel terms come from `wp_implicitShifting.computeShiftingPairTerms`.

Assembling `Hess(C)` from the pairwise kernel Hessian `H_ij` needs care, and
differs from a literal transcription of diffSPH's block layout in two ways
verified against a finite-difference Hessian of `C` and a dense exact solve:

- Self-pairs (`i == j`, zero separation) are dropped before assembly -- **and
  both bullets below are the same fact, not two unrelated pitfalls, per a
  correction made during `warpSPHCore/warpier_forward_mode_plan.md` Phase 4
  step 3.** `C_i = sum_j omega_j W(x_i - x_j, h_ij)` depends on `x_i` through
  every term, but the `j = i` term's *own* `x_j` argument is also `x_i` --
  writing that term as a function of the one shared variable,
  `f(x) = W(x - x, h) = W(0, h)`, makes it visibly constant (a particle's
  distance to itself is identically zero everywhere in configuration space,
  not just at the current position), so `f'(x) = f''(x) = 0` for *any*
  kernel. Expanding via the chain rule on `g(a, b) = W(a - b, h)` (the two
  occurrences of `x_i` held temporarily independent as `a`/`b`, then set
  equal) shows *why*: translation invariance forces `dg/db = -dg/da` and
  `d^2g/(da db) = -d^2g/da^2 = -d^2g/db^2`, so `f''(x) = H(0,h) - 2H(0,h) +
  H(0,h) = 0` identically, regardless of what `H(0,h)` numerically equals.
  **An earlier version of this docstring claimed instead that this was a
  numerical-safety drop -- that `sphKernelHessian`'s near-origin
  regularization branch "produced a large, effectively arbitrary value
  (floating-point noise divided by a near-zero epsilon)" at `r=0`. That
  claim was wrong**: `sphKernelHessian` returns a well-defined, finite,
  physically meaningful value there (the kernel's own curvature at its peak
  -- `warpSPHCore/wp_kernels.ipynb` checks this directly for `Wendland2` and
  finds a smooth `-15.0`, continuous with its `-14.88` neighbors either
  side). The value was never the problem; the identity above is why it still
  has to be excluded from `Hess(C)`'s diagonal regardless of how well-behaved
  it is -- it was never really part of `d(grad_i C)/dx_i` to begin with.
- The *same* identity is why `Hess(C)`'s row-`i` block is *not* `sum_j
  omega_j H_ij` placed at column `j` the way `grad C`'s row-`i` term is (that
  would make `Hess(C)` nonsymmetric and, empirically, solve in the wrong
  direction even for an almost-perfect lattice): the chain-rule expansion
  above gives `dg/da = -dg/db` and `d^2g/(da db) = -d^2g/da^2`, i.e. the
  *off-diagonal* mixed partial always carries a sign flip relative to the
  diagonal one. Differentiating `grad_i C = sum_j omega_j gradW(x_i - x_j)`
  w.r.t. `x_i` and w.r.t. a specific neighbor `x_k` separately gives
  `d(grad_i C)/dx_i = sum_{j != i} omega_j H_ij` (the diagonal block, self
  excluded per the identity above) and `d(grad_i C)/dx_k = -omega_k H_ik`
  (off-diagonal, negated). The assembled operator is therefore a
  graph-Laplacian-style matrix -- symmetric, with an exact null space along
  uniform translation (`diag_i = sum_{j!=i} H_ij` cancels the off-diagonal
  row exactly for a constant shift) -- which is why it's solved with
  BiCGStab rather than a direct solve. A raw, undamped Newton step from this
  operator is only reliably stable very close to the solution (confirmed by
  sweeping a jittered-lattice convergence test); the `implicitRelaxation`
  config field damps each step, matching this codebase's own IISPH Jacobi
  relaxation precedent (`modules/incompressible/incompressible.py`).

Free-surface/boundary handling: rows for `currentState.kinds != 0` particles
get a zero RHS/initial-guess (so the solver doesn't chase a target for
particles that get clamped to zero shift anyway by
`wrapper.solveShifting`'s post-hoc `update[kinds != 0] = 0`). diffSPH also
zeroed rows of the per-pair Hessian by `kinds`, but indexed it with a
particle-shaped mask against a pair-shaped tensor -- inconsistent shapes
that would raise unless `numPairs == numParticles`; that step is dropped
here rather than ported as-is.
"""

from typing import Any, Tuple
import torch
from torch.profiler import record_function
from warpSPHCore import *

from warpSPH.math import scatter_sum
from warpSPH.configurations.simulationConfig import SimulationConfig
from ...configurations.moduleConfigurations.shifting import ShiftingImplicitInitializer

from .wp_implicitShifting import computeShiftingPairTerms
from .bicgstab import bicgstabSolve
from .delta import computeDeltaShift

__all__ = ['computeImplicitShift']


def _multiplyLaplacianBlock(
    diagBlock: torch.Tensor, H: torch.Tensor, x: torch.Tensor,
    i: torch.Tensor, j: torch.Tensor, numParticles: int, dim: int,
) -> torch.Tensor:
    """`out_i = diagBlock_i @ x_i - sum_{j: neighbor of i} H_ij @ x_j`."""
    xr = x.view(numParticles, dim)
    out = torch.einsum('nab,nb->na', diagBlock, xr)
    for a in range(dim):
        for b in range(dim):
            out[:, a] -= scatter_sum(H[:, a, b] * xr[j, b], i, dim=0, dim_size=numParticles)
    return out.flatten()


def _buildSystem(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    domain: Any,
    adjacency: AdjacencyList,
    i: torch.Tensor,
    j: torch.Tensor,
    J: torch.Tensor,
    H: torch.Tensor,
    rho0: float,
    dim: int,
    numParticles: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = currentState.positions.device, currentState.positions.dtype

    if schemeConfig.shiftProperties.summationDensity:
        omega = currentState.masses / currentState.densities
    else:
        omega = currentState.masses / rho0

    Jw = scatter_sum(J * omega[j, None], i, dim=0, dim_size=numParticles)
    Hw = H * omega[j, None, None]
    B = Jw.flatten().clone()
    diagBlock = scatter_sum(Hw, i, dim=0, dim_size=numParticles)

    initializer = schemeConfig.shiftProperties.implicitInitializer
    if initializer in (ShiftingImplicitInitializer.deltaPlus, ShiftingImplicitInitializer.deltaMinus):
        delta, _ = computeDeltaShift(currentState, config, schemeConfig, domain, adjacency, iters=1)
        sign = -0.5 if initializer == ShiftingImplicitInitializer.deltaPlus else 0.5
        x0 = (delta.flatten() * sign).clone()
    else:
        x0 = torch.zeros(numParticles * dim, device=device, dtype=dtype)

    if torch.any(currentState.kinds != 0):
        boundary = currentState.kinds != 0
        B.view(numParticles, dim)[boundary] = 0
        x0.view(numParticles, dim)[boundary] = 0

    return Hw, diagBlock, B, x0


def computeImplicitShift(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    domain: Any,
    adjacency: AdjacencyList,
    iters: int = -1,
):
    """Solves for the equilibrium implicit-shifting position delta in a
    single BiCGStab solve (`iters` is accepted, matching
    `delta.computeDeltaShift`'s call signature, but is not otherwise used --
    the outer per-iteration adjacency rebuild already happens in
    `wrapper.solveShifting`'s loop). Returns `(delta, adjacency)`.
    """
    with record_function("[warpSPH] - (shift) - implicit"):
        numParticles = currentState.positions.shape[0]
        dim = currentState.positions.shape[1]

        rho0 = schemeConfig.fluid.restDensity

        _K, J, H = computeShiftingPairTerms(currentState, domain, config.kernel, adjacency)

        # Drop self-pairs: see the module docstring -- an exact translation-
        # invariance identity, not a numerical-safety measure against an
        # unstable sphKernelHessian value (it's well-defined and finite there).
        pairMask = adjacency.i != adjacency.j
        i, j, J, H = adjacency.i[pairMask], adjacency.j[pairMask], J[pairMask], H[pairMask]

        Hw, diagBlock, B, x0 = _buildSystem(
            currentState, config, schemeConfig, domain, adjacency, i, j, J, H, rho0, dim, numParticles,
        )

        if torch.any(currentState.kinds != 0):
            activeMask = currentState.kinds[i] == 0
        else:
            activeMask = torch.ones_like(i, dtype=torch.bool)
        iA, jA, HwA = i[activeMask], j[activeMask], Hw[activeMask]

        diagComponents = torch.diagonal(diagBlock, dim1=-2, dim2=-1).flatten()
        precond = torch.where(diagComponents.abs() > 1e-8, 1.0 / diagComponents, torch.zeros_like(diagComponents))
        if not torch.any(diagComponents.abs() > 1e-8):
            precond = None

        def matvec(x: torch.Tensor) -> torch.Tensor:
            return _multiplyLaplacianBlock(diagBlock, HwA, x, iA, jA, numParticles, dim)

        dx = config.dx.cpu().item() if isinstance(config.dx, torch.Tensor) else config.dx
        threshold = schemeConfig.shiftProperties.implicitSolverThreshold
        if threshold is None:
            threshold = dx / 2

        xk, solverIters, convergence = bicgstabSolve(
            matvec, B, x0,
            tol=schemeConfig.shiftProperties.implicitTolerance,
            rtol=schemeConfig.shiftProperties.implicitRelativeTolerance,
            maxiter=schemeConfig.shiftProperties.implicitMaxSolverIter,
            precond=precond if schemeConfig.shiftProperties.implicitUsePreconditioner else None,
            threshold=threshold,
            dim=dim,
        )

        update = -xk.view(numParticles, dim) * schemeConfig.shiftProperties.implicitRelaxation
        return update, adjacency
