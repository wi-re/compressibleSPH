"""Matrix-free Richardson iteration (steepest descent on the quadratic model,
`x <- x + omega * (b - A x)`, i.e. step preconditioner `M = I`), a robust
bounded last-resort fallback for the implicit shifting Newton solve
(`implicitShifting.computeImplicitShift` / `implicitShiftingAutomatic`).

Why a separate solver, and why `M = I` (NOT the production diagonal):
  - An eigenvalue probe of the *production* operator (the exact `matvec`
    assembled from `_buildSystem`/`_buildDiagBlock`, a jittered lattice)
    shows the diagonal-preconditioned iteration matrix `I - D^-1 A` has
    spectral radius > 1 (2.7 on the default positive-definite
    `legacyPairwise` operator, 6.6 on the indefinite `exactHessian`):
    damped Jacobi with the production Jacobi diagonal *diverges* here. The
    diagonal is only useful *inside* Krylov (where it reshapes the spectrum
    the subspace search works on), not as a fixed-point step direction. So
    Richardson uses `M = I` and an auto-tuned step size instead.
  - No breakdown modes (no rho/omega divisions, no least squares): it cannot
    hit the `-10`/`-11` BiCGStab breakdowns and cannot emit a non-finite
    iterate. Every step is a bounded move; the only failure mode is making
    insufficient progress, which the stagnation guard reports as `-15`.
  - Self-monitoring: the exact residual `||b - A x||` is computed every step
    (it drives both the convergence test and the step), so every history
    entry -- including the final stamped one -- is a true-residual
    certificate of the returned iterate.

Step size: `omega` may be passed explicitly. By default it is auto-tuned: a
short power iteration estimates the operator's spectral scale
`rho = max|lambda|` and seeds `omega = 1.5 / rho`, then a backtracking step
search refines it -- the step sign is chosen to reduce the residual, and if a
trial overshoots (residual grows) `omega` is halved and retried. This is
robust to the power-iteration underestimating `rho` on a clustered spectrum
(the failure mode a fixed `omega` has). For the positive-definite
`legacyPairwise` operator (eigenvalues in `[~0, lambda_max]`, exact
translation null space) the resulting step sits inside the Richardson
convergence window `(0, 2/lambda_max)` and is near-optimal; for the
indefinite `exactHessian` operator no step size converges, and the solve
correctly bails with `-15`. Because the sign is discovered by the same
backtracking search, the solver is correct for either definiteness without
the caller knowing it. `precond` is deliberately *not* part of this
interface (see above).

Status codes follow the `bicgstabSolve`/`gmresSolve` family:
  `>= 0`   converged at that step
  `-12`    per-particle `|x|` threshold bailout
  `-14`    max-iteration budget exhausted
  `-15`    stagnation (residual failed to beat its best over an 8-step window
            by more than 2% -- a genuine stall/divergence, not slow progress)
`convergence[-1]` is always the verified true residual `||b - A x||` of the
returned iterate. `iters` counts accepted `x`-updates (the sign-detection
trial is step 1).
"""

from typing import Callable, List, Optional, Tuple
import torch

__all__ = ['richardsonSolve']


def _estimateOmega(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    r: torch.Tensor,
    rnrm: torch.Tensor,
    tune_iters: int,
) -> float:
    # Power iteration on A from the initial residual: estimates the spectral
    # scale rho = max|lambda| (lambda_max for the SPD legacyPairwise
    # operator). The residual is a problem-specific start with generic
    # eigendirection content, so a few sweeps suffice; it is also not the
    # translation null vector (A @ 1 == 0), which a ones start would be.
    v = r / rnrm
    rho = float(rnrm)
    for _ in range(max(1, int(tune_iters))):
        w = matvec(v)
        wnrm = torch.linalg.norm(w)
        if wnrm == 0 or not bool(torch.isfinite(wnrm)):
            break
        rho = float(wnrm)  # v is unit-norm, so ||A v|| -> max|lambda|
        v = w / wnrm
    # 1.5 (not 2) keeps the seed just inside the Richardson window (0, 2/rho)
    # even if the power-iteration underestimates rho; backtracking in the
    # caller is the safety net for a coarser underestimate.
    return 1.5 / max(rho, 1e-12)


def richardsonSolve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 0.0,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    omega: Optional[float] = None,
    tune_omega: bool = True,
    tune_iters: int = 6,
    verbose: bool = False,
    threshold: Optional[float] = None,
    dim: int = 1,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solves `matvec(x) == b` with (auto-tuned, sign-auto-detected)
    Richardson iteration. See the module docstring for why `M = I` and for
    the status-code contract (which matches `bicgstabSolve`/`gmresSolve`)."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(tol), float(rtol) * float(bnrm2))
    convergence: List[torch.Tensor] = []
    if verbose:
        print(f'Richardson: initial |b| = {bnrm2}, atol = {atol}')
    if bnrm2 == 0:
        return xk, 0, convergence

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    def finish(x: torch.Tensor, status: int) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        # stamp every returned iterate with its verified true residual, so
        # callers always know the quality of what they received
        convergence.append(torch.linalg.norm(matvec(x) - b))
        return x, status, convergence

    def thresholdExceeded(x: torch.Tensor) -> bool:
        if threshold is None:
            return False
        return bool(torch.any(torch.linalg.norm(x.view(-1, dim), dim=-1) > threshold))

    # --- initial residual --------------------------------------------------
    r = b - matvec(xk)
    rnrm = torch.linalg.norm(r)
    convergence.append(rnrm)
    if rnrm < atol:
        if verbose:
            print(f'\t[  0] already converged, |r| = {rnrm}')
        return xk, 0, convergence
    if thresholdExceeded(xk):
        return finish(xk, -12)

    # --- step size + sign (auto-tuned, then backtracking) ------------------
    # Seed omega from a short power iteration (rough spectral scale), then run
    # a backtracking step search: take the sign that reduces the residual and,
    # if the step overshoots (residual grows), halve omega and retry. This is
    # robust to a coarse spectral estimate -- a clustered spectrum can make the
    # power-iteration underestimate lambda_max, and backtracking recovers the
    # in-window step size rather than oscillating into stagnation. Each trial
    # costs 2 matvecs; in the common case the first trial succeeds.
    if omega is None:
        omega = 1.0 if not tune_omega else _estimateOmega(matvec, r, rnrm, tune_iters)
    w = float(omega)
    if verbose:
        print(f'\tinitial step size: omega = {w:.3e}')

    rnrm0 = rnrm  # residual before any step; a trial must beat this
    for _ in range(8):
        x_plus = xk + w * r
        x_minus = xk - w * r
        r_plus = b - matvec(x_plus)
        r_minus = b - matvec(x_minus)
        rnrm_plus = torch.linalg.norm(r_plus)
        rnrm_minus = torch.linalg.norm(r_minus)
        if rnrm_plus <= rnrm_minus:
            xk, r, rnrm, sign = x_plus, r_plus, rnrm_plus, 1.0
        else:
            xk, r, rnrm, sign = x_minus, r_minus, rnrm_minus, -1.0
        if rnrm < rnrm0:
            break
        w *= 0.5  # overshoot: shrink the step and retry
    omega = w
    steps = 1
    convergence.append(rnrm)
    if verbose:
        print(f'\t[  1] sign = {"+" if sign > 0 else "-"}, omega = {omega:.3e}, |r| = {rnrm}')
    if rnrm < atol:
        return xk, steps, convergence
    if thresholdExceeded(xk):
        return finish(xk, -12)

    # --- main loop (1 matvec/step) -----------------------------------------
    best = rnrm
    stagnant = 0
    while steps < maxiter:
        xk = xk + sign * omega * r
        r = b - matvec(xk)
        rnrm = torch.linalg.norm(r)
        steps += 1
        convergence.append(rnrm)
        if verbose:
            print(f'\t[{steps:3d}] |r| = {rnrm}')
        if rnrm < atol:
            return xk, steps, convergence
        if thresholdExceeded(xk):
            return finish(xk, -12)
        if rnrm < best:
            best = rnrm
            stagnant = 0
        else:
            # tolerate small non-monotonic wiggles: a slowly-converging solve
            # (poorly-conditioned / singular operator) can wiggle a little
            # without being stuck. Treat it as stagnation only when the
            # residual has failed to beat the best for a window AND is
            # meaningfully worse than it (a genuine stall or divergence).
            stagnant += 1
            if stagnant >= 8 and rnrm > best * 1.02:
                if verbose:
                    print(f'\t[{steps:3d}] stagnation, best |r| = {best}')
                return finish(xk, -15)
    return finish(xk, -14)