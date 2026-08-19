"""Matrix-free, left-preconditioned, restarted GMRES -- a drop-in alternative
to `bicgstab.bicgstabSolve` for the implicit-shifting Newton step: same
`matvec`-closure + flat-diagonal-`precond` interface, same `(x, status,
convergence)` return convention, same status-code family (see
`bicgstabSolve`'s module docstring; GMRES additionally uses `-13` for
stagnation).

Why it exists here: the implicit-shift operator can sit exactly where
BiCGStab is weakest -- `ShiftingImplicitOperator.exactHessian` is a
graph-Laplacian-style matrix (symmetric but indefinite, with an exact
translation null space), and either operator is nonsymmetric whenever the
per-particle `omega_j` varies, so neither is SPD. GMRES has no rho/omega
breakdown modes, costs 1 matvec/iterate (BiCGStab's is 2), and its residual
estimate is monotone within a restart cycle.

Method (left-preconditioned, matching `bicgstabSolve`'s `psolve` usage):
Arnoldi on `M^{-1} A` starting from `M^{-1} r0`. After `j + 1` steps the
GMRES residual estimate is exactly the new Arnoldi column norm `h_{j+1,j}`
produced by MGS (nested-subspace property), so the per-iterate convergence
test is free. MGS is reorthogonalized once per step (fp32 production). The
cycle-boundary coefficient solve is the normal equations on the tiny
`(m+1) x m` upper-Hessenberg block via `torch.linalg.solve`.

Every `restart` steps (capped at `n - 1`) the iterate is updated with the
cycle's least-squares solution and the true residual is recomputed (one
extra matvec, keeps the bookkeeping honest against accumulated orthogonality
loss) before a fresh cycle starts; a full cycle with < 0.1% residual
reduction bails with `-13` instead of burning the budget. `x` only changes
at those cycle boundaries, so the `threshold` check runs there.
`convergence` holds the per-step (preconditioned) residual estimate, with
the final entry -- as for `bicgstabSolve` -- always the verified true
residual `||b - A xk||` of the returned iterate.
"""

from typing import Callable, List, Optional, Tuple
import torch

__all__ = ['gmresSolve']


def gmresSolve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 0.0,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    precond: Optional[torch.Tensor] = None,
    verbose: bool = False,
    threshold: Optional[float] = None,
    dim: int = 1,
    restart: int = 30,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solves `matvec(x) == b` matrix-free with restarted GMRES. `restart`
    is the Krylov length m (GMRES(m)); `tol`/`rtol`/`atol` and the status
    codes follow `bicgstabSolve`'s convention (see that module's
    docstring). `iters` counts total Arnoldi steps (each consumes one
    matvec), across restarts."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(tol), float(rtol) * float(bnrm2))
    convergence: List[torch.Tensor] = []
    if verbose:
        print(f'GMRES: initial |b| = {bnrm2}, atol = {atol}')

    if bnrm2 == 0:
        return b, 0, convergence

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    m = max(1, min(int(restart), n - 1))
    eps = torch.finfo(xk.dtype).eps
    psolve = (lambda x: precond * x) if precond is not None else (lambda x: x)

    def finish(x: torch.Tensor, status: int) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        # stamp every returned iterate with its verified true residual, so
        # callers always know the quality of what they received
        convergence.append(torch.linalg.norm(matvec(x) - b))
        return x, status, convergence

    def lsCoeffs(H: torch.Tensor, beta: torch.Tensor, cols: int) -> torch.Tensor:
        # min_y ||H[:, :cols] y - beta e1|| via the normal equations on the
        # tiny Hessenberg block; the caller guards non-finite output
        Hc = H[:, :cols]
        e1 = torch.zeros(Hc.shape[0], dtype=xk.dtype, device=xk.device)
        e1[0] = 1.0
        return torch.linalg.solve(Hc.T @ Hc, Hc.T @ (beta * e1))

    def thresholdExceeded(x: torch.Tensor) -> bool:
        if threshold is None:
            return False
        return bool(torch.any(torch.linalg.norm(x.view(-1, dim), dim=-1) > threshold))

    iters = 0
    prevStartResid: Optional[torch.Tensor] = None
    while iters < maxiter:
        # zero x0 is the common production case: A @ 0 == 0, skip the matvec
        if iters == 0:
            r = b.clone() if (x0 is None or not x0.any()) else b - matvec(xk)
        else:
            r = b - matvec(xk)
        startResid = torch.linalg.norm(r)
        if verbose:
            print(f'\t[cycle start {iters}] |r| = {startResid}')
        if prevStartResid is not None and startResid > 0.999 * prevStartResid:
            # a full cycle with < 0.1% residual reduction: GMRES(m) is
            # stagnating, burning the budget would just reproduce this
            if verbose:
                print(f'\t[cycle start {iters}] stagnation: |r| = {startResid} vs prev {prevStartResid}')
            return finish(xk, -13)
        prevStartResid = startResid

        if startResid < atol:
            convergence.append(startResid)
            return xk, iters, convergence

        v0 = psolve(r)
        beta = torch.linalg.norm(v0)
        if beta == 0:
            # degenerate preconditioner zeroed out the whole residual
            return finish(xk, -13)

        V = torch.zeros(n, m, dtype=xk.dtype, device=xk.device)
        H = torch.zeros(m + 1, m, dtype=xk.dtype, device=xk.device)
        V[:, 0] = v0 / beta
        optimisticBreak = False

        for j in range(m):
            if iters >= maxiter:
                break
            w = psolve(matvec(V[:, j]))
            h = V[:, :j + 1].T @ w
            w = w - V[:, :j + 1] @ h
            h2 = V[:, :j + 1].T @ w  # one reorthogonalization (fp32)
            w = w - V[:, :j + 1] @ h2
            h = h + h2
            hNew = torch.linalg.norm(w)
            H[:j + 1, j] = h
            H[j + 1, j] = hNew
            iters += 1
            convergence.append(hNew)
            if verbose:
                print(f'\t[{iters:3d}] resid est = {hNew}')

            if hNew < atol:
                y = lsCoeffs(H, beta, j + 1)
                if not bool(torch.isfinite(y).all()):
                    return finish(xk, -13)
                xk = xk + V[:, :j + 1] @ y
                if thresholdExceeded(xk):
                    return finish(xk, -12)
                trueResid = torch.linalg.norm(matvec(xk) - b)
                convergence.append(trueResid)
                if trueResid < atol:
                    if verbose:
                        print(f'\t[{iters:3d}] converged, true |r| = {trueResid}')
                    return xk, iters, convergence
                # the (preconditioned-norm) estimate under-reported: let the
                # true residual drive the restart
                optimisticBreak = True
                break

            if hNew < eps * beta:
                # invariant subspace reached without the estimate reaching
                # atol: take the best reachable iterate and report its true
                # residual (converged-to-precision vs genuinely stuck)
                y = lsCoeffs(H, beta, j + 1)
                if not bool(torch.isfinite(y).all()):
                    return finish(xk, -13)
                xk = xk + V[:, :j + 1] @ y
                trueResid = torch.linalg.norm(matvec(xk) - b)
                convergence.append(trueResid)
                if trueResid <= max(atol, eps * bnrm2):
                    if verbose:
                        print(f'\t[{iters:3d}] converged to precision limit, true |r| = {trueResid}')
                    return xk, iters, convergence
                return xk, -13, convergence

            if j + 1 < m:
                V[:, j + 1] = w / hNew

        if iters >= maxiter:
            break
        if optimisticBreak:
            continue

        # full m-step cycle: apply the least-squares update, then restart
        y = lsCoeffs(H, beta, m)
        if not bool(torch.isfinite(y).all()):
            return finish(xk, -13)
        xk = xk + V[:, :m] @ y
        if thresholdExceeded(xk):
            return finish(xk, -12)

    return finish(xk, -14)