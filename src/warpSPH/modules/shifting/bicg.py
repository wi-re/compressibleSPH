"""Matrix-free (left-)preconditioned bi-conjugate gradient (BiCG) for the
incompressible pressure solve. Same ``matvec``-closure + flat-diagonal-``precond``
interface and same ``(x, status, convergence)`` return convention as
``bicgstab.bicgstabSolve`` / ``gmres.gmresSolve``; the
``atol = max(atol, tol, rtol*||b||)`` floor and the status-code family match that
family.

BiCG is the only one of the four Krylov methods that needs the **adjoint**
operator ``A^T`` (the dual residual lives in the ``A^T``-Krylov space), so it is
the most expensive to wire in: ``warpSPHCore`` exposes no ready adjoint of the
pressure ``accel``/``shift`` pair, so the dispatch in ``incompressible.krylov``
supplies ``A^T`` via ``buildIISPHMatvecT`` -- currently a *self-adjoint
placeholder* (``A^T ~= A``). That placeholder is only correct if the operator is
symmetric, which the Phase-0 operator probe quantifies; a rigorously derived
transpose is the Phase-4 work item. Until then BiCG results are provisional.

Method: left-preconditioned BiCG on ``M^{-1} A x = M^{-1} b``. The adjoint of
``M^{-1} A`` is ``A^T M^{-T}``; for the diagonal IISPH preconditioner
``M^{-T} = M^{-1}``, so the adjoint matvec is ``psolve(matvecT(q))``. One forward
(``A p``) and one adjoint (``A^T p``) matvec per iterate.

Status codes: ``>= 0`` converged at that iterate; ``-10`` rho breakdown;
``-11`` alpha/denominator/beta breakdown; ``-12`` per-particle ``|x|`` threshold
bailout; ``-14`` max-iteration budget exhausted. ``convergence`` holds the
per-iterate (preconditioned) residual norm, with the final entry always the
verified true residual ``||b - A x||`` of the returned iterate.
"""

from typing import Callable, List, Optional, Tuple, Union
import math
import torch

__all__ = ['bicgSolve']

# relative breakdown margin (see bicgstabSolve): scale-free, only fires once the
# corresponding scalar is already ~1e6-scaled and the iterate is diverging
_REL_BREAKDOWN_EPS = 1e-6


def bicgSolve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    matvecT: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 0.0,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None,
    verbose: bool = False,
    threshold: Optional[float] = None,
    dim: int = 1,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solve ``matvec(x) == b`` matrix-free with (left-)preconditioned BiCG.
    ``matvecT`` is the adjoint operator (required by the dual-residual update).
    ``convergence`` holds the per-iterate (preconditioned) residual norm, with
    the final entry always the verified true residual ``||b - A x||``
    (matching ``bicgstabSolve``)."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(tol), float(rtol) * float(bnrm2))
    convergence: List[torch.Tensor] = []
    if verbose:
        print(f'BiCG: initial |b| = {bnrm2}, atol = {atol}')

    if bnrm2 == 0:
        return xk, 0, convergence

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10
    eps = torch.finfo(xk.dtype).eps

    # left-preconditioned operator and its adjoint (M diagonal -> M^{-T} = M^{-1})
    if precond is None:
        Apre = matvec
        ApreT = matvecT
        bpre = b
    elif callable(precond):
        Apre = (lambda p: precond(matvec(p)))
        ApreT = (lambda q: precond(matvecT(q)))
        bpre = precond(b)
    else:
        Dinv = precond
        Apre = (lambda p: Dinv * matvec(p))
        ApreT = (lambda q: Dinv * matvecT(q))
        bpre = Dinv * b

    def thresholdExceeded(x: torch.Tensor) -> bool:
        if threshold is None:
            return False
        return bool(torch.any(torch.linalg.norm(x.view(-1, dim), dim=-1) > threshold))

    def finish(x: torch.Tensor, status: int) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        # stamp every returned iterate with its verified true residual
        convergence.append(torch.linalg.norm(matvec(x) - b))
        return x, status, convergence

    # --- initial residual (preconditioned) -----------------------------------
    r = bpre - Apre(xk)
    rnrm = torch.linalg.norm(r)
    convergence.append(rnrm)
    if rnrm < atol:
        if verbose:
            print(f'\t[  0] already converged, |r| = {rnrm}')
        return xk, 0, convergence
    if thresholdExceeded(xk):
        return finish(xk, -12)

    # --- BiCG recurrence -----------------------------------------------------
    rhat = r.clone()          # dual (bi-) residual
    p = r.clone()             # primal search direction
    pT = rhat.clone()         # dual search direction
    rho_prev = 0.0
    alpha_prev = 0.0

    for k in range(maxiter):
        rho = torch.dot(rhat, r)
        if (not bool(torch.isfinite(rho))) or \
                abs(float(rho)) < _REL_BREAKDOWN_EPS * torch.linalg.norm(rhat) * torch.linalg.norm(r):
            if verbose:
                print(f'\t[{k:3d}] rho breakdown {rho}')
            return finish(xk, -10)

        ap = Apre(p)
        denom = torch.dot(rhat, ap)
        if (not bool(torch.isfinite(denom))) or \
                abs(float(denom)) < _REL_BREAKDOWN_EPS * abs(float(rho)):
            if verbose:
                print(f'\t[{k:3d}] alpha breakdown (denom) {denom}')
            return finish(xk, -11)
        alpha = rho / denom

        xk = xk + alpha * p
        r = r - alpha * ap
        rnrm = torch.linalg.norm(r)
        convergence.append(rnrm)
        if rnrm < atol:
            if verbose:
                print(f'\t[{k:3d}] converged, |r| = {rnrm}')
            return xk, k, convergence
        if thresholdExceeded(xk):
            return finish(xk, -12)

        if abs(rho_prev) > eps and abs(alpha_prev) > eps:
            beta = float((rho / rho_prev) * (alpha / alpha_prev))
        else:
            beta = 0.0
        if not math.isfinite(beta):
            if verbose:
                print(f'\t[{k:3d}] beta breakdown {beta}')
            return finish(xk, -11)

        p = r + beta * p
        # dual residual + dual search direction: the A^T matvecs (one adjoint
        # matvec per iterate)
        rhat = rhat - alpha * ApreT(pT)
        pT = rhat + beta * pT

        rho_prev = float(rho)
        alpha_prev = float(alpha)

    return finish(xk, -14)