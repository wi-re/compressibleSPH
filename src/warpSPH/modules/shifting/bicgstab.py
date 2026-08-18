"""Matrix-free (preconditioned) BiCGStab, ported from diffSPH's
`sparseSolver.bicg.bicgstab_shifting`: same recurrence (rho/alpha/omega
updates, rho/omega breakdown bail-outs, `threshold`-based divergence
bail-out), generalized to take a caller-supplied `matvec` closure and a flat
diagonal `precond` vector instead of diffSPH's hardcoded 2-component
scatter-sum and `(values, (row, col), n)` sparse-COO preconditioner -- both of
which were always diagonal in practice here, so `psolve` is just an
elementwise multiply.
"""

from typing import Callable, List, Optional, Tuple
import torch

__all__ = ['bicgstabSolve']


def bicgstabSolve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    precond: Optional[torch.Tensor] = None,
    verbose: bool = False,
    threshold: Optional[float] = None,
    dim: int = 1,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(rtol) * float(bnrm2))
    convergence: List[torch.Tensor] = []
    if verbose:
        print(f'BiCGStab: initial |b| = {bnrm2}, atol = {atol}')

    if bnrm2 == 0:
        return b, 0, convergence

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    eps = torch.finfo(xk.dtype).eps
    rhotol = eps ** 2
    omegatol = rhotol

    psolve = (lambda x: precond * x) if precond is not None else (lambda x: x)

    rk = matvec(xk)
    rk = b - rk if xk.any() else b.clone()
    r0 = rk.clone()
    pk = rk.clone()

    for iteration in range(maxiter):
        if torch.linalg.norm(rk) < atol:
            if verbose:
                print(f'\t[{iteration:3d}] converged, |r| = {torch.linalg.norm(rk)}')
            return xk, iteration, convergence

        rho = torch.dot(rk, r0)
        if torch.abs(rho) < rhotol:
            if verbose:
                print(f'\t[{iteration:3d}] rho breakdown {rho} | {rhotol}')
            return xk, -10, convergence

        phat = psolve(pk)
        apk = matvec(phat)
        rv = torch.dot(apk, r0)
        if rv == 0:
            if verbose:
                print(f'\t[{iteration:3d}] rv breakdown')
            return xk, -11, convergence
        alpha = rho / rv
        sk = rk - alpha * apk

        if torch.linalg.norm(sk) < atol:
            xk = xk + alpha * pk
            if verbose:
                print(f'\t[{iteration:3d}] converged, |s| = {torch.linalg.norm(sk)}')
            return xk, 0, convergence

        shat = psolve(sk)
        ask = matvec(shat)
        omega = torch.dot(ask, sk) / torch.dot(ask, ask)
        xk = xk + alpha * phat + omega * shat
        rho_prev = rho
        rk = sk - omega * ask
        beta = (torch.dot(rk, r0) / rho_prev) * (alpha / omega)
        pk = rk + beta * (pk - omega * apk)

        if iteration > 0 and torch.abs(omega) < omegatol:
            if verbose:
                print(f'\t[{iteration:3d}] omega breakdown {omega} | {omegatol}')
            return xk, -11, convergence

        residual = matvec(xk)
        resNorm = torch.linalg.norm(residual - b)
        convergence.append(resNorm)
        if verbose:
            print(f'\t[{iteration:3d}] residual: {resNorm}')

        if threshold is not None:
            dist = torch.linalg.norm(xk.view(-1, dim), dim=-1)
            if torch.any(dist > threshold):
                if verbose:
                    print(f'\t[{iteration:3d}] xk breakdown: max |dx| = {dist.max()} > {threshold}')
                return xk, -12, convergence

    if verbose:
        print(f'Reached maximum iterations {maxiter}, |r| = {torch.linalg.norm(rk)}')
    return xk, maxiter, convergence
