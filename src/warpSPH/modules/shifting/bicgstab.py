"""Matrix-free (preconditioned) BiCGStab, ported from diffSPH's
`sparseSolver.bicg.bicgstab_shifting`, generalized to take a caller-supplied
`matvec` closure and a flat diagonal `precond` vector instead of diffSPH's
hardcoded 2-component scatter-sum and `(values, (row, col), n)` sparse-COO
preconditioner -- both of which were always diagonal in practice here, so
`psolve` is just an elementwise multiply.

The per-iterate recurrence (rho/alpha/omega updates) is unchanged from the
port, so on non-pathological inputs the iterate sequence is bit-identical to
diffSPH's `bicgstab_shifting` (the check the regression note in
`docs/regression/implicit_shifting_operator_choice.md` relies on). The
changes below only alter what happens near breakdown and what is reported:

- `tol` is now enforced: `atol = max(atol, tol, rtol*||b||)`. The port
  accepted `tol` but never read it, so callers' absolute tolerance (e.g.
  `ShiftProperties.implicitTolerance`) had no effect. Both this function's
  and the production config's default are `0.0` (= "relative tolerance
  only"), which is the behavior the port actually had.
- Breakdown guards are relative as well as absolute, in addition to the
  port's absolute `eps**2` floor:
  - `|rho| < 1e-6*||r0||*||rk||` -- `|rho|/(||r0||*||rk||)` is
    `|cos(angle(rk, r0))|`, a scale-free indicator of the `alpha = rho/rv`
    blow-up; the port's bare `eps**2` (1.4e-14 in fp32) only caught rho
    values at the edge of underflow, so near-breakdown produced a
    ~1e6-scaled update instead of a clean bailout.
  - `|rv| < 1e-6*|rho|` replaces the port's `rv == 0` exact-zero test
    (same rationale: it caught nothing short of exact underflow).
- `omega` is now validated **before** the update that uses it (the port
  validated after, so a broken `omega` had already corrupted `xk`), with a
  `torch.isfinite` guard (the port's `abs(omega) < eps**2` is `False` for
  NaN/Inf, so those passed the check), and the port's `iteration > 0` skip
  is dropped.
- 2 matvecs/iterate instead of the port's 3: the per-iterate `convergence`
  entry is the recurrence residual norm (the port's 3rd matvec recomputed
  the true residual every iterate purely to record it). The true residual is
  now computed once, on every return: it verifies `||b - A xk|| < atol`
  before a recurrence-based convergence is declared (and the iterate is
  stamped with the verified value), and it is appended on breakdown/
  threshold/budget bailouts too, so `convergence[-1]` always reports the
  actual quality of the returned iterate.
- Status codes: `>= 0` converged at that iterate; `-10` rho breakdown;
  `-11` rv/omega breakdown; `-12` per-particle `|x|` threshold bailout;
  `-14` max-iteration budget exhausted (the port returned `maxiter` here,
  indistinguishable from a converged status). The port's s-convergence
  branch returned a literal `0` at every iterate; it now returns the actual
  `iteration`.
"""

from typing import Callable, List, Optional, Tuple
import torch

__all__ = ['bicgstabSolve']


# relative breakdown margin (see module docstring): ~8x float32 eps, i.e. it
# only fires once the corresponding update (alpha/omega) is already
# ~1e6-scaled and the iterate is diverging, so it never triggers on
# well-behaved solves
_REL_BREAKDOWN_EPS = 1e-6


def bicgstabSolve(
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
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solves `matvec(x) == b` matrix-free. `tol` is an absolute residual
    floor (0.0 = relative tolerance only), `rtol*||b||` the relative one,
    `atol` an additional explicit floor. See the module docstring for the
    status-code convention and the `convergence` list semantics."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(tol), float(rtol) * float(bnrm2))
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

    # zero x0 is the common production case: A @ 0 == 0, skip the matvec
    if x0 is None or not x0.any():
        rk = b.clone()
    else:
        rk = b - matvec(xk)
    r0 = rk.clone()
    pk = rk.clone()
    r0norm = torch.linalg.norm(r0)

    def finish(x: torch.Tensor, status: int) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        # stamp every returned iterate with its verified true residual, so
        # callers always know the quality of what they received
        convergence.append(torch.linalg.norm(matvec(x) - b))
        return x, status, convergence

    for iteration in range(maxiter):
        rkNorm = torch.linalg.norm(rk)
        if rkNorm < atol:
            # the recurrence residual can under-report near breakdown: verify
            # with the true residual before declaring convergence (one extra
            # matvec, only on the would-be final iterate); if the recurrence
            # is lying, fall through and keep iterating (the guards below
            # catch the near-singular case)
            trueResid = torch.linalg.norm(matvec(xk) - b)
            if trueResid < atol:
                if verbose:
                    print(f'\t[{iteration:3d}] converged, |r| = {trueResid}')
                convergence.append(trueResid)
                return xk, iteration, convergence

        rho = torch.dot(rk, r0)
        if torch.abs(rho) < rhotol or torch.abs(rho) < _REL_BREAKDOWN_EPS * r0norm * rkNorm:
            if verbose:
                print(f'\t[{iteration:3d}] rho breakdown {rho} | {rhotol}')
            return finish(xk, -10)

        phat = psolve(pk)
        apk = matvec(phat)
        rv = torch.dot(apk, r0)
        if torch.abs(rv) < _REL_BREAKDOWN_EPS * torch.abs(rho):
            if verbose:
                print(f'\t[{iteration:3d}] rv breakdown {rv}')
            return finish(xk, -11)
        alpha = rho / rv
        sk = rk - alpha * apk

        if torch.linalg.norm(sk) < atol:
            # verify before committing: in fp32 `sk = rk - alpha*apk` is a
            # cancellation of O(1) terms, so the computed ||sk|| can
            # under-report the true residual of the updated iterate
            x_cand = xk + alpha * pk
            trueResid = torch.linalg.norm(matvec(x_cand) - b)
            if trueResid < atol:
                xk = x_cand
                if verbose:
                    print(f'\t[{iteration:3d}] converged, |s| = {trueResid}')
                convergence.append(trueResid)
                return xk, iteration, convergence
            # otherwise the omega part of the step is not negligible: fall
            # through and complete the full iteration

        shat = psolve(sk)
        ask = matvec(shat)
        omega = torch.dot(ask, sk) / torch.dot(ask, ask)
        if (not bool(torch.isfinite(omega))) or torch.abs(omega) < omegatol:
            if verbose:
                print(f'\t[{iteration:3d}] omega breakdown {omega} | {omegatol}')
            return finish(xk, -11)

        xk = xk + alpha * phat + omega * shat
        rho_prev = rho
        rk = sk - omega * ask
        beta = (torch.dot(rk, r0) / rho_prev) * (alpha / omega)
        pk = rk + beta * (pk - omega * apk)

        convergence.append(torch.linalg.norm(rk))
        if verbose:
            print(f'\t[{iteration:3d}] residual: {convergence[-1]}')

        if threshold is not None:
            dist = torch.linalg.norm(xk.view(-1, dim), dim=-1)
            if torch.any(dist > threshold):
                if verbose:
                    print(f'\t[{iteration:3d}] xk breakdown: max |dx| = {dist.max()} > {threshold}')
                return finish(xk, -12)

    return finish(xk, -14)
