"""Preconditioner builders for the implicit-shifting Krylov solve.

Both builders take the assembled operator's diagonal blocks `diagBlock`
(shape `[n, dim, dim]` -- one `dim x dim` block per particle, where `dim` is
the *problem dimensionality* (1/2/3), not a hard-coded 3) and return the
preconditioner in a form the solvers (`bicgstabSolve`/`gmresSolve`) accept:

  - `buildScalarJacobiPrecond` returns a flat `[n*dim]` vector of `1/diag`
    (the historical production preconditioner); the solvers apply it as an
    elementwise multiply. `None` when the whole diagonal is ~0.
  - `buildBlockJacobiPrecond` returns a *callable* `psolve(r) -> M^-1 r`
    with `M = block-diag(diagBlock + ridge*I)`, applied as a batched inverse
    + `einsum`; `None` when the whole diagonal is ~0. For `dim == 1` the
    blocks are 1x1, so this reduces exactly to scalar Jacobi.

Why the block builder is honest-but-a-wash on this operator (see
`docs/regression/implicit_shifting_operator_choice.md`): for `legacyPairwise`
(the default) the diagonal block is `-omega_i * kernelHessian(x_i, x_i)`, the
kernel Hessian at the self-point, which is isotropic (`c * I_dim`) for any
radial kernel -- so the block has no intra-particle off-diagonals to exploit
and block Jacobi is *bit-identical* to scalar Jacobi. For `exactHessian` the
blocks are non-isotropic (a sum of neighbor Hessians), but inverting them is
slightly *worse*, because the coupling that controls convergence is the
inter-particle off-diagonal, which no diagonal/block-diagonal preconditioner
captures. It is provided as the correct *general form* for a block-structured
operator, not as a convergence win for the current operators.
"""

from typing import Callable, Optional
import torch

__all__ = ['buildScalarJacobiPrecond', 'buildBlockJacobiPrecond']

# |diag| below this is treated as "no meaningful diagonal" (matches the
# historical inline threshold in implicitShifting.py)
_MIN_DIAG = 1e-8


def buildScalarJacobiPrecond(diagBlock: torch.Tensor) -> Optional[torch.Tensor]:
    """Scalar Jacobi: `1/diag(A)` as a flat `[n*dim]` vector (the historical
    production preconditioner). Byte-identical to the inline computation it
    replaces. Returns `None` when every diagonal entry is ~0."""
    diagComponents = torch.diagonal(diagBlock, dim1=-2, dim2=-1).flatten()
    precond = torch.where(diagComponents.abs() > _MIN_DIAG, 1.0 / diagComponents, torch.zeros_like(diagComponents))
    if not torch.any(diagComponents.abs() > _MIN_DIAG):
        return None
    return precond


def buildBlockJacobiPrecond(
    diagBlock: torch.Tensor,
    ridge_frac: float = 1e-6,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Block Jacobi: invert the full `dim x dim` diagonal blocks and return a
    `psolve` callable (`None` when every block is ~0).

    A *scale-aware* ridge is added to each block's diagonal before inverting:
    `ridge = ridge_frac * (max |entry| in the block)`, floored at `_MIN_DIAG`.
    This keeps (near-)singular blocks -- which occur for the indefinite
    `exactHessian` operator -- invertible instead of producing Inf/NaN, while
    being a negligible relative perturbation for well-conditioned blocks (so
    on `legacyPairwise` the result is effectively the exact block inverse).

    If `torch.linalg.inv` still yields a non-finite entry for some block at
    the small ridge, it retries at a coarser ridge, then falls back to a
    pseudo-inverse (unconditionally stable). The inverse is computed once per
    Newton step, not per Krylov iterate, so the (dim <= 3) cost is negligible.
    """
    n, dim, _ = diagBlock.shape
    if not torch.any(diagBlock.abs() > _MIN_DIAG):
        return None
    device, dtype = diagBlock.device, diagBlock.dtype
    I = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)
    scale = diagBlock.abs().amax(dim=(-2, -1)).clamp_min(_MIN_DIAG).view(n, 1, 1)

    Minv: Optional[torch.Tensor] = None
    for frac in (ridge_frac, 1e-3):
        cand = torch.linalg.inv(diagBlock + (frac * scale) * I)
        if bool(torch.isfinite(cand).all()):
            Minv = cand
            break
    if Minv is None:
        Minv = torch.linalg.pinv(diagBlock + (1e-3 * scale) * I, rtol=1e-3)

    def psolve(r: torch.Tensor) -> torch.Tensor:
        return torch.einsum('nab,nb->na', Minv, r.view(n, dim)).flatten()

    return psolve
