"""Phase 4 step 3 of `warpSPHCore/warpier_forward_mode_plan.md`: "`Hess C . v`
is a JVP of that JVP" -- validated against `wp_implicitShifting.py`'s own `H`
output, the same way `test_implicitShiftingGradientJVP.py` validated step 2's
`grad C` against its `J` output.

`implicitShifting.py`'s Newton solve needs `Hess(C) @ x` (the BiCGStab
matvec, `implicitShifting._multiplyLaplacianBlock`): `out_i = diagBlock_i @
x_i - sum_{j in N(i)} H_ij @ x_j`, `diagBlock_i = sum_j omega_j H_ij`, built
from `wp_implicitShifting.computeShiftingPairTerms`'s hand-rolled per-pair
`sphKernelHessian` call plus a hand-written `torch.einsum`/`scatter_sum`
assembly -- no autograd involved, and (per the module's own docstring) two
non-obvious pitfalls found by comparing against a dense finite-difference
Hessian: self-pairs must be dropped before assembly (the kernel Hessian's
near-origin regularization is unstable, not exactly zero, there), and the
off-diagonal block is `-omega_k H_ik`, not the naively-placed `omega_j
H_ij` (needed for the assembled operator to come out symmetric).

`warpSPHCore.warpOperationHVP` (Phase 4 step 3,
`coreOperations.wp_densityHVP.computeSPHDensityPositionHVP`) reduces to
exactly `sum_j m_j * H_ij @ (v_i - v_j)` -- the same closed form as
`_multiplyLaplacianBlock(diagBlock, Hw, v, ...)` once expanded algebraically
-- but was derived by differentiating `warpOperationJVP`'s own position
tangent a second time ("a JVP of that JVP"), then reusing
`kernels.hessian.sphKernelHessian` (the same building block
`computeShiftingPairTerms` calls) through `warpOperationHVP`'s own
adjacency/self-pair handling, not by transcribing `_multiplyLaplacianBlock`.
Agreement here is therefore evidence the composed-JVP route reaches the same
operator, not a tautology -- and, per the plan's step 5, whether it does so
*without* needing the self-pair drop `computeImplicitShift` needs by hand is
exactly the thing worth checking explicitly below.
"""

from __future__ import annotations

import torch

from warpSPH.utils import buildDomainDescription
from warpSPH.configurations.simulationConfig import buildConfig
from warpSPH.sample.regular import sampleRegularParticles
from warpSPH.modules.shifting.wp_implicitShifting import computeShiftingPairTerms
from warpSPH.modules.shifting.implicitShifting import _multiplyLaplacianBlock
from warpSPH.math import scatter_sum
from warpSPHCore import ParticleState, SupportScheme, buildVerletList, warpOperationHVP, OperationProperties
from warpSPHCore.enumTypes import WarpOperation, OperationDirection


def _jitteredLatticeState(nx, dim, L, device, dtype, jitter, seed):
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                      dtype=dtype, dx=L / nx, cflFactor=0.3)
    template = sampleRegularParticles(nx=nx, domain=domain, targetNeighbors=config.targetNeighbors)
    n = template.positions.shape[0]
    particleDx = template.masses.pow(1 / dim).mean().item()

    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(template.positions.shape, generator=gen).to(device=device, dtype=dtype)
    positions = template.positions + jitter * particleDx * noise

    state = ParticleState(
        positions=positions, supports=template.supports, masses=template.masses,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        densities=torch.full((n,), 1000.0, device=device, dtype=dtype),
    )
    return state, config, domain


def test_densityPositionHVP_matches_implicitShiftings_ownHessianMatvec():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L, rho0 = 2, 8, 1.0, 1000.0

    state, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                    jitter=0.1, seed=1234)
    n = state.positions.shape[0]
    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)

    # --- Hand-built reference: implicitShifting.py's own H, assembled by ---
    # --- its own _multiplyLaplacianBlock matvec. ---------------------------
    _K, _J, H = computeShiftingPairTerms(state, domain, config.kernel, adjacency)
    pairMask = adjacency.i != adjacency.j  # implicitShifting.py drops self-pairs before assembly
    i, j, Hm = adjacency.i[pairMask], adjacency.j[pairMask], H[pairMask]
    omega = state.masses / rho0
    Hw = Hm * omega[j, None, None]
    diagBlock = scatter_sum(Hw, i, dim=0, dim_size=n)

    gen = torch.Generator(device='cpu').manual_seed(5678)
    v = torch.randn(n, dim, generator=gen).to(device=device, dtype=dtype)

    reference = _multiplyLaplacianBlock(diagBlock, Hw, v.flatten(), i, j, n, dim).view(n, dim)

    # --- Automatic: warpOperationHVP(Density) with omega as the mass channel, ---
    # --- v applied to both query and reference roles (the same particles moving --
    # --- together, matching implicitShifting's own usage). -----------------------
    omegaState = ParticleState(positions=state.positions, supports=state.supports, masses=omega,
                               kinds=state.kinds, densities=None)
    props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    automatic = warpOperationHVP(omegaState, props, domain, adjacency=adjacency,
                                 tangentQueryPositions=v, tangentReferencePositions=v)

    torch.testing.assert_close(automatic, reference, rtol=1e-4, atol=1e-6)
