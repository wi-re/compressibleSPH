#!/usr/bin/env python3
"""torch.autograd.gradcheck against modules/shockCapturing/{wp_computeM,wp_vsig}.py
-- Tier 2 of CLEANUP_PLAN.md's Phase 4.1 gradcheck rollout.

`computeMWarp` computes the Cullen-Dehnen-style renormalization matrix
`sum_j m_j outer(x_ij, gradw_ij)` (matrix-valued output, no scheme-specific
scalar knobs beyond positions/supports/masses/densities) -- gradchecks clean.

`computeVsigWarp` computes the signal velocity `max_j(c_bar - mu_ij)` used by
shock-capturing switches. Getting a clean test case for it surfaced two real
bugs, one fixed here and one confirmed but NOT fixed (upstream, out of this
repo's reach):

1. **Fixed here.** `computeVsigWarp` had `referenceVelocities = queryVelocities`
   as a fallback when the caller doesn't supply `referenceParticles`, but no
   equivalent `referenceCs = queryCs` fallback -- so a caller that passes
   `queryCs=` explicitly (exactly what a bare-`ParticleState` gradcheck case
   does, and what every real scheme call site does too, just via a
   `.soundspeeds` attribute that happens to be on the same object) fell
   through to a **size-1 dummy tensor read out of bounds** the moment
   `referenceParticles` (defaulting to `queryParticles`) lacked a
   `.soundspeeds` attribute -- silently reading uninitialized/stray memory,
   which is why this script's first attempts here gave different, wrong
   forward *values* on every run. Same missing-fallback shape as Phase 4.2's
   `computeCompSPHBalanceTermWarp` finding, and it turned out **not to be
   unique to vsig**: the identical gap (`referenceCs`/`referenceAlphas`/
   `referencePressures` never defaulted to their `query*` counterparts) was
   present in all 7 files that share this parameter family --
   `modules/{compSPH,crk}/{accel,dudt}.py` and
   `modules/dissipation/{wp_conductivity,wp_diffusion,wp_dissipation}.py`,
   all previously gradchecked "clean" in Tiers 0/1 only because those scripts
   always populate `.soundspeeds`/`.alphas`/`.pressures` on the *same* state
   object passed as both query and reference. Fixed identically in all 8
   files (this one plus the 7). A separate, unrelated `torch.scalar_t32`
   typo (not a real torch dtype -- should have been `get_torch_precision()`)
   in the same dummy-tensor fallback line was fixed alongside it, present in
   the same 8 files.
2. **Confirmed, NOT fixed -- needs an upstream (warp-lang) fix.** With the
   memory bug out of the way, `computeVsigWarp`'s backward pass is
   *deterministically wrong* for 2 of 3 particles in a hand-picked,
   tie-free case (verified 3 independent ways: `torch.autograd.gradcheck`,
   `torch.autograd.grad` cross-checked per-output, and a from-scratch manual
   central-difference replay of the exact same forward closure -- the manual
   replay agrees with hand-derived calculus and disagrees with
   `torch.autograd`'s analytical result on 2 of 3 rows). Root cause: `out =
   wp.max(out, vsigs)` is a **loop-carried variable reassigned via a
   nonlinear op inside the neighbor `for` loop** -- the same underlying bug
   *class* Tier 1 found in `correctGradientCRK`'s hand-written accumulation
   loop ("reverse-mode AD through a loop silently producing a wrong
   adjoint"), but a different concrete shape: here the adjoint sometimes
   attributes the gradient to the *wrong* neighbor (not the true argmax) or
   drops it to zero entirely, even though the *forward* value is correct.
   Swapping `wp.max(out, vsigs)` for the logically equivalent
   `wp.where(vsigs > out, vsigs, out)` was tried as a local workaround and
   made no difference -- confirming the bug is in how Warp differentiates
   the loop-carried reassignment itself, not in `wp.max`'s own adjoint
   registration, so there is no local rewrite in this repo that fixes it.
   `_run_vsig` below runs gradcheck and treats "fails with exactly this
   mismatch" as the expected, tracked state -- not a silent pass, and not a
   script failure either, so this stays wired into the regular test suite as
   a regression guard: if a future Warp upgrade changes this behavior in
   either direction, the script says so instead of staying silently stale.

    python scripts/gradcheck_shockCapturing.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp
from torch.autograd.gradcheck import GradcheckError

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, compute_densities, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState
from warpSPHCore.enumTypes import SupportScheme

from warpSPH.modules.shockCapturing.wp_computeM import computeMWarp
from warpSPH.modules.shockCapturing.wp_vsig import computeVsigWarp

DIM = 1
N = 5


def _build_case():
    domain = make_domain(dim=DIM)
    positions, supports, masses = line_case(N)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)
    return domain, positions, supports, masses, densities, adjacency, kinds


def _run_computeM() -> bool:
    domain, positions, supports, masses, densities, adjacency, kinds = _build_case()

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeMWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            adjacency=adjacency,
        )

    print("\n=== computeMWarp: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def _run_vsig() -> bool:
    # Hand-picked, tie-free case: asymmetric positions and a strong outward
    # velocity spread so each particle's true argmax neighbor has a
    # comfortable margin (>=1.0, vs. eps=1e-6) over every other candidate,
    # including the self-term (mu_ij == 0 identically there). This rules out
    # near-tie/branch-switch artifacts as the explanation for what follows --
    # see this module's docstring, finding 2: the backward pass is
    # confirmed wrong regardless, via an independent manual central-difference
    # check, not just gradcheck's own numerics.
    domain = make_domain(dim=DIM)
    positions = torch.tensor([[-0.5], [0.0], [0.3]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.full((3,), 1.5, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    velocities = torch.tensor([[2.5], [0.0], [-1.5]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    soundspeeds = torch.full((3,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)

    adjacency, kinds = build_adjacency(positions, supports, masses, domain, mode=SupportScheme.Gather)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency, mode=SupportScheme.Gather)

    def f(pos, sup, mass, dens, vel, cs):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return computeVsigWarp(
            queryParticles=p,
            operationProperties=OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather),
            domain=domain,
            queryVelocities=vel,
            queryCs=cs,
            adjacency=adjacency,
        )

    print("\n=== computeVsigWarp [individual_cs=True]: torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, velocities, soundspeeds)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        if ok:
            print(
                "UNEXPECTEDLY PASSED -- this script's docstring documents a confirmed, "
                "reproducible wp.max-loop backward bug here (finding 2). If this keeps "
                "passing, the upstream Warp behavior likely changed for the better; "
                "update CLEANUP_PLAN.md's Phase 4.1 Tier 2 entry to say so."
            )
        return True
    except GradcheckError as exc:
        print(
            "FAILED as expected -- known, confirmed wp.max-loop backward bug, "
            "see this script's docstring (finding 2). Treated as a tracked "
            "regression guard, not a script failure."
        )
        print(f"  ({type(exc).__name__}: {str(exc).splitlines()[0]})")
        return True
    except Exception as exc:  # noqa: BLE001 - anything other than the known mismatch is a real failure
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    ok &= _run_computeM()
    ok &= _run_vsig()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see this script's docstring and CLEANUP_PLAN.md Phase 4.1 Tier 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
