# Implicit shifting: why the "wrong" matrix converges better (warpSPH)

**Date:** 2026-08-19. Investigates a stability problem in
`modules/shifting/implicitShifting.py`'s implicit particle shifter (IPS):
solved correctly, it should still be unreliable far from equilibrium; ported
faithfully from `diffSPHLegacy`, it wasn't reliable at all. This note is the
"why," kept separate from the code's own docstrings (which state the
conclusion) because the derivation and the evidence trail are worth having
somewhere a future reader can follow without re-deriving them.

**Outcome:** `ShiftingImplicitOperator.legacyPairwise` is now the default
(`configurations/moduleConfigurations/shifting.py`); the mathematically exact
Newton Hessian is kept as `ShiftingImplicitOperator.exactHessian`, an
explicit opt-in for comparison. This note explains what each one actually
*is* and why the "wrong" one wins in practice.

## The setup

IPS solves for a position update `dx` that drives the SPH concentration
field `C_i = sum_j omega_j W_ij` toward uniformity, via one step of
`A @ dx = -grad(C)`. `grad(C)`'s row-`i` entry is `sum_j omega_j J_ij`,
`J_ij := J(x_i - x_j)` the per-pair kernel gradient (a function of the
separation vector alone). Two different matrices have been used for `A`:

- **`exactHessian`**: the true Newton Hessian of `C`.
- **`legacyPairwise`** (default): ported byte-for-byte from diffSPH's
  original `getShiftingMatrices`/`bicgstab_shifting`.

Both were derived and cross-checked against a finite-difference `Hess(C)` on
a 36-particle test case (see `implicitShifting.py`'s module docstring for
the full chain-rule derivation): `exactHessian` matches to 1.7e-4 relative
Frobenius error; `legacyPairwise` does not (222% relative error). So
`legacyPairwise` is **not** a different-looking way to write the same exact
Hessian — it's a genuinely different matrix. The rest of this note is about
what that matrix actually is, and why being "wrong" doesn't make it worse.

## The two matrices, side by side

Write `H_ij := H(x_i - x_j)`, the per-pair kernel Hessian (a function of the
separation vector, symmetric: `H_ij = H_ji`, since a radial kernel's second
derivative is even). Both operators are built from the *same* `H_ij` values
— per-pair kernel formulas were checked to agree between the two codebases
to float32 noise (~1e-5 relative) on identical inputs, so this is purely an
assembly difference, not a kernel-formula bug.

| | diagonal block (row `i`, column `i`) | off-diagonal block (row `i`, column `j != i`) |
|---|---|---|
| `exactHessian` | `sum_{j != i} omega_j H_ij` | `-omega_j H_ij` |
| `legacyPairwise` | `omega_i H_ii` (`H` at zero separation) | `+omega_j H_ij` |

`exactHessian`'s columns come from a careful bivariate chain rule through
`g(a,b) = W(a-b,h)`: differentiating w.r.t. the shared `a`-slot gives `+H`,
w.r.t. the `b`-slot gives `-H` (translation invariance forces the sign
flip), and the self-pair's *own* contribution cancels to exactly zero once
both slots are correctly recognized as the same variable (see the module
docstring). `legacyPairwise` uses the `+H` convention uniformly for every
column, including a nonzero self term at `H(0,h)` (the kernel's own
curvature at its peak) instead of the correctly-cancelling zero. That's the
entire difference: **one sign convention (`+H`) applied everywhere, instead
of `+H` on the diagonal and `-H` off it, with the self-pair's true
contribution (zero) replaced by a nonzero placeholder.**

(The table above is diffSPH's original convention, the clearest form for
explaining what the operator *is*. `implicitShifting.py`'s
`_buildDiagBlock` actually builds `legacyPairwise`'s diagonal already
negated, so that this codebase's uniform `update = -xk` /
`positions += update` convention stays correct for both operators — see
the sign note in the evidence trail below. Negating the whole matrix
doesn't change anything about its conditioning, only which overall sign of
it the solver sees, so it doesn't affect the argument that follows.)

## What `legacyPairwise` actually represents

Under uniform particle mass and a constant reference density (`omega_j = m/
rho0` for all `j` — the common case: `summationDensity=False`, this
codebase's default), `omega_i = omega_j = omega` for every pair, and the
whole operator collapses to something much more legible:

```
(A_legacy)_i @ x = omega * H_ii @ x_i + omega * sum_{j!=i} H_ij @ x_j
                  = omega * sum_j H_ij @ x_j        (sum over ALL neighbors, self included)
```

That is: `A_legacy = omega * H`, where `H` is the matrix with block `(i,j)
= H_ij` placed **directly** at every pair in the neighbor graph, self
included. Since `H_ij = H_ji`, this matrix is symmetric by construction —
not because of any graph-Laplacian cancellation (translation invariance),
but because the underlying kernel Hessian is itself a symmetric function of
the separation vector. `legacyPairwise` is, in this common case, nothing
more than *the kernel's own Hessian, sampled at the current pairwise
separations and used directly as a matrix.* It has no relationship to
`Hess(C)`'s combinatorial structure (diagonal = sum over neighbors,
off-diagonal = negated) at all — it only happens to be built from the same
per-pair `H_ij` values.

This reframing is what explains the stability gap:

- **`exactHessian`'s diagonal is configuration-dependent and unbounded.**
  `sum_{j!=i} omega_j H_ij` accumulates one large-magnitude `H_ij` term
  (kernel curvature is largest near its own peak) per nearby neighbor. A
  crowded or near-coincident configuration — exactly what "fully random
  initial positions" produces, before any relaxation has happened — can
  make this diagonal arbitrarily large; a configuration with very few
  neighbors can make it arbitrarily close to singular. Either extreme
  degrades the local quadratic model Newton's method relies on, which is
  a known, general limitation of Newton's method on non-convex objectives:
  it is only reliably convergent once already close to the solution, where
  the model is trustworthy. Damping (`implicitRelaxation`) and step-clamping
  (`ShiftProperties.threshold`) reduce the damage per step but don't fix
  this — confirmed empirically: even swept across relaxation factors from
  0.1 to 1.0 with the clamp active, `exactHessian` still stalls or
  oscillates on fully-random starts.
- **`legacyPairwise`'s diagonal is a fixed constant**, `omega * H(0,h)` —
  the kernel's own curvature at zero separation, entirely independent of
  how the particles are currently arranged. The whole operator's
  conditioning is bounded by the kernel's own smoothness, not by how
  pathological the current configuration is. It degrades gracefully instead
  of blowing up, which is exactly the property a far-from-equilibrium
  relaxation needs and a literal Newton step structurally cannot offer.

None of this makes `legacyPairwise` a "smoothing"/diffusion operator in the
strict graph-Laplacian sense (it does not reduce to `sum_j c_ij (x_i -
x_j)` with a shared coefficient the way, e.g., a Brookshaw-style SPH
Laplacian does — its diagonal and off-diagonal coefficients come from
unrelated points on the kernel's Hessian, `H(0,h)` vs. `H(x_i-x_j,h)`, not a
single shared `c_ij`). The accurate description is the one above: it's the
kernel's own Hessian used as a matrix, and that alone is enough to explain
why it's well-behaved regardless of particle configuration.

## The evidence trail (and a self-correction worth keeping)

Getting to the explanation above required ruling out several other
candidate explanations first, in order, each with a real check rather than
an assumption:

1. **Per-pair kernel formulas.** `warpSPHCore.sphKernel*` (warp) vs.
   `diffSPH.kernels.KernelWrapper` (torch) agree to float32 noise on
   identical inputs — not a kernel-formula bug.
2. **`exactHessian`'s assembly.** Matches a finite-difference `Hess(C)` to
   1.7e-4 relative Frobenius error — the current default (at the time) was
   provably correct, so "fix the math" wasn't the answer.
3. **`legacyPairwise`'s assembly**, first pass, driven through this
   codebase's `wrapper.solveShifting`: appeared to perform *worse*
   (monotonic divergence) than `exactHessian`. This was wrong, and the
   reason is worth recording: diffSPH's own `computeShifting` computes
   `update = -xk` and its `solveShifting` applies it as `positions -=
   update` (`diffSPH/v2/modules/shifting.py:999`) — the *opposite* of this
   codebase's `wrapper.solveShifting`, which does `positions += update`.
   The first attempt kept `update = -xk` while running through the `+=`
   convention, silently flipping the effective step direction. Once the
   sign was corrected (`legacyPairwise`'s diagonal block is built already
   negated — see `implicitShifting.py`'s `_buildDiagBlock` — specifically
   so this codebase's uniform `update = -xk` convention stays correct for
   both operators), the "worse" result reversed completely.
4. **RHS (`grad(C)`) and solver.** Before trusting the corrected result,
   both were checked directly against diffSPH's real, unmodified code on
   identical random positions: `grad(C)` matches to 5+ significant figures;
   `bicgstabSolve` (this codebase's port) produces bit-identical output to
   diffSPH's original `bicgstab_shifting` on the same linear system,
   including hitting the same divergence bailout at the same iteration.
   Neither was the source of the discrepancy.
5. **diffSPH's real pipeline**, run end-to-end (not reimplemented) on the
   same random positions: clean, monotonic convergence, density std
   0.30 -> 0.0009 over 15-40 outer iterations, reproduced bit-for-bit
   across independent runs. This is what made the sign bug in step 3
   findable — a real discrepancy between "my reimplementation" and "the
   real thing" that had to be a bug in the former, not a property of the
   operator.

With every other explanation ruled out by a concrete check rather than an
assumption, the sign-corrected A/B result was trustworthy: `legacyPairwise`
converges cleanly and monotonically from fully-random initial positions
where `exactHessian` stalls or oscillates on the same seeds, and the
structural explanation above (configuration-independent vs.
configuration-dependent conditioning) is consistent with every one of these
checks, not just the final A/B number.

## Initializer and preconditioner: investigated, not the answer

Two further levers were swept once `legacyPairwise` became the default, in
case either explained additional headroom:

- **`implicitUsePreconditioner`** (Jacobi): on vs. off gives bit-identical
  output on the fully-random stress case. Traced to why: the solver's
  `-12` "step-magnitude breakdown" divergence bailout fires early enough on
  these far-from-equilibrium starts that the preconditioner's effect on
  convergence *rate* never gets a chance to matter before the bailout
  returns the current iterate.
- **`implicitInitializer`** (seeding the Newton solve with a delta-SPH
  shift, `deltaPlus`/`deltaMinus`, instead of zero): <1% difference for
  `legacyPairwise` (already convergent enough that the initial guess barely
  matters), and does **not** rescue `exactHessian` either — all three
  initializers left it stalled on all tested seeds, some finishing worse
  than they started.

Neither is a path to further improvement on this specific problem; the
operator choice above is what mattered.

## Practical notes for picking a mode

- `legacyPairwise` (default) is the right choice for anything that might
  start far from a relaxed configuration: initial particle sampling,
  scenes with newly-inserted particles, or generally whenever the shifter
  can't assume it's already close to uniform.
- `exactHessian` is kept because it's the one with an automatic-
  differentiation counterpart (`implicitShiftingAutomatic.py`'s
  `computeImplicitShiftAutomatic`, sourced from `warpOperationJVP`/
  `warpOperationHVP` rather than a hand-assembled `H`) — there's no
  autodiff version of `legacyPairwise` to cross-check against, since it
  isn't an actual Hessian of anything. `test_implicitShiftingComparison.py`
  pins `exactHessian` for exactly this reason: that test validates the
  hand-built matvec against the autodiff-sourced one, which only makes
  sense when both sides are computing the same mathematical object.
- A configuration that is already close to uniform (e.g. mid-simulation,
  after the flow has relaxed once) is exactly the regime where
  `exactHessian`'s quadratic convergence would be an advantage over
  `legacyPairwise`'s first-order-ish behavior, if that ever matters in
  practice. Nothing here rules that out; it just wasn't the problem this
  investigation was chasing. A hybrid (start on `legacyPairwise`, switch to
  `exactHessian` once some convergence criterion is met) is a reasonable
  future direction that hasn't been explored.

## Reproducing this

None of the ad hoc scripts used during this investigation
(`compare_ips_hessian.py`, `apples_to_apples.py`, `compare_B_vector.py`,
`compare_solvers.py`, `stress_random_ips.py`,
`stress_random_ips_legacyop.py`, `sweep_init_precond.py`) are checked into
this repo — they lived in a scratch directory. The permanent regression
coverage is `tests/test_implicitShifting.py`'s
`test_implicitShiftingConvergesFromFullyRandomPositions` (3 seeds, asserts
`legacyPairwise` recovers from a fully random start) and the existing
`test_implicitShiftingComparison.py` (now pinned to `exactHessian`, checking
the hand-built and autodiff-sourced exact-Hessian code paths still agree).
Anyone extending this should start there, and from `implicitShifting.py`'s
module docstring for the underlying chain-rule derivation, rather than
re-deriving either from scratch.
