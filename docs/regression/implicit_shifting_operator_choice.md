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

## Comparison against the source paper

`legacyPairwise` is not something that drifted away from IIPS somewhere in
the diffSPH -> warpSPH port chain: it is what P. Rastelli, R. Vacondio, J.C.
Marongiu, G. Fourtakas & B.D. Rogers, *"Implicit iterative particle shifting
for meshless numerical schemes using kernel basis functions"*, Comput.
Methods Appl. Mech. Engrg. 393 (2022) 114716 — the paper this module
implements — actually derives and solves, in its own Eqs. (20)-(21) (1D)
and (40)-(42) (2D). `exactHessian` is a correction to a gap in that
derivation, found independently in this codebase and cross-checked against
finite differences; it is not "the paper's method, implemented correctly"
in the sense of matching what the paper's authors actually wrote down and
tested.

### Where the paper's own derivation drops the diagonal case

IIPS defines `f_i(X) = dC_i(X)/dx` (a function of the *entire* position
vector `X`, Eq. (13)) and Taylor-expands it in every particle coordinate
`x_j` (Eq. (12)):

```
f_i(X̄) = f_i(X) + sum_j [df_i(X)/dx_j] (x̄_j - x_j) + O(...)
```

To evaluate `df_i/dx_j`, the paper substitutes the SPH form of `f_i`
(Eq. (16)): `f_i(X) = sum_k [dW(x_i - x_k)/dx_k] omega_k`. Because `x_i`
appears in *every* term of that sum (through the shared argument
`x_i - x_k`), differentiating w.r.t. an arbitrary `x_j` genuinely splits
into two different cases, depending on whether `j` is the row particle `i`
itself or a neighbor `k != i`:

- **`j != i` (off-diagonal).** Only the single term `k = j` depends on
  `x_j`, through its explicit `x_k` slot. Differentiating that slot gives
  `d^2W(x_i - x_j)/dx_j^2 * omega_j = H_ij * omega_j`. This is the case the
  paper's Eq. (17)-(18) states and gets right.
- **`j = i` (diagonal).** *Every* term in the sum depends on `x_i`, through
  the shared `x_i` slot common to `x_i - x_k`. Differentiating that shared
  slot instead of the explicit `x_k` slot flips the sign (translation
  invariance: `d/da[W(a-b)] = -d/db[W(a-b)]`), and it does so once per
  neighbor: `df_i/dx_i = -sum_{k != i} H_ik * omega_k`. This is the case
  `exactHessian`'s derivation (module docstring, `implicitShifting.py`)
  handles explicitly; the paper's Eq. (17)-(18) does not.

Eq. (17)-(18) collapses both cases to a single rule — *"The only term in
Eq. (17) which is non-null is the one in which j = k"* — and applies it
uniformly to every `j`, diagonal included. That is the `j != i` reasoning
applied where the `j = i` reasoning is actually required. The consequence,
visible directly in the assembled matrix `A` of Eq. (21) (and its 2D
counterpart Eq. (42)), is a diagonal built as `d^2W_ii/dx_i^2 * omega_i =
H(0,h) * omega_i` — the kernel's own curvature at *zero* separation, using
the *same* `+H` sign convention as every off-diagonal entry — rather than
the sign-flipped, self-excluded, neighbor-summed quantity a complete chain
rule produces. The 2D derivation (Eq. (30)-(37)) makes the identical move
for both the second-derivative and cross-derivative blocks, so this is not
a 1D-only shortcut that the 2D formulation happens to avoid.

That single substitution is, entry for entry, the difference the "two
matrices, side by side" table above already documents:

| | paper's Eq. (21)/(42) diagonal | paper's Eq. (21)/(42) off-diagonal | matches |
|---|---|---|---|
| value | `H(0,h) * omega_i` | `+H_ij * omega_j` | — |
| sign convention | same `+H` as off-diagonal | same `+H` as diagonal | `legacyPairwise` |

The paper never assembles `exactHessian`'s `sum_{j != i} omega_j H_ij` /
`-omega_j H_ij` structure at all. Despite the sentence immediately after
Eq. (12) describing the procedure as "a Newton-Raphson procedure" (and the
text after Eq. (19) reiterating "this effectively corresponds to a
Newton-Raphson algorithm"), the linear system the paper actually writes
down and solves is *not* the Newton Hessian of its own stated objective
`grad(C) = 0`. In this codebase's terms it is `legacyPairwise` —
diagonally-dominant, self-included, uniform-sign — and diffSPH's
`getShiftingMatrices`, which `legacyPairwise` ports byte-for-byte,
implements exactly that system. `legacyPairwise` is therefore best read as
a faithful port of the paper, and `exactHessian` as a from-scratch,
independently-derived fix to a gap in the paper's own math (Eqs. (17)-(18),
(30)-(37)) that neither the paper nor diffSPH ever implemented.

### Why the paper's reported results never surface `exactHessian`'s failure mode

This also explains why the paper's own numerical tests (Section 4) never
show the instability this investigation found when actually assembling
`exactHessian`:

- Their matrix has a **fixed, configuration-independent diagonal**,
  `omega * H(0,h)` — the same property the "What `legacyPairwise` actually
  represents" section above identifies as the reason it degrades gracefully
  instead of blowing up. The paper's static test (Section 4.1) reports
  convergence to Cartesian-grid-level accuracy in 3-5 Newton-Raphson
  iterations, robustly across three resolutions and two initial-disorder
  levels (`sigma/Delta = 0.10, 0.25`, Figs. 5, 9), and the kinematic test
  (Section 4.2) reports similar robustness under continuous injected
  disorder (Table 3). That is exactly the clean, monotone behavior a
  bounded-diagonal operator produces, and exactly what this investigation's
  A/B test reproduces for `legacyPairwise` — including the "does not depend
  on beta / initializer" insensitivity documented in the section below,
  which is itself a symptom of a diagonal that doesn't depend on the current
  configuration.
- The paper's initial disorder is a **bounded perturbation of a Cartesian
  grid** (`sigma/Delta` up to 0.25 — still, by construction, a grid particle
  displaced by a fraction of its own spacing), not literally uniform-random
  placement in the domain. That is milder than the "fully random initial
  positions" stress case
  (`test_implicitShiftingConvergesFromFullyRandomPositions`) that exposed
  `exactHessian` stalling/oscillating in this investigation. Even had the
  paper's authors assembled the true Hessian, their own test cases may
  simply never have been harsh enough to expose its unbounded-diagonal
  failure mode — a crowded, near-coincident configuration is a much more
  extreme regime than a 10-25%-of-spacing grid jitter.

Net effect: the paper's empirical case for IIPS's robustness is evidence
for the operator this codebase calls `legacyPairwise`, not for the
mathematically "corrected" `exactHessian` this investigation additionally
derived and validated against finite differences. Defaulting to
`legacyPairwise` is therefore not merely an empirically-motivated
workaround for a limitation the paper itself doesn't have — it is the
operator the source paper's own convergence results (Figs. 5-9, Table 1-4)
actually characterize.

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
   Neither was the source of the discrepancy. (That bit-identity statement
   describes the original port; `bicgstabSolve` has since been hardened --
   relative breakdown guards, true-residual-verified convergence, explicit
   status codes, and a GMRES alternative behind
   `ShiftingImplicitSolver` -- see the module docstring. The per-iterate
   recurrence is unchanged, so non-pathological solves still follow the
   same iterate sequence.)
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

## Opting into the improved inner solve (fallback chain + `dynamic` scheme)

The investigation above is about *which operator* to solve. A separate,
orthogonal robustness problem is *what happens when the inner Krylov solve
bails out*: the original `computeImplicitShift` consumed a bailed-out iterate
(status `< 0` -- BiCGStab rho/omega breakdown, threshold bailout, or an
exhausted iteration budget) exactly as if it had converged. The fix is an
**opt-in** fallback chain, so the legacy path stays byte-identical for users
who don't ask for it.

**What the chain does.** On a primary-solver bailout, `solverDriver
.solveImplicitSystem` (now the shared entry both `computeImplicitShift` and
`computeImplicitShiftAutomatic` call) retries, and keeps the best iterate by
its stamped true residual:
  - `krylov`: retry with the *other* Krylov solver (BiCGStab<->GMRES) from a
    clean start. This is the high-value fallback -- the two solvers fail in
    different regimes. On the exact production operator at `jitter=0` (the
    documented BiCGStab rho-breakdown case), BiCGStab bails at
    `rel_resid ~ 6e-3` while the GMRES retry converges to `rel_resid ~ 3e-6`.
  - `krylov_richardson`: `krylov`, plus a bounded Richardson
    (`richardson.richardsonSolve`) polish from the best iterate, as a last
    resort. Richardson is deliberately last: an eigenvalue probe shows the
    production diagonal *diverges* as a Richardson step direction (so it uses
    `M = I` with an auto-tuned, backtracking step size), it is much slower
    than GMRES on the positive-definite `legacyPairwise` operator, and it does
    not converge at all on the indefinite `exactHessian` operator. It is a
    bounded, never-NaN backstop, not a primary.

**How to opt in (pick one):**
  - **Easiest -- use `ShiftingScheme.dynamic`.** This dispatches to
    `computeDynamicImplicitShift`, the same implicit path with the `krylov`
    fallback enabled by default. An explicit `implicitFallback` choice is
    respected (it is never downgraded).
    ```python
    schemeConfig.shiftProperties.scheme = ShiftingScheme.dynamic
    ```
  - **Keep `ShiftingScheme.implicit`, enable the fallback explicitly.**
    ```python
    schemeConfig.shiftProperties.scheme = ShiftingScheme.implicit
    schemeConfig.shiftProperties.implicitFallback = ShiftingImplicitFallback.krylov
    # ...or ShiftingImplicitFallback.krylov_richardson to also add the
    # bounded Richardson polish
    ```
  - **No change -- `ShiftingScheme.implicit` with the default
    `implicitFallback=none`** runs exactly one solver and uses its result
    unconditionally. This is the historical behavior, and it is what the
    existing `test_implicitShifting.py` suite (and any legacy user) still sees.

On any fallback activation a `UserWarning` is emitted with the primary's
status and stamped residual, so a recovered inner solve is no longer silent.

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
