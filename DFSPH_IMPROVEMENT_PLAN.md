# warpSPH — DFSPH (Incompressible) Improvement Plan

Two independent workstreams to harden and extend the incompressible
(DFSPH/divergence-free) SPH path:

1. Investigate and fix the density drift seen on the forced Kolmogorov flow
   case (`src/warpSPH/cases/kolmogorovIncompressible.py`).
2. Add mDBC boundary-particle handling to the incompressible scheme, with
   three selectable fidelity levels, validated on a bounded port of the
   random-velocity-field case (06-randomFlow).

Written up here so it can be picked up/continued across sessions.

## Status

**Part 1: primary fix landed and validated past the case's own `tLimit`.**
**Part 2: all 8 steps landed, plus an nx=128 production-resolution
follow-up and two bug fixes found along the way.** A new case
(`randomFlowIncompressible`, step 7) now exercises mDBC boundary particles
under DFSPH for the first time — previously the machinery from steps 1–6 was
a verified no-op, since no case sampled `kind==1` particles under
`divergenceFree`. All three `BoundaryPressureMode` values run to completion
on it at nx=128, matched to t≈1.5. The nx=128 follow-up found and fixed two
bugs in turn: (1) boundary-row pressure masking was zeroing
`BoundaryPressureMode.mdbcMlsPressure`'s projected pressure instead of
freezing it, making Option c a silent no-op; fixing that surfaced (2) an
undamped feedback loop between the boundary pressure projection and the
fluid pressure solve that NaN'd within ~7 steps once Option c was actually
live, fixed with a new under-relaxation factor
(`mdbcPressureRelaxation`, default `0.3`). With both fixes,
`mdbcMlsPressure` is now the most stable *and* most accurate of the three
modes. Separately, confirmed the DFSPH/deltaSPH density-band gap itself is
not a resolution artifact (nx=128 doesn't close it) and is not primarily a
`BoundaryPressureMode` effect — that gap is still open, see "Open questions"
below.

**Part 3: one bug fixed (velocity-resample sign/axis error in the VD+PS
particle-shift, `systems/incompressible.py`), verified to not explain either
of Part 2's open gaps; a second, unrelated DFSPH+free-surface case found
broken (`rotatingSquarePatch.py`) and root-caused but not fixed; `integrateRho`
audited and found to have a dead `True` branch.** See "Part 3" sections below.

**Part 4: root-caused and fixed (2026-08-27).** Redirected to the
periodic/boundary-less case (`kolmogorovIncompressible`) because it is the
load-bearing building block every other DFSPH case sits on. The previous
session found a live, reproducible NaN (`solveIncompressible`'s pressure mean
climbing to 2.4e6 and blowing up at step 574, nx=128) and ruled out the
non-negative clamp, the iteration budget, and the initial grid's symmetry as
causes. This session found the actual mechanism and landed a fix:

- `solveIncompressible` is an *implicit particle-shifting* solve, not a
  momentum pressure solve (both call sites feed its output into a position
  shift), so its "pressure" is a shifting potential.
- Its setpoint is unreachable. The SPH summation density's particle average
  is *minimised* by the lattice and rises quadratically with disorder (shown
  directly, with no dynamics involved, in
  `scripts/probe_densityBiasVsDisorder.py`; it follows from Parseval for any
  positive-definite kernel), so `mean_i rho_i == rho0` cannot be attained.
  The solver responds by winding up the one mode with a nonzero mean
  response — the near-constant one, whose response is weak — without bound.
  **This retires the previous session's option 1: the systematic density
  bias is structural, not a bug anyone can go find and fix.**
- Fixed with `ShiftPressureGauge.minShift` (subtract the fluid minimum:
  non-negative *and* gauge-fixed), **now the default**, scoped to solves
  where the constant mode is genuinely free. On `kolmogorovIncompressible`
  at nx=128 it turns a run that NaNs at step 574 into a stable 1000-step run
  with density held inside [0.980, 1.015] against the clamp's [0.240, 2.515].
  Mean-centering — the fix that works in `solveDivergenceFree` — was tested
  and is the *worst* option here.
- The default change reaches exactly two cases (the periodic,
  complete-support ones): `kolmogorovIncompressible`, which it fixes, and
  `tgv`, which it leaves alone — TGV's kinetic-energy decay still matches
  the analytic rate to within 0.5% of the clamp's answer at nx=64 and
  nx=128, still monotone, still inside the band `tests/test_physics.py`
  asserts. Every wall-bounded or free-surface solve falls back to the
  historical clamp by construction, so nothing else moves. Full suite
  (241 passed) and gradcheck pass.

**Part 5: bounded DFSPH stability root-caused; a working (opt-in)
configuration landed (2026-08-27).** `randomFlowIncompressible --bounded` NaNs on its own at
nx=128/t=5.54 (past the t≈1.5 Part 2 validated; not a regression, and not the
Part 4 mechanism). Cause: fluid leaks into the wall band and piles up there
at 30%+ over-density, monotonically, until it detonates. The deltaSPH sibling
on the same geometry never lets a single particle in, so the geometry and
boundary sampling are fine. It is a timestep threshold: stability turns over
between half a particle spacing and a whole one of per-step displacement, and
the default `cflFactor=0.3` sits just past it at 1.2 spacings. At
`cflFactor=0.125` the case runs to t=8 with penetration in a steady state.
Not the boundary-pressure mode (all three diverge), not
`mdbcNoPenetrationShift` (helps ~20%), and *not* the unbounded implicit shift
(capping it makes things worse — a useful negative result). Two fixes exist. A wall-aware `dt`
constraint would work but costs ~2.4x throughput, and is not landed. Better:
the scheme turns out to have *no velocity-level response to a density error*
at all (DFSPH proper applies its constant-density solve to the velocity; this
scheme repurposes it as a position shift), and restoring that
(`ShiftApplication.positionAndVelocity`, new) reaches t=8.0 at the *default*
CFL in 387 steps with 9x lower near-wall error and penetration in a steady
state. It is dissipative, though — it drives `tgv`'s kinetic-energy decay to
1.93x the analytic rate — so it ships opt-in, with the near-wall-only
refinement that would likely remove that cost written up but not done.
MLS-projection timing and divergence-free solver options were both tested
and neither matters (<=10%). A second round found the better formulation:
`inStepVelocity` (DFSPH proper — velocity correction inside the step, no
position shift) gives a 30x better near-wall density error than the default's
death state and holds `rho` in [0.986, 1.140] to t=8.0. Both velocity modes
damp `tgv` at ~3.3x the analytic decay rate, and that cost is now understood
as Part 4's unreachable setpoint being integrated into the momentum equation
— which is also the explanation for why this scheme uses a momentum-neutral
position shift in the first place.

**Part 6: scheme-naming/architecture question opened, nothing restructured
(2026-08-27).** `schemes/dfsph.py::dfsph_step` does not implement DFSPH — it
is a divergence-free projection plus a position-based particle shift — though
the registered scheme name (`divergenceFree`) is accurate, so the misnomer is
confined to the file/function names. "Real DFSPH" is now measurable rather
than hypothetical (`ShiftApplication.inStepVelocity`): much better at walls,
worse on `tgv`, and the specific pairing of it with MLS-pressure boundaries
NaNs at t=0.21, inverting Part 2's boundary-mode ranking. Recommended
sequencing and a list of questions for a literature session on DFSPH and its
derivatives are written up in Part 6; the owner wants that session before any
restructuring.

**Part 7: the literature session is done (2026-08-27).** Run against
Cornelis et al.'s VD+PS source-term paper and Band et al.'s MLS pressure
boundaries paper. Headline: **this scheme is not a mis-named DFSPH, it is a
faithful implementation of Cornelis et al.'s VD+PS**, step for step — so the
position shift is the method rather than a deviation from it, the velocity
modes are the departures, and the `tgv` over-dissipation Part 5 measured is
the published, expected behaviour of a velocity-applied density-invariant
solve (the paper's abstract says so; its Fig. 3 measures it). Part 6's naming
proposal changes target accordingly: the file is `vdps.py`, not a broken
`dfsph.py`. The unreachable setpoint is real in the paper too — it is avoided
by never putting the residual in momentum, by a tolerance set *above* the
structural floor (theirs 1e-3, this codebase's 5e-4 against a measured floor
of 2.7e-3), and by keeping the sampling ordered enough that the floor stays
under it. Six discrepancies against the papers are listed and ranked, three
of them one-line changes; the boundary-pressure feedback loop this codebase
damps turns out to be a documented failure mode of holding boundary pressure
as *state*, which the standard formulation avoids by recomputing it inside
every solver iteration. **"Next session" item 1 is superseded** — see Part 7.

**Part 8: the DFSPH and IISPH originals (2026-08-27).** Bender & Koschier
2015 and Ihmsen et al. 2014 arrived after Part 7 and **correct it**. Three
things. (1) Part 7's account of why published solvers converge was wrong —
DFSPH runs a *tighter* density tolerance than this codebase (1e-4 vs 5e-4)
and still converges in 4.5 iterations. The real cause is local: two case
files (`kolmogorovIncompressible.py:71`, `tgv.py:53`) normalise mass so
`rho0` equals the *lattice* summation density, which Part 4 proved is the
structural **minimum** — so the setpoint is pinned exactly at the
unreachable floor, by one line, in exactly the two cases where the wind-up
was ever seen. (2) DFSPH's Algorithm 1 runs its solvers in the order
density -> integrate -> divergence, deliberately, so the divergence solve
cleans up the density solve's velocity side-effect; `inStepVelocity` has
them **swapped**, so it is not "DFSPH proper" as Part 5/6 claim. Both
inputs to every `tgv` dissipation number in this document are therefore
suspect. (3) `computeAlpha` is confirmed to discretise IISPH's diagonal but
carries an extra power of `rho` and the wrong neighbour set on one sum —
measurable directly, and a candidate explanation for `omega < 0.355`
against both papers' 0.5. Also closed: one-sided source clamping and the
Krylov-on-clamped-solve path both have published negative results. The
scheme split the owner proposed is now fully specified rather than open.
**Part 8's items 1 and 2 were then run, and both were falsified** ("Part 8 —
results", end of document): moving the setpoint kills the run 150 steps
*sooner* (the source's negative mean is the shifting solve's de-clumping
drive), and `computeAlpha` measures as the true operator diagonal to within
1.6% with `rho` out to 1.30, so `omega_eff = omega` exactly. What those two
negative results expose is the **stopping criterion**: `minShift` runs its
full 64 iterations every step for 1000 steps and never terminates, yet those
iterations are productive (`rhoErr` 1.15e-3 at 64 against 3.80e-3 at 8), so
an *absolute* compression threshold is simply the wrong test. `rtol` already
exists in the config and the Jacobi path ignores it.
**Item 3 (the CFL constant) then landed as the session's one positive result**:
[BK]'s `dt <= 0.4 d/|v_max|` is stated in particle *diameters*, this codebase
applies `cflFactor` to the support radius `h`, and `n_h=4` makes the default
1.2 spacings per step — 3x the published limit. At `cflFactor=0.1` the bounded
case reaches t=8.0 **at stock settings** (near-wall `|rho-1|` 2.6e-2 against
0.30 at the default's death, beating the opt-in `positionAndVelocity` mode
without its 3.2x `tgv` dissipation), `kolmogorovIncompressible` improves 23%,
`tgv` is provably unaffected (its `dt` is pinned at 1e-3; the CFL never binds),
and Part 4's step-574 NaN disappears even under the *historical* gauge. A
fourth correction also landed: Part 8's solver-ordering claim was overstated —
under `semiImplicitEuler` the ordering is equivalent to [BK]'s up to loop
phase. Then a fifth correction, from the project owner: **this codebase's walls are
not support-truncated.** `BOUNDED_BAND = 5` samples the boundary as a solid
five-layer band against a 4-spacing support radius, so the Shepard sum is
~1.00 to the wall and the operator's constant mode is as near-null there
(`|A.1|/|A.rand|` 0.19) as in the bulk (0.17). That falsifies half of
`ShiftPressureGauge`'s stated justification for refusing to gauge bounded
solves — and re-testing the other half showed Part 4's "minShift diverges at
t=0.69" was measuring the 3x-oversized timestep: **at the published CFL,
`minShift` on the bounded case does not diverge, covers 38% more time per
step budget, has 19% lower density error, and costs half the wall time.**
Also this session: MINRES *without the non-negativity clamp* is the largest
accuracy win found (2.15x on the periodic case, ~nothing at a wall), exactly
as [I] Sec. 3.2 and [BK] Sec. 5 predict; and the deltaSPH shift-on-top was
found to be an inert-but-expensive no-op (now fixed, then tested and
rejected). **Start from "Part 8 item 3"'s landing recommendation, which is
now entangled with the `minShift` scoping — the two default changes should
land together or not at all.**

**Part 9: the boundary terms the papers do not compute (2026-08-28).** The
project owner's next objection, checked against SPlisHSPlasH and confirmed:
the published solvers do not evaluate every operator term for a *static*
boundary particle, and this codebase evaluates all of them. Two terms --
`computeAlpha`'s second sum (`dp_i/dx_j`) and the operator's neighbour-
acceleration term `a_j` -- describe a reaction a particle that never moves
cannot have; [BK] 3.2 says so in one sentence and SPlisHSPlasH's
`computeDFSPHFactor`/`compute_aij_pj` implement it literally. In the
wall-adjacent bin those terms are **40% of the diagonal** (55% inside the
band) and exactly zero beyond 3 spacings. Dropping both
(`BoundaryOperatorTerms.staticBoundary`, new, default-off) cuts the bounded
case's density error **5.9x** at the published CFL (1.78e-1 -> 3.00e-2,
`rho_max` 1.247 -> 1.007) for the same cost -- but only there: at the shipped
`cflFactor=0.3` it dies sooner than the baseline, the same entanglement
`minShift` has. Dropping the term from the diagonal *alone* NaNs in 47 steps,
which is measurement 1's 1.6x oversized relaxation showing up on schedule.
The sharpest result is that **the win belongs to the constant-density/shifting
solve, and the same change applied to the divergence-free solve alone diverges
at t=1.65** — so a future default belongs on `RelaxedJacobiSolverConfig` (per
solver), not where the experiment hook currently sits.
**Part 10: the initial sampling, and integrating the density (2026-08-28).**
Two owner requests. **(a) The sampling is exact and the mass is not.** The
as-sampled fluid is uniform to float precision -- std 0.000000, the same value
in every wall-depth bin -- at `rho = 1.001206`, because `sample/regular.py`
uses the nominal `rho0*dx^d` and never corrects for the kernel's discrete
normalisation. It shows up as a 1.2e-3 error on step 1 and a global factor
removes it exactly (to 7.5e-6). **There is no startup shock**: step 0 is
exactly uniform and the transient is the noise field spinning up. Removing the
bias is a 30% win on the periodic case and a **35% loss** on the bounded one --
the same setpoint knob Part 8 item 1 swept, in the direction it did not cover:
the un-normalised lattice sits *above* `rho0`, which biases the source negative,
which is the shifting solve's de-clumping drive. **(b) `integrateRho` is now
real** (`DensityEvolution`, default `summation` = unchanged). Pure `continuity`
fails everywhere but `tgv`, and the reason is measurable rather than arguable:
the carried density reports 7.1e-3 while the truth is 4.3e-2, because
`drho/dt = -rho div v` cannot represent particle *rearrangement* at zero
divergence -- and that is not the position shift's doing either (dropping it
via `inStepVelocity` leaves the drift at 3.5e-2). One real bug was found on that
path and fixed: `drhodt` was evaluated on the *pre-projection* velocity, so the
integrated density re-accumulated exactly the divergence the solve had just
removed -- worth **1800x** on the divergence-free residual (2.6e+2 -> 1.5e-1).
The owner's proposed split is confirmed by measurement: **`hybrid` -- integrate
for the divergence-free path, re-sum for the shift -- matches `summation`
exactly where support is complete** (periodic `|rho-1|` 1.917e-3 against
1.926e-3, `tgv` 1.588e-4 against 1.608e-4 with identical energy decay and 21%
less wall time) while its carried density drifts 3.3e-2, because the
divergence-free solve genuinely only needs `div v`. It still dies at 286 steps
at an mDBC wall; the leading hypothesis is that `computeMdbcDensity`
extrapolates the boundary rows from the drifted field.

**Part 11: the consistent rigid-fluid coupling paper (2026-08-28).** Bender,
Westhofen & Jeske 2023 arrived and **it is the derivation Part 9 was missing**:
their constraint-based DFSPH defines the density constraint for fluid particles
only, so `dC_i/dx_k = 0` for a static boundary, and their Eqs. 32 and 34 *are*
`BoundaryOperatorTerms.staticBoundary` term for term -- which Part 9 had
reached from SPlisHSPlasH's source without the theory. Their Eq. 33 (no
boundary pressure value anywhere) is an identity for this codebase's symmetric
gradient whenever `p_b = 0`, which the default already arranges. What is new is
the boundary *state*: the paper treats boundary particles as "static fluid
particles" at `rho_k = rho0`, where this codebase feeds them an
mDBC-extrapolated density that reaches 1.3+ in a compressed band -- and that
value reaches every sum in the solve through the apparent volume `m_j/rho_j`.
Shipped as `BoundaryPressureMode.consistent` (opt-in): on the bounded case at
the published CFL it is **6.2x better than the shipped configuration**
(`|rho-1|` 2.86e-2 against 1.78e-1) and 5% better than the operator terms
alone. The sharper result is at the other end of the table: **`mdbcMlsPressure`
-- the MLS extrapolation the paper is written against, and which Part 2 called
this codebase's most accurate mode -- is the worst configuration measured**
(1.86e-1, worse than the baseline), correcting Part 2, which measured it at 3x
the published CFL out to t=1.5. The paper's Akinci volume correction `m~_k`
(measured mean 1.10, max 1.46 on this five-layer band) splits: applied inside
the operator it is the best row in the table (2.38e-2), applied as the
particles' actual mass -- the faithful reading, where Eq. 14's density sum sees
it -- it **diverges in nine steps**, because Akinci's correction assumes a
one-layer sampling and this codebase's band already contains the volume it adds.
`consistent` is entangled with `cflFactor` the same way everything else in
Parts 8-9 is, which now makes **four** independent changes that are better at
the published CFL and worse at 3x it. A first explanation of
that split (mDBC boundary rows are static in position but not in velocity) was
**tested by the owner's follow-up question and falsified**: setting the
boundary velocity to the rigid body's, as DFSPH does, delays the divergence
from step 283 to 482 but does not prevent it, and neither the Jacobi stability
window (`rho(D^-1 A)` 6.3777 vs 6.3782) nor the iteration budget (32 -> 96 ->
192 delays, then *reverses*) accounts for it either. What is measured is a
weaker contraction: each divergence-free solve removes ~20% of its residual
under `staticBoundary` against ~50% under `full`, and the leftover accumulates
for ~250 steps before detonating. Applied to *both* solves the same change
*improves* both solvers' final residuals (5x and 1.7x), so the harm is a
property of the mismatched half-state. That follow-up also found two live
defects in the mDBC slip conditions themselves — the normal component is
projected out where the published form (and `freeSlip`'s own comment) reflects
it, and `noSlip`'s moving-wall term reads a ghost-row velocity that nothing
ever writes — neither of which, measured, is worth fixing blind.

---

## Next session: start here

Part 4 is closed and Part 5 is diagnosed but unfixed (both are written up at
the bottom of this document). Nothing below is blocking; pick by preference.

0. ~~A literature session on DFSPH and its derivatives~~ — **done, see
   Part 7.** It reordered everything below it; the ranked list of what to do
   next now lives at the end of Part 7 ("What this changes, in order").
   Shortest version: try the PS solver's **tolerance** (5e-4 against a
   measured 2.7e-3 structural bias) before anything else.
1. ~~Drive the velocity correction from the attainable part of the source
   only.~~ **Superseded by Part 7 Q1.** This was a fix for the velocity
   formulation, which is the one Cornelis et al.'s VD+PS was written to
   replace — and the residual it targets is handled in the literature by a
   convergence tolerance set above the structural floor, not by projecting
   the mean out. The original text is preserved in git history if the
   tolerance experiment fails to explain the wind-up.
2. **Optionally, the wall-aware `dt` constraint** (Part 5). Still valid,
   still ~2.4x throughput, and now largely superseded by the above — worth
   landing only if the near-wall scoping does not pan out. Part 5 lists the
   one measurement that would sharpen its form (a sweep at a different
   `n_h`, to separate "half a spacing" from "an eighth of h").
3. **The Krylov path never got the fix.** `solveIncompressible` returns
   through `solvePressureKrylov` with `gauge='nonnegative'` before reaching
   the relaxed-Jacobi loop, so `ShiftPressureGauge` does not reach it, and
   its post-hoc clamp has exactly the "floor, not a gauge" character that
   was just fixed on the Jacobi path. Its intra-solve iterate has still
   never been instrumented.
4. **Part 2's wall-adjacent density gap** and **Part 3's
   `rotatingSquarePatch` corner loss** are both still open, both untouched.

**Tooling** (all in `scripts/`, all confirmed working, none require source
edits to use):
- `scripts/probe_shiftPressureGauge.py` — end-to-end A/B of the two gauges
  through the *real* solver, on either `kolmogorovIncompressible` or
  `randomFlowIncompressible` (`--extra=--bounded`). Start here to re-check
  anything about the fix.
- `scripts/probe_incompressibleGaugeDrift.py` — the standalone
  reimplementation of `solveIncompressible`'s loop, where the gauges were
  prototyped. Toggles: `--gauge {clamp,center,center-clamp,minshift,quantile,none}`,
  `--project-source`, `--null-test` (applies the operator to a constant and
  a random field, to measure how near-null the constant mode is),
  `--no-clamp`, `--maxIters`, `--jitter`.
- `scripts/probe_boundedIncompressibleBlowup.py` — Part 5's tool: watches the
  bounded case blow up step by step, reporting where each step's worst
  particle is relative to the wall, how many have crossed it, and a
  wall-depth density profile from just before the explosion. Knobs for the
  things that turned out not to matter, so they stay cheap to re-check:
  `--mode`, `--noPenShift`, `--shiftCap`, `--cflFactor`, and `--case
  randomFlow` for the deltaSPH control.
- `scripts/probe_tgvShiftGauge.py` — TGV under both gauges, graded against
  the analytic kinetic-energy decay rate.
- `scripts/probe_densityBiasVsDisorder.py` — the no-dynamics demonstration
  that the density bias is structural (jitter sweep against `mean(rho-1)`).
- `scripts/probe_densitySign.py` — the original signed-vs-unsigned bulk bias
  check on the running case.
- `scripts/probe_pressureGaugeDrift.py` — the VD (divergence-free) solver's
  gauge; confirms it is not where the problem is.

---

## Background / current-state facts

- mDBC **density** extrapolation is already unconditionally wired into
  `dfsph_step` (`src/warpSPH/schemes/dfsph.py`, step 3) via
  `computeMdbcDensity`. Boundary-particle (`kind==1`) contributions to the
  integration update are already zeroed at the end of the step via
  `nonFluidMask` (applied to `dxdt`/`dvdt`/`drhodt`) — the "one-way coupling"
  skeleton exists structurally, but is never exercised today because no
  incompressible case currently samples boundary particles.
- ~~The pressure solvers ... have no `kind` masking at all~~ — **fixed as
  part of this session's Part 2 work**; see the Part 2 findings below.
- Known/documented leads for the Kolmogorov case already on file:
  - Relaxed-Jacobi `ω` stability window (`ω<0.355`; default `0.5` diverges) —
    see `scripts/probe_relaxedJacobiOmega.py`.
  - The velocity correction in `IncompressibleSystem.finalize`
    (`src/warpSPH/systems/incompressible.py`) is currently commented out —
    see `scripts/probe_kolmogorovIncompressibleVelCorrection.py`.
  - Divergence observed around step ~720 at `nx=128` — see
    `scripts/probe_kolmogorovIncompressible.py` and
    `scripts/probe_kolmogorovAdjacencyRebuild.py`,
    `scripts/probe_kolmogorovContinuation.py`,
    `scripts/probe_kolmogorovSpinup.py`.
- `src/warpSPH/cases/randomFlow.py` currently only wires up `deltaSPH`
  (weakly compressible), with a `--bounded` flag adding a `boundaryRegion`
  (see `src/warpSPH/cases/weaklyCompressible.py`). Porting to the
  incompressible scheme needs an incompressible-flavored
  `buildSystem`/`configureScheme` that also supports the bounded region.

---

## Part 1 — Kolmogorov incompressible stability investigation

1. **Reproduce baseline** with `scripts/probe_kolmogorovIncompressible.py` at
   the documented `nx=128` divergence point (~step 720), capturing density
   min/max/std, solver iteration counts, and `ω` per step as a reference
   trace.
2. **Vary the solver knobs** already exposed, cross-referencing prior probe
   findings:
   - Relaxed-Jacobi `ω` (0.3 vs 0.5 vs `optimal` mode — see
     `probe_relaxedJacobiOmega.py`).
   - Krylov solvers (CG/BiCGStab/GMRES/MINRES, incl. fp64 variant) via the
     `PressureSolverType` switch in `solveDivergenceFree` — MINRES performed
     best for TGV; check whether that holds for the forced/shear case.
   - Toggle the commented-out velocity correction in
     `IncompressibleSystem.finalize` using the monkeypatch pattern from
     `probe_kolmogorovIncompressibleVelCorrection.py`.
   - `integrateRho` on/off (continuity-equation density vs. recomputed
     density each step) in `dfsph_step`.
   - Adjacency rebuild frequency (`probe_kolmogorovAdjacencyRebuild.py`).
3. **Root-cause the density drift**: correlate density-deviation growth with
   (a) pressure-solver non-convergence / iteration blowup, (b) particle
   disorder/clustering (implicit shifting's velocity correction disabled),
   (c) the `ω` instability window being violated transiently as velocities
   grow post forcing-ramp-up.
4. **Land a fix**, most likely candidates ordered by expected effort/impact:
   - Enable the velocity correction, if it helps and is cheap.
   - Clamp/adapt `ω`, or default to `optimal` mode for this case.
   - Increase `maxIterations` / tighten `tolerance` when density error
     crosses a threshold.
5. Record findings (before/after metrics) — appended to this file's Part 1
   status, not a separate report.

### Part 1 findings (2026-08-26)

**The fix was already landed** in commit `122d326` ("fix DFSPH's implicit
particle-shift dropping its own velocity correction") one commit before this
plan was written — that commit *is* step 4's first candidate ("enable the
velocity correction, if it helps and is cheap"), found and verified by the
same investigation this plan was proposing. Summary of what was found and
confirmed this session:

- **Root cause**: `IncompressibleSystem.finalize`'s implicit particle-shift
  (a second, constant-density pressure solve applied as a position
  correction `dx`) computed its own kinematic velocity correction
  (`proj_vel = grad(V)·dx`, the standard Taylor correction needed to keep a
  shifted particle's velocity consistent with its new position) every step,
  then discarded it — `self.state.velocities -= proj_vel` was commented out.
  Particles were shifted without their carried velocity being corrected,
  which is consistent with the step-720 divergence being driven by particle
  disorder (the failure mode step 3's diagnosis pointed at), not a viscous
  or pressure-solver-convergence mechanism.
- **Verification (this session, reproducing/extending commit 122d326's own
  before/after)**: `scripts/probe_kolmogorovIncompressible.py --nx 128`
  (`xi=1.0`, `k=4`, physical viscosity from `alpha=0.01`, the case's own
  operating point) now runs the full 1000-step budget with no divergence
  (previously NaN at step 720); extended to 1600 steps (t≈13.6s, well past
  the registered case's `tLimit=10.0`) it stays bounded throughout, with
  transient `rhoMin` excursions down to ~0.69–0.89 (single/few-particle
  outliers — `rhoStd` stays ~1e-3–4e-3 the rest of the time) that recover
  within the step, never cascading to NaN.
- **Secondary check**: `relaxationMode=JacobiRelaxationMode.optimal` (the
  per-step exact residual-minimizer, opt-in, no `ω` stability window) was
  also run 1600 steps on the same case — likewise stable to t≈14.2s, with
  comparable transient `rhoMin` dips. No clear advantage over the default
  fixed `ω=0.3` at this operating point, so **not** changing the scheme
  default; noted here as a validated fallback if a future case needs it.
  `solveDivergenceFree`'s relaxed-Jacobi loop hits its `maxIterations=32`
  cap every single step in both configurations (never converges early to
  `tolerance=2.5e-3`) — stable regardless, but a candidate to revisit if a
  future case needs tighter pressure convergence than this one does.
- **Not pursued further**: Krylov solver comparison (CG/BiCGStab/GMRES/
  MINRES) and adjacency-rebuild-frequency sweep — the velocity-correction
  fix alone resolved the divergence with margin (1600 vs. the case's
  own-1000-step validation budget, past `tLimit`), so these knobs were not
  needed. Left here as still-open follow-ups if a *different* future
  incompressible case hits a similar disorder-driven divergence that the
  velocity correction alone doesn't fix.

1. **Config**: add a `BoundaryPressureMode` enum (`plain` / `mdbcDensity` /
   `mdbcMlsPressure`) to the incompressible scheme config, mirroring the
   style of `PressureSolverType`. ✅ **Done.**
2. **Solver masking** (needed for all three modes): in
   `solveDivergenceFree` and `solveIncompressible`, exclude `kind==1`
   (and `kind==2` ghost, if present) particles from the pressure unknowns —
   zero their contribution to the gauge mean, zero the `alpha`/diagonal
   update for them, and zero `a_p` for them post-solve, matching the
   one-way-coupling contract already enforced downstream by `nonFluidMask`
   in `dfsph_step`. ✅ **Done**, including the Krylov path.
3. **Option a (plain)**: boundary particles get normal SPH density (skip
   mDBC), participate as regular density/pressure sources, but their
   pressure is forced to `0` per step 2. ✅ **Done.**
4. **Option b (mdbc density)**: keep the existing `computeMdbcDensity` call
   (already unconditional today), same solver masking as (a). ✅ **Done**
   (this is the new default, matching the historical always-on behavior).
5. **Option c (mdbc + MLS pressure / Liu-Liu projection)**: after each
   pressure solve, project/interpolate the fluid pressure field onto
   boundary particles via the existing Liu-Liu MLS machinery
   (`src/warpSPH/modules/liu/interp.py`), following the same pattern as
   `computeMdbcDensity`, so boundary particles carry a physically
   consistent pressure for force computation on fluid neighbors, while still
   contributing `a_p = 0` to their own (non-existent) motion. ✅ **Done**
   (`computeMdbcPressure`) — see the caveat in the findings below.
6. **Wire into `src/warpSPH/schemes/dfsph.py`**: make the
   `computeMdbcDensity` call conditional on mode, and inject the MLS
   pressure projection call for mode `mdbcMlsPressure` right after
   `solveDivergenceFree`. ✅ **Done.**
7. **Port 06-randomFlow to incompressible**. ✅ **Done** —
   `src/warpSPH/cases/randomFlowIncompressible.py`, registered in
   `cases/__init__.py`'s `CASE_MODULES`. See findings below for why this is
   a new case file, not a `--scheme` flag on the existing one.
8. **Validate**: run bounded random flow under all three boundary modes,
   check density near walls, momentum leakage, and stability vs. the
   periodic baseline; compare against wcsph/deltaSPH bounded behavior as a
   sanity reference. **Partially done** — see findings below for what was
   and wasn't checked.

### Part 2 findings / implementation notes (2026-08-26)

- **Where the enum lives**: `BoundaryPressureMode` in
  `src/warpSPH/configurations/moduleConfigurations/solver.py`, as a new
  field `IncompressibleSolverConfig.boundaryPressureMode` (default
  `mdbcDensity`, preserving the scheme's historical always-on
  `computeMdbcDensity` behavior). Round-tripped through
  `incompressibleConfigToDict`/`dictToIncompressibleSPHConfig`
  (`configurations/incompressible.py`).
- **Masking mechanics**: implemented at the Python level (not inside the
  warp kernels) as a `fluidMask = particles.kinds == 0` applied in four
  places: `solveDivergenceFree`'s fixed-`ω` loop, its `optimal`-mode sibling
  `_solveDivergenceFreeOptimal`, `solveIncompressible`'s fixed-`ω` loop, and
  `solvePressureKrylov` (wrapping the matvec/matvecT so every Krylov method
  sees and returns 0 at boundary rows, decoupling them from the iteration
  entirely). In every case: the trial pressure step is zeroed at boundary
  rows *before* it is fed through `computePressureAccelIISPH` (not just
  masked on the output), the gauge-fixing mean/error is computed over fluid
  rows only, and `a_p` is re-zeroed at boundary rows post-solve. For
  `optimal` mode, `omega_k`'s dot products are also restricted to fluid rows
  — needed for its per-step-optimal derivation to stay correct once
  boundary rows are frozen, not just for the final answer.
  Verified as an exact no-op on every currently-exercised code path
  (no incompressible case samples `kind==1`/`kind==2` particles yet, so
  `fluidMask` is all-`True` everywhere today): `pytest tests/test_physics.py
  tests/test_incompressibleKrylov.py` (82 tests, all 5 `PressureSolverType`
  variants incl. `krylovFp64`) passes unchanged, and
  `scripts/probe_kolmogorovIncompressible.py` still runs clean.
- **`computeMdbcPressure`** (`src/warpSPH/modules/mdbc/pressure2025.py`):
  structurally mirrors `computeMdbcDensity` (same Liu-Liu MLS fit evaluated
  at `kind==2` ghost points, same `ghostIndices` indirection back to the
  owning `kind==1` particle, same low-neighbor-count Shepard/zero fallback
  ladder) but deliberately simpler — no hydrostatic/gravity correction,
  since pressure (unlike density) has no rest value to fall back on.
  **Caveat, not yet resolved**: because `solveDivergenceFree` holds boundary
  pressure fixed for the *current* step's solve, `computeMdbcPressure`'s
  projection (run right after) only takes effect for the *next* step's
  neighbor sums — a one-step lag, same as how `computeMdbcDensity` already
  lags fluid density by construction. This is believed correct/analogous
  but has not been validated against a real bounded flow (no case exercises
  it yet — that's step 7/8).
### Part 2, steps 7–8 findings (2026-08-26)

- **New case file, not a `--scheme` flag — resolved.** `--scheme` is
  already a generic `CaseSpec` override honoured by every case
  (`runner.py`: `_resolveScheme(spec.scheme or case.scheme)`;
  `cli.py`: `defaults.setdefault('scheme', case.scheme)`) — so
  `randomFlow.py`'s own docstring claim that its cases differ "only in...
  which scheme integrates it (`--scheme`)" already looked true at the CLI
  level. It isn't, for two independent reasons found while building the
  port (both now documented in `randomFlowIncompressible.py`'s module
  docstring):
  - `Case.timestep` is one hook shared by whichever scheme a case ends up
    running under. `randomFlow` leaves it unset, which is correct for its
    `deltaSPH` default (falls through to `modules.timestep.computeTimestep`'s
    `WeaklyCompressibleSystem` branch) but wrong for `divergenceFree`: an
    `IncompressibleSystem` isn't a `WeaklyCompressibleSystem`, so the
    dispatcher falls through to the *compressible* branch instead, which
    reads `system.state.internalEnergies` — an attribute `IncompressibleState`
    doesn't have. `randomFlow.py --scheme divergenceFree` with the case's own
    `adaptiveDt=True` default crashes with an `AttributeError` on the first
    step. This is exactly the failure `kolmogorovIncompressible.py` already
    worked around with its own CFL-only `timestep` hook — reused verbatim
    for the new case (the formula is generic: advective + viscous CFL, no
    acoustic term, nothing Kolmogorov-specific in it).
  - `randomFlow.initialConditions` ends with `setupTimestep`, which (per
    `setupWeaklyCompressibleTimestep`) sets a **fixed** `dt = targetDt` for
    the whole run and derives a sound speed from it that DFSPH never reads.
    Mechanically harmless (no crash), but not adaptive, and it unconditionally
    warns about a synthetic "Mach number" that means nothing for an
    incompressible solver.

  Everything else in `randomFlow.py` turned out to already be scheme-agnostic
  and is reused as-is by the new case (`buildSystem`, `noiseVelocities`,
  `BOUNDED_BAND`, and — via `weaklyCompressible.py` —
  `configureWeaklyCompressible`, `paramExtraData`,
  `weaklyCompressibleDiagnostics`, `OBSTACLE_PARAMS`): confirmed by reading
  `initializers/weaklyCompressible.py`'s `initializeSimulation`, which
  branches generically on `SimulationState is IncompressibleState` right
  alongside its `WeaklyCompressibleState` branch, and
  `rigidBody/ghostParticles.py`'s `addBoundaryGhostParticles`, which builds
  the `kind==2` mDBC ghost layer via `type(particleState)` — i.e. it was
  already scheme-agnostic and already exercised by every existing WCSPH
  `--bounded` case (`randomFlow --bounded`, `lidDrivenCavity`,
  `movingObstacle`, `drivenSquare`). **Note**: no case of *any* scheme
  actually consumes those ghost particles via mDBC before this one — mDBC
  (`computeMdbcDensity`/`computeMdbcPressure`) is only wired into
  `schemes/dfsph.py`, so WCSPH's bounded cases sample the same `kind==2`
  layer but never read it.
- **Smoke-tested** (`python -m warpSPH.cases.randomFlowIncompressible`,
  `nx=24`, CPU): periodic mode (10–20 steps) and `--bounded` mode (20 steps,
  1736 particles incl. boundary + ghost) both run to completion, no NaNs.
  `--bounded` under all three `boundaryPressureMode` values (`plain`,
  `mdbcDensity`, `mdbcMlsPressure`, 8 steps each) also complete cleanly —
  this is the first time any of the three has run on live boundary data
  rather than as a no-op. `tests/test_runner.py` (case-registry smoke tests)
  and `tests/test_physics.py`/`tests/test_incompressibleKrylov.py` (82 tests)
  still pass unchanged.
- **Resolution sweep + deltaSPH baseline (this session, nx=24 `--bounded`,
  `mdbcDensity`, time-matched to t≈1.37)**:
  - nx=24 → nx=64 (both 20 steps, so different final `t` since dt is
    velocity/support-adaptive): `maxVelocity`'s transient spike shrank
    sharply (2.22 → 1.05), consistent with the coarse grid's noise-field
    sampling near the wall being the main driver of that spike rather than a
    masking bug. Density excursion did **not** shrink with resolution
    (min/max ≈ 0.78/1.33 at nx=24 vs. 0.86/1.17 at nx=64 — better, but not by
    the margin the velocity spike improved by).
  - **DFSPH vs. deltaSPH, both nx=24, `--bounded`, run to the same t≈1.37**
    (5480 steps at deltaSPH's fixed `dt=0.00025` vs. DFSPH's 20 adaptive
    steps — step-count-matched comparisons are not meaningful here, since
    DFSPH's per-step `dt` is ~400x larger): deltaSPH's density stays in
    [0.998, 1.003] throughout, including through its own transient
    `maxVelocity` spike (1.0 → 3.63, i.e. a *larger* spike than DFSPH's
    2.22) — so a coarse-grid boundary velocity spike alone does not explain
    DFSPH's much looser density band ([0.78, 1.33] over the same run). This
    is suggestive that DFSPH's mDBC path is genuinely looser at controlling
    boundary-adjacent density than deltaSPH's EOS-driven boundary handling
    at this resolution, not just coarse-sampling noise — but it is one
    data point at one (very coarse) resolution, not a diagnosis. **Not
    pursued further this session.**
- **Still open** (the rest of step 8's brief, now narrower thanks to the
  above): root-cause the DFSPH/deltaSPH density-band gap (is it inherent to
  mDBC's density extrapolation being one-step-lagged, a resolution effect
  that needs nx=128 to judge fairly, or a real bug in the pressure-solver
  masking from steps 2–5?); compare `mdbcDensity` vs `mdbcMlsPressure` vs
  `plain` against *each other* at matched physical time (only compared at
  matched step-count=8 so far, which the paragraph above shows is not
  meaningful once `dt` differs run-to-run); and a production-resolution
  (`nx=128`) run to confirm behavior holds past this smoke-test scale. Left
  for a follow-up session. **Resolved/superseded by the nx=128 follow-up
  below**, done in this same session.

### Part 2, nx=128 production-resolution follow-up (2026-08-26)

Script: `scripts/probe_randomFlowIncompressibleBoundaryModes.py` (new), runs
`randomFlowIncompressibleCase`/`randomFlowCase` in-process via
`runner.run()`, `--bounded`, matched physical time, `store=False`/`plot=False`
so only the diagnostics trajectory is collected.

- **The density-band gap does not close with resolution — closes the first
  open question.** nx=128, `mdbcDensity`, t≈1.5: ρ∈[0.661, 1.475], *looser*
  than the nx=24 smoke test's [0.78, 1.33] over a comparable run, while
  deltaSPH stays at [0.999, 1.003] on the identical setup and t. This rules
  out "coarse-resolution artifact" outright — production resolution does not
  narrow the gap, it's if anything worse. Given the finding below (`plain`
  tracks `mdbcDensity` almost exactly), this looks like it's inherent to
  DFSPH's own bulk pressure-projection behavior near a wall on this problem,
  not specific to the mDBC machinery.
- **Found and fixed a real masking bug — answers the second open question.**
  At nx=128, `mdbcDensity` and `mdbcMlsPressure` were not just similar but
  **bit-identical at every recorded step** (float32 precision, min/max
  density traced step-by-step over 17 steps) before this session's fix. Root
  cause: the `kind==1`/`kind==2` boundary-row masking in all four
  pressure-solve code paths — `divergenceFree.py`'s fixed-ω loop and its
  `_solveDivergenceFreeOptimal` sibling, `incompressible.py`'s fixed-ω loop,
  and `krylov.py`'s `solvePressureKrylov` — froze boundary-row pressure at
  **literal `torch.zeros_like(...)`**, not at whatever
  `currentState.pressures` already held there. Since `computeMdbcPressure`
  writes its projected value to `currentState.pressures` *after*
  `solveDivergenceFree` returns (a deliberate one-step lag, per
  `computeMdbcPressure`'s own docstring), that projected value was being
  discarded at the very top of the *next* step's solve, before
  `computePressureAccelIISPH` ever saw it — so `mdbcMlsPressure`'s entire
  stated purpose ("boundary particles carry a physically consistent pressure
  for force computation on fluid neighbors") was silently a no-op. This also
  means the general density-band gap above is *not* explained by this bug:
  `plain` (which never calls `computeMdbcPressure` at all) tracks
  `mdbcDensity` too closely for a broken Option c to be the main driver of
  that gap.
  - **Fix landed** (`divergenceFree.py`, `incompressible.py`, `krylov.py`):
    the two relaxed-Jacobi solvers now do
    `torch.where(fluidMask, pressureB, boundaryPressure)` (`boundaryPressure
    = particles.pressures.clone()`, captured once per solve) instead of
    `torch.zeros_like(...)` at every re-masking point — since
    `computePressureAccelIISPH`/`computePressureShiftIISPH` sum over all
    neighbors regardless of `kind`, baking the frozen value directly into the
    pressure field they read is sufficient, no separate RHS term needed. The
    Krylov path is structurally different (a generic matrix-free solver
    needs a fixed operator on a fixed unknown subspace, so the boundary value
    can't just ride along inside the iterate) and got the standard
    Dirichlet-lifting treatment instead: `b = sourceTerm - A(boundaryOnly
    field)`, solve the homogeneous fluid-only subproblem as before, then
    re-pin the boundary rows of the result to `boundaryPressure` (was `0`)
    instead of `0` at the end.
  - **Verified as an exact no-op everywhere it was previously verified inert**:
    `pytest tests/test_physics.py tests/test_incompressibleKrylov.py` (82
    tests, all 5 `PressureSolverType` variants) passes unchanged — expected,
    since `fluidMask` is all-`True` on every case those tests cover, so every
    new `torch.where(fluidMask, X, boundaryPressure)` branch never takes the
    `boundaryPressure` arm there.
  - **Verified as no longer a no-op on the one case that does exercise it**:
    re-ran the nx=128 step-by-step trace above post-fix —
    `mdbcMlsPressure` now diverges from `mdbcDensity` starting at step 1
    (previously identical through at least step 16).
- **New finding, found by the fix**: `mdbcMlsPressure`, now actually live,
  is **numerically unstable** on this case — nx=128 `--bounded` NaNs by step
  7 (density spikes to 7.26, velocity to `inf`, then NaN), where `plain`/
  `mdbcDensity` stay bounded over the same steps. `plain`/`mdbcDensity`
  trajectories are byte-identical to their pre-fix values (confirms the fix
  is properly isolated to the one mode that actually writes a nonzero
  boundary pressure). Suspected cause, not yet confirmed: `computeMdbcPressure`
  (`modules/mdbc/pressure2025.py`) does a first-order MLS extrapolation of
  pressure to the wall with **no magnitude clamp or rest-value fallback** —
  its own docstring already flags this design choice ("pressure ... has no
  reference value of its own to fall back on"), unlike `computeMdbcDensity`,
  which anchors low-neighbor-count/noisy fits back toward `rho0`. A noisy or
  large `p_interp_grad` near a corner or a low-neighbor-count boundary point
  could extrapolate to an unbounded pressure, which (now that it actually
  feeds `computePressureAccelIISPH`) becomes a runaway force on nearby fluid
  particles. **Not investigated further this session** — this is a genuine
  solver-stability question (does Option c need a magnitude clamp, a
  relaxation/damping factor blending with the previous step's value, or a
  different fallback ladder near low-neighbor boundary points?) rather than
  a wiring bug, and needs its own targeted probe (in the style of Part 1's
  `probe_*.py` scripts, isolating `computeMdbcPressure`'s output particle by
  particle right before the divergence) before attempting a fix.

### Part 2, `mdbcMlsPressure` instability — root cause and fix (2026-08-26)

Script: `scripts/probe_mdbcMlsPressureInstability.py` (new). Drives
`randomFlowIncompressibleCase` manually (not through `runner.run()`, which
only exposes the diagnostics dict) so it can re-run `computeMdbcPressure`'s
own internals (`interpolateLiuLiu`'s `numNeighbors`/`A_g`/`b`, the projected
`p_proj` and its Taylor gradient term) per boundary particle on the exact
steps leading up to the NaN.

- **Root cause: an undamped feedback loop, not a low-neighbor-count
  artifact.** Traced boundary pressure step by step: `[-1.47,1.33]` →
  `[-2.48,2.59]` → `[-3.95,5.56]` → `[-4.08,10.85]` → NaN — magnitude roughly
  doubling every step. At step 5's worst offender, `numNeighbors=22` (well
  past the `threshold=9` fallback cutoff, i.e. a well-sampled point using the
  full MLS projection, not the Shepard/zero fallback) with `|grad p|=153`
  producing `p_proj=11.25`. The mechanism: a larger boundary pressure pushes
  nearby fluid particles harder (once the masking-bug fix above let it
  through) → a steeper local fluid pressure field next step → `computeMdbcPressure`'s
  one-step-lagged linear extrapolation projects an even larger boundary
  pressure from that steeper gradient → repeat. No stabilizing term existed
  anywhere in this loop; `computeMdbcDensity`'s analogous rest-density anchor
  (mentioned in `computeMdbcPressure`'s own docstring as something pressure
  structurally lacks) has no equivalent here.
- **Fix landed**: a new `IncompressibleSolverConfig.mdbcPressureRelaxation`
  field (default `0.3`, matching `divergenceFreeSolver`'s own default
  `relaxationFactor` — the same fix pattern this scheme already uses for its
  own Jacobi iterations), applied in `computeMdbcPressure`
  (`modules/mdbc/pressure2025.py`) as `new = old + factor*(projected - old)`
  before merging back onto `currentState.pressures`, round-tripped through
  `incompressibleConfigToDict`/`dictToIncompressibleSPHConfig`
  (`configurations/incompressible.py`).
- **Verified**: `pytest tests/test_physics.py tests/test_incompressibleKrylov.py
  tests/test_runner.py` (100 tests) passes unchanged (boundary particles
  don't exist on those cases, so this new code path is never hit — the
  `oldBoundary`/relaxation blend is a no-op there just like the masking fix
  above). `probe_mdbcMlsPressureInstability.py --nx 128 --nsteps 30`: no
  longer diverges — boundary pressure stays in roughly `[-0.1, 0.33]`
  throughout instead of doubling every step. Re-ran the nx=128
  `probe_randomFlowIncompressibleBoundaryModes.py` sweep at t≈1.5:
  `mdbcMlsPressure` now completes the full run (`diverged=False`, 80 steps)
  with density band **[0.688, 1.226]** — markedly tighter than both `plain`
  (`[0.661, 1.460]`) and `mdbcDensity` (`[0.661, 1.475]`) on the identical
  setup and physical time. `mdbcMlsPressure` is now both the most stable and
  the most accurate of the three modes at this operating point (though all
  three remain far looser than deltaSPH's `[0.999, 1.003]` baseline — see the
  still-open bulk-behavior question above, which this does not touch).
- **`mdbcPressureRelaxation=0.3` was not tuned** — it was chosen to match
  the scheme's existing convention rather than swept. A follow-up could
  check whether a larger factor (faster response, e.g. matching
  `pressureSolver`'s own `0.3`... they're already equal) or a smaller one
  (more damping) trades off stability margin against how quickly the
  boundary pressure tracks a genuinely changing near-wall flow; not needed
  to unblock this session's work since `0.3` already resolved the
  divergence with margin (bounded to at least t≈1.5, well past where the
  undamped version NaN'd at t≈0.1).

### Part 2, bulk density-band gap — wall-distance profile (2026-08-26)

Script: `scripts/probe_dfsphWallDensityProfile.py` (new). Drives both cases
manually to matched t, then bins each fluid particle's `|density-1|` by its
signed distance from the nearest domain wall
(`weaklyCompressible.domainBoundarySdf`, the same SDF the case's own geometry
is built from) to test whether the error is wall-localized or bulk-spread.

- **Short-time check (t≈0.1, nx=64) looked bulk-uniform**: DFSPH's mean error
  was ~4.6e-3 to 6.4e-3 across all 4 depth bins, no wall concentration;
  deltaSPH's was ~1.3e-4 (~35x tighter) and equally flat. This is what
  motivated the "general DFSPH trait, not a wall effect" reading in the
  (now superseded) note below.
- **Production-time check (t≈1.5, nx=128) tells a different, two-part
  story.** deltaSPH stays flat across all 8 bins as before (mean error
  5.5e-4 to 6.3e-4, no depth dependence, always `depth >= 0.0037` — i.e. no
  deltaSPH fluid particle ever crosses the nominal wall). DFSPH does not:
  - **A severe, genuinely wall-localized spike**: the nearest-wall bin
    (`depth` from `-0.162` to `-0.017`, i.e. **outside** the nominal interior
    domain by up to ~10 particle spacings) has mean error **0.236** —
    50-80x worse than every other bin. 318 of ~16,600 fluid particles
    (~2%) fall in this bin. This is `kind==0` fluid particles, not boundary
    particles — meaning a real minority of fluid particles are drifting
    *past* the nominal wall SDF's zero level and into the boundary band by
    production time, which is not something deltaSPH's fluid particles ever
    do on the identical setup (its depth range stays `>= 0.0037`
    throughout). This looks like a no-penetration containment issue specific
    to DFSPH, not (only) a density-accuracy issue.
  - **A smaller but still real bulk gap survives underneath that spike**:
    excluding the wall-adjacent bin, DFSPH's bulk bins (`depth` from `0.417`
    to `0.996`) settle to mean error ~3.7e-3 to 5.4e-3 — still ~7-10x worse
    than deltaSPH's bulk ~5.5e-4 at the same depths, confirming the short-time
    check's "general trait" reading was not wrong, just incomplete: there
    are two distinct effects layered on top of each other, and the
    wall-adjacent one apparently needs time (or accumulated near-wall
    activity) to develop, since it wasn't yet visible in the t≈0.1 check.
  - **Not investigated further this session**: why fluid particles cross
    the nominal wall boundary at all is the more actionable of the two
    findings and the natural next lead — `computeMdbcNoPenShift`
    (`modules/mdbc/wp_nopenshift.py`, a `@wp.kernel`, so any change there
    needs the `gradcheck` skill afterward) is the mechanism that is supposed
    to prevent exactly this, and hasn't been inspected this session. The
    smaller residual bulk gap (~7-10x) is left as the same still-open
    "general DFSPH bulk pressure-projection" question, now with a cleaner
    number attached.

### Part 2, no-penetration shift A/B test (2026-08-26)

Hypothesis floated (project owner): the original DFSPH paper (Bender &
Koschier) has no no-penetration term at all and relies on the pressure
projection alone to prevent penetration — is `computeMdbcNoPenShift` a
crutch that is actually making the wall-crossing problem above worse, not
better? Tested directly rather than argued from the kernel's source (which
is a soft per-particle velocity-damping correction, not a hard containment
constraint — plausible either way from reading it alone).

- **Change**: added `IncompressibleSolverConfig.mdbcNoPenetrationShift: bool`
  (default `True`, preserving historical behavior), gating the
  `computeMdbcNoPenShift`/`dvdt += nopenshift / dt` call in `dfsph_step`
  (`schemes/dfsph.py`). Round-tripped through
  `incompressibleConfigToDict`/`dictToIncompressibleSPHConfig`. Verified as
  an exact no-op on the existing suite (100 tests unchanged, default `True`
  matches prior always-on behavior).
- **A/B result** (`probe_dfsphWallDensityProfile.py --no-pen-shift both`,
  nx=128, `mdbcDensity`, t≈1.5, same seed, otherwise identical run):

  | | shift on (default) | shift off |
  |---|---|---|
  | particles crossing the wall (depth<0) | 441/16384 (2.7%) | 543/16384 (3.3%) |
  | worst penetration depth | 10.4 dx | 16.2 dx |
  | density max | 1.401 | 1.472 |
  | density min | 0.829 | 0.907 |
  | mean error among crossed particles | 0.214 | 0.204 |

  **The hypothesis is not supported by this test.** Turning the shift off
  makes wall-crossing strictly worse on two of three crossing metrics (more
  particles cross, and the worst one crosses nearly twice as deep) and on
  density max; it only improves density min. Net: not a clear win either
  way, and the crossing/depth numbers argue mildly *against* removing it,
  not for it. Left `mdbcNoPenetrationShift` defaulted `True` — the toggle
  itself stays in the config for further experimentation (e.g. combined
  with a fix to the underlying near-wall pressure behavior, rather than as
  a substitute for one), but this session's data doesn't support flipping
  the default or removing the mechanism.
- **Not investigated further this session**: *why* the shift doesn't fully
  contain particles even when active (its threshold conditions vs. DFSPH's
  much larger adaptive `dt` than this mechanism may have been tuned/verified
  against elsewhere) — the A/B test answers "is it net-harmful" (no) but not
  "why is 2.7% of the fluid still crossing with it on."

---

### Part 2, reference-paper comparison — VD+PS velocity-resample sign/axis bug (2026-08-26)

Prompted by comparing this scheme against Cornelis et al., "An Optimized
Source Term Formulation For Incompressible SPH" (2019 TVCJ) — the paper this
scheme's implicit-particle-shift design (`IncompressibleSystem.finalize`,
`systems/incompressible.py`) is modeled on (VD+PS: a divergence-free pressure
solve, then a second density-invariance/DI pressure solve used only to shift
particle positions, with the divergence-free velocity field resampled at the
shifted positions rather than replaced).

- **Stale-comment correction**: `schemes/dfsph.py`'s docstring/comments (and
  the dead code at `dfsph.py:188-217`) read as if the paper's second PPE
  (PS/DI) is simply unimplemented/disabled. It isn't — that code in
  `dfsph.py` is genuinely dead, but the real PS step lives in
  `IncompressibleSystem.finalize` (`systems/incompressible.py:238-276`),
  fully wired and active every step regardless of `shiftProperties.active`.
  The scheme does implement the paper's VD+PS structure end-to-end; a reader
  of `dfsph.py` alone would reasonably conclude otherwise.
- **Real bug found and fixed**: the paper's velocity-resample step (Eq. 17),
  `v(t+Δt) = v'(t+Δt) + ∇v'(t+Δt)·(x(t+Δt) − x**)`, was implemented at
  `systems/incompressible.py:272-275` as:
  ```python
  proj_vel = torch.einsum('nij, ni -> nj', gradVel, dx)
  self.state.velocities -= proj_vel
  ```
  Verified against the actual compiled `warpSPHCore` `GradientScheme.Difference`
  kernel with a synthetic linear velocity field (`v=A·x`, asymmetric `A` so
  transpose vs. sign errors are distinguishable): the kernel matches the
  paper's `(v_j−v_i)` convention with no extra sign, and produces
  `gradVel[n,i,j] ≈ ∂v_i/∂x_j`. Against that layout, the einsum contracts
  `dx` over the wrong axis (computes `Aᵀ·dx`, not `A·dx` — not simply a sign
  flip, a different vector), and the `-=` then applies it with the wrong
  sign on top. Numeric check: code's delta was `[+0.0226,+0.0240]` vs. the
  paper-correct `[-0.0063,+0.0401]` — disagreeing in both direction and
  magnitude. The now-dead `dfsph.py:203-217` code shows the same confused
  pattern (explicitly negates the same operator's output before an
  analogous einsum+add), suggesting this was never checked against the
  paper's derivation, not a one-off typo.
  **Fix landed**: `proj_vel = torch.einsum('nij, nj -> ni', gradVel, dx)`
  and `self.state.velocities += proj_vel`.
- **Verified as a no-op on the existing suite**:
  `pytest tests/test_physics.py tests/test_incompressibleKrylov.py
  tests/test_runner.py` (100 tests) passes unchanged — expected, this term
  lives downstream of everything those tests check bit-for-bit.
- **Checked against the plan's own two open bulk-gap/wall-crossing metrics —
  fix does not move either one.** `probe_randomFlowIncompressibleBoundaryModes.py
  --nx 128 --tlimit 1.5 --modes mdbcDensity`: ρ∈[0.659,1.465] post-fix vs.
  the previously-recorded [0.661,1.475] pre-fix — within noise.
  `probe_dfsphWallDensityProfile.py --nx 128 --tlimit 1.5 --no-pen-shift on`:
  459/16384 (2.8%) fluid particles crossed the wall post-fix vs. 441/16384
  (2.7%) pre-fix, worst-bin mean error 0.231 vs. 0.236, bulk (far-from-wall)
  bins still ~5.5e-3-6.6e-3 mean error, same order as the pre-fix ~3.7e-3-
  5.4e-3. **This is a real, verified bug and the fix is correct and stays
  landed, but it is not the explanation for either open item below** — both
  numbers are unchanged within run-to-run noise.
  Plausible reason: `dx = dt**2 * dvdt_incomp` is second-order in `dt`, so
  `proj_vel` is a small correction relative to the dominant bulk
  pressure-projection and mDBC boundary effects already responsible for
  those two gaps; the paper's own diagnostic for VD+PS's benefit was never a
  density snapshot either — it was the shear-wave sinus-amplitude decay
  over many seconds (Fig. 3), specifically because clustering/diffusion
  artifacts from a wrong resample accumulate gradually in the *velocity*
  field, not in an instantaneous density band. A `kolmogorovIncompressible`-
  or shear-wave-style long-horizon velocity-quality check (rather than
  another density-snapshot probe) is the more paper-faithful way to see this
  fix's actual effect, if it's worth pursuing further — not done this
  session.

## Open questions / decisions needed

- Whether Part 1's fix should be a case-specific override (e.g. Kolmogorov
  case sets `ω`/tolerance explicitly) or a general default change to the
  incompressible scheme config.
- ~~Part 2, step 8: DFSPH's `--bounded` density band is markedly looser than
  deltaSPH's ... resolution artifact ... or a real gap in the steps 2–5
  pressure-solver masking?~~ **Resolved/refined (2026-08-26)**: not a
  resolution artifact, not primarily a `BoundaryPressureMode` effect (see the
  nx=128 follow-up above), and — per the wall-distance profile just above —
  **not one effect but two**: (1) a small minority (~2%) of fluid particles
  drifting past the nominal wall SDF boundary by production time (t≈1.5),
  with a severe local density error (mean 0.236) right where they do, and
  (2) a smaller ~7-10x bulk gap vs. deltaSPH that holds at every depth
  including far from any wall. (1) is the more actionable open item — an A/B
  test (see "no-penetration shift A/B test" above) ruled out "just remove
  `computeMdbcNoPenShift`" as a fix (removing it made crossing depth/count
  and density max modestly worse, not better), but did *not* explain why
  ~2.7% of fluid particles still cross the wall with it active — that's the
  narrower question still open, and would need reading
  `computeMdbcNoPenShift`'s threshold conditions against DFSPH's actual
  adaptive `dt`/velocity scale on this case, not just toggling it on/off.
  (2) would still benefit from the `kolmogorovIncompressible`-vs-`--bounded`
  unbounded/bounded comparison this bullet originally proposed, to check
  whether it's present even with no wall at all.
- ~~Is `BoundaryPressureMode.mdbcMlsPressure` (Option c) salvageable as
  currently formulated, or does `computeMdbcPressure`'s pressure
  extrapolation need a stabilizing change?~~ **Resolved (2026-08-26)**, see
  "Part 2, `mdbcMlsPressure` instability — root cause and fix" above: an
  under-relaxation factor (`mdbcPressureRelaxation`, default `0.3`) damps the
  projection/fluid-solve feedback loop that was causing the divergence.
  `mdbcMlsPressure` now runs stably at nx=128 to at least t≈1.5 and gives the
  tightest density band of the three modes. The relaxation factor itself
  was not tuned/swept — see that section's last bullet for a possible
  follow-up, not blocking.
- **New (2026-08-26), not yet acted on**: is `IntegrationSchemeType` actually
  constrained to match the PPE derivations' assumption anywhere, or is that
  purely convention? The paper's PPE derivation (its Eq. 2-6) is specific to
  semi-implicit/symplectic Euler: velocity is updated first (including the
  pressure correction), then position is advanced using that *new* velocity
  — `Δt∇²p(t) = ρ0∇·v*` is only the right one-shot correction under that
  ordering. Checked: `SimulationConfig.integrationScheme`'s dataclass default
  is `rungeKutta4` (`configurations/simulationConfig.py:39`), and
  `buildConfig`'s own fallback when unset is `rungeKutta2`
  (`simulationConfig.py:93-94`) — neither is semi-implicit Euler. Every DFSPH
  case that currently exists (`tgv.py:171`, `kolmogorovIncompressible.py:181`,
  `randomFlowIncompressible.py:118`) explicitly overrides this to
  `integrationScheme='semiImplicitEuler'` in its own `defaults` dict, and
  `warpSPHIntegrators/euler.py`'s `integrateSemiImplicitEuler` was confirmed
  (via `util.py`'s `updateStateSemiImplicitEuler`) to genuinely apply the
  velocity update before the position update, matching the paper's ordering
  — so **this is not the explanation for any currently-open item**: every
  case actually exercised by this plan already gets it right, by convention.
  But nothing enforces that convention: `CaseSpec.integrationScheme` defaults
  to `'rungeKutta2'` independent of `scheme`
  (`runner/caseSpec.py:53`), a case's own default is only a `setdefault`
  (CLI/user override wins), and no code path checks
  `scheme == divergenceFree` against `integrationScheme` anywhere in
  `runner/`. A multi-stage RK integrator would call `dfsph_step` once per
  stage, each stage independently solving `solveDivergenceFree` as if *that
  stage's* output were the final velocity — but the actual state update is a
  Butcher-tableau-weighted blend across stages, so the blended velocity is
  not guaranteed divergence-free even though each individual stage's was;
  `IncompressibleSystem.finalize`'s PS solve would then run once per full
  step (right cadence) but on top of that not-actually-divergence-free
  blended velocity. `--scheme divergenceFree --integration-scheme
  rungeKutta4` (or any future DFSPH case that forgets the override) would
  silently run a scheme whose PPE derivation no longer holds, with no
  warning. Not fixed this session — flagging as a real latent gap, and a
  candidate for a one-line guard (assert/warn in `_divergenceFree()` or
  `dfsph_step`) if it's worth hardening against, rather than relying on every
  future case remembering the same three-case convention.

### Part 3 — `rotatingSquarePatch`: a DFSPH + free-surface case with severe, unexplained corner density loss (2026-08-26, new)

Prompted by "check whether the tested cases use the right integration scheme,
and keep digging on boundary/free-surface cases." All three cases actually
exercised so far in this plan (`tgv`, `kolmogorovIncompressible`,
`randomFlowIncompressible`) do correctly set
`integrationScheme='semiImplicitEuler'`, so that's not live anywhere in this
plan's own history. But grepping for every DFSPH-capable case turned up a
fourth one the plan never touched: `cases/rotatingSquarePatch.py`
(`scheme='deltaSPH'` by default, but its own docstring explicitly advertises
`--scheme divergenceFree` as a supported comparison mode — "the same geometry
under two schemes"). It's a rotating square patch of fluid with a genuine
open free surface (`freeSurface=True`, unlike every other case here, which
has none), and its docstring calls the corners "what makes this test hard."

- **First hypothesis (integration-scheme mismatch, per the item above) —
  tested and ruled out as the primary cause.** This case inherits
  `WEAKLY_COMPRESSIBLE_DEFAULTS['integrationScheme'] = 'rungeKutta2'` (no
  override of its own), so `--scheme divergenceFree` on it *is* a live
  instance of exactly the invalid combination flagged above. Confirmed it
  runs that way by default: `python -m warpSPH.cases.rotatingSquarePatch
  --scheme divergenceFree --nx 32 --tLimit 0.05` blows up (kineticEnergy
  16→~10⁴, maxVelocity 4.8→144-200, minDensity 1.0→0.18 within 100 steps).
  Re-ran with `--integrationScheme semiImplicitEuler` explicitly forced: it
  blows up *worse* (maxVelocity up to 452, kineticEnergy up to ~4.4e4) — so
  the integration-scheme mismatch is real and worth fixing independently,
  but it is not what's driving this particular blowup.
- **Second hypothesis (frozen, oversized `dt`) — real, but also not the
  primary cause.** `rotatingSquarePatch` has no `Case.timestep` hook (unlike
  `randomFlowIncompressible`, which needed one for exactly this reason — see
  Part 2 findings above), so per `runner/runner.py:285-286`
  (`if case.timestep is not None: ctx.config.dt = case.timestep(...)`), `dt`
  is **never adapted** for this case under any scheme — confirmed: "final dt"
  in the run report always equals the initial WCSPH-acoustic-CFL-derived
  value, unchanged after 100 steps. Forcing a 10x smaller fixed `dt` via
  `--targetDt 0.00005` delayed the blowup (NaN at step 143 instead of ~100)
  but did not prevent it. This is a second, real, independent bug (the same
  bug class Part 2 already fixed for `randomFlow`/`randomFlowIncompressible`,
  never ported to this case) but not sufficient on its own to explain the
  failure.
- **Root cause, localized precisely: severe, resolution-independent density
  loss at the square's four convex corners, present within ~8 steps,
  regardless of `dt` or integrator.** Instrumented the case (monkeypatched
  `diagnostics` to capture the live state) and looked at the lowest-density
  particles directly. At nx=32 (100 particles) and nx=96 (1024 particles),
  the four corner-most particles read essentially the **same** density in
  both cases — `ρ≈0.506` at nx=32 vs. `ρ≈0.506` at nx=96 — and the same
  neighbor counts (30 vs. a bulk median of 56-93). Resolution-independence
  is the tell: this isn't discretization noise, it's the particle's actual
  geometric situation — a particle sitting at a 90° convex free-surface
  corner has roughly a quarter of a full kernel-support disk's worth of
  fluid around it, so its raw SPH-summation density *correctly* reads well
  below `rho0` there. **Under the identical setup and resolution with
  `--scheme deltaSPH` (EOS-based), density at the same corner particles
  stays at 0.9998** — confirming this is specific to DFSPH's implicit
  pressure solve, not a property of the geometry itself: an EOS just turns a
  low corner density into a small, bounded pressure response, while DFSPH's
  PPE tries to actively correct it back toward `rho0`.
- **A plausible fix already exists in the code, commented out, and a quick
  test of it didn't resolve the corner blowup — needs more targeted work,
  not landed this session.** `modules/incompressible/divergenceFree.py` has
  `# pressureB[particles.surfaceIndicators == 1] = 0.0  # Set pressures to
  zero for surface particles` — the standard IISPH/PCISPH free-surface
  treatment (pressure should be ~0 at a genuine free surface, not driven
  back toward the bulk value). Enabling it and re-running the corner probe
  changed almost nothing (`ρ≈0.507` vs. `0.506`). Root cause of *that*:
  `detectFreeSurface` flags **96/100 particles at nx=32 and 536/1024 (52%)
  at nx=96** as surface on this small, thin patch — with `n_h=4.0` and
  `expansionIterations` dilation, a full support-radius-thick shell inward
  from the true edge gets flagged, which at this patch's size is most of
  the domain. A clamp gated on that mask isn't a clean, isolated test of
  the free-surface-corner hypothesis when it's firing on half-to-nearly-all
  of the particles; it also only exists in `divergenceFree.py`'s relaxed-
  Jacobi loop — `solveIncompressible` (`incompressible.py`, the DI/PS
  solver that runs unconditionally every step in `finalize`) and the Krylov
  path (`krylov.py`) have **no** free-surface-aware logic at all, commented
  out or otherwise, so even a working clamp in the VD solve wouldn't reach
  the PS shift that's applied to positions every step.
- **Not resolved this session.** Two independent, real, already-fixed-
  elsewhere-but-not-here bugs (integration scheme, frozen `dt`) are ruled out
  as the primary driver; the actual mechanism is corner-localized,
  resolution-independent, DFSPH-specific density loss that an EOS-based
  scheme doesn't share, and the one existing candidate fix in the codebase
  is disabled, mask-overbroad on this geometry, and only wired into one of
  three solver code paths. Next steps, in rough order of leverage: (a) check
  whether `SurfaceDetectionConfig`'s scheme/`expansionIterations` are simply
  too aggressive for a patch this thin at these resolutions (a narrower mask
  would make the existing clamp a fairer test); (b) if a narrower mask still
  doesn't fix it, extend the same pressure clamp (or a softer, relaxed
  version, following `mdbcPressureRelaxation`'s precedent from Part 2) to
  `solveIncompressible` and `solvePressureKrylov`, not just the VD solve;
  (c) separately land the two ruled-out-as-primary-but-still-real fixes
  (a `timestep` hook for this case, and the integration-scheme guard) since
  both are real bugs independent of the corner issue. This is the most
  concrete, reproducible lead this plan has found for a DFSPH-vs-deltaSPH
  robustness gap at a genuine free surface (as opposed to the still-open
  wall/mDBC-boundary gap in Part 2, which is a structurally different
  scenario — solid walls, not open free surface).

### Part 3 — `integrateRho` audit (2026-08-26)

Asked directly: do the tested cases follow the "not `integrateRho`" path
correctly, and is that path actually correct? `IncompressibleSolverConfig.
integrateRho` defaults to `False`
(`configurations/moduleConfigurations/solver.py:121`), and none of `tgv`,
`kolmogorovIncompressible`, or `randomFlowIncompressible` override it — so
yes, every tested case uses the "recompute density from scratch every step"
path, and `dfsph_step`'s own gating for it
(`if currentState.densities is None or not integrateRho: currentState.
densities = computeDensities(...)`) is correctly wired for that setting.

- **But the flag it's gating turns out not to matter either way — a real,
  if currently-inert, config/contract bug.** Traced what `integrateRho=True`
  would actually change: the generic integrator applies the continuity-
  equation update (`ρ(t)+Δt·drhodt`) to `self.state.densities` as part of
  `updateStateSemiImplicitEuler`'s `applyQuantityUpdate`, *before*
  `IncompressibleSystem.finalize` runs. But `finalize` immediately discards
  that: `self.state.densities.copy_(lastState.densities)`
  (`incompressible.py:128-131`) overwrites it with `lastState.densities` —
  the density `dfsph_step` computed/used at the *start* of the step, not the
  continuity-integrated value — and then `self.state.densities =
  computeDensities(self.state, ...)` (`incompressible.py:230`) overwrites
  *that* again with a fresh full SPH summation at the shift-advected `x**`,
  unconditionally, regardless of `integrateRho`. So whatever
  `integrateRho=True`'s continuity-equation branch computes is thrown away
  twice over before it ever reaches the density that's carried into the next
  step. **The config flag's docstring ("Whether to integrate density in the
  incompressible solver") is not actually true for the density that
  survives a step** — `finalize` always resums, unconditionally. Since every
  case that exists uses the default (`False`), this isn't producing wrong
  results anywhere today, but the flag itself is dead on the `True` branch,
  which would surprise anyone who sets it expecting continuity-based density
  evolution. Not fixed this session — either `finalize`'s unconditional
  resum should itself become conditional on `integrateRho`, or the flag's
  `True` branch and docstring should be removed/corrected to say what it
  actually (doesn't) do.
- **Secondary, smaller finding from the same trace**: `finalize` always
  computes `ρ**` (line 230) *before* applying the PS shift (`self.state.
  positions += dx`, line 274) — so the density that's carried into the next
  step is always labeled at the pre-shift position `x**`, one shift-worth
  stale relative to the actual carried-forward position `x(t+Δt)`. This
  matches the reference paper's own approach (it never re-verifies density
  after its analogous shift either), so it's not obviously wrong, but it's
  an uncorrected small error source every step, independent of
  `integrateRho`, and was previously undocumented.

### Part 4 — periodic, boundary-less pressure-gauge drift: found a live NaN mechanism in `solveIncompressible` (2026-08-26)

Redirected per project owner: the free-surface corner issue above is real but
superfluous to the actual question — dig into the **boundary-less** case
(no solid wall, no free surface) specifically, because a fully periodic
domain's PPE is only defined up to an additive constant (pure-Neumann null
space), and that constant can drift to a magnitude large enough to erode
float32 precision and cause a blowup. One existing mitigation (subtracting
the mean) was flagged as unreliable. Used `kolmogorovIncompressible`
(periodic, no boundary, no free surface — exactly the isolated case asked
for) and instrumented both pressure solvers non-invasively (Python-level
monkeypatching of `solveDivergenceFree`/`solveIncompressible`, no source
changes) to log `sourceTerm`/pressure mean, std, min/max, and iteration
count every step.

- **The VD (divergence-free) solver's gauge fix works correctly, and is not
  the drift source.** `solveDivergenceFree`'s relaxed-Jacobi loop subtracts
  `pressureB[fluidMask].mean()` every iteration (both the fixed-`omega` and
  `optimal` variants — confirmed by reading both loops). Measured over 300
  steps at nx=64: `pMean` stayed at `~1e-8` (float32 noise floor) the entire
  run, even as `pStd` grew ~4 orders of magnitude (`1e-4` → `4.1`) tracking
  the flow's genuine physical pressure buildup. Tried the also-present,
  also-commented-out `sourceTerm = sourceTerm - sourceTerm.mean()` fix on
  top of the existing per-iteration recentering: **no measurable effect**
  (`pStd`/`nIter` trajectories nearly identical with and without it) — this
  solver's gauge was already fine; the persistent non-convergence
  (`maxIterations=32` hit every step from step ~30 on, in both variants) has
  some other cause, not gauge inconsistency. Reverted the experiment (see
  the module docstring's own note that `sourceTerm.mean()` isn't currently
  subtracted).
- **`solveIncompressible` (the DI/PS solver, called every step from
  `IncompressibleSystem.finalize`) has no defense against drift at all, and
  it is live and reproducible.** Its gauge fix is `torch.clamp(pressureB,
  min=0.0)` — a physically-motivated "pressure isn't tensile" floor, not a
  mean-centering. The PPE operator here has the *same* constant null space
  as the VD solver (same `computePressureAccelIISPH`/
  `computePressureShiftIISPH` machinery, just scaled by `dt**2`), but a
  floor-only clamp provides zero resistance to the null-space mode
  drifting *upward*: nothing pulls a too-large mean back down.
  Mechanistically: `sourceTerm = rho0 - rhoStar` had a **persistently
  negative mean at every single sampled step across 300+ steps**
  (`stMean` between `-1e-8` and `-2.4e-2`, never once positive after the
  first couple of steps) — not from the `rhoStar` floor-clamp at 0.9
  (confirmed `nClamped=0/4096` throughout a 200-step nx=64 run, ruling that
  out directly), but from a genuine, systematic tendency for the predicted
  density to sit slightly *above* `rho0` — i.e. the same sign/character as
  this plan's already-open "DFSPH bulk density-band gap vs. deltaSPH"
  finding (Part 2), previously only characterized as an error *magnitude*,
  never checked for *sign*. Since `alphas` (the IISPH diagonal) is always
  negative by construction (`torch.clamp(..., max=-1e-6)`), a persistently
  negative `residual` mean divided by a persistently negative `alphas`
  pushes `pressureB`'s mean *up* every single iteration, and nothing in the
  loop (the floor clamp least of all) ever pushes it back down.
- **Confirmed this actually blows up.** nx=64/300 steps: `pMean` climbed
  from `~1e-3` to `9.8` (comparable in magnitude to `pStd` itself — i.e.
  roughly half of the reported "pressure signal" at that point is an
  unphysical uniform offset, not real spatial variation), while `nIter`
  pegged at `maxIterations=64` from step ~45 onward (never converging,
  consistent with the null-space component being structurally
  un-correctable by this iteration). **nx=128, run to 1000 steps: `pMean`
  climbed to `29.9` by step 560, then to `2.38e6`, then NaN at step 574** —
  a clean, direct reproduction of exactly the failure mode described
  ("pressure drifts to a very large constant... leads to float accuracy
  issues and blow up"). This run used the identical `kolmogorovIncompressible`
  configuration Part 1 validated as stable to 1600 steps / t≈14 — the
  discrepancy is almost certainly run-to-run chaotic sensitivity (forced
  turbulence; no seed was pinned to match Part 1's exact probe), not a
  difference introduced by the instrumentation (which is read-only
  Python-level wrapping, no source files were modified for this run — verified
  via `git diff --stat` immediately after). That the *timing* varies run to
  run doesn't weaken the finding: the mechanism is real, live, and reachable
  from the current default config on a case this plan already calls
  "validated," not a hypothetical.
- **Not fixed this session — needs a design decision, not a quick patch.**
  Unlike the VD solver, `solveIncompressible` can't just copy the
  "subtract the mean every iteration" fix verbatim: its non-negative clamp
  is physically load-bearing (pressure without tension), and naively
  recentering would fight that constraint. Candidate directions, roughly in
  order of how much they change the solver's character: (a) recenter around
  the *median* or some other robust statistic instead of literal zero,
  then re-floor — cheaper than a redesign, but ad hoc; (b) cap `pressureB`'s
  *mean* growth per iteration directly (a "soft ceiling" mirroring
  `mdbcPressureRelaxation`'s under-relaxation precedent from Part 2) rather
  than touching the field's shape; (c) address the root cause instead of
  the symptom — if DFSPH's bulk density genuinely runs systematically
  (not just noisily) dense relative to `rho0`, chasing that upstream (Part
  2's still-open bulk-gap item) would shrink `sourceTerm`'s persistent bias
  and starve this drift mechanism directly, closing two open items at once.
  (c) is the most promising lead precisely because it's the same signed
  quantity as the pre-existing open bulk-gap question — worth checking
  first whether that gap's *sign* (not just magnitude) matches what's
  measured here before designing a fix around either symptom.
- **Krylov path spot-checked, not fully ruled out.** `solvePressureKrylov`'s
  `gauge='center'` recentering is applied **once, after the whole solve**,
  not per-iteration like both relaxed-Jacobi loops — structurally the
  highest-risk path for intra-solve drift, since none of the underlying
  Krylov methods (BiCGStab/GMRES/CG/BiCG/MINRES, in `modules/shifting/`)
  know about the gauge constraint internally, and the module's own docstring
  already flags this operator as ill-conditioned (`kappa(M^-1 A) ~ O(1e8)`).
  A GMRES run (nx=64, 100 steps, on the VD/divergence-free solve, which
  *does* have per-solve-final gauge fixing) didn't show obviously worse
  drift than relaxed-Jacobi in final per-step stats (`pMean` stayed
  `~1e-7`, `pStd`/`pMax` comparable order to relaxed-Jacobi's), but that
  only observes the *post-solve* state, not what happens to the iterate
  *during* the ~33 internal iterations each step — the exact blind spot
  this mechanism lives in. Did not instrument `solveIncompressible` under a
  Krylov solver (it always uses the same `gauge='nonnegative'` post-hoc
  clamp regardless of `solverType`, so it likely inherits the identical
  drift mechanism found above, possibly faster given the higher intra-solve
  iteration count). Not tested this session.
- **Confirmed the sign match for fix direction (c).** Checked directly
  (`kolmogorovIncompressible`, nx=64, 300 steps, no boundary/wall involved):
  `mean(rho-1) = +2.71e-3` against `mean|rho-1| = 3.54e-3` — the *signed*
  bias accounts for ~76% of the *unsigned* error, i.e. this is
  predominantly a systematic bias, not zero-mean noise, and 82% of
  particles sit above `rho0`. This is the same sign, same order of
  magnitude, and same case as this section's `sourceTerm` measurements
  above — strong evidence that Part 2's still-open "DFSPH bulk density-band
  gap vs. deltaSPH" and this section's `solveIncompressible` pressure-mean
  runaway are two symptoms of one upstream cause (DFSPH's bulk pressure
  projection running systematically dense), not two unrelated bugs. Fixing
  the upstream bias is now the best-supported next step across both open
  items, rather than patching either symptom (a wall-density gap, an
  unbounded pressure gauge) independently.

### Part 4 continued — isolating the drift's actual cause: clamp, iteration budget, grid symmetry (2026-08-26)

Follow-up per project owner: does deactivating the non-negative clamp change
the drift; does raising `maxIterations` let the solver actually converge;
and is a perfectly-regular initial grid ("even distortion... difficult to
solve cleanly... tension... until noise breaks the symmetry enough") the
underlying driver — with the caveat that an alternative ("optimal") sampling
avoids the perfect-symmetry start but introduces its own initial density
noise instead. Built a self-contained reimplementation of `solveIncompressible`'s
loop in a scratch probe (toggleable clamp, overridable `maxIterations`, no
source edits) and re-ran `kolmogorovIncompressible` at nx=64/300 steps under
each variant; confirmed the reimplementation reproduces the real solver's
trajectory byte-for-byte on the baseline case before trusting its variants.

- **Removing the clamp does not stop the drift — refines, doesn't overturn,
  the Part 4 finding above.** `pMean` trajectory with `--no-clamp` is nearly
  identical to the clamped baseline (rises to `~8.5` vs. `~9.8`, settles to
  `~2.1` vs. `~3.1` by step 300; `pMax` was if anything slightly *higher*
  without the clamp at some steps, e.g. `57.7` vs. `31.97` at step 90). So
  the clamp isn't *causing* the upward drift — the mechanism identified above
  (persistent negative `sourceTerm` mean ÷ persistently negative `alphas`
  pushing the null-space mode up every iteration) operates independently of
  whether the floor is there. The clamp's only role is failing to defend
  against it, not driving it.
- **A perfectly-regular starting grid is not what's driving the
  non-convergence either — tested directly, not just reasoned about.**
  `kolmogorovIncompressible`'s own `buildSystem` hardcodes
  `sampleRegularParticles(..., jitter=0.0)`
  (`sample/weaklyCompressible.py:24`) — it never reads
  `CaseSpec.samplingScheme` at all (confirmed: passing `--samplingScheme
  jittered`/`optimal` through the real CLI path produced byte-identical
  output to the unset default, since the case simply doesn't wire that
  general mechanism up). Monkeypatched `sampleRegularParticles` directly to
  test its own `jitter` parameter instead (0.1, a real, immediate breaking
  of the initial lattice symmetry): this produces a large **one-time**
  initial pressure spike (`pMean=36.8`, `pMax=248` at step 0, vs. the
  regular grid's `pMean=8.6e-4` at step 0) — exactly as expected, since a
  jittered start has real initial density disorder the solver must correct
  immediately — but that spike resolves in **6 iterations**, not 64, and
  from step 1 onward the trajectory tracks the unjittered baseline closely
  (peak `pMean~7.9` vs. `~9.8`; `nIter` pegs at `maxIterations=64` from
  step ~10-20 onward either way). **So breaking the initial symmetry trades
  a one-step transient spike for essentially the same subsequent
  non-convergence/drift pattern, not a fix for it** — the persistent
  `sourceTerm` mean-bias and the resulting drift is not primarily an
  artifact of the t=0 lattice being perfectly symmetric; it re-emerges
  dynamically as the flow develops regardless of the starting condition.
  This matches the "optimal sampling starts with its own density noise"
  caveat: the caveat turns out to be the whole story, not just a footnote —
  neither starting condition avoids the drift, they just differ in whether
  the disorder is front-loaded (jittered) or develops over ~10-40 steps
  (regular).
- **`nIter` itself does not show the peak-then-decay pattern — `pMean`/`pStd`
  do.** Across every variant tested, once `nIter` first hits
  `maxIterations` (around step 10-40, depending on variant) it **stays
  pegged there for the rest of the run** — it never recovers to early
  convergence the way the hypothesis's "sudden resolution once noise breaks
  the symmetry" framing would predict. What *does* rise-then-decay is the
  pressure field's magnitude (`pMean`, `pStd`, `pMax`) and the density-std
  disorder proxy (`rhoStd`, peaking around the same step range, `~80-110`,
  across every variant) — consistent with the *physical* forcing/dissipation
  balance settling into a statistically-steady state over that window
  (kinetic energy build-up then equilibration, expected for a forced
  Kolmogorov flow), not with the *solver* ever actually resolving the
  tension it's failing to converge on. The solver stays stuck at its
  iteration cap throughout; the field's own magnitude just stops growing
  once the physical forcing/dissipation balance equilibrates.
- **`maxIterations=1024` (16x default) — makes the drift categorically
  worse, not better. The most decisive result of this whole investigation.**
  `pMean` peaked at `61.4` (vs. `9.83` at the default `maxIterations=64`),
  `pMax` reached `194` (vs. `34.1`), and `nIter` is pegged at the *new* cap
  (`1024`) throughout the second half of the run — still never converging,
  just grinding through 16x more unopposed accumulation per step. This is
  exactly what the identified mechanism predicts and nothing else would:
  the null-space (mean) component of `pressureB` is not attributable to any
  degree of freedom `dx_p`/the residual can actually correct (the operator's
  constant null space means the residual's null-space part is structurally
  unshrinkable under this iteration), so each additional iteration just adds
  another `omega*mean(residual)/mean(alphas)` to the mean, forever, with no
  natural stopping point — more iterations is strictly more of the problem,
  not less. **This settles the "does the solver just need to converge
  harder" question conclusively: no.** The fix has to be architectural (a
  real gauge-fixing mechanism for `solveIncompressible`, or removing the
  bias that's forcing the null-space mode to move in a consistent direction
  every iteration in the first place), not a solver-tuning knob — raising
  `maxIterations`/tightening `tolerance` moves in the wrong direction
  entirely.
- **Taken together, the three tests point at one conclusion**: neither the
  clamp, the iteration budget, nor the initial grid's symmetry is the root
  cause — all three are downstream knobs on a solver that has no mechanism
  at all to prevent its null-space mode from accumulating a persistent,
  per-iteration, per-step bias. The only two tests that changed the
  *outcome* materially were (a) more iterations, which made it worse in the
  expected direction (confirming the mechanism), and (b) the still-open
  cross-reference to Part 2's bulk density-band gap, which showed the same
  signed bias driving the `sourceTerm` in the first place. Both point at the
  same fix priority as the previous section: address the upstream density
  bias, and/or give `solveIncompressible` a real gauge fix (recenter-then-
  reclamp, a soft mean ceiling, or similar) — not a solver-tuning change.

### Part 4 resolved — the drift is integrator wind-up against an unreachable setpoint; `ShiftPressureGauge.minShift` fixes it (2026-08-27)

Picked up exactly where the previous session's "Next session: start here"
left it, and both of its top-two leads turned out to be answerable. The
short version: **option 1 (fix the upstream density bias) is a dead end,
because there is no bug to find — the bias is structural**; option 2 (give
`solveIncompressible` a real gauge fix) is the right fix, but only one of
the three candidate gauges works, and only on the case class the constant
mode is actually free in. Both are now measured, not argued.

**First, what `solveIncompressible` actually is.** It is not a momentum
pressure solve. Both of its call sites in the library use it as an
*implicit particle-shifting* solve: `systems/incompressible.py`'s `finalize`
calls it with `dvdt=0` and feeds its output straight into a position shift
(`dx = dt**2 * a_p`), and `cases/tgv.py`'s lattice relaxation does the same.
Its "pressure" is therefore a shifting potential. That reframes the whole
question — the constant mode of a shifting potential is not a physical
pressure level, and the non-negativity clamp is not a physical
no-tension constraint on a stress, it is "the shift never pulls particles
together."

**The bias is structural, and no kernel bug is behind it**
(`scripts/probe_densityBiasVsDisorder.py`, new). Take the case's own initial
state — a perfect lattice, mass-normalised so `mean(rho) == rho0` by
construction — displace every particle by increasing random jitter, and
recompute the density. No solver, no timestep, no scheme kernel involved:

| jitter/dx | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|---|---|
| `mean(rho-1)` | 5.6e-8 | 2.2e-6 | 8.7e-6 | 3.4e-5 | 2.1e-4 | 8.5e-4 | 3.3e-3 | 1.2e-2 |

Always positive, and rising as the *square* of the displacement (each
doubling of jitter multiplies it by ~3.9). This is exactly what the SPH
summation density has to do and could not do otherwise: for equal masses
`sum_i rho_i = m * sum_{ij} W(x_i - x_j)`, which by Parseval is
`m * sum_k What(k) |rhohat(k)|^2` with `What(k) >= 0` for a positive-definite
kernel (Wendland is one). A lattice puts all of its spectral weight on
reciprocal-lattice vectors where the bandwidth-limited `What` is ~0; *any*
disorder moves weight to small `k` where `What` is large, and every such
contribution is non-negative. **The lattice minimises the particle-averaged
summation density, so `mean_i rho_i == rho0` is unattainable for any
disordered configuration.** The previous session's `mean(rho-1) = +2.7e-3`
was not a symptom of a bug; it is the floor.

That also explains why the running simulation's signed/unsigned ratio (0.76)
is so much higher than raw jitter's (0.008-0.29): the shifting solve does
its job and removes the *correctable*, near-zero-mean part of the density
error, leaving the irreducible positive-mean part as most of what remains.

**So the solver is an integral controller with a setpoint it cannot reach.**
`sourceTerm = rho0 - rhoStar` keeps a persistent negative mean forever; the
only mode with a nonzero mean response is the (near-)constant one, whose
response is weak; so the iteration drives that mode to ever larger
amplitude, every iteration, every step, without a stopping point. This
predicts — and explains — every previously-collected observation at once:
`nIter` pegged at `maxIterations` forever, `maxIterations=1024` making it
*worse* rather than better, the clamp being irrelevant to the drift, and the
initial lattice's symmetry being irrelevant too.

Measured directly (`probe_incompressibleGaugeDrift.py --null-test`, new
flag): on the t=0 lattice `|A*1| = 6.8e-11` against `|A*rand| = 2.8e-5`
(constants annihilated to 6 orders); by step 59 of the developed flow
`|A*1|` has risen to ~5% of a same-magnitude non-constant field's response —
i.e. the constant mode is *near*-null rather than exactly null once the
configuration disorders, so the mean error is weakly correctable, but only
at an amplitude ~20x larger than a normal mode would need. Hence a pressure
mean that reaches 2.4e6 before float32 gives out.

**Correction to the previous session's note on reproducibility.** The nx=128
blowup was described there as probably "run-to-run chaotic sensitivity."
It is not: re-running it reproduced `pMean` peaking at `2.3838e6` and NaN at
**step 574**, matching the earlier run's reported figures exactly. This case
is deterministic, which is what made the A/B tests below trustworthy.

**Four candidate gauges, compared** (nx=64, 300 steps, second-half means;
all prototyped in `probe_incompressibleGaugeDrift.py --gauge` before any
source was touched):

| gauge | `mean(rho-1)` | `mean|rho-1|` | `|pMean|` | outcome |
|---|---|---|---|---|
| `clamp` (historical) | 2.88e-3 | 3.83e-3 | 3.9 | survives 300, drifts |
| `center` (what the VD solver does) | 2.66e-2 | 2.78e-2 | 4.3e5 | **NaN at step 155** |
| `center-clamp` | 3.91e-2 | 4.34e-2 | 1.4 | **NaN at step 136** |
| `minshift` (subtract the fluid min) | **1.52e-3** | **2.86e-3** | 13.6 | survives, no drift |

Mean-centering — the fix that works in `solveDivergenceFree` — is the *worst*
option here, and that is not a surprise once the two solvers' source terms
are compared: the VD solver's source is a divergence, mean-zero by pair
antisymmetry, so recentering costs it nothing; this solver's source is not,
and recentering both gives up the non-negativity that keeps the shift from
pulling particles together and cancels the background-pressure de-clumping
force the solve is relying on. Also tested and rejected: projecting the
source to zero mean (`--project-source`). It makes the solver *converge*
(nIter 64 -> 13.4, as the compatibility argument predicts) but the density
gets worse across the board (`mean|rho-1|` 3.9e-3 vs 2.9e-3, `rhoStd` 3.4x
worse), because the source's mean is not purely unreachable — part of it is
the genuine "de-clump" signal, and throwing it away stops the solve doing
its job.

**Landed: `ShiftPressureGauge` (`configurations/moduleConfigurations/solver.py`),
read by `solveIncompressible`.** `nonNegativeClamp` is the historical
`clamp(p, min=0)` and stays the default; `minShift` subtracts the fluid
minimum instead — still non-negative, but gauge-fixed, and it *translates*
the field's negative part instead of discarding it. Round-trips through
`incompressibleConfigToDict`/`dictToIncompressibleSPHConfig` (absent key
falls back to the default).

**Validated end-to-end through the real solver**, not just the probe's
reimplementation (`scripts/probe_shiftPressureGauge.py`, new), at nx=128:

| case | gauge | steps | diverged | rho range | worst `|rho-1|` |
|---|---|---|---|---|---|
| `kolmogorovIncompressible` | `nonNegativeClamp` | 575 | **yes** (t=5.05) | [0.240, 2.515] | 1.51 |
| `kolmogorovIncompressible` | `minShift` | 1001 | no (t=7.41) | [0.980, 1.015] | 1.98e-2 |
| `randomFlowIncompressible --bounded` | either | 258 | yes (t=5.54) | [0.139, 2.452] | 1.45 |

`pMean` under `minShift` settles at ~10-15 and oscillates there for the
whole 1000 steps with no trend at all, against the clamp's monotone climb to
2.4e6.

**And validated as harmless on `tgv`**, the only other case the gauge
reaches (periodic, complete support, and it uses `solveIncompressible`
twice over — once for its 32-step lattice relaxation, once per step inside
`finalize`). Unlike `kolmogorovIncompressible` there is a reference answer
here, so the check is against physics, not just against not-blowing-up:
`KE(t) = KE(0) exp(-4 nu k^2 t)`.

| nx / steps | gauge | rate/analytic | KE_end/KE_0 | rho range (final) |
|---|---|---|---|---|
| 64 / 200 | `nonNegativeClamp` | 0.5489 | 0.99561 | — |
| 64 / 200 | `minShift` | 0.5486 | 0.99562 | — |
| 128 / 1000 | `nonNegativeClamp` | 0.5873 | 0.97696 | [0.99811, 1.00118] |
| 128 / 1000 | `minShift` | 0.5898 | 0.97690 | [0.99768, 1.00192] |

Agreement to ~0.4%, both monotone, neither diverging, both well inside the
0.55-0.6 band `tests/test_physics.py::test_tgvKineticEnergyDecaysAtRoughly
TheAnalyticRate` documents and asserts. (`scripts/probe_tgvShiftGauge.py`.)

**On that basis `minShift` is the default** as of this session, rather than
opt-in. The usual argument for keeping historical behavior as the default
does not apply: the historical default provably NaNs a production case, and
the new one is a scoped no-op on every case it does not apply to, so the
blast radius is exactly the two cases above — one fixed, one unchanged.

**The gauge is scoped, deliberately, to solves where the constant mode is
actually free** — `solveIncompressible` falls back to the clamp when there
are pinned pressure rows (`kind != 0`) or free-surface particles. This is
measured, not assumed: forcing `minShift` through on the bounded
`randomFlowIncompressible` diverges at t=0.69 against the clamp's t=5.54.
Two independent reasons, and the second is the one that bites:

- Pinned rows are Dirichlet data, and Dirichlet data already fixes the
  constant — there is no null space left to gauge.
- Where kernel support is truncated (a wall, a free surface) the kernel
  gradients stop summing to zero, so a *constant* pressure exerts a large
  real force. The offset stops being a gauge choice and becomes a background
  pressure blowing wall-adjacent particles around.

Both variants of the boundary handling were tried — translating the frozen
boundary rows along with the fluid (so all pressure *differences* are
preserved exactly) and leaving them in place — and they fail identically, at
the same step, which is what pinned the cause on the second reason rather
than the first. With the scoping in place, `minShift` on the bounded case is
now byte-identical to the clamp (confirmed: same 258 steps, same density
extrema to 5 digits), i.e. a genuine no-op rather than a silently different
answer.

**Checks:** full suite `241 passed, 1 skipped`;
`scripts/gradcheck_incompressible.py` ALL PASSED. One test
(`test_incompressibleKrylov.py::test_minresGivensMatchesDenseLstsq`) failed
once and passed on re-run with identical code — a pre-existing flake in a
tight-tolerance dense-lstsq comparison that never calls `solveIncompressible`,
not a regression from this change.

### Part 4 — the physics behind the fix, in full (2026-08-27)

Written out properly because every one of the four gauge candidates is
defensible on general PPE grounds, and only the physics of *this* operator
picks between them. Four facts do all the work.

**1. The particle-averaged summation density is bounded below by the lattice
value, and rises quadratically with disorder.** For equal masses,

    sum_i rho_i = m * sum_{i,j} W(x_i - x_j)

and by Parseval, on a periodic domain, that is

    sum_i rho_i = m * sum_k What(k) * |rhohat(k)|^2,    rhohat(k) = sum_i e^{i k . x_i}

For a positive-definite kernel — and Wendland is one, which is exactly why
it is used — `What(k) >= 0` for every `k`. A perfect lattice concentrates all
of `|rhohat|^2` on reciprocal-lattice vectors, and the kernel is
bandwidth-limited, so `What` is ~0 there: the lattice is the *minimum* of
this sum. Any disorder moves spectral weight to small `k` where `What` is
large, and every such contribution is non-negative, so it can only push the
sum up. Expanding to second order in a displacement field gives the
quadratic growth, which is what the jitter sweep measures (~3.9x per
doubling, against 4x predicted).

Consequence: `mean_i rho_i == rho0` is not a hard target the solver is
missing, it is a target *below the attainable floor* for any configuration
that is not a perfect lattice. `sourceTerm = rho0 - rhoStar` therefore keeps
a permanently signed mean, by construction, forever. Nothing upstream is
broken; there is nothing upstream to fix.

**2. A uniform pressure exerts no force where the kernel support is
complete, and a large force where it is not.** The SPH pressure force on `i`
is a sum of pair terms weighted by `grad W_ij`. With complete support the
gradients sum to zero by symmetry, so a *constant* pressure field produces
(near-)zero acceleration — the operator has a constant null space, and the
constant is a gauge. Truncate the support — a wall, a free surface — and
that cancellation fails: the same constant now produces a large net force
along the truncation normal. This is the well-known background-pressure
mechanism transport-velocity formulations use deliberately (Adami et al.);
here it is what decides where a gauge shift is legitimate.

Measured, on the developed flow: `|A*1|` is 6.8e-11 against 2.8e-5 for a
random field on the t=0 lattice (annihilated to 6 orders), rising to only
~5% of a same-magnitude non-constant field's response by step 59 as the
configuration disorders. So in the bulk the constant is *near*-null: the mean
density error is weakly correctable, but only by a pressure ~20x larger than
a normal mode would need. That factor is the drift.

**3. This solver's "pressure" is a shifting potential, not a stress.** Both
call sites (`systems/incompressible.py`'s `finalize`, `cases/tgv.py`'s
relaxation) feed its output into a position shift, `dx = dt**2 * a_p`. So
the non-negativity is not "a fluid cannot sustain tension" — it is "the
shift never pulls particles together," which matters because a shift that
can pull is the tensile instability that clumps particles. And the constant
component of a shifting potential is not a physical pressure level at all,
which is what makes translating it a free choice in the bulk.

**Putting 1-3 together: the solver is an integral controller with a setpoint
it cannot reach.** Fact 1 keeps the residual's mean signed every iteration;
fact 2 says the only mode that answers a mean residual is the constant one,
weakly; so each iteration adds another `omega * mean(residual) / mean(alpha)`
to the field's mean, and never gets close enough to stop. Classic
integrator wind-up. This is what makes `maxIterations=1024` *worse* than 64
(16x more unopposed accumulation), what makes the clamp irrelevant (a floor
does not oppose upward motion), and what makes the initial lattice's
symmetry irrelevant (the bias re-develops dynamically). All three were
measured in the previous session and none of them fit any other explanation.

**4. Which gauge follows, and why the other three do not.** The fix must pin
the constant mode without giving up non-negativity (fact 3) and without
zeroing the background pressure that is doing real work near truncated
support (fact 2):

- *Mean-centering* (`p -= mean(p)`) pins the mode but forces half the field
  negative, giving up the anti-clumping property, and drives the background
  pressure to ~0. Diverges at step 155. It works in `solveDivergenceFree`
  only because that solver's source is a divergence — mean-zero by pair
  antisymmetry — so its constant mode is never driven in the first place.
- *Centre-then-clamp* pins the mode but then chops the field's shape every
  iteration, which is not a translation at all. Diverges at step 136.
- *Zero-meaning the source* is the textbook compatibility projection, and it
  does make the solver converge (nIter 64 -> 13.4). But fact 1 says only
  *part* of the source's mean is unreachable; the rest is the genuine
  de-clumping signal, and discarding it makes the density worse
  (`mean|rho-1|` 3.9e-3 vs 2.9e-3, `rhoStd` 3.4x worse).
- *Subtracting the fluid minimum* pins the constant mode, keeps the field
  non-negative, translates rather than chops, and leaves the background
  pressure at whatever the solution's own shape implies. This is `minShift`.

**And why it is scoped.** Fact 2 also says exactly where a gauge shift stops
being free: wherever support is truncated, the constant is not forceless, so
translating it injects a spurious force. Fact 3's Dirichlet rows say the same
thing from the algebraic side: pinned rows fix the constant, leaving no null
space to gauge. Hence the fallback to the clamp whenever there are pinned
rows or free-surface particles. The decisive check that this is fact 2 and
not merely fact 3: translating the boundary rows *along with* the fluid (so
all pressure differences are preserved exactly, which answers the Dirichlet
objection completely) fails at the very same step as leaving them in place.

### Part 4 — what is left after the fix (2026-08-27)

- **Settled: `minShift` is the default** (2026-08-27), against this
  codebase's usual convention of keeping historical solver behavior as the
  default (`PressureSolverType.relaxedJacobi`, `mdbcNoPenetrationShift`).
  The convention exists to avoid moving numbers under existing results, and
  here it does not earn its keep: the historical default provably NaNs
  `kolmogorovIncompressible` at nx=128, and the replacement is a scoped
  no-op everywhere it does not apply, so the only numbers that move are
  `kolmogorovIncompressible`'s (fixed) and `tgv`'s (unchanged to ~0.4% on
  the analytic decay rate, verified at nx=64 and nx=128).
  `nonNegativeClamp` stays selectable for reproducing older results.
- **New, unrelated to this fix: the bounded case diverges on its own.**
  `randomFlowIncompressible --bounded` at nx=128 reaches NaN at t=5.54 under
  the untouched default. Part 2 validated that case only to t≈1.5, so this
  is past the horizon anyone had looked at, not a regression — but it means
  the bounded case has its own failure mode that the shifting-gauge work
  does not touch (and cannot: the gauge correctly declines to act there).
  Reproduced on both the committed tree and the working tree, so it is not
  the uncommitted Part 3 velocity-resample fix either. Worth its own
  investigation, and it is the natural successor to Part 2's still-open wall
  gap.
- **The Krylov path still has the original problem.** `solveIncompressible`
  returns through `solvePressureKrylov` with `gauge='nonnegative'` *before*
  reaching the relaxed-Jacobi loop, so `ShiftPressureGauge` does not reach
  it. Its post-hoc clamp has the same "floor, not a gauge" character the
  relaxed-Jacobi path just got fixed, and its intra-solve iterate has still
  never been instrumented. Unchanged from the previous session's item 3.
- **Part 2's wall-adjacent density gap** is still open and still untouched
  by this. The previous session's item 4 (re-run the wall-distance profile
  once a fix lands, to see whether it narrows) is now answerable, but the
  expected answer is "no": the fix does not apply to wall-bounded solves at
  all, by construction.

## Part 5 — bounded DFSPH stability: the wall cannot stop a particle that crosses a full spacing in one step (2026-08-27)

Picked up from Part 4's leftover item. `randomFlowIncompressible --bounded`
at nx=128 reaches NaN at t=5.54 under the untouched default — past the t≈1.5
Part 2 ever validated, so new ground rather than a regression, and reproduced
on the committed tree, so not the Part 3 velocity-resample fix either. It is
also *not* the Part 4 mechanism: that fix declines to act on a wall-bounded
solve by construction, and the run is byte-identical with and without it.

**What actually happens: fluid leaks into the wall and piles up there.**
Instrumenting every step with each particle's distance to the wall
(`scripts/probe_boundedIncompressibleBlowup.py`, new; `domainBoundarySdf`,
the same measure Part 2's wall profile bins by) shows a monotone accumulation
of fluid particles *inside* the boundary band, from the very first steps:

| step | t | fluid particles past the wall | mean&#124;rho-1&#124; within 2dx of wall | in the bulk |
|---|---|---|---|---|
| 26 | 0.47 | 233 | 0.070 | 0.0036 |
| 101 | 1.91 | 496 | 0.133 | 0.0068 |
| 201 | 4.17 | 963 | 0.148 | 0.0086 |
| 256 | 5.51 | 1487 | 0.198 | 0.0144 |
| 257 | 5.54 | 4506 | 0.297 | 0.113 |

The error is wall-localized by more than an order of magnitude throughout,
and the count never recovers — it only grows. The wall-depth profile at the
last pre-blowup step makes the shape of it plain: the bulk is fine
(`mean rho = 1.011` beyond 4 spacings from the wall) while the fluid jammed
into the wall band sits at **`rho = 1.30 - 1.36`**, 30%+ over-dense, peaking
at 1.66. The final NaN is that pile letting go in a single step.

**Not the boundary treatment, and not the shift.** Three candidates tested
and all rejected:

- **`BoundaryPressureMode`**: all three of `plain`, `mdbcDensity` and
  `mdbcMlsPressure` diverge at t = 5.40 / 5.54 / 4.83. The mode changes
  nothing that matters here.
- **`mdbcNoPenetrationShift`**: on (default) gives 349 penetrating particles
  at t=1.0, off gives 437. It helps ~20% — a real effect, nowhere near
  enough, and consistent with Part 2's own suspicion that it is a crutch.
- **Capping the implicit shift.** The shift `dx = dt**2 * a_p` is unbounded
  in this solver and reaches ~6 particle spacings per step, which looked like
  the obvious culprit. It is not: capping it at 0.25 spacings/step still
  diverges (t=5.28), and capping it at 0.05 diverges *much earlier*
  (t=1.45). The large shift is doing real work; throttling it just leaves
  the density error uncorrected. Worth recording as a negative result,
  because the shift's size is genuinely alarming and the natural next move.

**The deltaSPH control settles what kind of problem this is.** The
weakly-compressible sibling (`randomFlow --bounded`) on the *same geometry*,
run to matched time: **zero** particles ever enter the wall band, and its
wall-adjacent density (`1.0012`, `mean|rho-1| = 1.2e-3`) is identical to its
bulk — no near-wall band at all. So the geometry and the boundary sampling
are fine; something about how DFSPH runs against this wall is not.

**It is the timestep, and the threshold is one particle spacing.** deltaSPH
runs this case at `dt = 2.5e-4` (acoustic CFL); DFSPH runs it at `dt ≈ 0.02`,
~80x larger, which is the entire point of an implicit scheme. Sweeping
`cflFactor` and reading the result in the unit that matters — how far the
fastest particle travels per step, which is `cflFactor * h = cflFactor * 4dx`
for this case's `n_h = 4`:

| `cflFactor` | advective step | outcome | penetrating particles | near-wall &#124;rho-1&#124; |
|---|---|---|---|---|
| 0.3 (default) | 1.2 spacings | **NaN at t=5.54** | 4506, growing | 0.30 |
| 0.25 | 1.0 spacings | **NaN at t=5.09** | 15994, growing | 0.41 |
| 0.125 | 0.5 spacings | survives to t=8.0 | 653, steady | 0.040 |
| 0.05 | 0.2 spacings | survives to t=8.0 | 201, *declining* | 0.022 |

The transition is sharp and sits between half a spacing and a whole one.
Below it the penetration count reaches a steady state (and at 0.05 actively
recovers, 255 -> 224 -> 201); above it the count grows without bound until
the pile detonates.

**Why one spacing is the right threshold, physically.** The wall's force on
a fluid particle is mediated by the boundary particles' kernel contributions,
computed from the configuration at the *start* of the step. The distance over
which that contribution changes from "no wall here" to "fully inside the wall"
is one particle spacing. A particle allowed to cross a full spacing per step
can therefore traverse the entire first boundary-particle layer before the
wall has ever exerted a force on it — the wall is not weak, it is *late*.
That also explains why the near-wall error scales roughly first-order in dt
(0.096 -> 0.017 -> 0.0071 across the sweep), why the boundary *pressure* mode
is irrelevant (the problem is not what value the wall holds, it is that the
particle is already past it), and why capping the shift alone does not help
(advection at 1.2 spacings/step is over the threshold on its own, before the
shift adds anything).

Note the two contributions stack: at the default CFL, advection moves the
fastest particle 1.2 spacings and the shift adds up to ~6 more. At
`cflFactor = 0.05` advection gives 0.2 and the shift falls to ~0.08-0.19,
since the shift scales as `dt**2`. Any fix has to bound the *total*
displacement, which is why capping either one alone failed.

**Recommended fix, not implemented — it is a cost decision, not just a
correctness one.** The principled version is a wall-aware timestep
constraint: when the solve has boundary particles, additionally limit `dt`
so the per-step displacement (advective *and* shift) stays under ~half a
particle spacing. Structurally this mirrors the Part 4 gauge fix — scoped to
the case class that needs it, a no-op on periodic cases, which have no
boundary particles and are already stable at the default CFL. The cost is
real: it is a ~2.4x step-count increase on bounded DFSPH runs
(`cflFactor` 0.3 -> 0.125), which is a deliberate trade of throughput for
not diverging, and the sort of default the owner should choose rather than
inherit. The natural home is `kolmogorovIncompressibleTimestep`
(`cases/kolmogorovIncompressible.py`), which `randomFlowIncompressible`
already borrows as its `timestep` hook.

Two caveats worth carrying into that work:

- The threshold was measured in particle spacings, but `n_h = 4` was fixed
  throughout, so "half a spacing" and "one eighth of `h`" are not
  distinguished by this data. Which one governs matters for the constraint's
  form, and a single sweep at a different `n_h` would settle it.
- Even at `cflFactor = 0.125` the steady state still holds ~650 particles
  inside the wall band. Stable is not the same as correct: the wall still
  leaks, it just leaks into an equilibrium instead of an avalanche. Part 2's
  still-open wall-adjacent density gap is very likely the same phenomenon
  measured at a time when it had not yet run away.

### Part 5 continued — what actually helps: the missing velocity-level density correction (2026-08-27)

Follow-up per project owner: try different points in the step at which the
MLS boundary estimate is taken, and different solver options. Both were
tested and neither moves the needle; a third thing, found while looking,
does.

**MLS timing: no effect.** As shipped, `computeMdbcPressure` runs *after*
`solveDivergenceFree`, so the boundary pressure the solve reads is a full
step stale — computed from the previous step's fluid pressure at the
previous step's positions. That is exactly the kind of lateness Part 5's
diagnosis points at, so it was the obvious thing to try. Running it before
the solve as well (`--mlsBeforeSolve`) changes essentially nothing: 232
penetrating particles at t=1.0 against 228 for the shipped ordering. The
staleness of the boundary pressure is not what lets particles through.

Also re-confirmed while there: `mdbcPressureRelaxation = 1.0` (undamped)
NaNs within 7-8 steps, with or without the earlier projection — Part 2's
damping is still load-bearing, and its 0.3 default is not obviously
improvable from this direction.

Worth recording: `mdbcMlsPressure` does have visibly better *early* wall
behavior than `mdbcDensity` (228 penetrating particles at t=1.0 against
349) yet still dies soonest of the three modes (t=4.83). Better early wall
behavior does not translate into surviving longer, which is another way of
saying the boundary pressure treatment is not the binding constraint.

**Solver options: marginal.** Quadrupling the divergence-free solver's
iteration cap (32 -> 128, and it does peg at the cap every step) buys ~10%:
312 penetrating particles against 349. `JacobiRelaxationMode.optimal` buys
nothing (353). Both together, 329. Consistent with the Part 5 reading: the
projection is not failing because it is under-converged, it is failing
because it is computed before the particle gets to the wall.

**Found while reading `finalize`: two real inconsistencies, neither of which
turned out to matter for this.** Recorded so nobody re-derives them.
`dfsph_step` gives boundary particles an mDBC-extrapolated density for the
divergence-free solve, but `finalize` recomputes *plain* summation densities
and never re-applies mDBC, so the shifting solve sees boundary rows whose
density is systematically low (truncated outward support). Applying mDBC
there too changes nothing measurable (353 penetrating against 349), but the
inconsistency is real. Separately, `systems/incompressible.py:235` assigns
`self.state.surfaceIndicator` where the declared field is
`surfaceIndicators` (plural) — so `finalize`'s own `detectFreeSurface` result
is written to a typo'd attribute and discarded, making that call dead weight
apart from the assignment. Neither is fixed here.

**What does help: the scheme has no velocity-level response to a density
error at all.** DFSPH proper runs two solves per step and applies both to
the velocity — a divergence-free projection and a constant-density
correction. This scheme applies only the first to velocity and repurposes
the second as a one-shot *position* shift. In a periodic domain that is
fine. Against a wall it is not, because `div v = 0` prevents *further*
compression but never undoes compression that already exists, so the only
mechanism that can relieve a wall-adjacent pile-up is moving particles —
and near a wall that is precisely what pushes them through it.

Tested by additionally applying the constant-density solution as a velocity
correction (`v += dt * a_p`), the new `ShiftApplication.positionAndVelocity`.
Matched-`dt` comparison (dt pinned at 0.015, since the variants change the
velocity field and therefore the CFL-derived `dt` — an unmatched comparison
flatters this option by ~30%):

| application | penetrating | near-wall &#124;rho-1&#124; | bulk | rho_max | shift (spacings) |
|---|---|---|---|---|---|
| `positionShift` | 308 | 7.50e-2 | 3.0e-3 | 1.227 | 1.24 |
| velocity only | 68 | 8.38e-2 | 4.7e-2 | 1.206 | 1.02 |
| `positionAndVelocity` | **94** | **2.17e-2** | **1.4e-3** | **1.067** | **0.045** |

Velocity-only is not an option — its bulk error is 16x worse (`rho_min`
0.475). Both together is a large improvement on every measure, and the
position shift's own magnitude collapses by ~28x: the velocity correction
relieves compression continuously, so the shift has almost nothing left to
do, which is the whole mechanism by which the wall stops being breached.

**At the default CFL it turns the divergence into a steady state.** 387
steps to t=8.0 with no timestep penalty at all — against 1352 steps for the
`cflFactor=0.125` workaround, and against NaN at t=5.54 for the shipped
default. Penetration rises early then plateaus around 250 and declines;
`rho_max` sits at 1.147 for the whole run instead of climbing past 1.6;
near-wall `mean|rho-1|` peaks at 0.048 and falls back to 0.033, against
0.30 at the baseline's death.

**But it is dissipative, and `tgv` is where that shows.** On the one case
here with an analytic reference, `positionAndVelocity` drives the
kinetic-energy decay rate to **1.93x the analytic rate** (against 0.59x,
the value `tests/test_physics.py::test_tgvKineticEnergyDecaysAtRoughlyThe
AnalyticRate` documents and asserts) and makes the decay **non-monotone**,
which that file's other TGV test asserts against. The correction is a
velocity the divergence-free projection never asked for, and it damps. On
periodic `kolmogorovIncompressible` it is a wash — mean density band ~30%
better, worst-case excursion ~3.6x worse.

So it ships **opt-in**, unlike Part 4's gauge: that one was free everywhere
it applied, this one is a measured trade. `ShiftApplication.positionShift`
remains the default. Suite (241 passed) and gradcheck pass unchanged, since
the default path is untouched.

**The obvious refinement, not done: apply the velocity correction only near
walls.** It is needed only where the position shift cannot act freely, and
restricting it there would leave the bulk untouched — making it a no-op on
periodic cases by construction, exactly the way Part 4's gauge scopes
itself, and very likely removing the `tgv` regression entirely since TGV has
no walls at all. The blocker is mechanical rather than conceptual:
`finalize` has no wall-distance measure to hand (the SDF lives on the case,
not the state), so this needs either a boundary-proximity field on the state
or a kernel-sum over `kinds == 1` neighbours. A smooth taper rather than a
hard mask, to avoid injecting a discontinuity into the velocity field.

### Part 5 continued (2) — the velocity correction's real cost, and why the scheme uses a position shift at all (2026-08-27)

Two refinements to `positionAndVelocity` were tried. The first failed, the
second is a clear improvement, and together they explain what the trade
actually is.

**Confining the correction to a wall band: tested, does not work.** The plan
above predicted this would "keep the benefit and remove the cost", since the
correction is only *needed* where the position shift cannot act freely.
Implemented against an SPH interpolation of the boundary indicator
(`kinds != 0`), which is a clean proximity measure — measured on this case it
is 0.32 within a particle spacing of the wall, 0.07 at one to two, and
**exactly zero beyond three**, so a taper on it vanishes identically in the
bulk and on any periodic case. It diverges at **t=4.17**, against t=8.0 for
the unscoped correction and t=5.54 for the shipped default. Widening the band
makes it worse still (t=2.74). The prediction was wrong, and the reason is
visible in the matched-`dt` table above: the unscoped correction improves the
*bulk* density error 2x as well (1.4e-3 against 3.0e-3). Its value is not
wall-local — a better-conditioned bulk is what leaves the wall region less to
absorb — so confining it to a thin shell throws the mechanism away and injects
a velocity discontinuity at the shell edge on top. The mode was implemented,
measured, and removed rather than left in the config as a knob that is worse
than both of its neighbours.

**Scaling it down: no sweet spot.** The bounded case is stable at quarter
strength (t=8.0 at lambda=0.25), but `tgv`'s decay rate is already 1.4x
analytic and non-monotone there, against 0.55x for the default. The two
requirements do not overlap anywhere.

**Applying it where DFSPH proper does: a large improvement, and it isolates
the real cost.** The correction was being added in `finalize`, *after* the
integrator. DFSPH proper computes it inside the step and folds it into the
`dvdt` the integrator advects with — and drops the position shift entirely,
two velocity-level solves per step and no repositioning. That is the new
`ShiftApplication.inStepVelocity`, and on the bounded case it is the best
of everything tried:

| mode | near-wall &#124;rho-1&#124; | bulk | penetrating | rho range | outcome |
|---|---|---|---|---|---|
| `positionShift` (default) | 0.30 | 0.113 | 4506 | [0.139, 2.452] | **NaN t=5.54** |
| `positionAndVelocity` | 3.3e-2 | 1.2e-3 | 239 | [0.936, 1.147] | t=8.0 |
| `inStepVelocity` | **9.7e-3** | **6.6e-4** | **63** | **[0.986, 1.140]** | t=8.0 |

Keeping the position shift *as well* as the in-step correction is a trap
worth naming: it corrects the same density error twice per step, and on `tgv`
that **injects** energy rather than removing it — kinetic energy grows 6.6x
over 200 steps. `finalize` therefore skips the shift in this mode.

**And the cost is not a placement artifact — it is Part 4's unreachable
setpoint, resurfacing in the momentum equation.** Both velocity modes damp
`tgv` at essentially the same rate (3.28x and 3.26x analytic, against 0.55x
for the default, both non-monotone). Moving the correction to the
theoretically correct place did not help at all, which rules out "the
divergence-free projection never gets to clean it up" as the explanation.
The actual reason is the one Part 4 established: the SPH summation density's
particle average *cannot* equal `rho0` for a disordered configuration, so the
constant-density solve never converges and always carries a residual. Feed
that permanent residual into the **momentum** equation and it is a permanent
unphysical forcing. Apply the identical residual as a **position shift** and
it is momentum-neutral — it only reorganises particles.

That is the real answer to "why does this scheme use a position shift at all
when DFSPH proper uses a velocity correction", and it reframes the whole
trade: the position shift is not a shortcut, it is the formulation that is
robust to an unreachable setpoint. Its weakness is purely that it cannot act
near a wall without pushing particles through one.

**The experiment this points at, not done:** drive the *velocity* correction
from the attainable part of the source only — project out the structurally
unreachable mean Part 4 measured (`probe_densityBiasVsDisorder.py`) — while
leaving the position shift on the raw source, where Part 4 showed the mean
carries the genuine de-clumping signal and must be kept. If the `tgv`
dissipation is the unreachable mean being integrated into momentum, that
projection should remove it and leave the bounded case's gain intact. It is a
single flag on `solveIncompressible`'s source term, and it is the one
experiment that could make a velocity mode safe enough to be the default.

Suite (241 passed) and gradcheck pass; the default path is untouched.

## Part 6 — what this scheme actually is, and whether "real DFSPH" should be a separate scheme (2026-08-27)

Raised by the project owner off the back of Part 5: given that the velocity
formulation turns out to be the one DFSPH proper uses, should the current
scheme be moved to a new name and an actual DFSPH — position-based boundaries
with MLS pressure — take the `dfsph` name? Recorded here with the
measurements that bear on it. **No restructuring done**: the owner wants a
literature session on DFSPH and its derivatives first, to establish what
belongs where. The questions that session should answer are listed at the end,
each one tied to something measured here rather than to general curiosity.

### The misnomer is real, but narrower than it looks

`schemes/dfsph.py::dfsph_step` implements a divergence-free velocity
projection plus a position-based implicit particle shift
(`solveIncompressible` applied in `finalize` as `dx = dt**2 * a_p`). That is
not DFSPH, which runs two solves per step and applies **both** to the
velocity. But the *registered* name is already
`IncompressibleSPHScheme.divergenceFree`, and that is accurate for what the
code does. So the misnomer lives only in the file and function names, not in
the user-facing scheme name. Renaming `dfsph.py`/`dfsph_step` to match the
registered name is a zero-risk correction and is worth doing independently of
anything below.

### "Actual DFSPH" is no longer hypothetical — it is `ShiftApplication.inStepVelocity`

Part 5 continued (2) implemented it: constant-density solve inside the step,
folded into the `dvdt` the integrator advects with, position shift dropped.
That is the formulation. So the question "what would registering a real
`dfsph` scheme ship?" has measured answers rather than expected ones:

- **Walls: much better.** near-wall `mean|rho-1|` 9.7e-3 against 0.30 at the
  current default's death, 63 particles ever inside the boundary band against
  4506, `rho` held in [0.986, 1.140], stable to t=8.0.
- **Bulk: worse, on the one case with a reference solution.** `tgv`'s
  kinetic-energy decay runs at 3.3x the analytic rate and is non-monotone,
  against 0.55x and monotone for the position shift.

That second line is the whole difficulty with claiming the `dfsph` name for
it today: it would be a scheme named after the canonical method that fails the
canonical test case. And Part 5 continued (2) established the cause is not
placement or tuning — it is Part 4's unreachable setpoint being integrated
into the momentum equation, which no amount of renaming touches.

### The specific pairing proposed — DFSPH + MLS-pressure boundaries — is currently the unstable one

Untested until now. Under `inStepVelocity`, at nx=128 to t=8.0:

| boundary mode | outcome | near-wall &#124;rho-1&#124; | penetrating |
|---|---|---|---|
| `mdbcDensity` | t=8.0 | 9.7e-3 | 63 |
| `plain` | t=8.0 | 4.6e-2 | 268 |
| `mdbcMlsPressure` | **NaN at t=0.21** | — | — |

Extra damping only delays it — `mdbcPressureRelaxation` 0.1 gives t=0.71 and
0.03 (10x the default) gives t=3.09 — so this is a genuine instability, not
under-damping.

**This inverts Part 2's ranking**, where `mdbcMlsPressure` was found to be the
most stable *and* most accurate of the three modes. It was, under the
position-shift formulation. Under the velocity formulation it is the only one
that fails. So the boundary treatment's ranking is not a property of the
boundary treatment alone — it depends on which formulation it is coupled to,
and any conclusion carried over from Part 2 has to be re-derived rather than
assumed.

### Recommended sequencing (not acted on)

1. Rename the file/function to match the already-correct registered scheme
   name. Free, no behaviour change.
2. Gate a second registered scheme on the source-projection experiment
   ("Next session" item 1). If projecting the structurally-unreachable mean
   out of the *velocity* path removes the `tgv` dissipation, then `dfsph`
   becomes a scheme that is better at walls and correct in the bulk, and
   registering it needs no argument. If it does not, registering it adds a
   scheme that is strictly worse on the reference case.
3. When it is registered: two `SchemeBundle`s sharing **one** step function,
   differing in their defaults (`shiftApplication`, boundary mode, likely
   `cflFactor`), rather than a forked `dfsph.py`. The two paths are ~90%
   common code and the real divergence is in defaults, which is what the
   bundle is for.

### Questions for the literature session

Each of these is a place where this codebase's measured behaviour needs an
answer from the published methods, not a general reading topic.

- **The unreachable setpoint.** Part 4 established that the SPH summation
  density's particle average is minimised by the lattice and rises
  quadratically with disorder, so `mean_i rho_i == rho0` is unattainable and
  the constant-density solve carries a permanent residual. How do published
  DFSPH implementations avoid integrating that residual into momentum? Warm
  starting, a convergence tolerance deliberately looser than the structural
  bias, one-sided source clamping (correct compression only, never
  expansion), rest-density calibration against the sampled configuration, or
  something else? This is the single question that decides whether a velocity
  formulation can be default-safe here.
- **Which density.** Is the constant-density solve driven by summation
  density or by an advected/integrated density in the published method? The
  structural bias argument applies to the first and not obviously to the
  second, and this codebase has an `integrateRho` switch (currently
  defaulting off, and Part 3 found a dead branch in it).
- **Shifting versus the constant-density solve.** Which derivatives keep a
  particle-shifting term *alongside* DFSPH's two velocity solves, and how do
  they sequence it? Measured here: doing both at full strength corrects the
  same density error twice per step and injects energy into `tgv` (kinetic
  energy grows 6.6x over 200 steps).
- **Boundary coupling.** Which boundary treatments do published DFSPH
  implementations actually pair with — mDBC, Adami-style pressure
  extrapolation, MLS projection, position-based/rigid-body coupling? Is the
  boundary-pressure/fluid-pressure feedback loop this codebase damps with
  `mdbcPressureRelaxation` a known failure mode, and what is the standard
  formulation that avoids it rather than damping it?
- **Reference values.** What decay rate do DFSPH papers report on Taylor-Green,
  and is over-dissipation a known artifact of velocity-level density
  correction? `tests/test_physics.py` documents this codebase's 0.55x as the
  Monaghan viscosity switch rather than an error, so the comparison needs the
  published number to be meaningful.
- **Timestep near boundaries.** Part 5 measured a sharp stability threshold at
  ~one particle spacing of per-step displacement, with the default
  `cflFactor=0.3` sitting just past it at 1.2 spacings. Do published DFSPH
  implementations carry an explicit wall-proximity timestep constraint, or
  does their boundary treatment remove the need for one?
- **Background pressure.** Part 4 found the constant pressure mode is not
  forceless where kernel support is truncated, and that the drifting constant
  was acting as a de-clumping background pressure. How does this relate to
  transport-velocity formulations, which add such a background pressure
  deliberately?


---

## Part 7 — the literature session (2026-08-27)

Ran against two papers supplied as full text, plus the code each question is
tied to. **Read this section before acting on Part 6's naming proposal or on
"Next session" item 1** — it changes the recommended order of both.

**Sources actually available.**

- **[C]** Cornelis, Bender, Gissler, Ihmsen, Teschner, *An Optimized Source
  Term Formulation For Incompressible SPH* (TVCJ 2018/19). The VD+PS paper.
- **[B]** Band, Gissler, Peer, Teschner, *MLS pressure boundaries for
  divergence-free and viscous SPH fluids* (C&G 76, 2018). Boundary handling
  for a DFSPH framework; its Algorithm 1 is the clearest published statement
  of the DFSPH step available here.

**Not available, and every claim resting on them is marked "[unverified]".**
Bender & Koschier 2017 (DFSPH proper), Band et al. 2018a (Pressure
Boundaries), Adami et al. 2012 (wall BC), Akinci et al. 2012 (rigid-fluid
coupling), Ihmsen et al. 2010 (adaptive timestep), Adami et al. 2013
(transport velocity). [B] describes the first four second-hand and that is
what is used below.

### The headline: this scheme is not a mis-named DFSPH, it is VD+PS

Part 6 framed the scheme as "DFSPH with the second solve repurposed as a
position shift". Against [C] that framing is backwards. What
`dfsph_step` + `IncompressibleSystem.finalize` implement is [C]'s VD+PS,
step for step and in the right order:

| [C] | this codebase |
|---|---|
| `v* = v + dt a_nonp` (Eq. 2) | `dvdt` assembled in `schemes/dfsph.py` |
| `dt grad^2 p* = rho0 div v*` (Eq. 12) | `solveDivergenceFree`, source `-divergence` |
| `v' = v* - dt grad p*/rho0` (Eq. 13) | `dvdt_pressure` |
| `x** = x + dt v'` (Eq. 14) | integrator |
| `dt grad^2 p** = (rho0 - rho**)/dt` (Eq. 15) | `solveIncompressible`, source `rho0 - rhoStar` |
| `x(t+dt) = x** - dt^2 grad p**/rho0` (Eq. 16) | `dx = dt**2 * dvdt_incomp`, `positions += dx` |
| `v(t+dt) = v' + grad v' . (x(t+dt) - x**)` (Eq. 17) | `proj_vel` einsum (fixed in Part 3) |

Including the detail that [C] Eq. 19 evaluates `grad v'` on the *current*
neighbourhood rather than the shifted one, justified in the paper purely on
cost — which is what `finalize` does.

Three consequences, all of which cut against Part 6's proposal:

1. **The position shift is not a deviation to be corrected. It is the
   method.** [C]'s entire contribution is *not* applying the density-invariant
   solve to velocity. So the default `ShiftApplication.positionShift` is
   paper-faithful, and `positionAndVelocity` / `inStepVelocity` are the
   departures.
2. **The `tgv` over-dissipation those two modes show is the published,
   expected behaviour of a velocity-applied DI solve**, not a defect of this
   implementation. [C]'s abstract: "the DI source term suffers from
   significant artificial viscosity". Its Fig. 3 measures it — the DI
   variant's shear-wave amplitude decays from ~0.95 to ~0.1 by t≈15 s while
   VD+PS still holds ~0.75 at t=40 s. Part 5's measured 3.3x is that effect.
3. **The naming question has a cleaner answer than either option in Part 6.**
   The file is not a broken `dfsph.py`; it is `vdps.py` (or
   `divergenceFree.py`, matching the already-correct registered scheme name).
   A real `dfsph` scheme would be a *different* method, not a rescue of this
   one — and per Q1 below it would be the method [C] was written to replace.

### Q1 — the unreachable setpoint

> **Superseded in part by Part 8.** The reasoning below about *why published
> solvers terminate* (a tolerance above the structural floor) is wrong:
> Bender & Koschier run a **tighter** tolerance than this codebase and still
> converge in 4.5 iterations. The real answer is that this codebase pins
> `rho0` to the lattice minimum in two case files. See Part 8. Everything
> else in this section — that the paper's PS solve carries the same bias, and
> that it is spent on positions rather than momentum — stands.

**[C]'s PS solve has exactly the same structural bias, and does not avoid it.
It avoids *integrating* it.** Its second PPE (Eq. 15) is driven by
`(rho0 - rho**)/dt` with `rho** = rho(t) - dt rho0 div v'` — a summation
density, same as here, so Part 4's Parseval argument applies to the paper
verbatim. What [C] does with the result is the whole trick (§6, "Relation of
VD+PS to existing PPE solvers with DI source term"): "we do not update the
velocities using the solution of the PPE solver with the DI source term …
this velocity update does not constitute a change in the velocity field.
Instead, it is just a resampling." The permanent residual is spent on
positions, where it is momentum-neutral. That is `positionShift`.

**Why their solve nonetheless terminates, and this one does not.** [C] §4:
the DI/PS stopping criterion is an average density error of **0.1%**, the VD
residual is 0.01, `omega = 0.5`, and the DI solver is warm-started at
`p_i(t+dt) = 0.5 p_i(t)`. [B] §6: no warm start at all, pressures initialised
to zero, `omega = 0.5`. Against this codebase:

| | [C] | here |
|---|---|---|
| PS/DI tolerance | 1e-3 (0.1% avg density error) | **5e-4** (`buildDefaultPSConfig`) |
| PS warm start | `0.5 p(t)` | none (`incompressible.py:167`, `pressures.clone() * 0.`) |
| omega | 0.5 | 0.3 (and Part 1 measured the window as `omega < 0.355`) |
| max iterations | not stated; Fig. 10 shows PS iterations falling to ~0 | 64, **pegged there from step ~30 onward** (Part 4) |

Now combine that with this document's own jitter table (Part 4): the bias
grows as the *square* of the disorder — `mean(rho-1)` is 8.5e-4 at
`jitter = 0.1 dx` and 3.3e-3 at `0.2 dx`. So:

- At [C]'s sampling quality (Fig. 4: max density flat at ~1000 for 40 s, i.e.
  a configuration the shift keeps near-lattice), the structural floor sits
  **just under** their 1e-3 tolerance. Their solver converges — Fig. 10 shows
  PS iterations dropping to ~0 after the initial transient while VD stays at
  10–20 — because the setpoint is unreachable by *less than the tolerance*.
- Here the developed flow's floor is 2.7e-3 (Part 4), **5.4x above** this
  codebase's own 5e-4 tolerance. The solve cannot terminate, so it runs 64
  wind-up iterations every step, forever.

**So the answer to Part 6's question is not one of the exotic options it
listed.** Neither paper uses one-sided source clamping, rest-density
recalibration against the sampled configuration (except at boundaries, see
Q2), or a special treatment of the mean for fluid particles. The published
answer is three ordinary things at once: (i) never put the DI residual in
momentum, (ii) set the tolerance above the structural floor, (iii) keep the
sampling ordered enough that the floor stays under the tolerance.

**This retargets "Next session" item 1.** Projecting the unreachable mean out
of the velocity path is a fix for a formulation the literature says not to
use. The cheaper and more paper-faithful experiment, which item 1 never
considered, is: **raise `pressureSolver.tolerance` from 5e-4 to at least the
measured structural bias (~3e-3) and see whether the wind-up simply stops.**
Prediction: `nIter` comes off the 64 cap, `pMean` stops climbing, and the
Part 4 blowup does not need a gauge at all. `ShiftPressureGauge.minShift`
would then be defending against a self-inflicted wound. That is one number,
and it is testable with the existing `probe_shiftPressureGauge.py` /
`probe_incompressibleGaugeDrift.py` pair (the latter needs a `--tolerance`
knob added; it has `--maxIters` but not that).

### Q2 — which density

Summation density, in both papers, with a one-step divergence prediction on
top: [C] Eq. 11 `rho* = rho(t) - dt rho0 div v*`; [B] Algorithm 1's density
source is the relative form `(1/dt^2)(1 - rho0/rho_f(t)) - (1/dt) div v**`.
Neither integrates density. **`integrateRho = False` is therefore the
paper-faithful default, and Part 3's dead `True` branch is a cleanup item,
not a physics decision.**

One difference, and it favours this codebase: `finalize` recomputes the
summation density at the *shifted* positions `x**` before the solve, then
adds `dt * drhodt` (`incompressible.py:79`). [C] predicts `rho**` from the
density at the *old* positions instead. This codebase's is the more accurate
of the two and the extra term is small once `div v' ~ 0`. Not a bug; noted so
it is not "fixed" toward the paper later.

**Rest-quantity calibration against the sampled configuration is real but
boundary-only.** [B] Eqs. 6–7 replace the boundary density ratio with a
volume ratio `rho0_b/rho_b = V_b/V_b^0`, with the rest volume
`V_b^0 = gamma / sum_bb W_bb_b` computed *from the actual boundary sampling*
and two hand-set coefficients `gamma`, `beta` for incomplete neighbourhoods.
That is Part 6's "rest-density calibration" hypothesis — but it is published
only for boundary particles, only in the Pressure Boundaries variant, and [B]
§6.1 blames `beta` (which assumes a planar boundary) for that variant's worst
error in the rotating-sphere test. Not a lead worth pulling for fluid rows.

### Q3 — shifting versus the constant-density solve

**No published variant here does both at full strength, and that is the
point.** [C] §1 explicitly reviews prior two-PPE combinations — its refs
[5] (Bender & Koschier, DFSPH), [18] (Hu & Adams), [23] (Kang & Sagong) —
and says they "typically result in inconsistent particle positions and
velocities, i.e. the particles are not advected with their velocity", then
offers VD+PS as the resolution. The shift *replaces* the DI velocity
application; it is never stacked on top of it.

So this codebase's measured "doing both injects energy into `tgv` (KE grows
6.6x over 200 steps)" is the expected outcome of correcting the same density
error twice per step, and `finalize`'s guard that drops the position shift
under `inStepVelocity` is correct.

Also relevant to a rejected Part 5 experiment: [C] §6 contrasts its global
PPE-based shift with the local, concentration-gradient shifting variants
(its refs [30] Nestor, [35] Skillen, [41] Xu), whose defining weakness is a
user-tuned shift magnitude — "if this user-defined parameter is too small,
the resulting sampling quality is not as good as it could be … if the
parameter is too large, over-correction occurs which can result in even worse
sampling qualities after the particle shift." Part 5's `--shiftCap` is that
parameter, reintroduced. The paper predicts it degrades sampling in both
directions, which is what Part 5 measured ("capping it makes things worse").
That negative result is now explained rather than just recorded.

### Q4 — boundary coupling

**What published DFSPH-family work pairs with.** [B] §3 enumerates four
treatments — Akinci mirroring, Adami SPH extrapolation (Eq. 3, which carries
an explicit hydrostatic `g . sum rho x W` term), Band Pressure Boundaries (a
PPE with boundary unknowns), and its own MLS extrapolation (Eqs. 19–21) — and
recommends the last. [C]'s VD+PS ships with Akinci one-layer boundaries
(§4, refs [3, 20]) and deliberately runs its headline test on a *periodic*
domain "such that boundary handling does not influence the solution".

**So there is no published VD+PS x mDBC/MLS pairing.** This codebase's
combination is novel, which is the context Part 6's inverted boundary-mode
ranking belongs in: nothing in the literature licenses carrying a boundary
ranking across formulations.

**Yes — the feedback loop is a known failure mode, and it is known as the
reason *not* to hold boundary pressure as state.** [B] §3.3: Band et al. [6]
"reported convergence issues in case of large volume ratios between fluid and
boundary particles", and handled it with a *separate* relaxation for boundary
rows, `omega_b = 0.5 V_b^0/h^3` against `omega_f = 0.5`. That is structurally
`mdbcPressureRelaxation = 0.3`. And [B] §5 lists as an advantage of MLS over
Pressure Boundaries that "it does not depend on a relaxation factor `omega_b`
for boundary particles."

**The formulation that avoids it rather than damping it is a placement
change.** [B] Algorithm 1 recomputes boundary pressure **inside every solver
iteration**, from the current iterate's fluid pressures:

```
while not converged do
    for all boundary particle b do   compute pressure p_b using MLS   (Eq. 21)
    for all fluid particle f    do   compute pressure acceleration a_f (Eq. 1)
    for all fluid particle f    do   p_f <- p_f + (omega/lambda_f)(s_f - div a_f)
```

with pressures initialised to zero each step (§6, no warm start). Under that
placement `p_b` is a pure *function* of `p_f` — no state, no lag, no
autonomous dynamics — and the composition is just part of the Jacobi
operator. [B] explicitly contrasts it with Pressure Boundaries, where
"pressure at the boundary is not re-computed in each iteration, but updated
with a Jacobi step (Eq. 8)" — i.e. carried as state, which is the variant
that needed `omega_b`.

This codebase runs `computeMdbcPressure` **once per step, after
`solveDivergenceFree`** (`schemes/dfsph.py`), carrying `p_b` across steps
under-relaxed. That is the Pressure-Boundaries-style stateful update, and it
reproduced its documented failure exactly: Part 2 traced boundary pressure
roughly doubling per step to NaN, and needed a relaxation factor to survive.

**This is not the change Part 5 tested and rejected.** `--mlsBeforeSolve`
moved the projection *earlier in the step* but left it outside the iteration
— it changed the lag, not the statefulness — and correctly found nothing
("232 penetrating particles against 228"). Moving it *inside* the Jacobi loop
is a different change and remains untried. [B] Table 1's timings say the cost
is affordable: MLS boundary pressure is 1.87 ms per iteration against 1.12 ms
for Adami extrapolation and 1.95 ms for a full boundary PPE.

**One more guard this codebase lacks.** [B] solves the MLS gradient system
(Eq. 20) with **SVD safe inversion**, because boundary particles whose fluid
neighbours are co-linear or co-planar give a singular 3x3. Its zeroth-order
term (Eq. 19, `alpha_b`, the Shepard-weighted average) is always well-posed;
only the gradient needs the guard. `modules/mdbc/pressure2025.py` falls back
on a *neighbour-count* threshold (9) and has no conditioning guard — and the
Part 2 blowup's worst offender had `numNeighbors = 22`, well past that cutoff,
with `|grad p| = 153`. A neighbour count does not detect a co-linear
neighbourhood. That is a concrete, cheap, paper-backed fix for the mode that
Part 6 found NaNs under `inStepVelocity` at t=0.21.

### Q5 — reference values

**Neither paper reports a Taylor-Green decay rate.** [B]'s quantitative
comparisons are density error (0.037% on the rotating sphere), iteration
counts and computation times (Table 1). So `tests/test_physics.py`'s 0.55x
still has no published counterpart, and the comparison Part 6 wanted cannot
be closed from these two.

**But [C] supplies the benchmark that was actually wanted** (§5.1, Figs. 2–4):
the **shear-wave decay** — a 2D square, periodic on all sides, sinusoidal
initial velocity `v_x = v0 sin(2 pi y / L)`, no gravity and no explicit
viscosity, so any amplitude decay is solver artifact. It grades the two
failure modes this document keeps conflating on *separate axes*:

- Fig. 3, sinus amplitude vs time — isolates artificial viscosity.
- Fig. 4, max density vs time — isolates disorder/volume error (VD climbs
  1000 → 1300; DI and VD+PS stay flat).

Published outcome: DI decays fastest, VD holds amplitude but loses sampling
quality, VD+PS holds both. **Over-dissipation from velocity-level density
correction is therefore a documented artifact, stated in [C]'s abstract.**

**Action: port the shear-wave case.** It is cheaper than `tgv`, periodic (so
boundary handling cannot contaminate it), has published curves to compare
against, and grades the three `ShiftApplication` modes on exactly the axis
they differ on. Part 3 already wanted it for the Eq. 17 resample fix, whose
effect it said "accumulates gradually in the velocity field, not in an
instantaneous density band"; it now has a second and stronger justification.
This is the highest-value new case to add.

### Q6 — timestep near boundaries

**Neither paper carries a wall-proximity timestep constraint.** [B] uses a
standard adaptive CFL (its ref [38], Ihmsen et al. 2010) and its answer to
bad wall behaviour is a better boundary pressure, not a smaller `dt` near the
wall — §6.6: under pressure mirroring the washing machine "requires a time
step that is half as large compared to our MLS extrapolation", a global 1.8x
speedup.

**The interesting finding is a unit mismatch, and it is arithmetic, not
opinion.** This codebase's advective limit is `dt = cflFactor * h / v_max`
(`cases/kolmogorovIncompressible.py:130`) where `h` is the **support radius**.
With the default `n_h = 4`, `n_h_to_nH` fixes `N_h = pi * n_h^2` in 2D, i.e.
`h = n_h * dx = 4 particle spacings`. So `cflFactor = 0.3` permits
**1.2 spacings of displacement per step** — which is exactly the 1.2 Part 5
measured independently. The published CFL for this family is stated in
particle *diameters*, not support radii, so it is sub-one-spacing by
construction. **[Verified in Part 8]** — Bender & Koschier §3.1 state it directly:
`dt <= 0.4 d/||v_max||`, `d` the particle diameter.

That much can be corroborated from [B] Table 2, which *is* attached, without
relying on the unverified constant. Its scenes: washing machine, 20 mm
spacing at 0.5 ms; vase, 20 mm at 0.47 ms; teacup, 3 mm at 0.28 ms; glasses,
1.5 mm at 0.16 ms. For any plausible velocity in those scenes (a 10 m vase
implies ~14 m/s at impact) the per-step displacement lands at roughly
**0.1–0.35 spacings**. No run in either paper goes near one spacing per step.

**This reframes Part 5's rejected wall-aware `dt` constraint.** Part 5 found
stability turning over between half a spacing and a whole one, and recorded
`cflFactor = 0.125` (= 0.5 spacings) as the working value at a "~2.4x
throughput cost". That threshold is not a peculiarity of this scheme — it is
where published practice already sits, and the 2.4x is measured against a
default that is 3–12x outside every published operating point. **The
constraint does not need to be wall-aware, and it does not need to be new:
express `cflFactor` against `dx` instead of `h`, or drop the default to
~0.1.** That is a smaller and better-supported change than either option
Part 5 costed.

### Q7 — background pressure

**Open; neither attached paper answers it.** Transport-velocity formulations
(Adami et al. 2013) are not among the sources and [B] cites Adami 2012 only
for its wall boundary condition. The two adjacent data points here:

- [B] Eq. 3 (Adami-style boundary extrapolation) carries an explicit
  hydrostatic term `g . sum_bf rho_bf x_bb_f W_bb_f` — a deliberately
  non-zero, geometry-derived boundary pressure offset, i.e. published
  precedent for a background pressure that is *set*, not allowed to drift.
- [C]'s PS solve is itself a de-clumping potential that is never applied to
  momentum — the closest published analogue to Part 4's finding that the
  drifting constant was acting as a de-clumping background pressure. The
  difference is that [C]'s is bounded because it converges (Q1).

Needs a transport-velocity paper to close.

### Discrepancies against the papers, ranked by expected impact

1. **MLS boundary pressure is computed outside the solver iteration** and
   carried as under-relaxed state across steps (Q4). [B] recomputes it inside
   every Jacobi sweep from the current iterate, which removes the feedback
   loop instead of damping it. Untested — and *not* what Part 5's
   `--mlsBeforeSolve` tested.
2. **The PS tolerance (5e-4) sits below the measured structural bias
   (2.7e-3)** (Q1), so the solve provably cannot terminate; [C] runs at 1e-3
   with a sampling whose floor is under it. One-number experiment.
3. **`cflFactor` is applied to `h`, not `dx`** (Q6), putting the default at
   1.2 spacings/step against ~0.1–0.35 in [B]'s own scenes.
4. **`computeAlpha`'s second sum includes boundary neighbours.** [B]
   Algorithm 1: `lambda_f = (||sum_j m_j grad W_fj||^2 + sum_ff m_f m_ff
   ||grad W_f ff||^2) / (-rho_f^2)` — the first sum runs over **all**
   neighbours, the second over **fluid neighbours only**. `wp_alpha.py:110-128`
   accumulates both over the same set, and `computeAlpha`'s wrapper takes
   `OperationProperties`' default `operationMode = AllToAll`, so boundary
   neighbours enter both. Near a wall that inflates `|alpha|`, and since the
   update is `p += omega * residual / alpha`, it **under-corrects pressure
   exactly where Part 5 found the correction failing.** Cheap to test and
   directly on Part 5's open item.
5. **`omega = 0.3` against both papers' 0.5**, with Part 1's measured
   stability window `omega < 0.355`. Both papers use relaxed Jacobi at 0.5
   without qualification. A solver whose window excludes the universal
   published value suggests the diagonal is scaled differently — item 4 is
   the first place to look, and there is a second candidate: this codebase's
   `alpha` carries an extra `1/rho_i` relative to [B]'s form (both agree at
   `rho = rho0 = 1`, which is why a normalised case would not show it).
   Hypothesis, not a finding.
6. **The Krylov path is applied to the clamped solve.** `solveIncompressible`
   routes through `solvePressureKrylov(..., gauge='nonnegative')`
   (`incompressible.py:97-101`) whenever `solverType != relaxedJacobi`. [B]
   §6.3 states the rule this breaks: PCG is used for the *divergence* solve
   "because we do not have to clamp pressure values in-between the iterations
   (unlike for the density-invariant error)". The codebase already enforces
   this for `JacobiRelaxationMode.optimal` (it raises, with that exact
   reasoning) but not for the Krylov path. **"Next session" item 3's concern
   is correct and now has a citation.** [B] Fig. 5 also quantifies what is
   being given up on the solve where PCG *is* valid: peak iteration counts
   ~140 (Jacobi) against ~10 (PCG).

### What this changes, in order

> **Superseded by Part 8's revised list.** Items 2–6 below survive; item 1
> (the PS tolerance) is demoted behind the setpoint fix, and a diagonal
> measurement and a solver-reordering experiment come ahead of both.

1. **PS tolerance vs the structural bias** (Q1, discrepancy 2). One number,
   answers Part 4's residual question and may retire the gauge workaround.
   Do this first; it is the cheapest and the most consequential.
2. **`cflFactor` units** (Q6, discrepancy 3). One line, and it supersedes
   Part 5's costed wall-aware `dt` proposal.
3. **`alpha`'s neighbour sets** (discrepancy 4). Directly on Part 5's open
   near-wall item, and a real deviation from the published diagonal.
4. **Port [C]'s shear-wave decay case** (Q5). The missing reference case,
   with published curves, on the axis the `ShiftApplication` modes differ on.
5. **Move `computeMdbcPressure` inside the solver iteration** (Q4,
   discrepancy 1), and add [B]'s SVD guard on the MLS gradient. The
   paper-standard fix for the loop `mdbcPressureRelaxation` currently damps.
6. **Rename toward VD+PS, not DFSPH** (headline). Part 6's step 1 stands but
   the target name changes; its step 2 (gate a second `dfsph` scheme on the
   source-projection experiment) should wait for item 1 above, which may
   remove the motivation entirely.

**Item 1 of "Next session: start here" is superseded**: projecting the
unreachable mean out of the velocity path is a fix for a formulation [C] was
written to replace. Try the tolerance first.

---

## Part 8 — the two originals (DFSPH, IISPH) (2026-08-27)

Part 7 was run against Cornelis (VD+PS) and Band (MLS boundaries) only, and
flagged six papers it was reasoning about second-hand. Two of those are now
available in full:

- **[BK]** Bender & Koschier, *Divergence-Free Smoothed Particle
  Hydrodynamics* (SCA 2015). DFSPH proper.
- **[I]** Ihmsen, Cornelis, Solenthaler, Horvath, Teschner, *Implicit
  Incompressible SPH* (TVCG 2014). IISPH — the solver this codebase's
  relaxed-Jacobi loop is a discretisation of.

**These correct one of Part 7's central claims and add four findings, one of
which is a one-line explanation for Part 4.** Read this before acting on
Part 7's ranked list; the order changes.

### Correction to Part 7 Q1 — the tolerance answer was wrong

Part 7 argued that published solvers terminate because their tolerance sits
*above* the structural bias, citing Cornelis's 0.1%. [BK] §4 refutes that
directly: DFSPH "enforced an average density error of less than **0.01 %** and
a density error due to the density change rate of less than 0.1 % in all
simulations" — i.e. a constant-density tolerance of **1e-4, five times
*tighter* than this codebase's 5e-4**, and far below the 2.7e-3 structural
floor Part 4 measured. And it converges: [BK] Table 1 reports **4.5 iterations**
for the constant-density solver at its largest timestep. A solver that hits
1e-4 in 4.5 iterations is not fighting an unreachable setpoint.

So the question sharpens rather than closes: if the structural floor is real
(and Part 4's Parseval argument is not in doubt), how is DFSPH's setpoint
attainable at 1e-4?

### The actual answer: this codebase calibrates `rho0` to the lattice

> **Tested and falsified — see "Part 8 — results" at the end of this
> document.** The diagnosis below is correct (the two cases do pin `rho0` to
> the lattice minimum) but the prescription is not: the source's permanent
> negative mean is the shifting solve's de-clumping drive, and cancelling it
> kills `kolmogorovIncompressible` 150 steps *sooner*.

`cases/kolmogorovIncompressible.py:71`:

```python
# Normalise mass so the sampled density lands on rho0, matching `tgv`'s
# own `buildSystem` for this same scheme.
system.state.masses = system.state.masses / densities.mean() * rho0
```

The initial state is an unjittered lattice. Part 4 proved the lattice
*minimises* the particle-averaged summation density. So this line sets `rho0`
to exactly that minimum — **it places the setpoint precisely at the
unattainable floor, by construction.** Any subsequent disorder can only raise
`mean rho` above `rho0`, never reach it, and the solver integrates that gap
forever.

Neither paper does this. [BK] §3 and [I] §2 both define `rho_i = sum_j m_j W_ij`
with `rho0` an independent fluid constant and `m` from the nominal particle
volume; nothing ties `rho0` to the sampled configuration's density. Their
setpoint therefore sits at a generic value that a disordered configuration
can straddle, which is why a 1e-4 tolerance is reachable in 4.5 iterations.

**`grep` says this is exactly the two cases where the wind-up was observed:**

```
src/warpSPH/cases/kolmogorovIncompressible.py:71
src/warpSPH/cases/tgv.py:53
```

and no others. Those are also, precisely, the two periodic complete-support
cases that `ShiftPressureGauge.minShift` was scoped to in Part 4. That is not
a coincidence worth ignoring: **the population that needed a gauge fix and the
population that pins `rho0` to the lattice minimum are the same two cases.**

So Part 4's conclusion needs one amendment. "The systematic density bias is
structural, not a bug anyone can go find and fix" is still true. But *the
setpoint's placement at the floor* is a code decision, in one line, in two
case files — and it is the half that makes the bias unreachable rather than
merely present. `minShift` is a correct and well-argued defence against the
symptom; it may not be needed once the setpoint moves.

**The experiment this licenses** (cheaper and more decisive than Part 7's
tolerance test, which should now be run second): drop the normalisation line,
or offset it — set `rho0` to the lattice density times `(1 + eps)` for `eps`
around the operating disorder's bias (~3e-3) — and re-run
`probe_shiftPressureGauge.py` on `kolmogorovIncompressible` at nx=128 under
the *historical* `nonNegativeClamp`. Prediction: `nIter` comes off the 64 cap,
`pMean` stops climbing, and the step-574 NaN does not occur without any gauge
at all. If it holds, Part 4's fix becomes a belt-and-braces measure rather
than load-bearing, and `tgv`'s decay rate should be re-graded on the corrected
setpoint before any conclusion about `ShiftApplication` is trusted — including
Part 5's and Part 6's, since **every `tgv` dissipation number in this document
was measured with the setpoint pinned to the floor.**

### DFSPH runs its two solvers in the opposite order, and `inStepVelocity` has it backwards

> **Overstated — see "Part 8 — correction" at the end of this document.**
> Under `semiImplicitEuler` the position advances with the *updated* velocity,
> which makes this ordering equivalent to [BK]'s up to loop phase. The
> surviving difference is that `DF` is computed before `CD` and so does not
> see it within the step -- a one-step lag, not an absence.

[BK] Algorithm 1, lines 11–23:

```
v* = v + dt F_adv/m
correctDensityError(alpha, v*)        <- constant density solver FIRST
x(t+dt) = x(t) + dt v*                <- integrate with the corrected v*
find neighborhoods; compute rho, alpha
correctDivergenceError(alpha, v*)     <- divergence solver LAST
v(t+dt) = v*
```

[BK] §3.1 states the reason outright: "The pressure forces determined by the
constant density solver must be integrated twice to get the required position
changes. Therefore, as a side-effect, it also modifies the velocities. For
this reason, first, we execute the constant density solver and modify the
velocities and positions, and then the divergence-free solver which corrects
the resulting velocities to obtain a divergence-free state."

**The density solve's velocity modification is a known side-effect, and the
divergence solve is deliberately placed after it to clean it up.** The step
ends on a divergence-free velocity.

`ShiftApplication.inStepVelocity` does the reverse: `solveDivergenceFree`
runs first in `schemes/dfsph.py`, then the constant-density correction is
folded into the same `dvdt` the integrator advects with. Nothing projects it.
That is exactly the concern `ShiftApplication`'s own docstring raises about
`positionAndVelocity` ("applied here, after the integrator, nothing ever
removes the non-divergence-free part of it") — and it applies to
`inStepVelocity` too, which was introduced to fix it.

**So `inStepVelocity` is not "DFSPH proper", as Part 5 continued (2) and
Part 6 both claim. It is DFSPH with the two solvers swapped**, and the
un-projected divergence injected every step by the density solve is a
credible mechanism for the 3.3x `tgv` decay that Part 6 treated as intrinsic
to velocity-level correction. Combined with the setpoint finding above, both
inputs to that measurement are now suspect.

This is a reordering, not a new algorithm, and it is the single most
informative experiment left on the velocity path.

### `computeAlpha` versus the published diagonal — three confirmations and one new deviation

> **The `rho`-power half is tested and falsified — see "Part 8 — results".**
> `computeAlpha` *is* the true diagonal (1.0000 +/- 0.016 with `rho` out to
> 1.30). The boundary-neighbour-set half is untested and still open.

Part 7 discrepancy 4 (boundary neighbours entering both sums) is now
confirmed by all three primary sources, with the mechanism stated:

- **[BK] §3.2**, right after Eq. 8: "since `F^p_{j<-i} = 0` if particle j is
  not dynamic, the equation for `kappa^v_i` must be adapted accordingly for
  static boundary particles." Static boundary particles take no reaction
  force, so they belong in the first sum and not the second.
- **[I] §4**: with Akinci coupling, boundary particles enter `d_ii` (the
  self-displacement term) via `-dt^2 sum_b Psi_b(rho0) (1/rho_i^2) grad W_ib`,
  and appear nowhere in `d_ij`. Same asymmetry, derived independently.
- **[B]** Algorithm 1's `lambda_f`, as recorded in Part 7.

`wp_alpha.py:110-128` accumulates `sumA` and `sumB` over one neighbour set,
under `OperationProperties`' default `operationMode = AllToAll`. Unchanged
finding, now much better supported.

**New, and separate: the density powers do not match IISPH's diagonal.**
Expanding [I] Eq. 12 with Eq. 9's `d_ii`/`d_ij`:

```
a_ii = -dt^2 (1/rho_i^2) [ ||sum_j m_j grad W_ij||^2 + m_i sum_j m_j |grad W_ij|^2 ]
```

The codebase (`apparentArea = m/rho`, `incompressible.py:66`; `wp_alpha.py`)
gives, for uniform `rho_j = rho`:

```
alpha = -dt^2 (1/(rho_i rho^2)) [ ||sum_j m_j grad W_ij||^2 + m_i sum_j m_j |grad W_ij|^2 ]
```

Identical brackets — including the `m_i` on the second term, which confirms
the codebase is discretising IISPH's diagonal and not [BK]'s `alpha` (they
differ there). But the prefactor carries **one extra power of `rho`**:
`1/rho^3` against `1/rho^2`. At `rho = rho0 = 1` the two agree exactly, which
is why no normalised test would ever show it. In the wall band, where Part 5
measured `rho` climbing to 1.30–1.60, `|alpha|` is too small by a factor
`1/rho`, so the Jacobi step `p += omega * residual / alpha` **over-corrects by
30–60%, exactly in the over-dense band that runs away.** Note this pushes the
opposite way to the boundary-neighbour issue above, so the two are not
interchangeable and both need testing separately.

**Do not act on this algebra — measure it.** The decisive test is cheap and
settles the `omega` question at the same time: extract the *true* diagonal of
the operator actually being iterated (`computePressureShiftIISPH` composed
with `computePressureAccelIISPH`) by applying it to unit vectors, and compare
against `computeAlpha`'s output particle by particle. If they differ by a
factor `c`, then the effective relaxation is `omega/c`, which would explain
Part 1's measured stability window `omega < 0.355` against the value **both**
originals report as optimal — [I] §3.1.1: "We observed an optimal convergence
for the relaxation factor `omega = 0.5` in all settings." A diagonal that is
off by ~1.4x would put 0.5 exactly at the observed edge.

### Two of Part 6's hypotheses have published negative results

- **One-sided source clamping** ("correct compression only, never expansion")
  was tried and rejected. [I] §3.2: "Intuitively, we could disallow a positive
  change of density due to pressure by clamping `b_i` to negative values with
  `b_i = min(0, rho0 - rho_i^adv)`. Unfortunately, this adaptation causes
  implausible alignments of single particles at the fluid surface for CG and
  Jacobi." Do not spend a session on it.
- **Krylov on the clamped solve** is confirmed invalid, closing "Next session"
  item 3 with a citation stronger than Part 7's. [I] §3.2: "For CG, however,
  clamping in between the iterations leads to invalid states. We also observed
  instabilities in case of any change in the final pressure field, such as
  clamping of negative pressure values." [BK] §5 says the same from the other
  direction — dropping the clamp is listed as *future work* precisely so that
  "more sophisticated solving algorithms like the conjugate gradient method
  could be employed". `solveIncompressible` currently routes to
  `solvePressureKrylov(..., gauge='nonnegative')` whenever `solverType !=
  relaxedJacobi` (`incompressible.py:97-101`). That path should raise, the way
  the `JacobiRelaxationMode.optimal` path already does.

### Warm start: every paper does one, this codebase does none

| | initial pressure/stiffness each step |
|---|---|
| [BK] DFSPH | **full** warm start — sum of last step's `kappa_i`, applied as a pre-solve velocity update (§3.2, §4: "reduced the number of iterations by a factor of approximately 3") |
| [I] IISPH | `p_i^0 = 0.5 p_i(t - dt)` — "close to optimal convergence" (§3.1.1) |
| [C] VD+PS | `p_i(t+dt) = 0.5 p_i(t)` for the DI solve — inherited from [I] |
| [B] | none, deliberately |
| here | none: `pressureA = particles.pressures.clone() * 0.` (`incompressible.py:167`) |

[BK] §4 notes the split explicitly: "While DFSPH performs best for a full warm
start, IISPH has its best performance when multiplying the solution of the
last step with a factor of 0.5". The codebase is at neither end. This is worth
a factor of ~3 in iteration count on a solve that currently runs 64 iterations
every step and never converges — but it is a *performance* fix, and it should
be applied **after** the setpoint is corrected, not before: warm-starting a
solver that is winding up would carry the wind-up across steps, which is the
one thing the current cold start prevents.

### The CFL constant, now verified

Part 7 flagged `dt <= 0.4 d/|v_max|` as `[unverified]`. Confirmed, in the
canonical source: [BK] §3.1, "adapt the time step size by the
Courant-Friedrich-Levy (CFL) condition `dt <= 0.4 d/||v_max||` [Monaghan
1992], where **d is the particle diameter**". [BK]'s own numbers are
self-consistent with it — breaking dam at particle radius 0.02 m runs best at
`dt = 4 ms`, which at a few m/s is ~0.4 particle diameters of displacement.
[I] §5.2.1 writes the same condition as `dt = min(0.4 h/|v_i^adv|)`; given
[BK] §3.4 defines `h` as the *support radius* and [BK] states the constraint
in diameters, that `h` is loose usage for the particle diameter, not the
support radius.

So Part 7 Q6 stands and is now on a firm footing: this codebase's
`dt = cflFactor * h / v_max` with `n_h = 4` gives **1.2 particle spacings per
step at the default `cflFactor = 0.3`, three times the published limit.**
Matching [BK] exactly means `cflFactor = 0.1`. Part 5's empirically-stable
`cflFactor = 0.125` (0.5 spacings) is *still looser* than the published
condition, which is a satisfying independent confirmation of both.

### Free-surface clustering is a documented DFSPH limitation

Relevant to Part 3's unexplained `rotatingSquarePatch` corner density loss.
[BK] §5: "In SPH simulations the density near a free surface is
underestimated which causes unnatural particle clustering artifacts. In our
implementation this problem is solved by clamping negative pressures to zero.
However, a better solution would be to introduce ghost particles as suggested
by Schechter and Bridson [2012]". [I] §6 says the same and calls it open. So
Part 3's corner loss is a known artifact of the method with a known published
remedy (ghost particles), not a bug specific to this implementation — worth
knowing before more time goes into root-causing it.

### Where this leaves the scheme split

The owner's proposal — make the current scheme VD+PS, and add DFSPH in its
pressure-boundaries form — is the right shape, and Part 8 makes it cheaper
than Part 6 assumed, because the second scheme is now fully specified by
[BK] Algorithm 1 rather than being an open design question.

**Scheme A — VD+PS** (`schemes/vdps.py`, registered `divergenceFree`, keep the
name for compatibility). This is the current code, unchanged. Part 7
established it is a faithful implementation of [C] step for step.
Defaults: `shiftApplication = positionShift`, `boundaryPressureMode =
mdbcDensity`, DI tolerance and `cflFactor` per the corrections above.

**Scheme B — DFSPH** (`schemes/dfsph.py`, registered `dfsph`). Differs from A
in five specific, enumerable ways, all of them now sourced:

1. **Solver order**: density solve, integrate, divergence solve (Algorithm 1).
   Not A's order. This is the substantive difference and the one Part 6 did
   not know about.
2. **No position shift.** The density solve reaches positions through the
   velocity, within the step.
3. **Warm start**: full, per [BK] §3.2 — accumulate `kappa_i` and apply it
   before the divergence solve.
4. **Tolerances**: 1e-4 average density error, 1e-3 on the density change
   rate ([BK] §4).
5. **Boundary coupling**: [B]'s MLS pressure boundaries, recomputed *inside*
   each solver iteration (Part 7 Q4), with [B]'s SVD guard on the gradient
   fit. Not the current one-per-step under-relaxed projection.

Shared, unforked: `computeAlpha`, `computePressureAccelIISPH`,
`computePressureShiftIISPH`, the relaxed-Jacobi loop, the boundary-mode
machinery, every case and diagnostic. Part 6's recommendation of two
`SchemeBundle`s over one step function still holds for items 2–5, but item 1
is a genuine control-flow difference, so expect one branch or two thin step
functions over a shared body rather than pure config.

**Sequencing note.** Do not build Scheme B before the setpoint correction and
the diagonal measurement land. Both feed directly into it: B is the
formulation that puts the density solve into momentum, so it is the one that a
setpoint pinned to the unreachable floor damages most, and it is the one whose
wall behaviour the `alpha` deviations bear on. Building it first would
measure the bugs, not the method.

### Revised order of work (supersedes Part 7's list)

> **Superseded by "Part 8 — results"'s own list.** Items 1 and 2 below were
> run and both failed; 3-6 stand.

1. **Move the setpoint off the lattice floor** (`kolmogorovIncompressible.py:71`,
   `tgv.py:53`). One line, two files, explains Part 4, and re-grades every
   `tgv` number in this document. Everything else is measured against it.
2. **Measure the true operator diagonal against `computeAlpha`.** Cheap,
   decisive, and settles `omega < 0.355` versus both papers' 0.5 at the same
   time. Then fix the boundary-neighbour sets and the `rho` power.
3. **`cflFactor` units** — `0.3 * h` = 1.2 spacings against the published
   `0.4 d`. One line; supersedes Part 5's costed wall-aware `dt` proposal.
4. **Re-run the `ShiftApplication` comparison** with 1–3 landed, including
   `inStepVelocity` **reordered to [BK] Algorithm 1**. Part 5's and Part 6's
   conclusions about velocity-level correction were all measured with a pinned
   setpoint and the wrong solver order; none of them should be carried
   forward as-is.
5. **Port [C]'s shear-wave decay case** (Part 7 Q5) — still the missing
   reference case, and now the right instrument for step 4.
6. **Then** the scheme split, and the MLS-inside-the-iteration boundary work
   that Scheme B needs.

Dropped from Part 7's list: nothing. Demoted: the PS tolerance experiment
(item 1 there) — still worth running, but as a confirmation *after* the
setpoint moves, since [BK]'s 1e-4 shows a tight tolerance is not itself the
problem. Closed: "Next session" item 3, the Krylov path, which should now
simply raise.

---

### Part 8 — results: items 1 and 2 run, both hypotheses falsified (2026-08-27)

Ran Part 8's first two items. **Both predictions are wrong, both for
instructive reasons, and the pair of negative results points at a third thing
that neither Part 7 nor Part 8 considered.** Nothing in `src/` changed;
tooling is `scripts/probe_incompressibleGaugeDrift.py --setpointEps` (new
flag) and `scripts/probe_operatorDiagonal.py` (new).

**Baseline first, to make the rest trustworthy.** `--nx 128 --nsteps 1000
--gauge clamp` reproduces Part 4 bit-for-bit: NaN at step **574**, `pMean`
peaking at **2.3838e6**. `minshift` reproduces its "~29" as `pMean` max
**28.87**. The instrument is sound.

#### Item 1 — moving the setpoint off the lattice floor: falsified

`--setpointEps eps` rescales mass by `1/(1+eps)` after the case builds, putting
the lattice at `rho0/(1+eps)` so a configuration carrying a bias of `eps`
lands *on* `rho0`. `eps=0` is byte-identical to the shipped case.

nx=128, 1000 steps, second-half means:

| gauge | eps | outcome | nIter | `pMean` max | `rhoBias` | `rhoErr` | `rhoMax` |
|---|---|---|---|---|---|---|---|
| clamp | 0 | **NaN @574** | 64.0 | 2.38e6 | 5.70e-3 | 6.31e-3 | 5.38e-2 |
| clamp | 0.005 | **NaN @424** | 64.0 | 4.06e3 | 5.14e-3 | 6.58e-3 | 7.11e-2 |
| minShift | 0 | finite 1000 | 64.0 | 28.9 | 9.24e-4 | 1.15e-3 | 7.91e-3 |
| minShift | 0.005 | finite 1000 | **6.0** | 9.67 | -2.69e-3 | 4.08e-3 | 2.88e-2 |

And an nx=64/300 sweep under the clamp, `eps` in {0, 1e-3, 2.7e-3, 5e-3, 1e-2,
2e-2}: `rhoBias` falls 2.88e-3 -> 3.57e-4 up to `eps=5e-3` then goes
non-monotone, `|pMean|` falls 3.88 -> 1.41, **`nIter` stays pegged at ~64
throughout**, and `rhoMax` degrades monotonically 2.1e-2 -> 9.8e-2.

**Under the clamp it kills the run 150 steps sooner.** The offset shrinks
`pMean`'s peak by 580x and still dies earlier, because the blowup at nx=128 is
a *density* runaway that `pMean` follows rather than causes (`rhoBias` climbs
3.1e-3 -> 5.5e-3 -> 1.1e-2 -> 2.6e-2 over steps 494-570, with `rhoStar`
starting to hit its 0.9 floor). The offset makes it worse because
`sourceTerm = rho0 - rhoStar`'s persistent *negative* mean **is the
de-clumping drive** — it is what makes the shifting solve push particles
apart. Cancelling it is a static version of the mean-centering Part 4 already
tested and called "the worst option", and it fails the same way for the same
reason. `rhoStd` at the `nIter` peak rises 1.22e-3 -> 4.22e-3: the
configuration disorders faster without that drive.

**Under `minShift` the `nIter=6` is not convergence — it is the stopping
criterion going slack.** With `eps=5e-3` the fluid sits *below* `rho0`
(`rhoBias = -2.69e-3`), so `sourceTerm` has a positive mean, so `-residual` is
negative for most particles, so the compression-only metric
(`error = mean(clamp(-residual, min=-threshold))`, `incompressible.py:218-220`)
floors nearly every contribution at `-threshold` and reports convergence at
`minIterations`. The accuracy confirms it is not solving anything better:
`rhoErr` 3.5x worse (1.15e-3 -> 4.08e-3), `rhoMax` 3.6x worse. It converges by
declaring victory.

**So Part 8's headline recommendation is withdrawn.** The *diagnosis* stands —
`kolmogorovIncompressible.py:71` and `tgv.py:53` do pin `rho0` to the lattice
minimum, and that is why the source carries a permanent negative mean. The
*prescription* was wrong: for a position-shift (PS/VD+PS) scheme that
permanent negative mean is load-bearing, not a defect. Which sharpens
Part 7's conclusion rather than replacing it: **the papers differ precisely on
where the constant-density solve's permanent residual is allowed to land.** In
VD+PS it lands on positions, where it is momentum-neutral *and* does useful
de-clumping work. In DFSPH it lands in momentum, where it is a permanent
unphysical force. The setpoint question is therefore live only for a velocity
path (Scheme B), and moot for the current default.

#### Item 2 — `computeAlpha` versus the true diagonal: falsified, decisively

`probe_operatorDiagonal.py` extracts the diagonal *exactly* rather than
estimating it: for a sampled row `i`, apply the real operator
(`dt**2 * computePressureShiftIISPH(computePressureAccelIISPH(.))`) to the unit
vector `e_i` and read entry `i`. One matvec per row, sampled across the
density range.

| case | rho range | `diag(A)/alphas` | best fit |
|---|---|---|---|
| `kolmogorovIncompressible`, 80 steps | [0.965, 1.021] | **1.00011 +/- 0.0051** | constant (2.6x better than `k*rho`) |
| `randomFlowIncompressible --bounded`, 120 steps | [0.959, **1.303**] | **0.99987 +/- 0.0160** | constant (5.2x better) |
| boundary rows (`kind != 0`) | — | **0.99993 +/- 0.0314** | — |

**`computeAlpha` is the true diagonal.** Part 8's "extra power of `rho`"
algebra was simply wrong — the operator uses the same apparent-volume
convention `computeAlpha` does, so the `rho` powers are self-consistent even
though they do not match [I] Eq. 12 read literally. The bounded case has real
lever arm (`rho` to 1.30, where a `1/rho` error would show as a 30% deviation)
and shows none: at `rho = 1.303` the ratio is 0.931, not 1.303.

Two consequences:

- **`omega_eff = omega` exactly**, so Part 1's measured window `omega < 0.355`
  is *not* a mis-scaled-diagonal artifact and needs another explanation.
  A candidate worth one measurement, not pursued here: [I] §2.2 notes
  "typically a particle has 30-40 neighbors", while this codebase's `n_h = 4`
  gives `pi * n_h^2 ~ 50` in 2D. The IISPH operator reaches neighbours-of-
  neighbours, so its spectral radius is sensitive to that count.
- **The boundary-neighbour-set question is untouched by this.** The probe
  shows `computeAlpha` is a faithful diagonal *of the operator as
  implemented*; whether that operator's boundary coupling matches [I] Eq. 16
  and [BK] §3.2 is a separate question and still open.

#### What the two negative results expose instead: the criterion, not the setpoint

`minShift` at the shipped defaults runs **64 iterations every step for 1000
steps** and never satisfies its tolerance. That was never recorded — Part 4
established the gauge bounds the wind-up, not that the solve terminates. It
does not.

The obvious follow-up was that those iterations are wasted. They are not:

| maxIterations | `rhoErr` | `rhoMax` | `pMean` max | outcome |
|---|---|---|---|---|
| 64 (default) | 1.15e-3 | 7.91e-3 | 28.9 | finite 1000 |
| 16 | 2.29e-3 | 1.08e-2 | 15.9 | finite 1000 |
| 8 | 3.80e-3 | 1.87e-2 | 9.57 | finite 1000 |

Accuracy degrades ~2x at 16 and ~3.3x at 8, so **the iterations are doing real
work and the cap should not be lowered.** The solve is converging in the sense
that matters — the error falls steadily with iteration count — while never
meeting a criterion that is structurally unreachable.

**So the broken component is the stopping criterion.** An *absolute*
compression threshold (`error < 5e-4`) cannot be met when the source carries a
structural mean the operator cannot remove, no matter how well the solve is
going. A *relative* criterion — stop when the residual has fallen by a set
factor from its own initial value — is reachable by construction and measures
the thing the iterations are actually delivering. `RelaxedJacobiSolverConfig`
**already has `rtol`** (`solver.py:223`), documented as "Relative residual
tolerance for the Krylov solvers"; the relaxed-Jacobi path ignores it. Wiring
it in, as a disjunction with the existing absolute test, is a small change
with a clear prediction: `nIter` comes off the cap on the steps where the
solve has converged as far as it usefully can, and the accuracy numbers above
say what that costs at each stopping point.

This also reframes Part 8's puzzle about [BK]'s 1e-4 tolerance one more time.
Both papers' criteria are one-sided on the *average* (`rho_avg - rho0 > eta`,
[BK] Alg. 3; [I] §5.1), so under-dense particles cancel over-dense ones
without limit. This codebase floors each particle's negative contribution at
`-threshold`, which forbids exactly that cancellation. That is a real and
deliberate-looking difference in strictness, it is the difference that makes
the structural bias binding here and not there, and it is one line.

#### Revised next steps

1. **Wire `rtol` into the relaxed-Jacobi path** as an alternative stopping
   test, and re-measure the table above against it. This is now the item that
   Part 8's item 1 slot was reaching for.
2. **The one-sided-average vs floored-average criterion.** Match the papers
   (plain signed average) behind a flag and compare; it is the other end of
   the same question.
3. **`cflFactor` units** (Part 8 item 3) — unaffected by any of the above,
   still a one-line change against a now-verified published constant.
4. **Re-measure `ShiftApplication` with `inStepVelocity` reordered to
   [BK] Algorithm 1** (Part 8 item 4). Untouched by these results and still
   the most informative experiment on the velocity path.
5. Dropped: the setpoint offset (item 1) and the `alpha` rescale (item 2's
   `rho`-power half). Both tested, both wrong. The boundary-neighbour-set half
   of item 2 survives and needs a different probe than
   `probe_operatorDiagonal.py`.

---

### Part 8 — correction: the solver-ordering claim was overstated (2026-08-27)

Part 8 claimed `ShiftApplication.inStepVelocity` "is DFSPH with the two
solvers swapped" and that the constant-density correction's divergence is
"never projected". **Checking the integrator's actual semantics shows that is
wrong**, and the in-code comment at `schemes/dfsph.py:190-196` — which says the
correction "is visible to the *next* step's divergence-free projection" — was
right. Recorded before anyone spends a session reordering on my say-so.

**What the integrator actually does.** `kolmogorovIncompressible.py:181` selects
`semiImplicitEuler`, whose position update is `semi_implicit_position_step`
(`warpSPHIntegrators/specs.py:102-111`): `derivative_dt = 0.0`,
`current_velocity_dt = dt`. So the position advances with the **updated**
velocity, and `dfsph_step`'s `dxdt = currentState.velocities.clone()` is
multiplied by zero. **Do not delete it on that basis**: all three
incompressible cases (`tgv.py:171`, `kolmogorovIncompressible.py:181`,
`randomFlowIncompressible.py:118`) set `semiImplicitEuler` explicitly, but
`CaseSpec`'s default is `rungeKutta2` (`caseSpec.py:53`), which *does*
consume `dxdt`. It is inert for every case that exists today and live for
any future one that takes the default — worth a comment, not a deletion.
(Usefully, this also rules out an integrator confound between Part 5's wall
measurements and Part 6's `tgv` ones: all three cases use the same one.)
The step is therefore

```
v(t+dt) = v(t) + dt (F_adv + diss + DF + CD)      # DF, CD both evaluated at x(t)
x(t+dt) = x(t) + dt v(t+dt)
```

against [BK] Algorithm 1's

```
v*      = v(t) + dt F_adv
v*     += dt CD                                   # CD evaluated at x(t)
x(t+dt) = x(t) + dt v*                            # DF of this step not yet applied
v*     += dt DF                                   # DF evaluated at x(t+dt)
v(t+dt) = v*
```

**Unrolled over steps these are the same sequence of solves, differing in
phase.** Both advect with a velocity carrying exactly one step's worth of
un-projected `CD`; both project it on the following divergence solve. The
claim that the codebase never projects it does not survive contact with the
loop.

Two genuine differences survive, and both are smaller than what was claimed:

1. **The `DF` correction participates in the codebase's position update and
   not in [BK]'s.** The codebase computes `DF` at `x(t)` and folds it into the
   velocity that then advects; [BK] applies its `DF` after the position update,
   so that step's `DF` only reaches positions on the following step.
2. **`DF` is evaluated on stale neighbourhoods.** The codebase solves it at
   `x(t)`; [BK] at `x(t+dt)`, after rebuilding (Algorithm 1 lines 16-20). [BK]
   also recomputes `rho` and `alpha` there, which the codebase does not.

Neither is obviously worth a scheme fork, and neither is a credible mechanism
for a 3.3x `tgv` decay on its own. **So Part 8's item 4 is demoted**: the
reordering is a small, phase-level change, not the structural difference it
was written up as, and the `tgv` dissipation of the velocity modes still lacks
an explanation. The one thing Part 8 got right on this axis is narrower: in
the codebase `DF` is computed *before* `CD` and so does not see it within the
step, whereas [BK]'s does — a one-step lag, not an absence.

**Method note, since this is the third correction in a row.** All three
over-claims (Part 7's tolerance argument, Part 8's `rho`-power algebra,
Part 8's ordering) came from reasoning off published pseudocode without
checking what this codebase's surrounding machinery actually does with the
values. The two that were *measured* rather than argued
(`probe_operatorDiagonal.py`, `--setpointEps`) settled in one run each. Prefer
the measurement.

---

### Part 8 item 3 — the published CFL constant fixes the bounded case on its own (2026-08-27)

The first positive result of this session, and the simplest thing tried.

[BK] §3.1 gives the constraint as `dt <= 0.4 d/||v_max||` with `d` the particle
**diameter**. This codebase computes `dt = cflFactor * h / v_max` against the
**support radius**, and `n_h = 4` makes `h = 4` particle spacings, so the
default `cflFactor = 0.3` permits 1.2 spacings per step — 3x the published
limit. Matching [BK] means `cflFactor = 0.1`.

Ran `probe_boundedIncompressibleBlowup.py --nx 128 --tlimit 8.0 --cflFactor
0.1` at **stock scheme settings** — `positionShift`, `mdbcDensity`, no opt-in
flags:

| configuration | outcome | near-wall `mean｜rho-1｜` | `rho` range | steps to t=8 |
|---|---|---|---|---|
| default `cflFactor=0.3` (Part 5) | **NaN at t=5.54** | 0.30 at death | — | — |
| **`cflFactor=0.1`, else stock** | **t=8.003** | **2.6e-2** | [0.963, 1.155] | 1624 |
| `positionAndVelocity` @ 0.3 (Part 5) | t=8.0 | 3.3e-2 | max 1.147 | 387 |
| `inStepVelocity` @ 0.3 (Part 6) | t=8.0 | 9.7e-3 | [0.986, 1.140] | — |

**The default scheme, with one number changed to the published value, is
stable to t=8 and beats `positionAndVelocity` on near-wall density error** —
without either velocity mode, without their `tgv` dissipation (0.55x analytic
against 3.2-3.4x), and without an opt-in flag. `inStepVelocity` still has the
best wall numbers by ~2.7x.

The mechanism is exactly Part 5's diagnosis, read forwards: the implicit shift
`PS.shiftDx` settles at **0.20 particle spacings** per step instead of the 1.2
Part 5 measured at the death, so the shift can no longer throw a particle
through a wall in one step.

Wall-depth profile at t=8 confirms a steady state rather than an accumulation
(`nOutside` flat at 482-488 over the final 25 steps, `rhoMax` flat at ~1.15):

| depth (spacings) | n | mean rho | `mean｜rho-1｜` |
|---|---|---|---|
| [-6,-1) | 64 | 1.14 | 1.4e-1 |
| [-1,0) | 85 | 1.050 | 5.1e-2 |
| [0,1) | 349 | 1.021 | 2.4e-2 |
| [1,2) | 531 | 1.013 | 1.4e-2 |
| [2,4) | 903 | 1.002 | 2.3e-3 |
| [4,inf) | ~14100 | 1.000 | ~6e-4 |

The bulk is clean to ~6e-4; the error is confined to the two bands nearest the
wall plus a stalled group of 64 particles sitting outside it at `rho ~ 1.14`.

**Caveat on the penetration count, so it is not misread**: `nOutside = 482` is
an *instantaneous* count at t=8. Part 5's 4506 / 239 / 63 were "particles
*ever* inside the boundary band" — a cumulative statistic. The two are not
comparable and the table above deliberately omits it. The near-wall
`mean|rho-1|` column *is* the same statistic across all four rows.

**Cost: 4.2x the steps** (1624 against `positionAndVelocity`'s 387 to the same
`t`). That is the real trade, and it is a better-posed one than Part 5's
"wall-aware `dt` at ~2.4x throughput": this is a global constant matching
published practice, not a new term, and it buys the bulk physics that both
velocity modes give up.

**Recommendation.** Change the incompressible cases' `cflFactor` default from
0.3 to 0.1, or — better, since it removes the trap rather than papering over
it — express the advective limit against the particle spacing instead of `h`,
so the number means what it means in every paper. Not landed here: it changes
the default for `tgv` and `kolmogorovIncompressible` too, and those should be
re-graded first (`tgv`'s decay rate is asserted in `tests/test_physics.py`).
That re-grading is the natural next step and is cheap.

#### Status of Part 8's items

| item | result |
|---|---|
| 1. move the setpoint off the lattice floor | **falsified** — kills the run 150 steps sooner |
| 2. `computeAlpha` vs the true diagonal | **falsified** — it *is* the diagonal, to 1.6% |
| 3. `cflFactor` units | **confirmed** — fixes the bounded case at stock settings |
| 4. reorder `inStepVelocity` to [BK] Alg. 1 | **demoted** — equivalent up to loop phase |
| 5. port [C]'s shear-wave case | not started |
| 6. scheme split | not started |

Next, in order: re-grade `tgv` and `kolmogorovIncompressible` at `cflFactor
0.1` (cheap, and gates landing item 3); then the relative stopping criterion
(`rtol`, from the Part 8 results section); then the shear-wave case.

#### Re-grading the two periodic cases at `cflFactor = 0.1`

The gate on landing item 3 was "does dropping the default break `tgv`, whose
decay rate `tests/test_physics.py` asserts?" **It cannot: `cflFactor` does not
bind on `tgv` at all.**

| case | cfl | steps | `dt` (min/max/mean) | result |
|---|---|---|---|---|
| `tgv` nx=64, t=2 | 0.3 | 2001 | 1e-3 / 1e-3 / 1e-3 | — |
| `tgv` nx=64, t=2 | 0.1 | 2001 | 1e-3 / 1e-3 / 1e-3 | identical |
| `kolmogorovIncompressible` nx=128, t=2 | 0.3 | 109 | 1e-3 / 0.1 / 1.86e-2 | worst `｜rho-1｜` 5.17e-3 |
| `kolmogorovIncompressible` nx=128, t=2 | 0.1 | 323 | 1e-3 / 0.1 / 6.22e-3 | worst `｜rho-1｜` **3.98e-3** |

`tgv`'s `dt` is pinned at exactly its configured `dt = 1e-3` for every step of
both runs — the advective limit is never the binding constraint there, so the
CFL constant is inert and the asserted decay band cannot move. (Measured
anyway, for the record: `measured/analytic = 0.78` at nx=32, monotone, inside
the test's [0.33, 0.87].)

`kolmogorovIncompressible` does bind, and 0.1 is **better**: the density band
tightens 23% (5.17e-3 -> 3.98e-3) for 3x the steps, exactly the `dt` ratio.
Both stable.

So item 3's blast radius is one case, in the improving direction, with `tgv`
untouched. **The re-grading gate is cleared.**

#### The published CFL also removes Part 4's NaN — without any gauge

Asked because it follows: if 1.2 spacings per step is the real defect, does the
*historical* `nonNegativeClamp` survive at 0.1? It does. nx=128, 1800 steps at
`cflFactor=0.1` (≈ t=11.2, i.e. **more** physical time than the baseline's
death at step 574 ≈ t=10.7):

| gauge | cfl | outcome | `pMean` max | `rhoErr` | `rhoMax` | nIter |
|---|---|---|---|---|---|---|
| clamp | 0.3 | **NaN @574** | 2.38e6 | 6.31e-3 | 5.38e-2 | 64 |
| clamp | **0.1** | **finite 1800** | **47.8** | **1.42e-3** | **5.34e-3** | 64 |
| minShift | 0.3 | finite 1000 | 28.9 | 1.15e-3 | 7.91e-3 | 64 |

`pMean` plateaus at ~25 from step 1620 on rather than growing. So **the
timestep, not the gauge, is sufficient to stop the Part 4 blowup** — the
constant mode still winds up, it just no longer wins the race against the
step.

This does **not** retire `ShiftPressureGauge.minShift`, and the two results fit
together cleanly rather than competing:

- On **periodic, complete-support** cases `minShift` is the *cheaper* remedy:
  it costs nothing, where `cflFactor=0.1` costs 3x the steps for a comparable
  answer (`rhoErr` 1.15e-3 against 1.42e-3). Part 4's fix stands as the right
  default for exactly the two cases it was scoped to.
- On **wall-bounded** cases `minShift` is a no-op *by construction* — it falls
  back to the clamp whenever there are pinned pressure rows or free-surface
  particles (`incompressible.py:160-166`). So the bounded case has always been
  running the historical clamp *and* a timestep 3x past the published limit,
  and the CFL is the only one of the two remedies available to it.

That is why item 3 fixes the case Part 5 could not, and why Part 4's fix could
never have. `nIter` stays pegged at 64 in every row above, which is the
stopping-criterion problem from the previous section — independent of both
`dt` and the gauge, and still the outstanding item.

**Landing recommendation, now fully gated.** Change the advective limit to be
expressed against particle spacing rather than `h` (equivalently, default
`cflFactor` 0.3 -> 0.1 for the incompressible cases). Evidence: fixes the
bounded case at stock settings; improves `kolmogorovIncompressible` 23%;
provably inert on `tgv`; and removes the Part 4 NaN even under the historical
gauge. Cost is 3-4x steps, which is the published operating point. Not landed
in this session — it is a default change and the owner should make that call
with these numbers in hand.

---

### Part 8 — a different linear solver, and the deltaSPH shift on top (2026-08-28)

Two suggestions from the project owner, both aimed at the outstanding item
(the PS solve running its cap every step). **One works and is the largest
accuracy win found so far; the other does not, and turns out to have never
been running at all.** New tooling:
`scripts/probe_incompressiblePressureSolvers.py`.

#### Prior work this rests on, which I should have read first

`INCOMPRESSIBLE_SOLVER_PLAN.md` already characterises this operator and every
method on it: symmetric to fp32 (`‖A−Aᵀ‖/‖A‖ ≈ 1e-6`), **negative-semi-definite
with a gauge mode**, `kappa(sym) ≈ 2.4e7`, `kappa(M⁻¹A) ≈ 1.1e8`; MINRES best on
residual (9.7e-4 @200 iters), CG strong, BiCGStab stagnating then diverging by
1200; `krylovFp64` worth ~10x.

It also **already contains the measurement Part 8 spent an item re-deriving**:
"`computeAlpha` vs true `Diag(A)` | rel-L2 **3.3e-7** | the IISPH diagonal is
the exact operator diagonal". And it already explains `omega < 0.355` exactly —
`omega < 2/rho(D⁻¹A)` with `rho(D⁻¹A) ≈ 5.636`, a degenerate high-frequency
lattice cluster, `dt`-invariant and robust to deformation. So Part 8's item 2
was a *known* result, and my "extra power of rho" hypothesis was contradicted by
a measurement already in the repo. The two independent measurements agree
(3.3e-7 there, 1.6% there-and-back here), which is worth something, but the
session cost was avoidable.

What was genuinely untested is the thing the owner asked for: **any of these
solvers running end-to-end inside a case.** The plan's numbers are all single
solves on a seeded state.

#### Krylov end-to-end: the clamp is the whole story

`kolmogorovIncompressible`, `cflFactor=0.1`, `maxIterations=64`:

| solver | gauge | nx=64/200: `rhoErr` | outcome |
|---|---|---|---|
| relaxedJacobi (`minShift`) | per-iteration | 5.27e-3 | 201 steps |
| relaxedJacobi (`clamp`) | per-iteration | 9.60e-3 | 201 steps |
| minres | post-hoc `nonnegative` | 3.59e-1 | survives, **garbage** |
| cg | post-hoc `nonnegative` | — | **NaN at step 4** |
| bicgStab | post-hoc `nonnegative` | — | **NaN at step 4** |
| minres | **none** | **2.89e-3** | 201 steps |
| cg | **none** | 1.13e-2 | 201 steps |
| bicgStab | **none** | 5.82e-3 | 201 steps |

**Both papers predicted this exactly.** [I] Sec. 3.2: for CG "clamping in
between the iterations leads to invalid states", and they "observed
instabilities in case of any change in the final pressure field, such as
clamping of negative pressure values" — the shipped path's *post-hoc*
`gauge='nonnegative'` is precisely that change. [BK] Sec. 5 names dropping the
clamp as the precondition: "without pressure clamping more sophisticated
solving algorithms like the conjugate gradient method could be employed and
would enhance the convergence rate even more."

Confirmed at production resolution — nx=128, 600 steps, `cflFactor=0.1`, no
clamp:

| solver | steps | nIter mean | @cap | `rhoErr` | `rhoMax` | wall |
|---|---|---|---|---|---|---|
| relaxedJacobi (`minShift`) | 601 | 60.2 | 86% | 3.79e-3 | 7.34e-3 | 31.3 s |
| **minres (no clamp)** | 601 | 66.0 | 100% | **1.76e-3** | **3.35e-3** | 54.3 s |
| bicgStab (no clamp) | 601 | 56.0 | 75% | 4.49e-3 | 1.38e-1 | 64.5 s |

**MINRES is 2.15x better on mean density error and 2.2x on the worst case, for
1.73x the wall time.** BiCGStab is parity at best and spikes to `rhoMax`
1.38e-1 — consistent with the plan's finding that it stagnates then diverges on
this spectrum. This is the largest single accuracy improvement measured in this
document.

Three caveats, none of which are small:

1. **It requires dropping the non-negativity.** A shifting potential that may
   go negative can *pull* particles together. On this periodic, complete-support
   case that is evidently fine; [BK] Sec. 5 clamps specifically because "the
   density near a free surface is underestimated which causes unnatural particle
   clustering", and names ghost particles as the better fix. **So this must be
   tested on a free-surface case (`rotatingSquarePatch`) before it could ever
   be a default.**
2. **It does not fix the stopping criterion.** MINRES is at the 66-iteration cap
   100% of the time. The win is accuracy per unit work, not termination — the
   criterion problem from the previous section is untouched.
3. `minres` at 1.73x wall time is roughly break-even against just running the
   Jacobi path longer; that comparison was not made and should be
   (`relaxedJacobi` at `maxIterations=128` versus `minres` at 64).

#### The deltaSPH shift on top: it was never running

**First finding: `shiftProperties.active` was inert on this scheme, and paid
full price for it.** `IncompressibleSystem.finalize` bound `solveShifting`'s
result to `dx` (line 161), then **shadowed it** with `dx = dt**2 * dvdt_incomp`
(line 271); the only `positions += dx` that would have applied the deltaSPH
shift is in the commented-out block at line 320. So enabling the flag ran a
full shifting solve every step and discarded it. **Any previous test of "the
deltaSPH shift on top of the incompressible scheme" was measuring a no-op.**

Fixed (`systems/incompressible.py`): the shift now has its own name
(`dxDeltaShift`) and is added to the implicit shift before the position update
and the Eq. 17 velocity resample. Opt-in and default-inert — every
incompressible case ships `shifting=False`, and `dxDeltaShift is None`
short-circuits, so the default path is byte-identical.

**Second finding: with it actually applied, it does not help.**
`kolmogorovIncompressible` nx=128, 600 steps, `cflFactor=0.1`:

| shift | `shiftProperties.CFL` | `｜rho-1｜` 2nd half | worst | outcome |
|---|---|---|---|---|
| off | — | 3.79e-3 | 7.34e-3 | 601 steps |
| on | 0.3 (default) | 3.65e-3 | 6.90e-3 | 601 steps |
| on | 1.0 | 4.17e-3 | 7.83e-3 | 601 steps |
| on | 3.0 | 1.36e-2 | 2.60e-1 | **NaN at 567** |

4% better at the default magnitude — inside run-to-run noise — then
monotonically worse, then divergent. And on the bounded case at
`cflFactor=0.3`, where clustering *is* the failure mode, it does not rescue
anything: baseline NaN at step 258, shift(0.3) at **234**, shift(1.0) at 247.

**This is [C] Sec. 6's argument, measured.** Cornelis et al. contrast their
global PPE-based shift with the local concentration-gradient variants (their
refs [30] Nestor, [35] Skillen, [41] Xu) exactly on this point: the local ones
carry a user-tuned magnitude, and "if this user-defined parameter is too small,
the resulting sampling quality is not as good as it could be … if the parameter
is too large, over-correction occurs which can result in even worse sampling
qualities". The `shiftProperties.CFL` sweep above is that sentence as a table.
The implicit VD+PS shift is already a global, parameter-free version of the
same correction, so stacking a tuned local one on top has nothing to add and a
magnitude to get wrong. Same conclusion as Part 5's rejected `--shiftCap`, from
the opposite direction.

#### Where this leaves the outstanding item

Neither suggestion fixes the stopping criterion, which remains the one thing
that has survived every experiment: relaxed-Jacobi, MINRES, BiCGStab, every
`dt`, every gauge, and both shift configurations all sit at their iteration cap.
The relative-residual criterion (`rtol`, already in the config and ignored by
the Jacobi path) is still the untried item.

Revised standing:

1. **`cflFactor` units** — confirmed, ready to land, still unlanded.
2. **MINRES without the clamp** — best accuracy result so far; needs the
   free-surface test and the equal-cost Jacobi comparison before it can be
   recommended as anything but opt-in.
3. **Relative stopping criterion** — untried, and now the only remaining
   candidate for the cap problem.
4. deltaSPH shift on top — **tested and rejected**; the flag fix stays (it
   makes an inert-but-expensive option honest), the configuration does not.

#### Scoping the MINRES win: it is a complete-support result only

Same comparison on the bounded case (`randomFlowIncompressible --bounded`,
nx=128, 900 steps, `cflFactor=0.1`, no clamp):

| solver | steps | @cap | `rhoErr` | `rhoMax` | wall |
|---|---|---|---|---|---|
| relaxedJacobi | 901 | 100% | 1.778e-1 | 2.474e-1 | 113.7 s |
| minres (no clamp) | 901 | 100% | 1.721e-1 | 2.348e-1 | 140.0 s |

**3% better for 23% more time — nothing.** Both stable at the published CFL,
both dominated by the near-wall band rather than by solver quality.

So the 2.15x is a *periodic, complete-support* result, and it does not
transfer to a wall. That is the same conclusion Part 5 reached from the other
direction ("the projection is not failing because it is under-converged, it is
failing because it is computed before the particle gets to the wall") and it
sharpens the recommendation: **MINRES-without-the-clamp is worth having for
`kolmogorovIncompressible`/`tgv`-class problems and is not a wall fix.** At a
wall, the boundary treatment is still the binding constraint, and `cflFactor`
is still the thing that mattered.

---

### Part 8 — correction: the wall is NOT support-truncated, and that unblocks `minShift` on bounded cases (2026-08-28)

Raised by the project owner against Part 8's "a complete-support result only"
framing, and against a premise this document has leaned on since Part 4.
**The objection is correct, and acting on it overturns a shipped design
decision.** Tooling: `scripts/probe_wallSupportCompleteness.py` (new).

#### The premise was false

`ShiftPressureGauge`'s docstring justifies downgrading `minShift` to the clamp
on any wall-bounded solve with two reasons. The second is:

> where the support is truncated (against a wall, at a free surface) the kernel
> gradients no longer sum to zero, so a *constant* pressure exerts a large real
> force

That is imported from the literature's one-layer boundaries (Akinci et al., as
used by both [BK] and [C]). **This codebase does not use one-layer boundaries.**
`randomFlow.BOUNDED_BAND = 5` samples the boundary as a solid five-layer band,
against a support radius of `h = n_h * dx = 4` spacings — the band is *wider
than the kernel*, so a fluid particle on the wall has a full neighbourhood. It
is a deliberate deviation, commonly made, paid for in particle count.

Measured on `randomFlowIncompressible --bounded`, nx=128, after 120 steps:

| depth (spacings) | n | Shepard `sum V_j W_ij` | `｜sum V_j grad W_ij｜` | `｜A.1｜/｜A.rand｜` |
|---|---|---|---|---|
| (<0, inside wall) | 84 | 1.00857 | 5.63 | 0.245 |
| [0,1) | 436 | **1.00100** | 6.29 | **0.194** |
| [1,2) | 621 | 0.99123 | 3.78 | 0.536 |
| [2,3) | 455 | 0.99786 | 1.67 | 0.189 |
| [4,6) | 931 | 0.99989 | 1.29 | 0.167 |
| [10,inf) bulk | 11608 | 0.99997 | 1.80 | 0.166 |

Shepard sits at ~1.00 right down to the wall (min 0.970 across all fluid), so
**support is complete**. And the quantity that actually governs the gauge —
the real operator applied to a constant, relative to its response to a
same-scale random field — is **0.194 in the wall-adjacent bin against 0.166 in
the bulk**. The constant mode is no less null at the wall than anywhere else,
so a uniform pressure does *not* exert a large force there.

One nuance worth keeping: `|sum V_j grad W_ij|` **is** elevated ~4x at the wall
(6.3 against 1.3). But that is not missing neighbours — it is the volume field
being discontinuous across the interface, since boundary particles carry
mDBC-extrapolated densities and so different `V_j = m_j/rho_j`. Complete
support does not imply a consistent volume field. The composite operator
absorbs it; the raw gradient sum does not.

**So the docstring's second reason is wrong for walls.** It remains correct for
free surfaces, which are genuinely truncated. Its *first* reason — Dirichlet
rows pin the constant, so there is no free null space left to gauge — is
untouched and is on its own sufficient to justify a guard.

#### But the guard's empirical evidence was measured at 3x the published CFL

Part 4 recorded that forcing `minShift` through on the bounded case "diverges
at t=0.69, against t=5.5 for the clamp". That was at `cflFactor=0.3` — which
Part 8 item 3 has since established is 3x [BK]'s `dt <= 0.4 d/|v_max|`. Re-run
at both, with a new opt-in `forceShiftPressureGauge` bypass, nx=128, 900 steps:

| cfl | gauge | steps | diverged | t_final | `rho` range | `｜rho-1｜` 2nd half | wall |
|---|---|---|---|---|---|---|---|
| 0.3 | clamp | 258 | yes | 5.535 | [0.139, 2.452] | 5.38e-1 | 35.4 s |
| 0.3 | minShift | 38 | yes | **0.690** | [0.681, 1.257] | 2.16e-1 | 3.0 s |
| 0.1 | clamp | 901 | no | 4.690 | [0.902, 1.247] | 1.78e-1 | 113.1 s |
| **0.1** | **minShift** | 901 | **no** | **6.458** | [0.863, **1.154**] | **1.43e-1** | **61.2 s** |

The `cfl=0.3` row reproduces Part 4's t=0.69 exactly, so the original
measurement was sound — but it was measuring the timestep, not the gauge. At
the published CFL `minShift` is better than the clamp on every axis recorded:

- **it does not diverge**, over 900 steps;
- it covers **t=6.458 against 4.690** in the same step budget, i.e. the adaptive
  `dt` grows ~38% further because the solve is better conditioned;
- density error **1.43e-1 against 1.78e-1** (19% better) and a tighter band
  (`rho_max` 1.154 against 1.247);
- and it costs **half the wall time** (61.2 s against 113.1 s) for the same
  number of steps — the solve is terminating earlier.

**So `ShiftPressureGauge.minShift`'s scoping to "periodic, complete-support
cases" is too narrow, and was too narrow for a reason that has now been
measured false twice over** — the support premise, and the divergence evidence.
The remaining valid objection is the Dirichlet-rows argument, and the data says
it does not bite in practice here.

Not landed: widening `minShift`'s applicability is a default change that is
entangled with the `cflFactor` default change (it is only better *at* the
published CFL — at 0.3 it is catastrophically worse). The two should land
together or not at all, and that is the owner's call. `forceShiftPressureGauge`
ships default-off so the pairing can be re-measured on demand.

#### What this retracts from the previous section

Part 8's MINRES scoping said the 2.15x accuracy win "is a *periodic,
complete-support* result, and it does not transfer to a wall". **The premise of
that sentence is wrong** — the bounded case has complete support. The
observation stands (MINRES gave 3% there, against 115% on the periodic case);
the explanation does not. The better-supported explanation is Part 5's, which
never depended on support: near a wall the error is set by the boundary
treatment and by particles crossing it, not by how well the PPE is solved —
Part 5 measured quadrupling the divergence-free iteration cap buying only ~10%.
Solver quality cannot fix an error that is not a solver error.

## Part 9 — the terms the papers do not compute for boundary particles (2026-08-28)

Raised by the project owner: the published solvers do not evaluate every
operator term for boundary particles, "particularly on the diagonal elements",
with SPlisHSPlasH (`~/dev/SPlisHSPlasH`) as the reference implementation. Both
halves check out. **Two terms are involved, this codebase computes both, and
removing them is worth 5.9x on the bounded case's density error — but only in
one of the step's two solves, and only at the published CFL.** New:
`BoundaryOperatorTerms` (config, default `full` = unchanged) and
`scripts/probe_boundaryOperatorTerms.py`.

### What SPlisHSPlasH actually computes

Read from source, with Akinci2012 boundaries (the `Bender2019`/`Koschier2017`
map variants do the same thing with `Vj`/`gradRho` in place of the neighbour
sum):

| term | SPlisHSPlasH | boundary neighbour `j` |
|---|---|---|
| `computeDFSPHFactor`, first sum `grad_p_i` | `TimeStepDFSPH.cpp:774-812` | **in** |
| `computeDFSPHFactor`, second sum `sum_grad_p_k` | same | **out** |
| `computePressureAccel` | `TimeStepDFSPH.cpp:954-1010` | in, but with `p_i/rho_i^2` only |
| `compute_aij_pj` (the matvec) | `TimeStepDFSPH.cpp:1042-1099` | in, but with `a_i` only |
| IISPH `dii` | `TimeStepIISPH.cpp:170-206` | **in** |
| IISPH `dij_pj` | `TimeStepIISPH.cpp:360-380` | **out** (no pressure to carry) |
| IISPH `pressureSolveIteration` sum | `TimeStepIISPH.cpp:401-430` | in, but with `dij_pj_i` only |

(The `TimeStepDFSPH` citations are the `USE_AVX` variants; the scalar copies at
`:1106-1423` are term-for-term identical, checked.)

The rule behind every row is one sentence, and [BK] 3.2 states it: "since
`F^p_{j<-i} = 0` if particle j is not dynamic, the equation for `kappa^v_i`
must be adapted accordingly for static boundary particles." A particle that
never moves takes no reaction force, so every term describing *the neighbour's*
response to `p_i` is absent; every term describing `i`'s own response to the
neighbour stays.

`compute_aij_pj` is worth quoting because it is this codebase's operator, line
for line — "`\sum_j a_ij * p_j = h^2 \sum_j V_j (a_i - a_j) * gradW_ij`", which
is exactly `dt**2 * computePressureShiftIISPH(computePressureAccelIISPH(p))`.
Its boundary loop is `V_j * a_i . gradW`: same sum, `a_j` deleted.

One inconsistency worth recording, since it is a trap for anyone reading
SPlisHSPlasH as ground truth: **its IISPH keeps `-d_ji` in `aii`'s boundary
loop** (`TimeStepIISPH.cpp:287-292`) — the neighbour-reaction term its own
matvec drops. Its DFSPH does not have that mismatch. [I] 4 puts boundary
particles in `d_ii` only, so the paper agrees with the DFSPH file.

### What this codebase computed, and the two terms that differ

`wp_alpha.py` accumulates both sums over one `AllToAll` neighbour set, and
`computePressureShiftIISPH` takes the full `(a_j - a_i)` difference for every
neighbour. So relative to the published formulation there are exactly two
extra terms, and they are one physical statement applied to the diagonal and
to the off-diagonal:

- `alpha`'s second sum `sum_j V_j^2/m_j |gradW_ij|^2` — `dp_i/dx_j`;
- the divergence's `a_j` term — the neighbour's own pressure displacement.

`schemes/dfsph.py:256-259` zeroes `dxdt`/`dvdt` for every `kind != 0` row, so
boundary *and* ghost particles are static here by construction and both terms
are unearned. `BoundaryOperatorTerms.staticBoundary` drops both together;
`diagonalOnly`/`operatorOnly` drop one each, as diagnostics.

### Measurement 1 — how much of the diagonal the disputed term is

`probe_boundaryOperatorTerms.py --mode diag`, bounded case, nx=128, 120 steps,
`cflFactor=0.1`. Diagonals extracted exactly (one unit-vector matvec per row
per operator), binned by distance to the wall in particle spacings:

| depth | n | `alpha` full | `alpha` static | static/full | `diag_full/a_full` | `diag_stat/a_stat` | `diag_stat/a_full` | `diag_full/a_stat` |
|---|---|---|---|---|---|---|---|---|
| (<0, inside) | 84 | -7.678e-3 | -3.449e-3 | **0.449** | 0.984 | 0.801 | 0.372 | 224.0 |
| [0,1) | 436 | -8.244e-3 | -4.978e-3 | **0.604** | 1.009 | 1.003 | 0.615 | 1.665 |
| [1,2) | 621 | -8.253e-3 | -7.662e-3 | 0.928 | 1.000 | 0.994 | 0.899 | 1.112 |
| [2,3) | 455 | -8.523e-3 | -8.488e-3 | 0.996 | 1.002 | 1.002 | 0.994 | 1.009 |
| [3,4) and beyond | — | — | — | **1.000** | 1.003 | 1.003 | 1.003 | 1.003 |

Two things. **The disputed term is 40% of the diagonal in the wall-adjacent
bin** (55% for particles that have crossed the wall), and exactly zero beyond
3 spacings — so this is a wall effect and nothing else; on a periodic case the
setting is a no-op by construction, not by tuning. And **both self-consistent
pairings are the true diagonal to within 1%** (columns 4 and 5), which is what
makes `staticBoundary` a different *equation* rather than a rescaled solver.
The two mismatched pairings are not: at the wall `diagonalOnly` runs the Jacobi
step at `omega/0.61` (1.6x too large) and `operatorOnly` at `omega/1.67`.

### Measurement 2 — end to end

Same case, nx=128, 900 steps, stock settings otherwise (so
`shiftPressureGauge` falls back to the clamp, as shipped).

At the published CFL (`cflFactor=0.1`, Part 8 item 3):

| mode | steps | diverged | `rho` range | `｜rho-1｜` 2nd half | t_final | wall s |
|---|---|---|---|---|---|---|
| `full` (shipped) | 901 | no | [0.902, 1.247] | 1.78e-1 | 4.690 | 113.1 |
| **`staticBoundary`** | 901 | **no** | **[0.950, 1.007]** | **3.00e-2** | **6.347** | 118.6 |
| `diagonalOnly` | 48 | **yes** | [0.905, 1.140] | 9.29e-2 | 0.287 | 4.0 |
| `operatorOnly` | 901 | no | [0.946, 1.009] | 2.94e-2 | 6.372 | 117.6 |

**`staticBoundary` is a 5.9x reduction in density error at the same step
budget and the same cost**, with `rho_max` down from 1.247 to 1.007 — i.e. the
near-wall pile-up that Part 5 root-caused largely stops happening — and 35%
more simulated time per step budget because the adaptive `dt` grows.
`diagonalOnly` NaNs at step 47, exactly as measurement 1 predicts (a 1.6x
oversized relaxation at the wall, on top of a 0.5 relaxation factor that is
already at the edge of the measured window). `operatorOnly` matches
`staticBoundary` on accuracy while running its wall rows at ~0.6x the intended
relaxation, i.e. it buys the same equation with extra damping it did not ask
for.

At the shipped CFL (`cflFactor=0.3`, 3x [BK]'s published limit) everything
dies, and `staticBoundary` dies *sooner*:

| mode | steps | t_final | `rho` range | `｜rho-1｜` 2nd half |
|---|---|---|---|---|
| `full` | 258 | 5.535 | [0.139, 2.452] | 5.38e-1 |
| `staticBoundary` | 80 | 1.412 | [0.840, 1.271] | 8.69e-2 |
| `diagonalOnly` | 14 | 0.224 | [0.670, 1.143] | 2.75e-1 |
| `operatorOnly` | 454 | 5.894 | [0.143, 2.477] | 3.26e-1 |

Same shape as the `minShift` result in the previous section: the smaller
`|alpha|` at the wall means bigger Jacobi steps, and at 1.2 particle spacings
of displacement per timestep that is not survivable. (`operatorOnly` living
longest here is the same fact from the other side — its oversized diagonal is
extra damping.) **So this is a published-CFL result, entangled with the
`cflFactor` default exactly the way `minShift` is: the three would land
together or not at all.**

### Measurement 3 — it belongs to only one of the two solves

The step runs two solves, and the config setting reaches both. Scoping it to
one at a time (`--solvers`, which forces the other back to `full` around its
call), `cflFactor=0.1`, same 900 steps:

| `staticBoundary` applied to | steps | diverged | `rho` range | `｜rho-1｜` 2nd half | t_final |
|---|---|---|---|---|---|
| neither (`full` baseline) | 901 | no | [0.902, 1.247] | 1.78e-1 | 4.690 |
| `solveDivergenceFree` only | 283 | **yes** | [0.731, 2.321] | 1.46e-1 | 1.645 |
| **`solveIncompressible` only** | 901 | no | [0.948, 1.011] | **2.88e-2** | **6.586** |
| both | 901 | no | [0.950, 1.007] | 3.00e-2 | 6.347 |

**The win belongs to the constant-density/shifting solve, and applying the
change to the divergence-free solve *alone* is actively harmful** — on its own
it turns a finite run into a divergence at t=1.65.

> **The mechanism first proposed here was wrong and is retracted** — it
> attributed the divergence-free harm to mDBC boundary particles carrying an
> extrapolated *velocity* (true: mean `|v|` 0.063 against the fluid's 0.327 at
> that solve's call site, and exactly 0 at the other's) and so not being
> "static" in the quantity that solve projects. Zeroing them does not fix it.
> Three candidate mechanisms were then tested and all three fail; see the
> addendum below for what survives.

**So the correct scoping is per-solve, and a future default should live on
`RelaxedJacobiSolverConfig` (which already exists per solver) rather than on
`IncompressibleSolverConfig`.** The single global knob shipped here is an
experiment hook, and it is safe as one — the `both` row survives, is 5.9x
better than the baseline, *and* converges better than it on both solves (final
residuals 1.57e-2/1.44e-3 against 7.52e-2/2.51e-3, see the addendum). The
harm is a property of the mismatched half-state, not of the divergence-free
solve as such.


### Part 9 addendum — the boundary velocity, and whether the slip conditions work (2026-08-28)

The project owner's follow-up: how much of the divergence-free harm above is
the *boundary velocity*? In DFSPH a boundary particle's velocity is the rigid
body's — a constant zero for a static wall — not an extrapolated fluid
velocity, so the question is decidable by setting it to zero and re-running.
And separately: do the free-slip (mirror into the fluid) and no-slip (flip the
fluid velocity) conditions work, and do they change convergence? New tooling:
`scripts/probe_boundaryVelocityModes.py`, plus `--mode spectrum` and
`--mode dfTrace` on `scripts/probe_boundaryOperatorTerms.py`.

**Everything below is measured at `cflFactor=0.1`, nx=128, 900 steps, on
`randomFlowIncompressible --bounded`, which ships `BCType.freeSlip`.**

#### The boundary velocity is not the mechanism — the previous section's explanation is retracted

| `BCType` | terms | `rho` range | `｜rho-1｜` 2nd half | t_final | DF resid | PS resid |
|---|---|---|---|---|---|---|
| `freeSlip` (shipped) | `full` | [0.902, 1.247] | 1.78e-1 | 4.690 | 7.52e-2 | 2.51e-3 |
| `freeSlip` | `staticBoundary` | [0.950, 1.007] | 3.00e-2 | 6.347 | 1.57e-2 | 1.44e-3 |
| `zeros` (the DFSPH convention) | `full` | [**0.563**, 1.289] | 2.48e-1 | 5.028 | 8.93e-2 | 3.05e-3 |
| `zeros` | `staticBoundary` | [0.957, 1.006] | **2.58e-2** | 6.386 | 1.53e-2 | 1.47e-3 |
| `noSlip` | `full` | [0.863, 1.237] | 1.73e-1 | 4.348 | 8.57e-2 | 2.37e-3 |
| `noSlip` | `staticBoundary` | [0.952, 1.007] | 3.24e-2 | 6.286 | 1.81e-2 | 1.48e-3 |

Two things fall out immediately. **The boundary-velocity condition is a small
effect and the operator terms are a large one**: across all three conditions
`staticBoundary` buys 5.4-6.4x, while switching condition moves `｜rho-1｜` by
at most 40% at fixed terms. And **`zeros` is the best of the three under
`staticBoundary`** (2.58e-2), which is the consistent answer — with a genuinely
motionless wall the static-boundary operator is exactly right — but it beats
`freeSlip` by only 14%.

The decisive run is the scoped one. Under `zeros`, `staticBoundary` applied to
the divergence-free solve *alone* still diverges — at step 482/t=2.91 rather
than 283/t=1.64, i.e. **delayed by 70% and not prevented**. So the retracted
explanation had the right ingredient list and the wrong conclusion: the
extrapolated velocity does make it worse, and removing it does not make it go
away.

#### Two more mechanisms tested, both eliminated

**Not the Jacobi stability window.** Both solvers iterate
`p <- p + omega * D^-1 r`, which converges iff `omega < 2/rho(D^-1 A)`, and
`rho` is scale-free (`A` and `D` both carry the `dt` factor) so one number
covers both. Power iteration on the fluid subproblem, same captured state:

| operator | `rho(D^-1 A)` | window `2/rho` | `omega`/window | dominant mode's energy within 2 spacings of the wall |
|---|---|---|---|---|
| `full` | 6.3777 | 0.3136 | 0.957 | 0.2% (7.0% of rows are there) |
| `staticBoundary` | 6.3782 | 0.3136 | 0.957 | 0.2% |

Identical to four digits, and the dominant mode is a *bulk* high-frequency mode
either way, not a wall mode. **Incidental finding worth its own line: the
shipped `relaxationFactor = 0.3` sits at 96% of the stability window on this
state.** `JacobiRelaxationMode`'s docstring quotes "~15% margin", measured on
the TGV operator family; on a wall-bounded state the margin is 4%.

**Not the iteration budget either.** If the new operator merely converged more
slowly, more sweeps would fix it. Raising the divergence-free cap from its
default 32 (`staticBoundary`, divergence-free only):

| DF `maxIterations` | steps | t_final | `rho` range |
|---|---|---|---|
| 32 (default) | 283 | 1.645 | [0.731, 2.321] |
| 96 | 597 | 3.370 | [0.272, 1.314] |
| 192 | 513 | 3.375 | [0.141, 3.145] |

Tripling delays it, and 6x is *worse* than 3x. Not an under-iteration.

#### What is actually measured: a weaker contraction whose leftover accumulates

`--mode dfTrace` records every divergence-free solve's first and last residual
over 300 steps (constant-density solve pinned to `full` throughout):

| | residual last/first, steady state | solves whose residual grew | max`｜a_p｜` at step 250 | outcome |
|---|---|---|---|---|
| `full` | 0.39 – 0.53 | **0/300** | 17.1 | finite |
| `staticBoundary` (DF only) | 0.77 – 0.82 | 4/283 (all late) | 19.8 -> **1.04e4** at step 276 | NaN at 282 |

Each solve still converges internally — the iteration is not unstable, matching
the spectrum result — but it removes only ~20% of the incoming divergence
instead of ~50%, the incoming residual creeps from 2.4e-2 to 3.8e-2 over 250
steps, and the applied acceleration ramps and then detonates. **That is the
honest end of this thread: the observable is a halved contraction rate in the
mismatched half-state, and no mechanism proposed so far predicts it.** Note the
sign of the effect flips when both solves are changed together — the `both` row
above has the *best* divergence-free residual of any configuration (1.57e-2
against `full`'s 7.52e-2, a 4.8x improvement), so whatever this is, it is a
property of running the two solves on inconsistent operators rather than of the
static-boundary operator itself. Which is another argument for the per-solver
config split, and against ever setting this globally to a mixed state.

#### The slip conditions do not implement their own formulas

`--mode verify` decomposes each `BCType`'s output against the wall normal,
reported as least-squares slopes over the Shepard-interpolated fluid velocity
at the ghost point. The published mDBC forms are exact integers, so the table
grades itself:

| `BCType` | normal | tangential | published | `｜u_b｜` mean | rows set |
|---|---|---|---|---|---|
| `zeros` | +0.0000 | +0.0000 | 0 / 0 | 0 | 0/2660 |
| `constant` | (n/a) | (n/a) | 0 / 0 | 0 | 0/2660 |
| `noSlip` | **-0.0000** | -1.0000 | **-1** / -1 | 9.43e-2 | 2660/2660 |
| `freeSlip` | **+0.0000** | +1.0000 | **-1** / +1 | 9.43e-2 | 2660/2660 |
| `extended` | -0.6490 | +0.5605 | (extrapolated) | 1.13e-1 | 2660/2660 |

Both slip conditions get the tangential component exactly right and **both drop
the normal component instead of reflecting it**. `freeSlip` computes
`u_f - 1*(u_f.n) n` while the comment directly above it says
`u_g = u_f - 2 * (u_f . n_b) * n_b` (`modules/mdbc/velocity.py:118`); `noSlip`
computes `2 u_wall - u_f` and then applies the same projection, undocumented.
The physical consequence is specific: a boundary particle never opposes an
approaching fluid particle's normal velocity, so the wall contributes only half
the compression signal to `computeMomentumIncompressible`'s divergence — which
is the source term both solvers are driven by, and it is the same wall-normal
response that `computeMdbcNoPenShift` exists to supply by other means.

**Fixing it is not an improvement, measured** (`freeSlipReflect` runs the
published `-2` form, patched in; `src/` unchanged):

| `BCType` | `mdbcNoPenetrationShift` | `｜rho-1｜` | t_final | min `rho` |
|---|---|---|---|---|
| `freeSlip` | on (default) | 1.78e-1 | 4.690 | 0.902 |
| `freeSlipReflect` | on | 1.70e-1 | 4.327 | 0.839 |
| `freeSlip` | off | 1.93e-1 | 5.088 | 0.666 |
| `freeSlipReflect` | off | 2.30e-1 | 3.868 | 0.468 |

So the reflecting form is a wash with the no-penetration shift on and *worse*
with it off — which also disposes of the tidy hypothesis that
`mdbcNoPenetrationShift` is a crutch standing in for the missing reflection. It
is not: removing the crutch does not make the reflection pay. And paired with
`staticBoundary` the reflecting form is worse still (`｜rho-1｜` 7.98e-2 and
`t_final` 1.43 against 3.00e-2 and 6.35 — it survives 900 steps but the
adaptive `dt` collapses). **Report the deviation, do not "fix" it blind.**

#### `noSlip`'s moving-wall term is dead code

`noSlip` computes `2 * currentState.velocities - u_f` and then reads the result
at the *ghost* rows, so the `u_wall` it uses is the ghost row's stored velocity.
Nothing writes a body velocity there: `rigidBody/update.py:51-56` refreshes
ghost velocities only for `BCType.constant` bodies. Measured on
`lidDrivenCavity` (nx=64, 50 steps), the one case that uses `noSlip` on a
moving wall:

```
pre-BC |v|     fluid: mean=6.27e-04 max=1.47e-02
pre-BC |v|  boundary: mean=2.68e-01 max=1.00e+00     <- the lid is here
pre-BC |v|     ghost: mean=0.00e+00 max=0.00e+00     <- and not here
```

The lid velocity reaches the boundary particles and never the ghosts, so the
`2 u_wall` term contributes exactly zero and `noSlip` degenerates to
`u_b = -u_f_tangential` — a *stationary* no-slip wall — no matter how fast the
wall moves. The case is not visibly broken because `enforceDirichlet` runs
*after* `computeBoundaryVelocities` (`schemes/deltaSPH.py:110-117`,
`schemes/dfsph.py:85-96`) and re-imposes the lid velocity on the boundary rows
it just overwrote. So today this is latent: it would bite the first moving
no-slip wall that is not backed by a Dirichlet condition. Moving *rigid bodies*
are unaffected — they use `BCType.constant`, which is the branch that does
refresh ghost velocities.

#### Convergence behaviour, since it was asked directly

Iteration counts do not discriminate: **every configuration in every table
above runs its full budget every step** (32 divergence-free, 64
constant-density) and never reaches its tolerance, so the count is pegged and
the *residual* is the only usable signal. On that measure the boundary-velocity
condition is again a small effect (DF residual 7.5e-2 / 8.9e-2 / 8.6e-2 for
free-slip / zeros / no-slip at fixed `full` terms) and the operator terms are a
large one (7.5e-2 -> 1.6e-2). This also re-confirms Part 8's stopping-criterion
finding from a third direction: a solver that never terminates cannot report
that a change helped it, which is why every number here is a residual.

### Status

Not landed as a default; nothing in the shipped behaviour moves
(`BoundaryOperatorTerms.full`). Verified: `scripts/gradcheck_incompressible.py`
passes, including a new second case covering the `includeBoundaryReaction=False`
branch with mixed `kinds` in one launch (a conditional accumulation inside the
neighbour loop is exactly the shape that file's docstring is about), and the
full suite passes (241 passed, 1 skipped). Periodic cases are untouched by construction — with no
`kind != 0` particles both code paths are identical, which measurement 1's
"1.000 beyond 3 spacings" column also shows empirically.

**Unrelated flake seen while verifying, worth its own investigation:** one full
suite run in three failed `test_implicitShiftingComparison.py`'s two
`implicitShiftAutomatic` assertions. Its relative density std falls smoothly
for six steps (0.0257 -> 0.0145) and then jumps to 0.217 on step 7 and stays
there. The file passes in isolation and the suite passes on a re-run, and
nothing in Part 9 touches the shifting path, so this is a pre-existing
intermittent blow-up in the autodiff-built Newton shift, not a regression --
but "passes 2 runs in 3" is not passing.

Open, in the order they matter:

1. **Move the knob to `RelaxedJacobiSolverConfig`** and default
   `pressureSolver` to `staticBoundary` while `divergenceFreeSolver` stays
   `full`. That is the configuration measurement 3 endorses, and it cannot be
   expressed today. Note the addendum's caveat: `both` is *also* better than
   the baseline on every axis including both solvers' residuals, so the choice
   between "PS only" and "both" is a 4% accuracy question, not a stability one.
2. **Re-measure against `minShift` and the CFL together.** All three of
   `cflFactor=0.1`, `ShiftPressureGauge.minShift` (via
   `forceShiftPressureGauge`) and `staticBoundary` are individually better at
   the published CFL and individually worse at 0.3. Nobody has run the 2x2x2.
3. **The divergence-free half-state's contraction collapse is unexplained.**
   Three mechanisms tested and eliminated (addendum). The remaining lead is
   that it only appears when the two solves run *inconsistent* operators, which
   suggests looking at what the constant-density solve leaves behind for the
   next step's divergence-free solve rather than at either solve alone.
4. **`relaxationFactor = 0.3` has a 4% stability margin on a bounded state**,
   not the ~15% `JacobiRelaxationMode`'s docstring quotes from the TGV family.
   Independent of everything else in Part 9 and probably the cheapest
   robustness win available; `--mode spectrum` measures it on demand.
5. **The two mDBC slip defects** (normal component projected rather than
   reflected, contradicting `freeSlip`'s own comment; `noSlip`'s `2 u_wall`
   term reading a ghost-row velocity nothing writes). Neither is worth fixing
   for stability — the reflecting form measures *worse* — but the comment and
   the code should be made to agree, and the moving-wall path needs either a
   fix or a docstring saying it only works behind a Dirichlet.
6. **Ghost particles (`kind == 2`) are lumped in with boundaries** by the
   `kj == 0` test. That is right for this scheme (`dfsph_step` freezes them
   too), but it is an assumption no measurement here separates.

## Part 10 — the initial sampling, and integrating the density instead of re-summing it (2026-08-28)

Two requests from the project owner, both about where the density field comes
from rather than what the solvers do with it. New tooling:
`scripts/probe_initialSampling.py`, `scripts/probe_densityEvolution.py`.

### The sampling is exact, the mass is not, and correcting it makes the bounded case worse

`sample/regular.py:70` gives every particle the nominal mass `rho0 * dx^d`.
That is the continuum mass of its cell, not the mass that makes the *discrete*
summation `sum_j m_j W_ij` land on `rho0`. `kolmogorovIncompressible.py:71` and
`tgv.py:53` already correct for it; nothing built through `buildRegionSystem`
does, which is every `randomFlow` variant.

Measured on `randomFlowIncompressible --bounded`, nx=128, as sampled:

| | |
|---|---|
| `m / dx^2` | 1.000000 (exactly nominal) |
| fluid summation density | **1.001206**, std **0.000000**, min = max |
| by wall depth, `[0,1)` through `[10,inf)` | 1.001206 in **every** bin |
| normalisation factor, mean / bulk / median rules | 0.998796 (identical to 6 digits) |

So the sampling itself is perfect — a uniform lattice with a uniform density —
and the wall band does not perturb it at all, which is the third independent
confirmation that `BOUNDED_BAND = 5` gives complete support. The entire error
is a single global **+0.12%**, and any of the three candidate rules removes it
exactly, in one shot, because the summation is linear in mass.

It shows up exactly where it should and then stops mattering. The first twelve
steps' `max(|rho_max - 1|, |rho_min - 1|)`:

| rule | step 0 | 1 | 2 | 3 | 6 | 9 | 11 |
|---|---|---|---|---|---|---|---|
| `none` (shipped) | 0 | **1.21e-3** | 8.08e-3 | 9.42e-3 | 8.80e-3 | 1.37e-2 | 2.29e-2 |
| `meanAll` | 0 | **7.51e-6** | 5.10e-3 | 1.00e-2 | 2.48e-2 | 3.93e-2 | 4.90e-2 |

**There is no startup shock to fix.** Step 0 is exactly uniform, step 1 shows
the 1.2e-3 sampling bias precisely as predicted, and normalisation removes it
to 7.5e-6 — and then the normalised run's error grows about twice as fast from
step 2 onward. Over 900 steps:

| case | rule | peak `｜rho-1｜` in first 50 | first-50 mean | 2nd-half mean |
|---|---|---|---|---|
| bounded | `none` | 1.09e-1 | 6.37e-2 | **1.78e-1** |
| bounded | `meanAll` | 1.46e-1 | 9.53e-2 | 2.40e-1 (**35% worse**) |
| periodic | `none` | 5.87e-3 | 3.95e-3 | 4.13e-3 |
| periodic | `meanAll` | 6.69e-3 | 2.87e-3 | **2.89e-3 (30% better)** |

**Normalisation helps the periodic case and hurts the bounded one**, and that
sign flip is the same knob Part 8 item 1 swept, now measured in the direction
it did not cover. `none` leaves the lattice 0.12% *above* `rho0`, i.e. the
setpoint sits *below* the structural floor, which biases `sourceTerm =
rho0 - rhoStar` negative — and Part 8 established that the source's negative
mean is the shifting solve's de-clumping drive. Removing it at a wall, where
that drive is doing the most work, costs 35%. Removing it in a periodic domain,
where the drive is not needed to hold particles off a boundary, is a 30% win.
Not landed either way: `none` is shipped, the periodic cases already normalise
in their own `buildSystem`, and the bounded evidence argues against changing
`buildRegionSystem`.

### `integrateRho` is now real, and the WCSPH convention does not survive contact with this scheme

Part 3 audited `integrateRho` and found the `True` branch inert: `finalize`
re-summed unconditionally, twice per step in total. `DensityEvolution`
(new, default `summation` = byte-identical history) makes the choice real, with
the two re-sums separately controllable because they answer different questions:

- `summation` — re-sum at the top of `dfsph_step` and again in `finalize`.
- `continuity` — integrate `drho/dt`, never re-sum. The WCSPH standard.
- `hybrid` — carry the integrated density through the step (so `drhodt`, the
  mDBC extrapolation and the divergence-free solve all run on it) but give the
  *constant-density/shifting* solve a fresh summation that is then discarded.

`integrateRho=True` now maps to `continuity` via `resolveDensityEvolution`,
which is what its name always claimed.

#### One real bug found on that path, worth 1800x on the solver residual

`dfsph_step` computes `drhodt = computeMomentum(...)` from
`currentState.velocities` — the velocity *before* the divergence-free
projection — and that is what the integrator advances density with. Under
`summation` it does not matter (the result is discarded). Under `continuity` it
is fatal: the particles are advected with the projected velocity, whose
divergence the solve just drove to ~0, so **the carried density integrates
exactly the error the solver was there to remove, every step.** Re-evaluating
`drhodt` on the projected velocity (same operator the solvers form `rhoStar`
with) on the bounded case:

| | steps survived | DF final residual |
|---|---|---|
| before | 25 | **2.61e+2** |
| after | 63 | **1.46e-1** |

The divergence-free solve goes from meaningless to healthy — `summation`'s own
residual on the same case is 7.5e-2 — and the run lasts 2.5x longer. Also
added, and much less useful: the shift's own first-order density resample
(`rho += grad rho . dx`, the density half of the `proj_vel` correction that has
always been there). Worth 24 -> 25 steps on its own. Kept because it is
correct and free, not because it helped.

#### The drift is intrinsic, and it is not the shift

Every number below is reported twice — against the **carried** density (what
the solvers and the case diagnostics see) and against a fresh **summation** on
the same positions. nx=128, 900 steps, `cflFactor=0.1`:

| case | mode | steps | `｜carried-1｜` | `｜true-1｜` | drift mean | DF resid | t_final | wall s |
|---|---|---|---|---|---|---|---|---|
| bounded | `summation` | 901 | 7.04e-3 | 7.04e-3 | 0 | 7.52e-2 | 4.690 | 114.6 |
| bounded | `continuity` | **63** | 7.06e-3 | 4.30e-2 | 4.74e-2 | 1.46e-1 | 0.370 | 3.9 |
| bounded | `hybrid` | **286** | 2.27e-2 | 2.52e-3 | 2.14e-2 | 1.67e-1 | 1.713 | 30.2 |
| periodic | `summation` | 901 | 1.93e-3 | 1.93e-3 | 0 | 8.79e-3 | 5.873 | 73.2 |
| periodic | `continuity` | **470** | 9.05e-3 | 3.66e-2 | 3.93e-2 | 7.25e-1 | 2.851 | 38.2 |
| periodic | **`hybrid`** | **901** | 3.52e-2 | **1.92e-3** | 3.33e-2 | **8.49e-3** | 5.872 | 72.6 |

Look at the `continuity` rows' first two columns: **the carried density reports
7.1e-3 while the truth is 4.3e-2.** It is not noisy, it is *smooth* — and
wrong. `drho/dt = -rho div v` describes advection of a density field; the real
density error in this scheme is dominated by particle *rearrangement* at
essentially zero divergence, which that equation cannot represent at all. The
carried field converges to a plausible-looking lie.

The obvious suspect is the position shift, which moves particles every step
with no `drhodt` to match. **It is not the cause.** Under
`ShiftApplication.inStepVelocity`, which drops the position shift entirely,
`continuity` on the bounded case still dies (step 290) with the same drift
(3.46e-2). The rearrangement the continuity equation cannot see is the flow's
own, not the shift's.

#### `hybrid` is free where support is complete, and fails at an mDBC wall

The periodic `hybrid` row is the result worth keeping: **`｜true-1｜` 1.917e-3
against `summation`'s 1.926e-3, the same `t_final` to four digits, and a
slightly better divergence-free residual** — while its carried density has
drifted 3.3e-2 from the truth. That is the whole design point stated as a
measurement: *the divergence-free solve does not care*, because all it needs is
`div v`, which the continuity equation tracks exactly; only the shifting solve
needs a density that matches the particle positions, and `hybrid` hands it one.

On `tgv` (nx=64, 200 steps) the same holds with a cost saving attached:

| mode | `｜true-1｜` | DF resid | KE last/first | wall s |
|---|---|---|---|---|
| `summation` | 1.608e-4 | 1.338e-3 | 0.9956 | 2.4 |
| `continuity` | 1.310e-3 (8x worse) | 1.328e-3 | 0.9956 | 1.8 |
| **`hybrid`** | **1.588e-4** | 1.336e-3 | 0.9956 | **1.9 (-21%)** |

Identical kinetic-energy decay in all three, so the mode is energy-neutral on
the case `tests/test_physics.py` grades. The 21% is the second summation being
skipped; at nx=128 on `randomFlow` the same saving is ~1%, because there the
two pressure solves (32 and 64 iterations, two SPH passes each) dominate and a
density pass is noise.

**But `hybrid` dies at 286 steps on the bounded case.** Leading hypothesis,
untested: `computeMdbcDensity` runs at the top of the step, on the *carried*
density, so under `hybrid` the boundary rows are extrapolated from a field that
has drifted — and the wall force is built from those boundary densities. The
periodic case has no mDBC, which is exactly the difference. The cheap test is
to re-sum for the extrapolation only and see whether the bounded case then
behaves like the periodic one.

### Status

Nothing shipped moves: `DensityEvolution.summation` is the default and the
non-default branches are guarded, `integrateRho` keeps its serialised field,
and the full suite passes. Also fixed while here: `densityEvolution`,
`boundaryOperatorTerms` and `forceShiftPressureGauge` were not serialised by
`incompressibleSPHConfigToDict`, so they silently reset to defaults on a
TOML/HDF5 round-trip; all three now round-trip.

Open, in the order they matter:

1. **Test the mDBC hypothesis for `hybrid`.** If re-summing for the boundary
   extrapolation alone fixes the bounded case, `hybrid` becomes a general
   default candidate rather than a periodic-only one.
2. **`hybrid` on the periodic cases is a free option today** — same accuracy,
   same energy, one fewer neighbour pass. It wants a `tgv`-style regression
   test before it could ever be a default, since its failure mode (a carried
   density that looks *better* than the truth) is invisible to any diagnostic
   that reports the carried field.
3. **Adopt the mass normalisation for periodic `buildRegionSystem` cases only**
   (30% better), and leave the bounded ones alone (35% worse). Or, better,
   understand the sign flip first — it is the same setpoint question Part 8
   item 1 opened and neither session has closed.
4. **`DensityEvolution` + `BoundaryPressureMode.plain` is a trap**: `plain`
   skips the mDBC extrapolation, and the non-summation modes skip the re-sum,
   so `kind != 0` rows would never be updated at all. Documented in the enum,
   not guarded in code.

## Part 11 — Bender, Westhofen & Jeske 2023, "Consistent SPH Rigid-Fluid Coupling" (2026-08-28)

The paper [BWJ23] (VMV 2023) derives DFSPH from a density *constraint* instead
of from the PPE, and the derivation settles the boundary question Part 9 was
circling. New: `BoundaryPressureMode.consistent`,
`modules/incompressible/consistent.py`, `scripts/probe_consistentCoupling.py`.

### Two thirds of it was already here

Their constraint `C_i = rho0 - sum_j m_j W_ij - sum_k m~_k W_ik` is defined for
**fluid particles only**, and a static boundary particle's position is constant,
so `dC_i/dx_k = 0`. Term by term against this codebase:

| paper | what it says | this codebase |
|---|---|---|
| Eq. 32, diagonal | boundary in `‖sum_j m_j gradW_ij + sum_k m~_k gradW_ik‖²`, **not** in `sum_j ‖m_j gradW_ij‖²` | **`BoundaryOperatorTerms.staticBoundary`'s alpha half** (Part 9) |
| Eq. 34, Laplacian | boundary term is `sum_k m~_k a^p_i . gradW_ik` — no `a^p_k` | **`staticBoundary`'s operator half** (Part 9) |
| Eq. 33, pressure accel | `-sum_k m~_k (p_i/rho_i²) gradW_ik` — **no boundary pressure at all** | an *identity* for this codebase's symmetric gradient whenever `p_b = 0`, i.e. under `plain`/`mdbcDensity` |
| Eq. 14, density | `rho_i = sum_j m_j W_ij + sum_k m~_k W_ik` | `computeDensities` — same, with nominal `m~_k = rho0 dx^d` |
| Sec. 3.3, boundary state | "static fluid particles" at `rho_k = rho0`, `m~_k = rho0 / sum_l W_kl` | **differs**: mDBC-extrapolated `rho_k` (1.3+ in a compressed band), nominal mass |
| Eq. 35, dynamic boundaries | `f_{k<-i} = m_i m~_k (p_i/rho_i²) gradW_ik` onto the body | **absent**: `integrateRigidBody(rigidBody, 0, 0, dt)` — this scheme is one-way coupled by construction |

So Part 9 arrived at the paper's operator from SPlisHSPlasH's source without
the derivation, and the paper is the derivation. **That closes Part 9's "the
`staticBoundary` win is unexplained at the theory level" gap**: it is not a
tweak, it is what the constraint formulation produces when you stop pretending
a wall can move.

What is genuinely new is the *state* boundary rows enter the solve with, and
it does reach the solve: every SPH sum in it weights a neighbour by its
apparent volume `m_j / rho_j`. `BoundaryPressureMode.consistent` is the paper
end to end — `staticBoundary` terms forced on, `p_b` pinned at exactly 0,
`rho_k = rho0` for the duration of the solve (restored afterwards, so the mDBC
extrapolation still serves everything outside it).

### It is the best configuration measured, and `mdbcMlsPressure` is the worst

`randomFlowIncompressible --bounded`, nx=128, 900 steps, `cflFactor=0.1`:

| configuration | `rho` range | `｜rho-1｜` 2nd half | t_final | DF resid | PS resid | wall s |
|---|---|---|---|---|---|---|
| `mdbcDensity` + `full` (shipped) | [0.902, 1.247] | 1.78e-1 | 4.690 | 7.52e-2 | 2.51e-3 | 122.7 |
| `mdbcDensity` + `staticBoundary` | [0.950, 1.007] | 3.00e-2 | 6.347 | 1.57e-2 | 1.44e-3 | 118.7 |
| **`consistent`** | [0.955, 1.008] | **2.86e-2** | **6.463** | 1.55e-2 | 1.44e-3 | 119.1 |
| `consistent` + `akinciBoundaryVolume` | [0.951, 1.019] | **2.38e-2** | 6.330 | **1.24e-2** | 1.85e-3 | 120.9 |
| `mdbcMlsPressure` + `full` | [0.878, **1.334**] | **1.86e-1** | 5.336 | 9.98e-2 | 2.41e-3 | 115.6 |

**`consistent` is 6.2x better than the shipped configuration** on near-wall
density error, and the `rho_k = rho0` convention is worth a further 5% on top
of the operator terms alone (2.86e-2 against 3.00e-2) plus 2% more simulated
time. Modest — the operator terms are the big win — but free and in the same
direction.

**The other end of the table is the sharper result.** `mdbcMlsPressure` — Band
et al.'s MLS pressure extrapolation, which is precisely the method [BWJ23] is
written against — is the **worst** row measured, worse than the shipped
baseline (1.86e-1 vs 1.78e-1), with the worst divergence-free residual
(9.98e-2) and a `rho_max` of 1.33. **This corrects Part 2**, which called
`mdbcMlsPressure` "the most stable *and* most accurate of the three modes";
that was measured at `cflFactor=0.3` out to t≈1.5. Over 900 steps at the
published CFL it is 6.5x worse than the paper's method, which needs no boundary
pressure at all. The paper's argument was that extrapolation is unnecessary
overhead; here it is not merely unnecessary, it is actively harmful.

### The Akinci volume correction: helps in the operator, fatal in the density

The paper specifies `m~_k = rho0 / sum_l W_kl` (`l` over boundary neighbours
only). Measured on this five-layer band: `m~/m_nominal` mean **1.102**, min
0.999, max **1.456** — the interface layer gains ~46%, the interior layers
nothing, exactly as the correction is designed to behave at a sampling's outer
face.

Where it is applied decides everything:

| where `m~_k` is applied | result |
|---|---|
| inside the pressure solve only (`akinciBoundaryVolume`) | **best row in the table**, 2.38e-2 |
| as the particles' actual mass, so Eq. 14's density sum sees it too | **diverges at step 9** (`consistent`), and 1.82e-1 with `mdbcDensity` |

**So the faithful reading of the paper is the one that fails here, and the
reason is the deviation this codebase already made.** Akinci's correction
assumes a *one-layer* boundary, where the single layer must stand in for the
whole solid half-space. `randomFlow.BOUNDED_BAND = 5` samples five layers, so
the volume behind the interface is already represented by real particles;
adding the correction to the density double-counts it, the fluid sees a
~15% phantom compression at the wall on step 0, and the solver tears the
configuration apart in nine steps. Confined to the operator, the same factor is
not a density claim at all — it just weights the wall more heavily in `A` and
`alpha`, i.e. a stiffer wall — and that is worth 17%.

`akinciBoundaryVolume` therefore ships default-off and is **not** presented as
"the paper's convention": it is the paper's factor applied only where this
codebase's thick band lets it be applied.

### Same CFL entanglement as everything else in Parts 8-9

At the shipped `cflFactor=0.3` (3x [BK]'s published limit) all three die:

| configuration | steps | t_final | `rho` range | `｜rho-1｜` 2nd half |
|---|---|---|---|---|
| `mdbcDensity` + `full` | 258 | 5.535 | [0.139, 2.452] | 5.38e-1 |
| `consistent` | 90 | 1.557 | [0.842, 1.241] | 9.25e-2 |
| `consistent` + `akinciBoundaryVolume` | 184 | 3.677 | [0.894, 1.139] | 3.50e-2 |

Fewer steps, far tighter density — the same shape Part 9 measured for
`staticBoundary` and Part 8 for `minShift`. **The list of changes that are
better at the published CFL and worse at 3x it is now four long**
(`cflFactor=0.1`, `ShiftPressureGauge.minShift`, `BoundaryOperatorTerms.
staticBoundary`, `BoundaryPressureMode.consistent`), and they all point the
same way: the timestep default is what is holding the rest back.

### Status

Nothing shipped moves: `BoundaryPressureMode.mdbcDensity` is still the default,
`consistent` is a new opt-in value, `akinciBoundaryVolume` defaults False, and
the full suite passes. `solveIncompressible`/`solveDivergenceFree` are now thin
wrappers around `_solveIncompressibleImpl`/`_solveDivergenceFreeImpl` so the
boundary-state substitution can be scoped to the solve; every other mode passes
straight through.

One latent bug found and fixed while building this: `akinciBoundaryMass`'s
`BoundaryToBoundary` kernel sum is zero for fluid and ghost rows, and
`rho0 / 0` is a mass large enough to end a simulation. It never reached a
neighbour sum — `OperationDirection.AllToAll` and the default
`TrueAllToToAll` both exclude `kind == 2`, and fluid rows keep their own mass —
but it is now a fallback to the nominal mass rather than a landmine.

Open:

1. **The four-way default change.** `cflFactor=0.1` + `minShift` +
   `staticBoundary` + `consistent` are each individually better at the
   published CFL. Nobody has run them together, and that is now the single
   highest-value experiment in this document.
2. **`consistent` makes `BoundaryPressureMode` mostly moot.** If it lands,
   `mdbcMlsPressure` has no measured case for existing (worst row here, and the
   feedback loop Part 2 had to damp with `mdbcPressureRelaxation` is a direct
   consequence of holding boundary pressure as state, which the paper's
   formulation does not do). It should be deprecated rather than kept as a
   tuning knob.
3. **Eq. 35, two-way coupling, is absent** and unrelated to any of the above:
   `integrateRigidBody(rigidBody, 0, 0, dt)` never applies a fluid pressure
   force to a body. Every case here has static walls, so nothing is wrong
   today, but a moving-body case would silently get one-way coupling.
