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

---

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
  including far from any wall. (1) is the more actionable open item now —
  points at `computeMdbcNoPenShift` (`modules/mdbc/wp_nopenshift.py`)
  potentially not fully containing fluid particles under DFSPH — and hasn't
  been inspected yet. (2) would still benefit from the
  `kolmogorovIncompressible`-vs-`--bounded` unbounded/bounded comparison
  this bullet originally proposed, to check whether it's present even with
  no wall at all.
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
