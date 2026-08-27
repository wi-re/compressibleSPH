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

---

## Next session: start here

Part 4 is closed and Part 5 is diagnosed but unfixed (both are written up at
the bottom of this document). Nothing below is blocking; pick by preference.

0. **A literature session on DFSPH and its derivatives**, to establish what
   belongs where before any restructuring — the project owner's stated next
   step. Part 6 lists the questions, each tied to a specific measurement in
   this document rather than to general reading. The first of them (how
   published implementations avoid integrating the constant-density solve's
   permanent residual into momentum) is the same question item 1 below
   attacks empirically, so the two inform each other.
1. **Drive the velocity correction from the *attainable* part of the source
   only.** This is the highest-value item left, and the one experiment that
   could make a velocity mode safe enough to default to.
   `ShiftApplication.inStepVelocity` already gives the bounded case a 30x
   better near-wall density error and turns its NaN into a steady state, but
   it damps `tgv` at 3.3x the analytic rate. Part 5 continued (2) argues that
   dissipation *is* Part 4's unreachable setpoint integrated into the
   momentum equation — the constant-density solve never converges, and its
   permanent residual becomes a permanent unphysical force. Projecting the
   structurally-unreachable mean out of the source for the velocity path
   (while leaving the position shift on the raw source, where Part 4 showed
   the mean carries the de-clumping signal) should remove it. One flag on
   `solveIncompressible`'s source term. Note two refinements already tried
   and *rejected*, so they are not re-run: confining the correction to a wall
   band (diverges sooner than not doing it at all — its value is not
   wall-local) and scaling it down (no lambda satisfies both cases).
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

