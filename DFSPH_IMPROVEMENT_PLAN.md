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
**Part 2: config + solver-masking + MLS-pressure-projection machinery
implemented and regression-tested (steps 1, 2, 3, 4, 5, 6); case port and
validation (steps 7, 8) not started — no case currently samples boundary
particles, so the new machinery is exercised only as a verified no-op so
far.** See the per-part sections below for details.

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
7. **Port 06-randomFlow to incompressible**: add an incompressible
   `buildSystem`/`configureScheme` hook (new function alongside
   `configureWeaklyCompressible`, or an incompressible-specific variant)
   that reuses `boundaryRegion`/`domainBoundarySdf` from
   `weaklyCompressible.py`. Register as a new case (e.g.
   `randomFlowIncompressible`) or a `--scheme` switch on the existing case —
   confirm which fits the `Case` protocol (`src/warpSPH/runner/case.py`)
   before deciding. **Not started.**
8. **Validate**: run bounded random flow under all three boundary modes,
   check density near walls, momentum leakage, and stability vs. the
   periodic baseline; compare against wcsph/deltaSPH bounded behavior as a
   sanity reference. **Not started** (blocked on step 7).

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
- **Why steps 7–8 weren't attempted this session**: porting `randomFlow` to
  the incompressible scheme needs actual boundary-particle *sampling*
  (assigning `kind==1`/`kind==2`, `ghostIndices`, `ghostOffsets` at
  particle-generation time for a bounded domain), which is a materially
  different, larger task than the solver-side wiring above — it touches
  `initializers/weaklyCompressible.py`'s region-sampling path and the
  `Case`/`RunContext` machinery, not just the incompressible modules. Left
  for a follow-up session; the open question below on new-case-vs-flag
  should be resolved first.

---

## Open questions / decisions needed

- Part 2, step 7: new case file vs. flag on existing `randomFlow.py` case —
  to be resolved by checking `Case`/scheme-selection conventions before
  implementation.
- Whether Part 1's fix should be a case-specific override (e.g. Kolmogorov
  case sets `ω`/tolerance explicitly) or a general default change to the
  incompressible scheme config.
