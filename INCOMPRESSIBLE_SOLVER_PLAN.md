# warpSPH — Incompressible (IISPH) Pressure-Solver Krylov Plan

Plan for adding opt-in **CG / BiCG / BiCGStab / GMRES** Krylov pressure solvers
to the DFSPH incompressible scheme (today a matrix-free **relaxed Jacobi**),
reusing the existing matrix-free Krylov library from the implicit-shifting
work, with **relaxed-Jacobi kept as the byte-identical shipped default**.
Written up here (rather than only in an ephemeral plan) so it can be picked
up and followed structurally in a later session. **Not started — no code
changes have been made yet.**

The solver set is deliberately **one solver per phase (Phases 1–4)** so any
solver can be targeted independently whenever its time comes. **BiCG is
Phase 4 (last)** because it is the only method that needs the operator
**transpose** `Aᵀ` (a real derivation); CG / BiCGStab / GMRES only ever apply
the existing matrix-free forward `matvec`, so they slot in directly.

## Why

- The live pressure solve (`solveDivergenceFree`) is **damped Jacobi**: it
  converges linearly and slowly (defaults up to 32–64 iterations, each a full
  2-pass SPH matvec). A Krylov method converges in a small multiple of
  `√κ` matvecs, and we already have a principled preconditioner — the IISPH
  diagonal `a_ii` that Jacobi is already dividing by.
- The operator is **already matrix-free** (two SPH neighbor-loop passes, no
  matrix ever built) and the diagonal is precomputed in `O(n)`, so a Krylov
  method preserves the memory property exactly — it just calls the same
  `matvec` closure.
- The **implicit-shifting work already built generic, pure-torch, matrix-free
  Krylov solvers** (`modules/shifting/bicgstab.py`, `gmres.py`,
  `richardson.py`) whose interface is *exactly* "caller-supplied `matvec`
  closure + flat diagonal `precond` vector". Reusing them (not rewriting the
  solve loops) is the whole point of this plan.
- **People have tried CG before.** This plan scopes that honestly: it adds an
  **operator probe** (Phase 0) that measures symmetry + definiteness so CG
  viability is a *measured* decision, and it makes **BiCGStab/GMRES** the
  robust defaults for a non-SPD operator (matching the shifting experience,
  where the same operator family was found nonsymmetric/indefinite).

## Current state (what runs today)

Live path: `schemes/dfsph.py:160` → `modules/incompressible/divergenceFree.py`
(`solveDivergenceFree`, the **relaxed Jacobi**). The commented-out
`modules/incompressible/incompressible.py` (`solveIncompressible`) is the
density-error variant (adds a `clamp(p, min=0)` nonlinearity); it is **not**
on the live `dfsph_step` path today.

`dfsph_step` (`schemes/dfsph.py:159-169`):
```python
currentState.pressures = torch.zeros_like(currentState.densities) if currentState.pressures is None else currentState.pressures
dvdt_pressure, pressure, errors, pressures = solveDivergenceFree(
    particles=currentState, config=config, schemeConfig=schemeConfig,
    adjacency=adjacency, dvdt=dvdt + dvdt_diss, dt=dt)
currentState.pressures = pressure
```

`solveDivergenceFree` solves the **linear system** `A_op · p = b` for the
**scalar** pressure field `p ∈ ℝⁿ`:
```python
predictedVelocities = v + dt*dvdt
b   = sourceTerm = -divergence(predictedVelocities)   # = -rho0*div(v*)  [momentum/incompressible.py:20]
D   = alphas = dt * computeAlpha(...)                 # IISPH a_ii, NEGATIVE, precomputed ONCE [wp_alpha.py:315]
for i in range(maxIters):
    a_p  = computePressureAccelIISPH(p)        # 1 SPH pass (Scatter Symmetric gradient) [pressure/iisph.py:17]
    dx_p = dt * computePressureShiftIISPH(a_p) # 1 SPH pass (Scatter Difference divergence) [incompressible/drift.py:22]
    residual = b - dx_p
    p = p + omega * residual / D               # damped Jacobi step, omega = relaxationFactor
    p = p - p.mean()                           # gauge fix (kills the constant null-space mode)
    if mean|residual| < tol and i >= minIters: break
# final: a_p = computePressureAccelIISPH(p)  -> returned as dvdt_pressure
```
The "matrix" is **never materialized**. The matvec is a two-kernel composition
and the only thing built is an `O(n)` diagonal.

## Key observations

### The linear system, in standard form

The unknown is the scalar pressure `p`; the operator, preconditioner, and RHS
are all already computed (or trivially computable) matrix-free:

| standard concept | in this codebase | note |
|---|---|---|
| unknown `x` | `pressure` (scalar `[n]`) | not a vector field |
| `A x` (matvec) | `p → dt_scale · computePressureShiftIISPH(computePressureAccelIISPH(p))` | 2 SPH passes; `dt_scale = dt` (div-free) or `dt**2` (incompressible variant) |
| preconditioner `M⁻¹` | `1/(dt_scale · computeAlpha(...))` | the IISPH `a_ii`, **negated** → negative diagonal |
| RHS `b` | `sourceTerm` | `-rho0·div(v*)` (div-free) or `rho0 − rho*` (incompressible variant) |
| current iteration | `p ← p + ω·M⁻¹(b − A p)` | damped Jacobi, `ω = relaxationFactor`, `M = D = diag(A)` |
| gauge | `p −= p.mean()` | constant null space, `A·1 = 0` |

**Preconditioner gotcha (critical):** both `bicgstabSolve` and `gmresSolve`
apply the flat vector as `psolve(v) = precond * v` (**multiply**, not divide).
So we must pass `precond = 1/D = 1/(dt_scale·computeAlpha)`, *not* `D`. This
matches exactly what the Jacobi does (`residual / alphas`). `computeAlpha`
returns a negative value (clamped `max=-1e-6` upstream), so `precond` is a
finite negative reciprocal — safe.

The state (`positions`, `densities`, `masses`, `adjacency`) is **fixed**
throughout the pressure iteration (only `p` updates), so `A` and `b` are
genuinely constant → a proper linear solve. The only nonlinearity is the
post-iteration gauge re-center (linear, harmless) and — in the inactive
incompressible variant — the `clamp(p, min=0)` (a true inequality, see Scope).

### What is already reusable (the big win)

`modules/shifting/` already contains generic, **pure-torch** (import only
`typing` + `torch`, zero SPH dependency → no circular-import risk when
imported from `modules/incompressible/`) matrix-free Krylov solvers with the
exact interface we need:

- `bicgstab.py:62  bicgstabSolve(matvec, b, x0, tol, rtol, atol, maxiter, precond, verbose, threshold, dim) → (x, iters, convergence)` — 2 matvec/iter, robust non-SPD.
- `gmres.py:41     gmresSolve(matvec, b, ..., precond, ..., restart) → (x, iters, convergence)` — 1 matvec/iter, no breakdown modes, O(m·n) memory.
- `richardson.py   richardsonSolve(...)` — bounded auto-tuned fallback (optional last resort).
- `solverDriver.py  solveImplicitSystem / runKrylov` — primary + BiCGStab↔GMRES(+Richardson) fallback chain, returns the best iterate by *stamped true residual* (`convergence[-1]`). **Mirror this dispatch/fallback pattern** in the incompressible glue.

Both solvers return a `convergence` list whose **last entry is always the
verified true residual `‖b − A x‖`** of the returned iterate — that is what we
report as solver quality and use for the comparison harness.

### Operator character → which methods

`A_op = dt·div( −grad(p)/ρ )` is a weighted graph-Laplacian.
- **Continuum:** a self-adjoint (symmetric) elliptic operator, PSD with a
  1-D constant null space.
- **Discrete SPH:** generally **nonsymmetric** (the Symmetric-gradient and
  Difference-divergence are not exact discrete adjoints; `/ρ` is applied on
  one side; BCs break symmetry) and can be **indefinite** — the *same operator
  family* the shifting work probed (`exactHessian`) and found
  nonsymmetric/indefinite with a translation null space.

Consequences (this drives the phase order):
- **CG** needs SPD. The null space is harmless (converges on the
  null-orthogonal complement; re-center the result). But nonsymmetry/
  indefiniteness breaks its guarantees → **fragile; gate on the Phase-0 probe.**
- **BiCGStab** — robust non-SPD, low memory → **best all-round default**.
- **GMRES(m)** — most robust (no breakdown), 1 matvec/iter, O(m·n) memory →
  **best fallback / ill-conditioned choice**.
- **BiCG** — robust in principle but **needs `Aᵀ`**; see next subsection.

### The BiCG-adjoint finding (why BiCG is Phase 4 / last)

BiCG (unlike BiCGStab/GMRES/CG) updates a **shadow residual**
`r̃ ← r̃ − α·Aᵀ p̃`, so it needs the **adjoint matvec `Aᵀ`** every iteration.
Investigation of `warpSPHCore` (`/home/lu26029/dev/warpSPHCore`):
- `enumTypes.py` has **no adjoint/transpose operator scheme** — only forward
  `Gradient`/`Divergence` with `GradientScheme` (Naive/Symmetric/Difference/
  Summation) and `SupportScheme` (Gather uses h_i, Scatter uses h_j,
  MeanSymmetric, KernelMeanSymmetric, SuperSymmetric, PartialSymmetric).
- `warpier_adjoint.md` is about Warp **autodiff** adjoints, *not* an explicit
  operator-transpose. So there is **no ready matrix-free `Aᵀ`**.

Therefore BiCG's `matvecT` must be **derived**, in descending preference:
1. **Verified discrete adjoint** — identify the transpose SPH ops and prove
   `⟨A x, y⟩ = ⟨x, Aᵀ y⟩` numerically against an explicitly assembled `A`
   (test-only, small `n`). Most work, but makes BiCG production-grade.
2. **Self-adjoint approximation** `matvecT = matvec` — valid when the Phase-0
   probe shows a small symmetry residual `‖A − Aᵀ‖/‖A‖` (plausible, since the
   continuum operator *is* self-adjoint). Cheapest; ship with a documented
   caveat and let the probe set the threshold.
3. **Explicit-assembly `Aᵀ = A.t()`** — test/benchmark only (breaks
   matrix-free for production).

Because (1) is real deriving work and (2)/(3) are the realistic near-term
paths, **BiCG is scheduled last** and can be delivered at whatever fidelity the
probe warrants.

### Cost / memory (matrix-free property is preserved)

| method | matvec/iter | SPH passes/iter | new memory | breakdown modes |
|---|---|---|---|---|
| relaxed-Jacobi (default) | 1 | 2 | O(n) | none (but slow convergence) |
| CG | 1 | 2 | O(n) | none — **needs SPD** |
| BiCG | 2 | 4 | O(n) | **needs `Aᵀ`** + rho/alpha |
| BiCGStab | 2 | 4 | O(n) | rho/omega (already guarded in the port) |
| GMRES(m) | 1 | 2 | O(m·n) | none (most robust) |

No matrix is ever built in any path — only the `O(n)` diagonal and (GMRES) the
`[n, restart]` Arnoldi basis. Each matvec transiently allocates the `[n, dim]`
accel intermediate (the composition), never an `[n, n]` operator.

## Scope

**In scope (this plan):** all four Krylov methods as opt-in; the `PressureSolverType`
enum + config; the `krylov.py` glue (matvec/precond/RHS builders + dispatch);
the operator probe; the comparison harness; the legacy byte-identity regression
guard; docs.

**Out of scope / deferred (each is a real, separate piece of work):**
- A **fused single-kernel matvec** (`div(−grad(p)/ρ)` in one pass) to drop the
  `[n, dim]` intermediate + a launch — optimization only; the two-kernel
  composition is correct to start.
- A **true constrained solve** for the inactive `solveIncompressible`
  `clamp(p, min=0)` inequality — here we approximate with a linear solve +
  post-projection clamp, documented as such.
- **Production BiCG** if no verified discrete adjoint is established — it then
  ships at self-adjoint-approximation fidelity (with caveat) or test-only.

**Invariant across all phases:** the shipped default stays
`PressureSolverType.relaxedJacobi`, and that path stays **byte-identical** to
today's `solveDivergenceFree`/`solveIncompressible` output. Every phase is
additive behind the enum; nothing changes for existing users.

## Status

| Phase | Solver / work | Status |
|---|---|---|
| 0 | Foundations: glue, enum, operator probe, baseline capture | ⏸ Not started |
| 1 | **BiCGStab** (first solver; validates the glue) | ⏸ Not started |
| 2 | **GMRES** | ⏸ Not started |
| 3 | **CG** (gated on Phase-0 probe) | ⏸ Not started |
| 4 | **BiCG** (last — needs `Aᵀ`) | ⏸ Not started |
| 5 | Comparison harness, regression guard, docs | ⏸ Not started |

## Phases

Each solver is its own phase so it can be started/landed independently. Phases
1–4 are additive (each just adds a `PressureSolverType` case); Phase 0 is the
shared substrate and Phase 5 is the wrap-up.

### Phase 0 — Foundations & operator probe (no behavior change)

The shared substrate. Nothing user-visible changes; default stays relaxed-Jacobi.

- [ ] **0a — `modules/incompressible/krylov.py`** (new). Builders over the fixed
  state:
  - `buildIISPHMatvec(state, config, schemeConfig, adjacency, dt_scale, supportScheme)`
    → closure `matvec(p) = dt_scale·computePressureShiftIISPH(computePressureAccelIISPH(p))`
    (Scatter, same `adjacency` as the Jacobi).
  - `buildIISPHPrecond(state, config, schemeConfig, adjacency, dt_scale)` →
    `1.0/(dt_scale·computeAlpha(...))` (the **`1/D`** the solvers multiply by).
  - `buildIISPHMatvecT(...)` → **stub** (returns `matvec` self-adjoint approx for now;
    Phase 4 replaces with the derived adjoint).
  - `solvePressureKrylov(...)` → dispatch scaffold (raises `NotImplementedError`
    for solver types not yet added) + post-solve gauge re-center (`p −= p.mean()`)
    + final `a_p = computePressureAccelIISPH(p)` + return `(a_p, p, errors, pressures)`
    (maps the solver `convergence` list → `errors`). Signature mirrors the
    `solverDriver.solveImplicitSystem` pattern (primary + optional fallback).
- [ ] **0b — `configurations/moduleConfigurations/solver.py`**: add
  `PressureSolverType(Enum)`: `relaxedJacobi`(default) / `cg` / `bicg` / `bicgStab`
  / `gmres`; add `solverType`, `rtol`, `atol`, `restart` to the per-solver config
  (both `pressureSolver` + `divergenceFreeSolver`), reusing `maxIterations`→`maxiter`,
  `tolerance`→`tol`. Update `__all__`, `buildDefault*`.
  - Also update the round-trip in `configurations/incompressible.py`
    (`incompressibleConfigToDict` / `dictToIncompressibleSPHConfig`, ~line 200).
- [ ] **0c — `tests/test_incompressibleOperatorProbe.py`** (new). Build a small case
  (reuse the `_gradcheck_common` line-case helpers, `n≈20–40`, like
  `scripts/gradcheck_incompressible.py`), **assemble `A` explicitly** (apply
  `matvec` to each basis vector — test-only), then record:
  - symmetry residual `‖A − Aᵀ‖_F / ‖A‖_F`;
  - on the null-orthogonal subspace `Q = I − 11ᵀ/n`: `λ_min`, `λ_max`, condition
    number (eigvalsh of `Q·(A+Aᵀ)/2·Q`, or Rayleigh quotients).
  - **Emit/record a verdict** (a small helper the later phases can import):
    `cg_viable = (sym_residual < τ_sym) and (λ_min > 0)`;
    `bicg_selfadjoint_ok = (sym_residual < τ_sym)`. This is what Phase 3 and
    Phase 4 read — it turns "should we trust CG / BiCG" into a measured fact.
- [ ] **0d — baseline capture** for the regression guard: snapshot the
  relaxed-Jacobi `(a_p, pressure, errors, pressures)` on the small case (a
  fixture / recorded values) so Phase 5 can assert byte-identity of the default.

### Phase 1 — BiCGStab (first solver; proves the glue end-to-end)

Chosen first because it reuses the already-hardened `bicgstabSolve` and is the
robust non-SPD default — lowest risk way to validate `krylov.py`.

- [ ] Add the `bicgStab` case to `solvePressureKrylov` → call `bicgstabSolve(matvec, b, x0, tol=…, rtol=…, maxiter=…, precond=1/D, threshold=…, dim=1)`.
- [ ] Wire `modules/incompressible/divergenceFree.py` to branch on `solverType`:
  `relaxedJacobi` → **existing code unchanged**; `bicgStab` → `solvePressureKrylov`.
  Warm-start `x0 = currentState.pressures` (mirrors the Jacobi's `*0.75`).
- [ ] Tests: `matvec` closure == direct composition; convergence quality
  (stamped `‖A p − b‖` ≤ relaxed-Jacobi residual); gauge `mean(p) ≈ 0`; status
  codes on a forced-breakdown/budget case.
- [ ] Verify the **default path is unchanged** (run with `relaxedJacobi`, diff
  against the Phase-0d baseline).

### Phase 2 — GMRES

Reuses `gmresSolve`; the robust, breakdown-free alternative (ill-conditioned).

- [ ] Add the `gmres` case → `gmresSolve(matvec, b, x0, …, restart=cfg.restart, precond=1/D)`.
- [ ] Wire `divergenceFree.py` (`gmres` branch).
- [ ] Tests: convergence; restart sweep (small `m` vs larger) for the
  residual-vs-memory tradeoff; confirm no breakdown on the indefinite probe case
  where BiCGStab might struggle.

### Phase 3 — CG (gated on the Phase-0 probe)

CG is the cheapest (1 matvec/iter, low memory, no breakdown) *if* the operator
is SPD. The Phase-0 probe decides whether it is.

- [ ] **New `modules/shifting/cg.py`** → `cgSolve(matvec, b, x0, tol, rtol, atol,
  maxiter, precond, verbose, threshold, dim)`, matching the `bicgstabSolve`
  signature/status-code convention exactly (left-preconditioned, `psolve(v)=
  precond·v`; breakdown code if `‖p_k‖` collapses; `convergence[-1]` = stamped
  true residual). ~45 lines. (Optional: extend `solverDriver.runKrylov` to add a
  `cg` case, or keep the incompressible dispatch self-contained — decide in 0a.)
- [ ] Add the `cg` case to `solvePressureKrylov`; wire `divergenceFree.py`.
- [ ] **Gate:** read the Phase-0 verdict.
  - If `cg_viable`: test CG on the real operator, assert convergence + residual.
  - If **not** SPD: still ship `cgSolve` (it's correct and useful for SPD
    subcases / as a reference), but the real-operator test is `xfail`/skipped
    with the probe's numbers attached, and the regression note says so.
- [ ] Unit-test `cgSolve` itself on a **known-SPD** matvec (a small SPD matrix
  assembled explicitly) so the solver is validated independent of the SPH operator.

### Phase 4 — BiCG (LAST — the transpose)

The only method needing `Aᵀ`. Scheduled last so the direct methods land first
and the adjoint derivation is isolated.

- [ ] **New `modules/shifting/bicg.py`** → `bicgSolve(matvec, matvecT, b, x0, …,
  precond, …)` (~50 lines; same status-code convention; `convergence[-1]` =
  stamped true residual).
- [ ] **Derive/choose `matvecT`** per the preference list in "Key observations":
  - [ ] **(1) verified discrete adjoint** — find the transpose SPH ops, then
    assert `|⟨A x, y⟩ − ⟨x, Aᵀ y⟩| / |⟨A x, y⟩| < tol` on the probe case against
    an assembled `A`. If it holds, `buildIISPHMatvecT` returns the true adjoint.
  - [ ] **(2) self-adjoint approx** — if (1) is infeasible and the probe's
    `sym_residual` is small, `buildIISPHMatvecT` returns `matvec`, with a
    documented caveat + a runtime `warnings.warn` when `sym_residual` is large.
  - [ ] **(3) explicit-assembly `Aᵀ = A.t()`** — test/benchmark-only fallback.
- [ ] Add the `bicg` case to `solvePressureKrylov`; wire `divergenceFree.py`.
- [ ] Tests: adjoint-identity check (if (1)); convergence; **BiCG vs BiCGStab**
  head-to-head (BiCGStab is the stabilized BiCG — expect it to be at least as
  robust; record where plain BiCG wins/loses).

### Phase 5 — Comparison harness, regression guard, docs

- [ ] **`tests/test_incompressibleSolverComparison.py`** (new) — the "full
  comparison in one pass": run **relaxed-Jacobi + CG + BiCG + BiCGStab + GMRES**
  on the same case and print a table (iters, final `‖r‖`, matvec count,
  wall-time); assert all finite + the probe's viability verdicts.
- [ ] **Regression guard:** default (`relaxedJacobi`) output is byte-identical to
  the Phase-0d baseline, on both `divergenceFree.py` and `incompressible.py`.
- [ ] **`incompressible.py` (density-error variant):** wire the same enum
  branch; linear solve + `clamp(p, min=0)` as a **post-projection**, documented
  as an approximation (the inequality is not enforced inside the Krylov
  iteration). Lower priority (path is currently inactive in `dfsph_step`).
- [ ] **Docs:** update `modules/incompressible/` + `configurations/.../solver.py`
  docstrings; add `docs/regression/incompressible_pressure_solver_choice.md`
  (honest per-method findings — which converged, iteration/residual numbers, the
  probe verdict, and why BiCG's fidelity is what it is), mirroring
  `docs/regression/implicit_shifting_operator_choice.md`.

## File map

| file | role | phase |
|---|---|---|
| `modules/incompressible/krylov.py` | matvec/precond/matvecT builders + `solvePressureKrylov` dispatch | 0, then 1–4 |
| `configurations/moduleConfigurations/solver.py` | `PressureSolverType` + fields | 0 |
| `configurations/incompressible.py` | config dict round-trip (~line 200) | 0 |
| `modules/incompressible/divergenceFree.py` | active solver; enum branch | 1 (2,3,4 add cases) |
| `modules/incompressible/incompressible.py` | density-error variant; enum branch | 5 |
| `modules/shifting/cg.py` | `cgSolve` | 3 |
| `modules/shifting/bicg.py` | `bicgSolve` (needs `matvecT`) | 4 |
| `modules/shifting/{bicgstab,gmres,richardson,solverDriver}.py` | reused as-is | 1–2 |
| `tests/test_incompressibleOperatorProbe.py` | symmetry + spectrum probe, verdicts | 0 |
| `tests/test_incompressibleKrylov.py` | per-solver unit tests | 1–4 |
| `tests/test_incompressibleSolverComparison.py` | all-methods table | 5 |
| `docs/regression/incompressible_pressure_solver_choice.md` | findings | 5 |

## Risks / open questions

- **Is the discrete operator SPD?** Unknown until Phase-0c runs. If not, CG is
  benchmark-only (still shipped) and BiCGStab/GMRES are the real answers — this
  is the likely outcome given the shifting experience.
- **BiCG adjoint fidelity** is the main open technical item; Phase 4 resolves
  it against the probe. If no clean discrete adjoint exists, BiCG ships at
  self-adjoint-approximation (or test-only) fidelity — acceptable, and it is the
  least-attractive method anyway (BiCGStab dominates it).
- **`/ρ` weighting + BCs** can push the operator indefinite; the probe's
  `λ_min < 0` would confirm it (and rule out CG even if symmetry looks fine).
- **Warm start** (`x0 = previous pressure`) should help convergence (as it does
  for the Jacobi); verify it actually does in the comparison table.

## Verification (per phase)

- Every phase: `python -m py_compile` on touched files; the phase's tests green;
  **default-path byte-identity** re-checked against the Phase-0d baseline.
- Phase 0: probe numbers printed + recorded (drives 3 and 4).
- Phase 5: full comparison table; `docs/regression` note written; then run the
  broader incompressible test slice (and `scripts/run_tests.sh`) to confirm no
  cross-module regressions.




