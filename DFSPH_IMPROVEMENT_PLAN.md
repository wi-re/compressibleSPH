# warpSPH — Incompressible (VD+PS / DFSPH) Improvement Plan

Working document for the incompressible SPH path (`schemes/dfsph.py`,
registered as `IncompressibleSPHScheme.divergenceFree`). Twelve sessions of
investigation, 2026-08-26 to 2026-08-28.

This file was rewritten on 2026-08-28 to remove superseded reasoning and
retracted claims. **The full narrative, including every hypothesis that was
later falsified, is in git history** (`git log -p DFSPH_IMPROVEMENT_PLAN.md`);
what is kept here is what a reader needs in order to act. Sections are ordered
so the durable content comes first and the current-state overview comes last.

**Units note.** `cflFactor` on the incompressible cases multiplies the particle
**diameter** `dx`, which is the form Bender & Koschier state their condition in
(`dt <= 0.4 d/|v_max|`). Before Part 12 it multiplied the support radius
`h = n_h dx`; with `n_h = 4` the two conventions differ by 4x for the same
number, so **every `cflFactor` recorded in git history means 4x more travel per
step than the same number means today**. Every number in this file is restated
in the current (diameter) units:

| this file says | git history says | what it is |
|---|---|---|
| **`cflFactor = 0.4`** ("the published CFL") | `cflFactor = 0.1` | 0.4 spacings of travel per step |
| **`cflFactor = 1.2`** ("the legacy CFL") | `cflFactor = 0.3` | 1.2 spacings per step, 3x published |
| `cflFactor = 0.5` | `cflFactor = 0.125` | Part 5's empirical stability point |

---

## 1. Lessons learned — the durable physics

These are the findings that everything else rests on. Each was measured, not
argued, and each survived every subsequent session.

### 1.1 The density setpoint is structurally unreachable

For equal masses, `sum_i rho_i = m sum_{ij} W(x_i - x_j)`, which by Parseval on
a periodic domain is `m sum_k What(k) |rhohat(k)|^2` with `What(k) >= 0` for any
positive-definite kernel (Wendland is one). A perfect lattice puts all spectral
weight on reciprocal-lattice vectors where the bandwidth-limited `What` is ~0;
*any* disorder moves weight to small `k` where `What` is large, and every such
contribution is non-negative.

**The lattice minimises the particle-averaged summation density.** So
`mean_i rho_i == rho0` is unattainable for any disordered configuration, and
`sourceTerm = rho0 - rhoStar` carries a permanently signed mean. Measured with
no dynamics at all (`scripts/probe_densityBiasVsDisorder.py`) — jitter sweep
against `mean(rho-1)`:

| jitter/dx | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|---|---|
| `mean(rho-1)` | 5.6e-8 | 2.2e-6 | 8.7e-6 | 3.4e-5 | 2.1e-4 | 8.5e-4 | 3.3e-3 | 1.2e-2 |

Always positive, rising as the square of the displacement (~3.9x per doubling
against 4x predicted). In the developed `kolmogorovIncompressible` flow the
floor is ~2.7e-3, against a configured PS tolerance of 5e-4.

**There is no upstream bug to find.** Both `kolmogorovIncompressible.py:71` and
`tgv.py:53` additionally pin `rho0` *to* that lattice minimum by normalising
mass — which places the setpoint exactly at the floor. That is a real code
decision, but removing it makes things worse (§2), because the source's
negative mean is load-bearing.

### 1.2 The permanent residual is safe on positions and unsafe in momentum

This is the single most important consequence of §1.1, and it is what the
literature split turns on.

- Applied as a **position shift** (`dx = dt^2 a_p`), the residual is
  momentum-neutral — it only reorganises particles, and its negative mean *is*
  the de-clumping drive that keeps the sampling ordered.
- Applied to **velocity**, the same residual is a permanent unphysical forcing.
  Measured: both velocity modes damp `tgv`'s kinetic energy at ~3.3x the
  analytic rate (against 0.55x for the position shift) and make the decay
  non-monotone.

That is why this scheme uses a position shift, and it is not a shortcut — it is
the formulation that is robust to an unreachable setpoint. Cornelis et al.
state the same conclusion from the other direction: their abstract says "the DI
source term suffers from significant artificial viscosity", and their Fig. 3
measures it.

### 1.3 This scheme is VD+PS, not a mis-named DFSPH

`dfsph_step` + `IncompressibleSystem.finalize` implement Cornelis et al.'s
VD+PS step for step and in the right order:

| [C] | this codebase |
|---|---|
| `v* = v + dt a_nonp` (Eq. 2) | `dvdt` assembled in `schemes/dfsph.py` |
| `dt grad^2 p* = rho0 div v*` (Eq. 12) | `solveDivergenceFree`, source `-divergence` |
| `v' = v* - dt grad p*/rho0` (Eq. 13) | `dvdt_pressure` |
| `x** = x + dt v'` (Eq. 14) | integrator |
| `dt grad^2 p** = (rho0 - rho**)/dt` (Eq. 15) | `solveIncompressible`, source `rho0 - rhoStar` |
| `x(t+dt) = x** - dt^2 grad p**/rho0` (Eq. 16) | `dx = dt^2 dvdt_incomp`, `positions += dx` |
| `v(t+dt) = v' + grad v' . (x(t+dt) - x**)` (Eq. 17) | `proj_vel` einsum |

Including [C] Eq. 19's detail of evaluating `grad v'` on the *current*
neighbourhood rather than the shifted one.

The registered scheme name `divergenceFree` is accurate. The misnomer is
confined to the file/function names `dfsph.py`/`dfsph_step`, whose correct
target is `vdps.py` — **not** a rescue of a broken `dfsph.py`. A real DFSPH
scheme would be a *different* method, and specifically the one [C] was written
to replace.

### 1.4 `solveIncompressible` is a shifting potential, not a stress

Both call sites (`systems/incompressible.py::finalize`, `cases/tgv.py`'s lattice
relaxation) feed its output into a position shift. So:

- its non-negativity is not "a fluid cannot sustain tension" — it is "the shift
  never pulls particles together", which is the tensile-instability guard;
- its constant mode is not a physical pressure level, which makes translating
  it a free choice wherever the operator's constant mode is null.

### 1.5 This codebase's walls have complete support, so a constant pressure is forceless there

With complete support the kernel gradients sum to zero, so a constant pressure
field produces ~zero acceleration: the operator has a constant null space and
the constant is a gauge. Truncate the support and the same constant produces a
large net force along the truncation normal.

The literature's one-layer boundaries (Akinci, as used by [BK] and [C]) *are*
truncated. **This codebase's are not.** `randomFlow.BOUNDED_BAND = 5` samples a
five-layer solid band against `h = 4` spacings — the band is wider than the
kernel. Measured (`scripts/probe_wallSupportCompleteness.py`, bounded case,
nx=128, 120 steps):

| depth (spacings) | n | Shepard `sum V_j W_ij` | `\|A.1\|/\|A.rand\|` |
|---|---|---|---|
| (<0, inside wall) | 84 | 1.00857 | 0.245 |
| [0,1) | 436 | **1.00100** | **0.194** |
| [2,3) | 455 | 0.99786 | 0.189 |
| [10,inf) bulk | 11608 | 0.99997 | 0.166 |

Shepard is ~1.00 down to the wall, and the constant mode is as near-null there
(0.194) as in the bulk (0.166). **Free surfaces are genuinely truncated; walls
here are not.** One nuance: `|sum V_j grad W_ij|` *is* elevated ~4x at the wall
(6.3 vs 1.3), but that is the volume field being discontinuous across the
interface (boundary rows carry mDBC-extrapolated densities, hence different
`V_j = m_j/rho_j`), not missing neighbours. The composite operator absorbs it.

### 1.6 A wall cannot stop a particle that crosses a full spacing in one step

The wall's force is mediated by boundary-particle kernel contributions computed
from the configuration at the *start* of the step. The distance over which that
contribution goes from "no wall" to "fully inside the wall" is one particle
spacing. A particle allowed to cross a full spacing per step traverses the
entire first boundary layer before the wall ever pushes on it — **the wall is
not weak, it is late.**

Measured sweep on `randomFlowIncompressible --bounded`, nx=128:

| `cflFactor` (spacings/step) | outcome | penetrating | near-wall `\|rho-1\|` |
|---|---|---|---|
| 1.2 (legacy default) | **NaN at t=5.54** | 4506, growing | 0.30 |
| 1.0 | **NaN at t=5.09** | 15994, growing | 0.41 |
| 0.5 | survives to t=8.0 | 653, steady | 0.040 |
| 0.2 | survives to t=8.0 | 201, *declining* | 0.022 |

The transition is sharp and sits between half a spacing and one. [BK] §3.1's
published constant is `dt <= 0.4 d/|v_max|` with `d` the particle **diameter**;
[B] Table 2's own scenes land at 0.1–0.35 spacings/step. No published run goes
near one spacing.

Caveat carried forward: `n_h = 4` was fixed throughout, so "half a spacing" and
"one eighth of `h`" are not distinguished by this data. One sweep at a
different `n_h` would settle which governs.

### 1.7 The stopping criterion is broken, and it is the last unexplained thing

`solveIncompressible` runs its full 64 iterations every step for 1000 steps and
never satisfies its tolerance — under every gauge, every `dt`, every solver
(relaxed-Jacobi, MINRES, BiCGStab), and both shift configurations. But the
iterations are productive:

| maxIterations | `rhoErr` | `rhoMax` | `pMean` max |
|---|---|---|---|
| 64 (default) | 1.15e-3 | 7.91e-3 | 28.9 |
| 16 | 2.29e-3 | 1.08e-2 | 15.9 |
| 8 | 3.80e-3 | 1.87e-2 | 9.57 |

So the solve *is* converging in the sense that matters while never meeting a
criterion that is structurally unreachable. Two concrete defects:

1. An **absolute** compression threshold cannot be met when the source carries
   a structural mean the operator cannot remove. `RelaxedJacobiSolverConfig`
   already has `rtol` (`solver.py:223`); the relaxed-Jacobi path ignores it.
2. Both papers' criteria are one-sided on the *average*
   (`rho_avg - rho0 > eta`, [BK] Alg. 3; [I] §5.1), so under-dense particles
   cancel over-dense ones. This codebase floors each particle's negative
   contribution at `-threshold` (`incompressible.py:218-220`), forbidding
   exactly that cancellation. That is the difference that makes the structural
   bias binding here and not there, and it is one line.

**A solver that never terminates cannot report that a change helped it**, which
is why every convergence number in this document is a residual, not a count.

### 1.8 Static boundary particles do not take reaction forces

[BK] §3.2, one sentence: "since `F^p_{j<-i} = 0` if particle j is not dynamic,
the equation for `kappa^v_i` must be adapted accordingly for static boundary
particles." Every term describing *the neighbour's* response to `p_i` is absent
for a static neighbour; every term describing `i`'s own response stays.
SPlisHSPlasH implements this literally
(`TimeStepDFSPH.cpp::computeDFSPHFactor`, `compute_aij_pj`), and [BWJ23] Eqs.
32/34 *derive* it — their density constraint is defined for fluid particles
only, so `dC_i/dx_k = 0` for a static boundary.

This codebase computed both terms. The two that differ:

- `computeAlpha`'s second sum `sum_j V_j^2/m_j |gradW_ij|^2` — i.e. `dp_i/dx_j`;
- the divergence operator's `a_j` term — the neighbour's own pressure
  displacement.

In the wall-adjacent bin they are **40% of the diagonal** (55% inside the band)
and **exactly zero beyond 3 spacings**, so this is a wall effect only and a
strict no-op on periodic cases.

> Trap for anyone reading SPlisHSPlasH as ground truth: its **IISPH** keeps
> `-d_ji` in `aii`'s boundary loop (`TimeStepIISPH.cpp:287-292`) — the term its
> own matvec drops. Its DFSPH has no such mismatch, and [I] §4 agrees with the
> DFSPH file.

### 1.9 The continuity equation cannot see particle rearrangement

`drho/dt = -rho div v` describes advection of a density field. The real density
error in this scheme is dominated by particle *rearrangement* at essentially
zero divergence, which that equation cannot represent at all. Measured: under
pure `continuity` the carried density reports `|rho-1|` = 7.1e-3 while a fresh
summation on the same positions reads 4.3e-2. The carried field converges to a
smooth, plausible-looking lie.

This is not the position shift's doing — dropping the shift entirely
(`inStepVelocity`) leaves the drift at 3.5e-2. It is the flow's own
rearrangement.

The corollary is the useful part: **the divergence-free solve genuinely only
needs `div v`**, which the continuity equation tracks exactly. Only the
shifting solve needs a density that matches the positions. Hence `hybrid`.

### 1.10 Method note: measure, do not derive from pseudocode

Three over-claims in a row (Part 7's tolerance argument, Part 8's `rho`-power
algebra, Part 8's solver-ordering claim) all came from reasoning off published
pseudocode without checking what the surrounding machinery does with the values.
Every claim that was *measured* instead settled in one run. `INCOMPRESSIBLE_
SOLVER_PLAN.md` already contained one of the results a session was spent
re-deriving. **Read the repo's own prior measurements first, then measure.**

---

## 2. Negative results — do not re-run these

Each was tested, and each failed. Recorded so nobody spends a session on them
again.

| hypothesis | result |
|---|---|
| **Move the setpoint off the lattice floor** (`--setpointEps`) | Kills `kolmogorovIncompressible` **150 steps sooner**. The source's negative mean is the shifting solve's de-clumping drive; cancelling it is a static mean-centering. Under `minShift` the apparent `nIter` 64→6 is the stopping criterion going slack, not convergence (`rhoErr` 3.5x worse). |
| **`computeAlpha` carries an extra power of `rho`** | Falsified decisively. `probe_operatorDiagonal.py` extracts the diagonal exactly: `diag(A)/alphas` = 1.00011 ± 0.0051 periodic, 0.99987 ± 0.0160 bounded with `rho` out to **1.303**. `omega_eff = omega` exactly. (`INCOMPRESSIBLE_SOLVER_PLAN.md` had already measured rel-L2 3.3e-7.) |
| **Mean-centering the shift pressure** (what the VD solver does) | Worst of four gauges: NaN at step 155. The VD solver's source is a divergence, mean-zero by pair antisymmetry, so recentering costs it nothing; this solver's is not. |
| **Centre-then-clamp** | NaN at step 136. Chops the field's shape every iteration rather than translating it. |
| **Zero-meaning the source** (textbook compatibility projection) | Makes the solver *converge* (nIter 64→13.4) and the density *worse* (`mean\|rho-1\|` 3.9e-3 vs 2.9e-3, `rhoStd` 3.4x). Only part of the source's mean is unreachable; the rest is the real de-clump signal. |
| **Capping the implicit shift** (`--shiftCap`) | 0.25 spacings/step still diverges (t=5.28); 0.05 diverges *much earlier* (t=1.45). This is [C] §6's argument about user-tuned shift magnitudes, measured. |
| **deltaSPH shift stacked on the implicit shift** | 4% (noise) at default magnitude, then monotonically worse, then NaN at `shiftProperties.CFL=3.0`. Same [C] §6 argument from the other side. Note it was also *never running* — `finalize` shadowed `dx`; fixed, then rejected. |
| **Confining the velocity correction to a wall band** | Diverges at t=4.17 vs t=8.0 unscoped; widening the band is worse (t=2.74). The correction's value is not wall-local — it improves the *bulk* 2x — and a shell edge injects a velocity discontinuity. Mode implemented, measured, removed. |
| **Scaling the velocity correction down** | No sweet spot. Bounded case stable at λ=0.25; `tgv` already 1.4x analytic and non-monotone there. |
| **Moving the MLS projection earlier in the step** (`--mlsBeforeSolve`) | 232 penetrating vs 228. It changed the *lag*, not the *statefulness* — the real fix is moving it inside the Jacobi iteration (§4, item 5), which is untried. |
| **Removing `mdbcNoPenetrationShift`** | Strictly worse on 2 of 3 crossing metrics (543 vs 441 crossing; worst depth 16.2dx vs 10.4dx). Not a crutch worth removing. |
| **Reflecting the mDBC normal component** (the published `-2` form) | A wash with the no-pen shift on, *worse* with it off, much worse paired with `staticBoundary` (`\|rho-1\|` 7.98e-2 vs 3.00e-2). Report the deviation; do not "fix" it blind. |
| **Boundary velocity is the divergence-free half-state's problem** | Setting boundary velocity to the rigid body's (`zeros`, the DFSPH convention) delays the divergence from step 283 to 482 and does not prevent it. |
| **The Jacobi stability window explains it** | `rho(D^-1 A)` = 6.3777 (`full`) vs 6.3782 (`staticBoundary`) — identical to four digits, dominant mode is a bulk mode either way. |
| **The iteration budget explains it** | DF cap 32→96 delays (283→597 steps); 192 is *worse* than 96. Not an under-iteration. |
| **Applying `staticBoundary` to the divergence-free solve alone** | Turns a finite 901-step run into divergence at t=1.65. See §4 item 3 — still unexplained. |
| **Akinci `m~_k` as the particles' actual mass** (the faithful reading) | Diverges in **9 steps**. The correction assumes a one-layer sampling; `BOUNDED_BAND = 5` already contains the volume it adds, so the density sum double-counts it (~15% phantom compression at step 0). |
| **One-sided source clamping** | Published negative result — [I] §3.2: "causes implausible alignments of single particles at the fluid surface for CG and Jacobi." |
| **Krylov on the clamped solve** | Published negative result, confirmed end-to-end: post-hoc `gauge='nonnegative'` gives `rhoErr` 3.6e-1 (MINRES, garbage) or NaN at step 4 (CG, BiCGStab). [I] §3.2 and [BK] §5 both state it. |
| **Reordering `inStepVelocity` to [BK] Alg. 1** | Demoted, not run. Under `semiImplicitEuler` the position advances with the *updated* velocity, making the codebase's ordering equivalent to [BK]'s up to loop phase. The surviving difference is a one-step lag (`DF` computed before `CD`), not an absence of projection. |
| **Krylov `minres` as a wall fix** | 3% better for 23% more time on the bounded case, against 115% better on the periodic one. At a wall the error is set by the boundary treatment, not by how well the PPE is solved. |

---

## 3. Corrections — claims this document previously made and retracted

Kept because each was acted on or nearly acted on.

1. **"`mdbcMlsPressure` is the most stable and most accurate mode"** (Part 2) —
   **wrong.** That was measured at the legacy CFL out to t≈1.5. Over 900 steps
   at the published CFL it is the **worst configuration measured**
   (`\|rho-1\|` 1.86e-1, worse than the shipped baseline's 1.78e-1, `rho_max`
   1.334). Under `inStepVelocity` it NaNs at t=0.21, inverting the ranking a
   third way. **A boundary-mode ranking does not transfer across formulations
   or across timesteps.**
2. **"Published solvers converge because their tolerance sits above the
   structural floor"** (Part 7 Q1) — **wrong.** [BK] runs 1e-4, five times
   *tighter* than this codebase's 5e-4, and converges in 4.5 iterations. The
   real defect is the stopping criterion's form (§1.7).
3. **"`computeAlpha` carries an extra power of `rho`"** (Part 8) — **wrong**,
   see §2.
4. **"`inStepVelocity` is DFSPH with the solvers swapped, and nothing projects
   the density correction"** (Parts 5/6/8) — **overstated**, see §2.
5. **"Kernel support is truncated at a wall"** (`ShiftPressureGauge`'s
   docstring, Part 4) — **false for this codebase's walls**, see §1.5. Half of
   `minShift`'s scoping justification does not hold.
6. **"`minShift` diverges on bounded cases at t=0.69"** (Part 4) — **that was
   measuring the timestep.** At the published CFL `minShift` on the bounded
   case does not diverge, covers 38% more simulated time per step budget, has
   19% lower density error, and costs half the wall time.
7. **"The MINRES win is a complete-support result"** (Part 8) — the
   *observation* stands (2.15x periodic, 3% bounded) but the *explanation* is
   wrong, since the bounded case has complete support. The better explanation
   is §1.6's: solver quality cannot fix an error that is not a solver error.
8. **"The bounded divergence-free harm is the extrapolated boundary
   velocity"** (Part 9) — **retracted**, see §2. Three mechanisms tested, all
   eliminated; the observable that survives is §4 item 3.
9. **"`consistent + akinciBoundaryVolume` is the best configuration measured"**
   (Part 11) — **true only under the clamp gauge.** Paired with
   `ShiftPressureGauge.minShift` it NaNs at step 137. And `consistent`'s own
   contribution over `staticBoundary` (4.7% under the clamp) falls to 0.007% —
   i.e. nothing — once the gauge is fixed. Both measured in Part 13's factorial
   (§4), where Part 11's own numbers reproduce exactly, so this is an
   interaction rather than a contradiction.
10. **"The nx=128 blowup is run-to-run chaotic sensitivity"** (Part 4) — no.
   The case is deterministic; re-running reproduces `pMean` = 2.3838e6 and NaN
   at step 574 exactly. That determinism is what made every subsequent A/B
   trustworthy.

---

## 4. The combined default change — measured

**The first of the four has landed** (`cflFactor`, Part 12 — see §10), and
**the factorial has now been run** (Part 13, below): the combination is worth
40x, `consistent` turns out inert against `staticBoundary`, and
`akinciBoundaryVolume` diverges once the gauge is fixed.
**The other three are each better at the published CFL and each worse at 3x
it. Nobody has run them together, and that is the single highest-value
experiment left in this document.**

All rows: `randomFlowIncompressible --bounded`, nx=128, 900 steps.

| change | at published CFL (0.4) | at legacy CFL (1.2) |
|---|---|---|
| **`cflFactor` 1.2 → 0.4** *(landed, Part 12)* | bounded case reaches t=8.0 at *stock* settings, near-wall `\|rho-1\|` 2.6e-2 (vs 0.30 at the legacy default's death); `kolmogorovIncompressible` 23% better; `tgv` provably inert (its `dt` is pinned at 1e-3, the CFL never binds); also removes Part 4's step-574 NaN under the *historical* gauge | — |
| **`ShiftPressureGauge.minShift` on bounded solves** (via `forceShiftPressureGauge`) | no divergence over 901 steps, t=6.458 vs clamp's 4.690, `\|rho-1\|` 1.43e-1 vs 1.78e-1, **half the wall time** | diverges at t=0.69 vs clamp's t=5.54 |
| **`BoundaryOperatorTerms.staticBoundary`** | `\|rho-1\|` 3.00e-2 vs 1.78e-1 (**5.9x**), `rho_max` 1.247→1.007, 35% more simulated time, same cost | dies at t=1.41 vs `full`'s t=5.54 |
| **`BoundaryPressureMode.consistent`** | `\|rho-1\|` **2.86e-2** (6.2x vs shipped), t=6.463; +`akinciBoundaryVolume` gives **2.38e-2**, the best row measured | dies at t=1.56 (t=3.68 with Akinci) |

The mechanism behind the entanglement is the same in all four: each makes the
solve less damped at the wall (smaller `|alpha|` → larger Jacobi step), which
is not survivable at 1.2 spacings of displacement per step.

### The factorial — run (Part 13, `probe_fourWayDefaults.py`)

**`cfl × gauge × boundary` is a 2 x 2 x 4, not a 2^4.**
`BoundaryPressureMode.consistent` *forces* `BoundaryOperatorTerms.
staticBoundary` inside both solvers (§6), so crossing them yields
bit-identical rows. The boundary axis is one ladder: `shipped`
(`mdbcDensity`+`full`) / `static` (`mdbcDensity`+`staticBoundary`) /
`consistent` / `akinci` (`consistent`+`akinciBoundaryVolume`).

Bounded `randomFlowIncompressible`, nx=128, 900 steps. **Ten of the sixteen
rows are configurations this document had already recorded, and all ten
reproduced to every digit on file** — so the harness is validated and the new
rows are trustworthy.

**At the published CFL (0.4):**

| gauge | boundary | `rho` range | `\|rho-1\|` 2nd half | t_final | DF resid | outcome |
|---|---|---|---|---|---|---|
| shipped | shipped | [0.902, 1.247] | 1.7782e-1 | 4.690 | 7.52e-2 | ok ✓*Part 9* |
| shipped | static | [0.950, 1.007] | 3.0033e-2 | 6.347 | 1.57e-2 | ok ✓*Part 9* |
| shipped | consistent | [0.955, 1.008] | 2.8620e-2 | 6.463 | 1.55e-2 | ok ✓*Part 11* |
| shipped | akinci | [0.951, 1.019] | 2.3847e-2 | 6.330 | 1.24e-2 | ok ✓*Part 11* |
| minShift | shipped | [0.863, 1.154] | 1.4318e-1 | 6.458 | 1.19e-2 | ok ✓*Part 8* |
| **minShift** | **static** | **[0.995, 1.007]** | **4.4810e-3** | 6.150 | 6.50e-3 | **best measured** |
| minShift | consistent | [0.995, 1.006] | 4.4813e-3 | 6.171 | 6.54e-3 | ok |
| minShift | akinci | — | — | 0.817 | 1.9e+6 | **NaN @ 137** |

Three results, none of which either change shows on its own:

1. **The interaction is the finding.** `minShift` alone is worth 1.24x and
   `static` alone 5.9x; composing independently predicts **2.42e-2**. Measured:
   **4.48e-3 — 5.4x better than that prediction**, 40x the shipped default and
   5.3x the best configuration previously known. The two changes are not
   additive knobs, they are one fix applied at two points: the gauge stops the
   constant mode winding up, and the static-boundary operator stops the wall
   manufacturing the error the gauge was absorbing.
2. **`consistent` is inert once the gauge is right** — 4.4813e-3 against
   `static`'s 4.4810e-3, a 0.007% difference. Under the clamp it was worth 4.7%
   (§ Part 11). So [BWJ23]'s `rho_k = rho0` boundary *state* was compensating
   for the gauge, not contributing in its own right. Its operator terms, which
   `static` already has, are the whole of its value.
3. **Part 11's best row does not survive the gauge fix.**
   `akinciBoundaryVolume` was the best measured configuration (2.38e-2) — under
   the clamp, where it still reproduces exactly. Under `minShift` it NaNs at
   step 137 with a divergence-free residual of 1.9e6. **This is the fourth time
   a boundary ranking in this document has inverted** when something underneath
   it changed (Part 2 → Part 11 on `dt`; Part 6 on formulation; now Part 11 →
   here on the gauge).

**At the legacy CFL (1.2), all eight diverge — including the shipped default.**

| gauge | boundary | steps | t_final | `rho` range | `\|rho-1\|` |
|---|---|---|---|---|---|
| shipped | shipped | 258 | 5.535 | [0.139, 2.452] | 5.38e-1 ✓*Part 9* |
| shipped | akinci | 184 | 3.677 | [0.894, 1.139] | 3.50e-2 ✓*Part 11* |
| shipped | consistent | 90 | 1.557 | [0.842, 1.241] | 9.25e-2 ✓*Part 11* |
| shipped | static | 80 | 1.412 | [0.840, 1.271] | 8.68e-2 ✓*Part 9* |
| minShift | shipped | 38 | 0.690 | [0.681, 1.257] | 2.16e-1 ✓*Part 8* |
| minShift | akinci | 34 | 0.603 | [0.951, 1.285] | 3.60e-2 |
| minShift | static | 33 | 0.585 | [0.934, 1.267] | 3.42e-2 |
| minShift | consistent | 30 | 0.529 | [0.950, 1.349] | 4.43e-2 |

> **Do not rank the `1.2` half by `|rho-1|`.** That column is a mean over the
> second half of the steps a run *survived*, so a row that dies at step 30
> reports its error over steps 15-30, before the error has developed, and a row
> that reaches 258 reports over steps 129-258. The column is only comparable
> between rows of similar length. `t_final` is the meaningful metric here, and
> every row is a divergence.

**This reframes the entanglement, and it removes the trade-off.** The story was
"each change is better at 0.4 and worse at 1.2", which reads as a trade. It is
not one: **at 1.2 there is no viable configuration at all.** The shipped
default's apparent survival to t=5.54 is the misleading part — it is surviving
with `rho` in [0.139, 2.452], i.e. 145% over-dense and 86% evacuated, which is
not a simulation anyone should want. The better configurations "die sooner"
only in the sense that they stop rather than continue producing nonsense.

So the question was never "0.4 with the new defaults against 1.2 with the old
ones". **The legacy CFL was already broken; the published CFL is what makes any
of this measurable, and it has landed.** Nothing is blocking the rest.

### Open items, ranked

1. ~~**Run the 2x2x2x2.**~~ **Done — see above.** The follow-through is to
   **land `minShift`-on-bounded + `staticBoundary` as the defaults**, which is
   now a 40x result with a validated harness behind it and no measured downside
   at the shipped CFL. Two specifics: widening `minShift` means relaxing
   `solveIncompressible`'s guard for pinned rows (its free-surface half must
   stay — §1.5), and `staticBoundary` should land per-solver, per item 2.
   `consistent` and `akinciBoundaryVolume` should **not** land: the first is
   inert against `static` and the second diverges.
2. **Move `BoundaryOperatorTerms` to `RelaxedJacobiSolverConfig`** (per solver)
   and default `pressureSolver` to `staticBoundary` while
   `divergenceFreeSolver` stays `full`. That is the configuration measurement
   endorses and it cannot be expressed today. Caveat: `both` is *also* better
   than the baseline on every axis including both solvers' residuals, so
   "PS only" vs "both" is a 4% accuracy question, not a stability one.
3. **The divergence-free half-state's contraction collapse is unexplained.**
   Under `staticBoundary` applied to the divergence-free solve alone, each
   solve removes ~20% of its incoming residual against ~50% under `full`; the
   incoming residual creeps 2.4e-2 → 3.8e-2 over 250 steps and then detonates
   (max `|a_p|` 19.8 → 1.04e4 at step 276, NaN at 282). Each solve still
   converges internally. Three mechanisms tested and eliminated (§2). The
   surviving lead: the harm appears **only** when the two solves run
   inconsistent operators — applied to both, the divergence-free residual is
   the *best* of any configuration (1.57e-2 vs `full`'s 7.52e-2). So look at
   what the constant-density solve leaves behind for the *next* step's
   divergence-free solve, not at either solve alone.
4. **Wire `rtol` into the relaxed-Jacobi path** as a disjunction with the
   existing absolute test (§1.7), and re-measure. Then the one-sided-average
   vs floored-average criterion behind a flag.
5. **Move `computeMdbcPressure` inside the solver iteration** ([B] Alg. 1
   recomputes `p_b` from the current iterate every sweep, so it is a pure
   function of `p_f` with no state and no lag) and add [B]'s **SVD safe
   inversion** on the MLS gradient system. This codebase falls back on a
   neighbour-count threshold (9) with no conditioning guard, and Part 2's
   worst offender had `numNeighbors = 22` with `|grad p| = 153` — a count does
   not detect a co-linear neighbourhood. *But see §4's note: if `consistent`
   lands, `mdbcMlsPressure` should be deprecated rather than repaired.*
6. **`relaxationFactor = 0.3` has a 4% stability margin on a bounded state**,
   not the ~15% `JacobiRelaxationMode`'s docstring quotes from the TGV family
   (`omega`/window = 0.957). Independent of everything else and probably the
   cheapest robustness win available. `probe_boundaryOperatorTerms.py --mode
   spectrum` measures it on demand.
7. **Test the mDBC hypothesis for `DensityEvolution.hybrid`.**
   `computeMdbcDensity` runs at the top of the step on the *carried* density,
   so under `hybrid` the boundary rows are extrapolated from a drifted field.
   The periodic case has no mDBC, which is exactly the difference. Cheap test:
   re-sum for the extrapolation only.
8. **Port [C]'s shear-wave decay case** (§5 Q5). Still the missing reference
   case: 2D, fully periodic, no gravity or explicit viscosity, so any decay is
   solver artifact — and it grades artificial viscosity (Fig. 3, sinus
   amplitude) separately from disorder/volume error (Fig. 4, max density) on
   exactly the axis the `ShiftApplication` modes differ on, with published
   curves to compare against.
9. **Warm start.** [BK] does a full one (worth ~3x in iteration count), [I] and
   [C] do `0.5 p(t-dt)`, [B] does none; this codebase does none
   (`incompressible.py:167`). Apply **after** the stopping criterion is fixed —
   warm-starting a solver that is winding up carries the wind-up across steps,
   which the cold start currently prevents.
10. **Rename `dfsph.py`/`dfsph_step` → `vdps.py`** (§1.3). Zero-risk, and the
    registered scheme name already needs no change.
11. **The scheme split.** Add a real `dfsph` scheme once 1–4 land. Fully
    specified by [BK] Alg. 1: density solve → integrate → divergence solve; no
    position shift; full warm start; tolerances 1e-4 / 1e-3; [B]'s MLS
    boundaries recomputed inside each iteration. Shares `computeAlpha`, both
    IISPH kernels, the Jacobi loop, and every case. **Do not build it before
    the stopping criterion and the boundary defaults land** — it is the
    formulation that puts the density solve into momentum, so it is the one
    most damaged by the current defects.
12. **`MINRES` without the non-negativity clamp** — 2.15x on periodic density
    error for 1.73x wall time. Needs (a) a free-surface test, since a shifting
    potential that may go negative can *pull* particles together, and (b) an
    equal-cost comparison against `relaxedJacobi` at `maxIterations=128`.
    Opt-in at best.

### Known-open, lower priority

- **`rotatingSquarePatch` corner density loss** (Part 3). Resolution-
  independent `rho ≈ 0.506` at the four convex free-surface corners, present
  within ~8 steps, under any `dt` or integrator; `--scheme deltaSPH` on the
  same geometry holds 0.9998. **[BK] §5 documents this as a known method
  limitation** ("the density near a free surface is underestimated which
  causes unnatural particle clustering") with a published remedy (ghost
  particles, Schechter & Bridson 2012) — so it is not an implementation bug.
  Two real, independent bugs on that case are ruled out as the primary driver
  but still want fixing: it has **no `Case.timestep` hook** (so `dt` is never
  adapted under any scheme) and it inherits `integrationScheme='rungeKutta2'`.
  The existing candidate fix (`pressureB[surfaceIndicators == 1] = 0.0`,
  commented out in `divergenceFree.py`) is untestable as-is: `detectFreeSurface`
  flags 96/100 particles at nx=32 and 52% at nx=96 on this thin patch, and it
  is wired into only one of three solver paths.
- **Nothing enforces `semiImplicitEuler`.** The PPE derivation is specific to
  it. All three incompressible cases set it explicitly, but `CaseSpec`'s
  default is `rungeKutta2` and no code path checks `scheme == divergenceFree`
  against `integrationScheme`. A multi-stage RK integrator would solve each
  stage as if it were final and then blend, so the blended velocity is not
  divergence-free. Candidate: a one-line assert/warn in `dfsph_step`.
- **`solveIncompressible` should raise on the Krylov path**, the way the
  `JacobiRelaxationMode.optimal` path already does, since it routes through
  `solvePressureKrylov(..., gauge='nonnegative')` — the clamped-solve
  combination both papers rule out.
- **Two mDBC slip defects** (`modules/mdbc/velocity.py`). `freeSlip` computes
  `u_f - 1*(u_f.n)n` while the comment above it says `u_f - 2*(u_f.n)n`;
  `noSlip` applies the same projection undocumented. Fixing measures *worse*
  (§2), but the comment and the code should be made to agree.
  Separately, **`noSlip`'s `2 u_wall` term is dead code**: it reads the *ghost*
  row's velocity, and `rigidBody/update.py:51-56` refreshes ghost velocities
  only for `BCType.constant`. Measured on `lidDrivenCavity`, ghost `|v|` is
  exactly 0 while boundary `|v|` reaches 1.0. It is latent today only because
  `enforceDirichlet` runs after `computeBoundaryVelocities` and re-imposes the
  lid. It would bite the first moving no-slip wall not backed by a Dirichlet.
- **Two-way coupling is absent.** [BWJ23] Eq. 35's `f_{k<-i}` is never applied;
  `integrateRigidBody(rigidBody, 0, 0, dt)`. Fine today (all walls static), but
  a moving-body case would silently get one-way coupling.
- **`DensityEvolution` + `BoundaryPressureMode.plain` is a trap** — `plain`
  skips the mDBC extrapolation and the non-summation modes skip the re-sum, so
  `kind != 0` rows would never be updated at all. Documented in the enum, not
  guarded in code.
- **Ghost particles (`kind == 2`) are lumped in with boundaries** by the
  `kj == 0` test. Correct for this scheme (`dfsph_step` freezes them too), but
  no measurement here separates them.
- **Two intermittent test flakes**, both pre-existing, neither a regression:
  `test_implicitShiftingComparison.py`'s `implicitShiftAutomatic` assertions
  (~1 run in 3; relative density std falls smoothly for six steps then jumps
  0.0145 → 0.217 on step 7) and
  `test_incompressibleKrylov.py::test_minresGivensMatchesDenseLstsq`.

---

## 5. The literature, and where this codebase deviates

**Sources read in full.**

- **[C]** Cornelis, Bender, Gissler, Ihmsen, Teschner, *An Optimized Source
  Term Formulation For Incompressible SPH* (TVCJ 2018/19) — VD+PS. **This is
  the paper this scheme implements.**
- **[BK]** Bender & Koschier, *Divergence-Free SPH* (SCA 2015) — DFSPH proper.
- **[I]** Ihmsen et al., *Implicit Incompressible SPH* (TVCG 2014) — IISPH;
  the solver this codebase's Jacobi loop discretises.
- **[B]** Band, Gissler, Peer, Teschner, *MLS pressure boundaries* (C&G 76,
  2018).
- **[BWJ23]** Bender, Westhofen & Jeske, *Consistent SPH Rigid-Fluid Coupling*
  (VMV 2023) — derives DFSPH from a density constraint; **this is the
  derivation behind `staticBoundary`.**
- Plus SPlisHSPlasH source (`~/dev/SPlisHSPlasH`) as reference implementation.

**Still unavailable:** Adami et al. 2012 (wall BC), Akinci et al. 2012
(rigid-fluid coupling), Ihmsen et al. 2010 (adaptive timestep), Adami et al.
2013 (transport velocity — needed to close the background-pressure question).

### The published CFL is calibrated against a metric a free surface silences

[BK]'s `dt <= 0.4 d/|v_max|` was validated alongside a convergence tolerance on
an *average density error*, and that average is computed compression-only.
SPlisHSPlasH, `TimeStepDFSPH.cpp:603-618`:

```cpp
const Real residuum = min(s_i - aij_pj, static_cast<Real>(0.0));   // r = b - A*p
density_error -= density0 * residuum;
...
avg_density_err = density_error / numParticles;
```

`min(..., 0)` keeps only compressive residuals; the sum is then divided by
**all** particles. A particle in expansion contributes exactly zero to the
numerator and a full unit to the denominator, so the reported error is diluted
in proportion to how much of the domain is expanding — and a free surface is
precisely the configuration that puts a large population there. Measured,
nx=128:

| case | fraction `rho < rho0` | honest `mean｜rho-rho0｜` | compression-only mean | dilution |
|---|---|---|---|---|
| `randomFlowIncompressible --bounded`, 200 steps | **21.7%** | 3.00e-3 | 2.66e-3 | **1.13x** |
| `rotatingSquarePatch --scheme divergenceFree`, 20 steps | **100.0%** | 3.29e-2 | **0.00e+00** | **total** |
| same, 60 steps | 93.9% | 4.53e-2 | 9.74e-5 | **465x** |

**On the free-surface case the metric reads exactly zero while the true mean
density error is 3.3%** — not approximately zero, zero, because after 20 steps
not one of the 1764 fluid particles is above rest density, so every term in the
numerator is clipped away. The bounded case is the other extreme: the metric is
within 13% of honest.

Two caveats. `rotatingSquarePatch` under `divergenceFree` is itself broken
(§1.9), so its 3-4% error is not what a healthy free-surface simulation would
show — but the 100%-below-`rho0` structure that silences the metric is a
property of sampling a free surface, not of that bug. And **this codebase's own
metric is worse than the published one**: `clamp(-residual, min=-threshold)`
lets an expanding particle *subtract* `threshold` rather than contribute zero.

The consequence is a calibration-scale mismatch of two to three orders of
magnitude between the scene type the constant was tuned on and the scene type
this codebase runs. It is a concrete reason to expect 0.4 to be permissive
here, and the reason §4 item 4 (the stopping criterion) is not cosmetic.

**The sweep that prediction called for has been run** (§10 has the table and the
cost analysis). The part that belongs here is the dilution column itself:
`frac rho < rho0` measures **5-13% on the bounded case**, against the ~100% a
free-surface scene shows in the table above. That is the calibration mismatch
quantified on the scene this codebase actually runs, and it is why the published
0.4 is permissive here without being wrong where it was tuned.

The sweep's own answer — a 2.83x gain for the first halving and ~1.25x for each
one after, i.e. a knee at 0.2 — was measured at the *shipped* boundary
configuration, where the near-wall band dominates the error. §4's factorial
then removed 40x of that band, so the sweep's case for a smaller default no
longer stands on its own numbers and should be re-run against the new
configuration (§10, item 2).

### Answers to the questions that drove the literature sessions

- **Q1, the unreachable setpoint.** [C]'s PS solve has exactly the same
  structural bias and does not avoid it — it avoids *integrating* it (§6: "we
  do not update the velocities using the solution of the PPE solver with the DI
  source term … it is just a resampling"). Not warm starting, not one-sided
  clamping, not rest-density recalibration. See §1.2.
- **Q2, which density.** Summation, in both papers, with a one-step divergence
  prediction on top ([C] Eq. 11, [B] Alg. 1). Neither integrates density, so
  `DensityEvolution.summation` is the paper-faithful default. One difference
  that **favours this codebase**: `finalize` recomputes the summation density
  at the *shifted* positions before the solve, where [C] predicts from the old
  positions. Do not "fix" that toward the paper. Rest-quantity calibration
  against the sampled configuration is real but **boundary-only** ([B] Eqs.
  6-7), and [B] §6.1 blames its planar-boundary coefficient `beta` for that
  variant's worst error.
- **Q3, shifting vs the constant-density solve.** No published variant does
  both at full strength — that is [C]'s whole point (§1: prior two-PPE
  combinations "typically result in inconsistent particle positions and
  velocities"). Measured here: doing both grows `tgv`'s kinetic energy 6.6x
  over 200 steps. `finalize` correctly drops the shift under `inStepVelocity`.
- **Q4, boundary coupling.** **There is no published VD+PS × mDBC/MLS pairing**
  — [C] ships Akinci one-layer boundaries and runs its headline test on a
  *periodic* domain "such that boundary handling does not influence the
  solution". This codebase's combination is novel, which is the context for
  every boundary-mode ranking here being formulation- and `dt`-dependent. The
  boundary-pressure feedback loop `mdbcPressureRelaxation` damps is a
  **documented** failure mode of holding boundary pressure as *state* ([B]
  §3.3, contrasting Pressure Boundaries' `omega_b` with its own
  recompute-in-loop MLS).
- **Q5, reference values.** Neither [BK] nor [B] reports a Taylor-Green decay
  rate, so `tests/test_physics.py`'s 0.55x has no published counterpart. [C]
  supplies the benchmark that was actually wanted — the shear-wave decay
  (§4 item 8).
- **Q6, timestep near boundaries.** **Neither paper carries a wall-proximity
  constraint.** [B]'s answer to bad wall behaviour is a better boundary
  pressure, not a smaller `dt` (§6.6: pressure mirroring "requires a time step
  that is half as large compared to our MLS extrapolation"). The real finding
  was the units mismatch — see §1.6. This supersedes Part 5's costed
  wall-aware `dt` proposal (~2.4x throughput) entirely.
- **Q7, background pressure.** **Open.** Two adjacent data points: [B] Eq. 3's
  Adami-style extrapolation carries an explicit hydrostatic term (published
  precedent for a background pressure that is *set*, not allowed to drift), and
  [C]'s PS solve is itself a de-clumping potential never applied to momentum.
  Needs a transport-velocity paper.

### Remaining deviations, with status

| deviation | status |
|---|---|
| `cflFactor` applied to `h`, not `dx` | **fixed in the working tree** (Part 12, uncommitted) |
| `computeAlpha` / operator include static-boundary reaction terms | `BoundaryOperatorTerms.staticBoundary`, **opt-in**; belongs per-solver (§4 item 2) |
| Boundary rows enter the solve at mDBC-extrapolated `rho_k` (1.3+), not `rho0` | `BoundaryPressureMode.consistent`, **opt-in** (§4) |
| MLS boundary pressure computed once per step and carried as under-relaxed state | **open** (§4 item 5) |
| No SVD guard on the MLS gradient fit | **open** (§4 item 5) |
| No warm start | **open** (§4 item 9), deliberately deferred |
| `omega = 0.3` against both papers' 0.5 | **explained, not a bug.** `INCOMPRESSIBLE_SOLVER_PLAN.md`: `omega < 2/rho(D^-1 A)` with `rho(D^-1 A) ≈ 5.636`, a degenerate high-frequency lattice cluster, `dt`-invariant. Candidate contributor: `n_h = 4` gives ~50 neighbours in 2D against [I] §2.2's "typically 30-40", and the IISPH operator reaches neighbours-of-neighbours. |
| Krylov routed at the clamped solve | should **raise** (§4, known-open) |
| Stopping criterion is absolute and floors negative contributions | **open**, §1.7 / §4 item 4 |
| Two-way rigid coupling absent | **open**, no case needs it |

---

## 6. Configuration surface

Every switch added by this work, with its default and what it measured.
All are round-tripped through `incompressibleConfigToDict` /
`dictToIncompressibleSPHConfig`.

| setting | default | notes |
|---|---|---|
| `boundaryPressureMode` | `mdbcDensity` | `plain` / `mdbcDensity` / `mdbcMlsPressure` / `consistent`. **`consistent` is the best measured** (§4); `mdbcMlsPressure` is the worst and should be deprecated. |
| `boundaryOperatorTerms` | `full` | `staticBoundary` is the published formulation, 5.9x at the published CFL. `diagonalOnly`/`operatorOnly` are deliberately-mismatched diagnostics — `diagonalOnly` runs the wall's Jacobi step 1.6x too large and NaNs in 47 steps. |
| `akinciBoundaryVolume` | `False` | `consistent` only. Measured `m~/m_nominal` mean 1.102, max 1.456 on the five-layer band. Best row in the table *inside the operator*; fatal as actual mass (§2). |
| `shiftPressureGauge` | **`minShift`** | The Part 4 fix, and the one default that changed. `nonNegativeClamp` is the historical clamp and stays selectable. Scoped to solves with no pinned rows and no free surface. |
| `forceShiftPressureGauge` | `False` | Bypasses that scoping. Exists because half its justification measured false (§1.5) and the other half's evidence was measured at 3x the CFL (§3.6). |
| `shiftApplication` | `positionShift` | The paper-faithful default. `positionAndVelocity` and `inStepVelocity` are much better at walls and dissipative in the bulk (§1.2). |
| `densityEvolution` | `summation` | `continuity` (WCSPH standard) fails everywhere but `tgv`; `hybrid` matches `summation` exactly where support is complete, for ~21% less wall time on `tgv`, and dies at 286 steps at an mDBC wall (§4 item 7). |
| `mdbcPressureRelaxation` | `0.3` | Load-bearing for `mdbcMlsPressure` — at 1.0 it NaNs in 7-8 steps. Never swept; chosen to match the solver's own `relaxationFactor`. |
| `mdbcNoPenetrationShift` | `True` | Removing it is worse (§2). |
| `integrateRho` | `False` | Legacy alias; `resolveDensityEvolution` maps `True` → `continuity`. |
| `cflFactor` (incompressible cases) | **`0.4`** | Working tree only; multiplies `dx`. See §7. |

`ShiftApplication` comparison, bounded case, nx=128 (legacy CFL, so `positionShift`
is at its death state):

| mode | near-wall `\|rho-1\|` | bulk | penetrating | `rho` range | outcome | `tgv` decay/analytic |
|---|---|---|---|---|---|---|
| `positionShift` (default) | 0.30 | 0.113 | 4506 | [0.139, 2.452] | NaN t=5.54 | **0.55x**, monotone |
| `positionAndVelocity` | 3.3e-2 | 1.2e-3 | 239 | [0.936, 1.147] | t=8.0 | 3.28x, non-monotone |
| `inStepVelocity` | **9.7e-3** | **6.6e-4** | **63** | [0.986, 1.140] | t=8.0 | 3.26x, non-monotone |

For comparison: `positionShift` at the **published** CFL reaches t=8.0 with
near-wall `|rho-1|` = 2.6e-2 — better than `positionAndVelocity` and without
its dissipation — for 4.2x the steps.

`DensityEvolution`, nx=128, 900 steps, published CFL. Note the split between
the **carried** density (what solvers and diagnostics see) and a fresh
summation on the same positions:

| case | mode | steps | `\|carried-1\|` | `\|true-1\|` | drift | DF resid | wall s |
|---|---|---|---|---|---|---|---|
| bounded | `summation` | 901 | 7.04e-3 | 7.04e-3 | 0 | 7.52e-2 | 114.6 |
| bounded | `continuity` | **63** | 7.06e-3 | 4.30e-2 | 4.74e-2 | 1.46e-1 | 3.9 |
| bounded | `hybrid` | **286** | 2.27e-2 | 2.52e-3 | 2.14e-2 | 1.67e-1 | 30.2 |
| periodic | `summation` | 901 | 1.93e-3 | 1.93e-3 | 0 | 8.79e-3 | 73.2 |
| periodic | `continuity` | **470** | 9.05e-3 | 3.66e-2 | 3.93e-2 | 7.25e-1 | 38.2 |
| periodic | **`hybrid`** | 901 | 3.52e-2 | **1.92e-3** | 3.33e-2 | **8.49e-3** | 72.6 |

---

## 7. Bugs found and fixed along the way

Landed, verified, and not worth re-litigating — but worth knowing about.

| bug | worth |
|---|---|
| The implicit shift computed its own kinematic velocity correction every step and then discarded it (`self.state.velocities -= proj_vel` commented out) | Fixed the original nx=128 step-720 divergence outright (commit `122d326`) |
| Boundary-row masking froze pressure at literal `torch.zeros_like` rather than at `currentState.pressures` | Made `mdbcMlsPressure` a **silent no-op** — `mdbcDensity` and `mdbcMlsPressure` were bit-identical at every step. Fixed in all four solver paths; Krylov got proper Dirichlet lifting (`b = source - A(boundaryOnly)`, re-pin at the end) |
| The exposed loop: with the projection actually live, boundary pressure doubled every step to NaN in 7 steps (`numNeighbors=22`, `\|grad p\|=153`) | Fixed with `mdbcPressureRelaxation` |
| [C] Eq. 17's velocity resample contracted `dx` over the **wrong axis** and applied it with the **wrong sign** (`einsum('nij,ni->nj')` and `-=`, i.e. `-A^T dx` where `+A dx` was wanted) | Verified against the compiled kernel with an asymmetric linear field; real bug, correctly fixed, but moves neither open metric — `dx = dt^2 dvdt` is second-order in `dt` |
| `drhodt` was evaluated on the **pre-projection** velocity, so an integrated density re-accumulated exactly the divergence the solve had just removed | **1800x** on the divergence-free residual (2.61e+2 → 1.46e-1); only reachable once `DensityEvolution` made the branch real |
| `shiftProperties.active` ran a full deltaSPH shifting solve every step and **discarded it** — `finalize` bound it to `dx` then shadowed `dx` | Any prior test of "the deltaSPH shift on top" was measuring a no-op. Fixed (`dxDeltaShift`), then the configuration was tested and rejected |
| `densityEvolution`, `boundaryOperatorTerms`, `forceShiftPressureGauge` were not serialised by `incompressibleSPHConfigToDict` | Silently reset to defaults on a TOML/HDF5 round-trip |
| `akinciBoundaryMass`'s `BoundaryToBoundary` sum is zero for fluid and ghost rows, and `rho0/0` is a simulation-ending mass | Never reached a neighbour sum, but is now a fallback rather than a landmine |
| `integrateRho`'s `True` branch was dead — `finalize` re-summed unconditionally, twice per step | Now real via `DensityEvolution` |
| `self.state.surfaceIndicator` assigned where the field is `surfaceIndicators` | `finalize`'s `detectFreeSurface` result is written to a typo'd attribute and discarded. **Not fixed.** |
| `finalize` recomputes plain summation densities and never re-applies mDBC, so the shifting solve sees systematically-low boundary rows | Real inconsistency; applying mDBC there changes nothing measurable (353 vs 349 penetrating). **Not fixed.** |

---

## 8. Tooling

All in `scripts/`, all confirmed working, none require source edits to use.

**Start here for anything about the current open items:**
- `probe_boundaryOperatorTerms.py` — `--mode diag` (exact diagonals binned by
  wall depth), `--mode spectrum` (`rho(D^-1 A)` and the `omega` margin),
  `--mode dfTrace` (per-solve first/last residuals), `--solvers` to scope
  `staticBoundary` to one solve.
- `probe_fourWayDefaults.py` *(new, Part 13)* — the `cfl x gauge x boundary`
  factorial. `--cfls/--gauges/--boundary` subset it; rows print as they
  complete. Ten of its cells are prior published rows, so it doubles as a
  regression check on this whole document.
- `probe_consistentCoupling.py` — [BWJ23]'s `consistent` mode end to end.
- `probe_cflCondition.py` *(new, Part 12)* — `--mode verify` checks that
  `dt |v_max| / dx == cflFactor` exactly whenever the advective term binds
  (run: 39/40 steps, 0.4000 against 0.4); `--mode sweep` runs each `cflFactor`
  to the same simulated time and reports the sub-rest-density fraction
  alongside the error, printing rows as they finish (run: 0.4 / 0.2 / 0.1, §10).
  It must not be given `--nSteps` -- `runner.py:246` only honours `tLimit` when
  `nSteps` is absent, so a step cap silently converts a time-matched sweep into
  a step-matched one, which is the comparison the mode exists to avoid.
- `probe_boundedIncompressibleBlowup.py` — step-by-step wall penetration,
  per-step worst particle, wall-depth density profile. Knobs for everything
  that turned out not to matter, so they stay cheap to re-check: `--mode`,
  `--noPenShift`, `--shiftCap`, `--cflFactor`, `--case randomFlow` for the
  deltaSPH control.

**Solver and gauge:**
- `probe_shiftPressureGauge.py` — end-to-end A/B of the two gauges through the
  real solver, either case (`--extra=--bounded`).
- `probe_incompressibleGaugeDrift.py` — standalone reimplementation of
  `solveIncompressible`'s loop, verified byte-identical to the real solver on
  the baseline. `--gauge {clamp,center,center-clamp,minshift,quantile,none}`,
  `--project-source`, `--null-test`, `--no-clamp`, `--maxIters`, `--jitter`,
  `--setpointEps`. Has no `--tolerance` knob; add one if §4 item 4 is run.
- `probe_incompressiblePressureSolvers.py` — Krylov methods end-to-end inside a
  case (the prior plan's numbers are all single solves on a seeded state).
- `probe_operatorDiagonal.py` — exact diagonal extraction, one matvec per row.
- `probe_relaxedJacobiOmega.py` — the `omega` stability window.

**Density and sampling:**
- `probe_densityBiasVsDisorder.py` — the no-dynamics demonstration that the
  bias is structural (§1.1).
- `probe_densityEvolution.py` — `DensityEvolution` modes, carried vs true.
- `probe_initialSampling.py` — as-sampled uniformity and mass normalisation.
- `probe_densitySign.py` — signed vs unsigned bulk bias on the running case.

**Boundaries:**
- `probe_wallSupportCompleteness.py` — Shepard sums and `|A.1|/|A.rand|` by
  wall depth (§1.5).
- `probe_boundaryVelocityModes.py` — `BCType` A/B; `--mode verify` decomposes
  each condition against the wall normal and grades it against the published
  integer slopes.
- `probe_dfsphWallDensityProfile.py` — `|rho-1|` binned by signed wall distance,
  DFSPH vs deltaSPH.
- `probe_randomFlowIncompressibleBoundaryModes.py` — the three legacy boundary
  modes at matched physical time.
- `probe_mdbcMlsPressureInstability.py` — re-runs `computeMdbcPressure`'s
  internals per boundary particle on the steps before a NaN.

**Case-specific / historical:** `probe_tgvShiftGauge.py`,
`probe_pressureGaugeDrift.py`, `probe_kolmogorovIncompressible.py`,
`probe_kolmogorovAdjacencyRebuild.py`, `probe_kolmogorovContinuation.py`,
`probe_kolmogorovSpinup.py`, `probe_kolmogorovIncompressibleVelCorrection.py`.

**Always run after touching a `@wp.kernel`/`@wp.func`:** the `gradcheck` skill
(`scripts/gradcheck_incompressible.py`). Part 9 added a second case there
covering the `includeBoundaryReaction=False` branch with mixed `kinds` in one
launch — a conditional accumulation inside a neighbour loop is exactly the
shape that file is about.

---

## 9. Session index

One line each, for locating the full write-up in git history.

| part | date | outcome |
|---|---|---|
| 1 | 08-26 | Kolmogorov nx=128 step-720 divergence — the fix (the dropped velocity correction) was already in commit `122d326`; verified to 1600 steps. |
| 2 | 08-26 | mDBC boundary handling for the incompressible scheme: `BoundaryPressureMode`, solver-row masking in all four paths, `computeMdbcPressure`, and `randomFlowIncompressible` — the first case to sample `kind==1` under `divergenceFree`. Two bugs found and fixed. Its boundary-mode ranking is **retracted** (§3.1). |
| 3 | 08-26 | `rotatingSquarePatch` corner loss root-caused, not fixed; `integrateRho`'s dead branch found; the [C] Eq. 17 sign/axis bug fixed. |
| 4 | 08-26/27 | The periodic pressure-gauge drift: integrator wind-up against an unreachable setpoint. `ShiftPressureGauge.minShift` landed **as the default**. |
| 5 | 08-27 | Bounded DFSPH stability: the wall is late, not weak (§1.6). `ShiftApplication.positionAndVelocity` and `inStepVelocity` landed opt-in. |
| 6 | 08-27 | Scheme-naming/architecture question opened; the DFSPH × boundary-mode matrix measured. Nothing restructured. |
| 7 | 08-27 | Literature session ([C], [B]). Headline: **this is VD+PS, not a mis-named DFSPH** (§1.3). Its Q1 tolerance argument is retracted. |
| 8 | 08-27/28 | The originals ([BK], [I]). Setpoint and `alpha` hypotheses both falsified; **the published CFL constant landed as the one positive result**; MINRES-without-clamp measured; the wall-truncation premise corrected. |
| 9 | 08-28 | The terms the papers do not compute for static boundaries: `BoundaryOperatorTerms.staticBoundary`, 5.9x at the published CFL. Its boundary-velocity explanation is retracted. |
| 10 | 08-28 | Initial sampling (exact; the mass is not) and `DensityEvolution`. The `drhodt` pre-projection bug, worth 1800x. |
| 11 | 08-28 | [BWJ23] — the derivation behind Part 9. `BoundaryPressureMode.consistent`, the best configuration measured. |
| 13 | 08-28 | The factorial (§4) and the CFL sweep (§5). `minShift`+`staticBoundary` is 40x the default and 5.4x better than the two changes composing independently; `consistent` is inert and `akinci` diverges once the gauge is fixed; the legacy CFL has no viable configuration at all. Ten prior rows reproduced exactly. |
| 12 | 08-28 | The CFL condition rewritten in [BK]'s units (particle diameters) and **landed as the default**; verified per step and bit-for-bit against the old units. The compression-only error metric measured as diluted 465x on a free surface and 1.13x on the bounded case (§5). |

---

## 10. Overview — where things stand

### What is shipped and stable

The incompressible path is **VD+PS** (Cornelis et al.), faithfully implemented,
registered as `divergenceFree`. Exactly one of this work's changes altered a
shipped default; every other new switch is opt-in and default-inert, and the
bug fixes are behaviour-preserving on every case that existed before them.

- **`ShiftPressureGauge.minShift` is the default** (Part 4). It turns
  `kolmogorovIncompressible` at nx=128 from a NaN at step 574 into a stable
  1000-step run with density in [0.980, 1.015]; it is a byte-identical no-op on
  every wall-bounded or free-surface solve, and it leaves `tgv`'s analytic
  decay rate alone to 0.4%.
- **Several real bugs are fixed** (§7), the largest being the Eq. 17 resample,
  the boundary-row masking, and the `drhodt` pre-projection evaluation.
- Full suite passes (241 passed, 1 skipped) and `gradcheck_incompressible.py`
  passes. Two known intermittent flakes, both pre-existing (§4).

**Case status at the shipped defaults:** `tgv` and `kolmogorovIncompressible`
(periodic) are healthy. `randomFlowIncompressible --bounded` is the case that
exercises everything and is where all the remaining error lives.
`rotatingSquarePatch --scheme divergenceFree` (free surface) is broken and is a
known method limitation, not an implementation bug.

### Part 12, the CFL condition — landed and verified

`kolmogorovIncompressibleTimestep` now applies `cflFactor` to the particle
**diameter** `dx` rather than the support radius `h`, which is the form [BK]
state their condition in, and `kolmogorovIncompressible` /
`randomFlowIncompressible` ship `cflFactor = 0.4` — literally their constant.
The old form was wrong in two independent ways: it was `n_h` times larger (4x
here, so the legacy `0.3` permitted 1.2 spacings of travel per step), and it
silently rescaled with the neighbour count, so the same number meant different
physics at different `n_h`. The viscous limit stays in `h`; it is a diffusion
condition over the smoothing length, not an advection condition over the
spacing.

**Verified two ways.** `probe_cflCondition.py --mode verify` reports the
dimensionless travel `dt |v_max| / dx` per step: **39 of 40 steps are
advection-limited and the travel is 0.4000 against the configured 0.4** (the
fortieth is step 0, which takes the case's seeded `targetDt`). And the change
is *exactly* a relabeling — re-running the §4 baseline row under the new
default reproduces the pre-change run bit for bit:

| | steps | `rho` range | `｜rho-1｜` 2nd half | t_final | DF resid |
|---|---|---|---|---|---|
| pre-Part-12, `cflFactor=0.1` (h-based) | 901 | [0.90228, 1.24742] | 1.7782e-01 | 4.6895 | 7.5192e-02 |
| post-Part-12, `cflFactor=0.4` (dx-based) | 901 | [0.90228, 1.24742] | 1.7782e-01 | 4.6895 | 7.5192e-02 |

So every measurement in this document recorded at "the published CFL" is a
measurement at today's `cflFactor=0.4`. `SimulationConfig.cflFactor`'s
description now records that what it multiplies is scheme-dependent
(compressible and weakly-compressible still use `h`), `tgv` is untouched (no
`timestep` hook, `dt` pinned at 1e-3), and the suite passes.

**The sweep is now run, and §5's prediction is confirmed: there is a lot of
room below 0.4.** Three earlier attempts failed for a reason that turned out
not to be GPU contention — `--mode sweep` was passing `--nSteps`, and
`runner.py:246` only honours `tLimit` when `nSteps` is *absent*
(`timeLimited = spec.nSteps is None and case.timestep is not None`), so every
"run to t=3" was silently a 20 000-step run. With that removed, the same
simulated time on `randomFlowIncompressible --bounded`, nx=128, t=3.0, stock
configuration otherwise:

| `cflFactor` | steps | `rho` range | mean`｜rho-1｜` | mean`(rho-1)` | frac `rho<rho0` | wall s |
|---|---|---|---|---|---|---|
| **0.4** (published) | 457 | [0.902, 1.202] | 5.17e-3 | 4.84e-3 | 12.2% | 166.7 |
| 0.2 | 903 | [0.969, 1.091] | **1.83e-3** | 1.58e-3 | 13.1% | 276.2 |
| 0.1 | 1830 | [0.960, 1.044] | 1.37e-3 | 1.21e-3 | 8.2% | 511.5 |

**Halving the published timestep cuts the density error 2.8x for 1.66x the
wall time. Halving it again buys 1.33x for another 1.85x.** So the knee is at
about 0.2 and the published 0.4 is not near it — on this case 0.4 is a
permissive setting, not a safe one, exactly as §5's calibration-scale argument
predicts. The tail says the same thing more strongly than the mean: `rho_max`
falls 1.202 → 1.091 → 1.044, so at 0.4 the worst particles are still strongly
timestep-limited.

Two things worth keeping from the cost column. The scaling is **sub-linear** —
2x the steps costs 1.66x the time, because a smaller `dt` hands the pressure
solvers an easier problem and they terminate sooner — so halving `cflFactor` is
cheaper than it looks. And this is the first measurement in this document where
the two solvers' iteration counts are *not* pegged at their caps, which is a
second, independent way of seeing §1.7.

The upper side was already measured, repeatedly: at 3x the constant the bounded
case diverges at t=5.5 with `|rho-1|` 5.4e-1 against 1.8e-1 at the constant
itself. Both sides now agree that 0.4 is the edge of a plateau rather than the
middle of one.

*(An independent second sweep, run concurrently, reproduced this table to
three significant figures and extended it to `cflFactor=0.05`: 1.12e-3 for
3692 steps, i.e. a further 1.22x. The knee at 0.2 is firm. Both runs also found
the `--nSteps` defect above independently.)*

### Part 13, the factorial — run, and it settles the default question

**Done, and it changes the answer to item 1 below.** Full tables and analysis
in §4; the short version:

- **`minShift` (forced on bounded) + `staticBoundary` = 4.48e-3 against the
  shipped default's 1.78e-1 — 40x** — and **5.4x better than the two changes
  composing independently would predict** (2.42e-2). Neither shows this alone.
  It is the best configuration measured in this document by 5.3x.
- **`consistent` is inert** against `staticBoundary` once the gauge is fixed
  (0.007%, against 4.7% under the clamp), and **`akinciBoundaryVolume`
  diverges** at step 137. Part 11's "best configuration measured" was an
  artifact of the clamp gauge — its own numbers reproduce exactly, so this is
  an interaction, not a contradiction.
- **At the legacy CFL, 0 of 8 configurations survive**, the shipped default
  included. "Better at 0.4, worse at 1.2" was never a trade-off: 1.2 was
  already broken, and the shipped default merely fails *later* there, with
  `rho` in [0.139, 2.452].
- **Ten of the sixteen cells are configurations this document had already
  published, and all ten reproduced to every recorded digit.**

### What is left, in order

1. **Land `minShift`-on-bounded + `staticBoundary` as the defaults.** The 40x
   result, with a harness validated against ten prior rows and no measured
   downside at the shipped CFL. Needs: relaxing `solveIncompressible`'s gauge
   guard for pinned rows while keeping its free-surface half (§1.5); moving
   `BoundaryOperatorTerms` to `RelaxedJacobiSolverConfig` so it can be set per
   solver (§4 item 2); and a `tgv` re-grade, though both periodic cases are
   strict no-ops for `staticBoundary` (no `kind != 0` particles) and `minShift`
   is already their default. **Do not land `consistent` or
   `akinciBoundaryVolume`** — inert and divergent respectively.
2. **The `cflFactor = 0.2` question is now weaker, and should wait.** The
   sweep's 2.8x was measured at the *shipped* boundary configuration, where the
   near-wall band dominates the error. Item 1 removes 40x of that band. Whether
   a smaller `dt` still buys 2.8x on top of it is unmeasured and is the natural
   re-run after item 1 lands; the case for departing from a cited constant
   should rest on that number, not this one.
3. **Fix the stopping criterion** (§1.7 / §4 item 4) — `rtol` as a disjunction,
   then the one-sided average. Still the one thing that has survived every
   experiment: relaxed-Jacobi, MINRES, BiCGStab, every `dt`, every gauge, both
   shift configurations, and now all sixteen cells of the factorial. (The sweep
   is the one place iteration counts came off their caps — by shrinking `dt`,
   not by fixing the criterion.)
4. **The divergence-free half-state's contraction collapse** (§4 item 3) is
   still unexplained, and item 1 makes it *more* urgent: landing
   `staticBoundary` per-solver is exactly the mismatched-operator configuration
   that triggers it. Verify the landed default is the `both`-solves
   configuration, or explain the half-state first.
5. **Then** the shear-wave case, the warm start, the rename, and the scheme
   split — in that order, and not before 1–4 land.

### What is next, concretely

Do 1. The measurement that gated every default decision in this document has
been made, it points one way, and what it endorses is two config changes plus a
guard relaxation. Item 4 is the risk to watch while doing it.
