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

**Narrowed by Part 16.** The `tgv` measurement is real and reproduces, but the
mechanism above does not explain it on its own. On the `shearWave` case — an
exact solution whose pressure is *constant*, so there is no pressure error and
no advection to conflate with dissipation — **all three `ShiftApplication`
modes dissipate identically**, to 0.1% at `nu = 0.01`, and at `nu = 0` the
position shift is the **most** dissipative of the three. So "applied to
velocity the residual is a permanent unphysical forcing" cannot be asserted as
a general property of the modes: on a flow with no pressure gradient it costs
nothing measurable. Whatever drives `tgv`'s 3.3x requires a real pressure field
or a real advection term, and which one is unmeasured. What *does* separate the
modes on the shear wave is the other axis — the position shift carries 1.8x the
volume error and 2.4x the wall time. See §4, Part 16.

**Confirmed on the bounded case, at 2.1x (Part 18).** At a pinned `dt` on the
inviscid bounded case — where this scheme has no artificial viscosity, so all
loss is numerical — the velocity modes retain 41% of the initial kinetic energy
at t=6 against the position shift's 81%. That is what this section claims, and
it is the measurement that justifies the default: they buy 2x lower density
error for 2.1x the energy loss, with **identical** wall behaviour (zero
particles past the wall for all three modes, over 1201 steps).

**But the `tgv` number is still withdrawn (Part 17).** The 3.2x/3.4x does not
reproduce: at `tgv`'s own default resolution the three modes' decay ratios are
0.615 / 0.693 / 0.674, within 12% of each other, and the velocity modes' ratio
moves by 2x between nx=128 and nx=256 while `positionShift` holds 0.58-0.62
everywhere. A penalty that halves under refinement **cannot** be the permanent
residual this section attributes it to, because that residual is
resolution-independent — §1.1 argues it and Part 16 measured it flat across an
8x range. What survives is narrower and still real: the velocity modes are
**non-monotone at every resolution**, so a decaying flow with no forcing gains
energy on some steps. That is the defect; the mechanism above is not its
cause.

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

### 1.7 The constant-density solve does not converge — it integrates

The claim this section used to make was "the stopping criterion is broken, and
it is the last unexplained thing". Part 15 measured it, and the criterion turns
out not to be the defect. Two things are:

**The periodic cases converge; only the bounded one does not.** At the shipped
defaults `kolmogorovIncompressible` at nx=128 terminates *both* solvers at 3
iterations on every one of 200 steps, with the constant-density statistic at
2.26e-5 against a 5e-4 tolerance — a factor of 22 inside it. The
"never terminates under every gauge, every `dt`, every solver" claim was taken
under the clamp gauge and never re-checked after `minShift` landed. Like every
other error in this document, non-termination is a **near-wall** phenomenon
(§1.6).

**On the bounded case the constant-density solve is an integrator, not a
solve.** Measured along a fixed iterate path (`probe_stoppingCriterion.py
--mode trace`, 200 steps, nx=128):

| iteration k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| `mean\|r\|` | 1.085e-3 | 1.077e-3 | 1.074e-3 | 1.071e-3 | 1.069e-3 | 1.065e-3 | 1.056e-3 |
| pressure range `p_max - p_min` | 0.97 | 1.76 | 3.37 | 6.68 | 13.9 | 28.8 | 59.1 |

**64 sweeps remove 2.7% of the residual and grow the pressure field 61x,
linearly in `k` at about 0.92 per iteration.** That is what an iteration whose
increment `omega r / alpha` is nearly constant does — and the increment is
nearly constant because `A p` is nearly zero for the field being built, i.e.
the source lives almost entirely in the operator's near-null space, which is
§1.1 measured inside a single solve instead of across steps.

Three consequences:

1. **No residual criterion can fire here, and no repair of the criterion
   changes that.** The residual is not shrinking, so the tolerance's *value*
   and *form* are both beside the point.
2. **`maxIterations` on `pressureSolver` is a gain, not a budget.** It sets the
   shift amplitude. That is why §1.7's old table showed accuracy improving with
   iteration count — that was the gain rising, not a solve converging — and it
   is a physical parameter wearing a numerical parameter's name.
3. **The divergence-free solve is the opposite and is fine.** Its residual
   contracts 2.9x over 32 sweeps and its pressure range grows 34% and settles.
   It converges; it is merely stopped early by its cap.

**The floor is not the defect.** §1.7 used to name it as the one-line fix:
this codebase floors each particle's negative residual contribution at
`-tolerance` (`mean(clamp(-r, min=-tolerance))`) where both papers take a plain
one-sided average, so under-dense particles cannot cancel over-dense ones.
Measured along the same fixed path, the three statistics on the
constant-density solve are 1.033e-3 (floored), 1.020e-3 (one-sided, the
published form) and 1.086e-3 (mean absolute) — **within 6% of each other, and
all a factor of two above the tolerance**. Removing the floor changes nothing.
Retracted; see §3.

**Where the floor does matter is the divergence-free solve, and switching it
buys the published iteration count at the published price.** There the
one-sided average is 1.96e-3 against mean-absolute's 1.57e-2 — **8x smaller,
purely by cancellation** — and it is already below the 2.5e-3 tolerance at the
*first* iteration. Run end to end at the shipped tolerances (900 steps,
bounded, nx=128):

| PS criterion | DF criterion | PS iters | DF iters | band, 2nd half | t_final | wall s |
|---|---|---|---|---|---|---|
| `flooredOneSided` | `meanAbsolute` *(shipped)* | 64.0 | 31.9 | **4.4810e-3** | 6.1498 | 126.3 |
| `flooredOneSided` | `oneSided` | 64.0 | **3.0** | 6.8773e-3 | 6.0397 | 93.3 |
| `oneSided` | `meanAbsolute` | 64.0 | 31.9 | **4.4810e-3** | 6.1498 | 125.6 |
| `oneSided` | `oneSided` | 64.0 | 3.0 | 6.8773e-3 | 6.0397 | 93.0 |

Two things to read off. **Swapping the constant-density criterion is
bit-identical** — 4.4810e-3 and t=6.1498 to every digit either way — which is
the cleanest possible confirmation that it never fires. And **adopting the
published criterion on the divergence-free solve collapses it to 3.0
iterations**, which is [BK]'s reported 4.5 to within the difference between two
cases, **for 1.53x the density error**. The published iteration count is
purchasable here by choosing the statistic, and what it costs is exactly the
iterations it skips. That is a reason to be careful about comparing iteration
counts across papers, not a reason to adopt the criterion.

**The iteration budget is nevertheless well chosen** — measured, not assumed
(`--mode budget`, 900 steps, bounded, uncontended):

| PS cap | DF cap | band, 2nd half | wall s |
|---|---|---|---|
| 128 | 32 | 3.54e-3 | 203.2 |
| **64** | **32** *(shipped)* | **4.48e-3** | **127.2** |
| 96 | 16 | 4.20e-3 | 147.3 |
| 64 | 16 | 4.84e-3 | 108.1 |
| 32 | 32 | 6.54e-3 | — |
| 16 | 32 | 1.06e-2 | — |
| 8 | 32 | 1.71e-2 | — |

Cutting the constant-density budget is expensive (16x fewer iterations is 6.8x
the error) and doubling it saturates (1.26x the accuracy for 1.6x the cost);
the divergence-free budget is nearly flat (32 → 8 costs 24%). The obvious
reallocation — buy constant-density iterations with divergence-free ones — does
not pay: (96, 16) is 6% better than the shipped pair for 16% more time. The
shipped (64, 32) sits on the frontier.

**So the practical statement is the one the numbers support: this solver's
iteration count is a tuning parameter that happens to be spelled as a
convergence budget, the tolerance on it is decorative on the bounded case and
binding on the periodic ones, and the honest fix is to say so rather than to
adjust a threshold that nothing can reach.**

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

### 1.10 The scheme is over-dissipative where the flow is violent, and that was invisible until a free surface was put in front of it

Nineteen sessions tuned this scheme against periodic and wall-bounded cases,
where it is now good: the bounded `randomFlowIncompressible` holds a density
band of 4.5e-3 and the periodic cases are healthy. The first genuinely
recognisable scenario it was asked to run — a dam break — exposes something
none of those cases could.

- **It works**, which is itself new: `dambreak --scheme divergenceFree` runs
  3000 steps to t=1.5 with fluid density in [0.907, 1.004] and produces a
  recognisable collapse and run-out. The only other free-surface incompressible
  case, `rotatingSquarePatch`, is broken.
- **The run-out is about half speed.** The surge front travels 1.50 by t=0.7
  against `deltaSPH`'s 2.82 on identical geometry, resolution and `dt`.
- **88% of the kinetic energy disappears between t=0.5 and t=0.8**, exactly
  when the falling column should be turning into horizontal run-out, while
  `deltaSPH` is still gaining. The peak is 2x higher (7.42 against 3.61) and
  then collapses.

Three things this is *not*, each checked: not a timestep artifact (the run is
5x **finer** in time than [BK]'s condition requires, because `dambreak` has no
incompressible `timestep` hook); not the `ShiftApplication` mode (Part 18 has
`positionShift` as the least dissipative of the three); and not divergence or
instability (nothing blows up, the density band is good throughout).

The related observation, and the lead: **the free-surface density deficit
disappears.** A particle at a flat free surface reads about `0.5 rho0` under a
summation density because half its kernel support is empty, and it does — the
minimum fluid density is 0.518 at t=0.02. By t=0.2 it is ~0.98 and it stays
there, while the surface keeps only three quarters of the bulk's neighbour
count, i.e. while it is still geometrically a surface. Its neighbours must
therefore be packed closer than the bulk's. That is the constant-density solve
compacting the surface layer to reach a setpoint the geometry forbids — §1.1's
unreachable setpoint again, now at a free surface instead of in a disordered
bulk, and the same thing §4 records it doing at the rotating patch's corners.

Whether that compaction is what costs the energy is **now settled, and the
answer is no**: Part 21 forced the compaction's clamp off and the run NaNs in
4 steps (there is no dissipation left to measure), and Part 22's energy budget
measures the wall-side shifts (no-penetration) at negligible — the channel is
the incompressibility cycle itself (the DF projection plus the Eq. 17
position-shift resample), with the Monaghan viscosity secondary. The compaction
is real and the clamp that produces it is load-bearing; it is not the sink.

See §4, Part 19, Part 21, Part 22, and `probe_dambreakEnergyBudget.py`.

### 1.11 Method note: measure, do not derive from pseudocode

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
| **Adopting the published stopping criterion** | Bit-identical on the constant-density solve (its test never fires under any of the three statistics) and 1.53x *worse* on the divergence-free one, where it collapses the solve to 3.0 iterations by cancellation. It buys [BK]'s iteration count and pays for it in density error (§1.7, Part 15). |
| **Re-tuning the iteration budget** | The shipped (64, 32) is on the accuracy/cost frontier. 128 PS buys 1.26x for 1.6x the wall time; 16 DF loses 8%; the reallocation the "one converges, one does not" picture suggests, (96, 16), is 6% better for 16% more time. No free win (§1.7). |
| **`forceShiftPressureGauge` at a free surface** (`dambreak`) | NaN in 4 steps. Bypassing the clamp fallback does not reduce the over-dissipation Part 19 measures — the run does not survive long enough to reach it. Rules out "relax the free-surface pressure handling" as a fix and confirms the clamp is load-bearing, not optional damping (§4, Part 21). |
| **`dambreak`'s published CFL** (`cflFactor = 0.4`, [BK]'s constant, safe on every other incompressible case here) | NaN by step 30; even `0.3` NaNs by step 76. Unlike every wall-bounded case measured so far, this one needs `cflFactor = 0.2` (§4, Part 20) — the falling column's impact is sharper than `randomFlowIncompressible`'s bounded shear and the CFL's lagged `vMax` does not see it coming. |
| `convergenceCriterion` (per solver) | `flooredOneSided` (PS) / `meanAbsolute` (DF) | Each solver's historical statistic, now one setting instead of two inline tests. `oneSided` is the published form. On the constant-density solve the swap is **bit-identical** (its criterion never fires); on the divergence-free solve it collapses the solve to 3.0 iterations for 1.53x the density error (§1.7). | 3% better for 23% more time on the bounded case, against 115% better on the periodic one. At a wall the error is set by the boundary treatment, not by how well the PPE is solved. |

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
11. **"The win belongs to `solveIncompressible`"** (Part 9) — **retracted.**
   Under the clamp, `staticBoundary` scoped to the shifting solve gave 2.88e-2
   against 3.00e-2 for both, which read as "the divergence-free half
   contributes nothing". Under `minShift` the same split is 6.49e-3 against
   4.48e-3, i.e. the divergence-free half contributes 1.45x, and dropping it
   gives the *worst* divergence-free residual of any gauged row. The operator
   wants to be the same on both sides (Part 14).
12. **"The harm is a property of running the two solves on inconsistent
   operators"** (Part 9) — **narrowed to the clamp.** The divergence-free
   half-state NaN'd at t=1.65 under the clamp gauge; under `minShift` the same
   configuration runs all 901 steps. Whether the halved per-sweep contraction
   behind it is also gone is unmeasured (§4 item 3).
13. **"The stopping criterion is broken, and the floor is the one-line fix"**
   (§1.7, Parts 4-13) — **retracted twice over.** The floored one-sided
   average, the published unfloored one and the mean absolute all read within
   6% of each other on the constant-density solve and all sit a factor of two
   above the tolerance, so the floor is not what keeps it from terminating.
   And the premise was already half false: at the shipped defaults the
   *periodic* cases terminate both solvers in 3 iterations. What is actually
   happening is not a criterion defect at all (new §1.7).
14. **"Both velocity modes damp `tgv`'s kinetic energy at ~3.3x the analytic
   rate"** (Part 5, §1.2, §6) — **withdrawn.** It does not reproduce at any
   configuration tried, and the statistic is not stable: at `tgv`'s own default
   nx=256 the three modes read 0.615 / 0.693 / 0.674, and the velocity modes'
   ratio moves 2x with resolution and 12% with duration. The qualitative
   observation it accompanied — that their decay is non-monotone — reproduces
   everywhere and is the part worth keeping. The bounded half of the same
   comparison (§6's table) is superseded for a different reason: it was taken
   at the legacy CFL. See Part 17. **The underlying claim is nevertheless
   right** — Part 18 measures the velocity modes losing 59% of an inviscid
   flow's kinetic energy against the default's 19%. It was the number and the
   mechanism that were wrong, not the conclusion.
15. **"`positionAndVelocity` has a 5x larger worst-case density excursion"**
   (Part 17) — **retracted, and it was my own protocol error.** The three modes
   were compared at the case's *adaptive* `dt`, so they ran at three different
   timesteps: the velocity modes damp the flow, the CFL condition hands a
   slower flow a larger `dt`, and the excursion followed from the timestep. At
   pinned `dt` `positionAndVelocity` has the *lowest* max `rho` of the three.
   `probe_boundedIncompressibleBlowup.py`'s docstring requires a fixed `dt` for
   exactly this comparison and the requirement was not followed. See Part 18.
16. **"The knee is at about 0.2 and the published 0.4 is not near it"**
   (Part 12) — **retracted.** True at the shipped boundary configuration, and
   an artifact of it: the 2.83x that a halved timestep bought was the near-wall
   band, and once the band is gone halving buys 1.17x. The sweep's own numbers
   reproduce; the inference does not survive the defaults it was measured
   under. Same shape as corrections 1 and 9: a conclusion drawn at one
   configuration, inverted by a change underneath it — and, as there, the
   thing underneath was the boundary treatment.

---

## 4. The combined default change — measured

**Three of the four have landed and the fourth is rejected.** `cflFactor`
went in with Part 12; the factorial (Part 13, below) measured the combination
at 40x and Part 14 landed the two changes behind it — `ShiftPressureGauge.
minShift` on wall-bounded solves and `BoundaryOperatorTerms.staticBoundary` on
both solvers. `BoundaryPressureMode.consistent` is not landing: it is inert
once the gauge is right, and its `akinciBoundaryVolume` variant diverges.
The table below is the history of how each was measured *alone*, which is
what makes Part 13's interaction legible; the shipped configuration is the
`minShift` + `staticBoundary` row of Part 13's factorial, not any row here.

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

### The per-solver split, and the landing (Part 14, `probe_perSolverBoundaryTerms.py`)

**Landed.** `ShiftPressureGauge.minShift` now applies on wall-bounded solves,
and `BoundaryOperatorTerms.staticBoundary` is the default on **both** solvers.
Running `randomFlowIncompressible --bounded` at nx=128 with no overrides at all
now reproduces Part 13's best cell exactly — `rho` in [0.99491, 1.00712], band
4.4810e-3, t=6.1498, DF residual 6.4951e-3 — so the 40x is the shipped
behaviour, not a configuration someone has to know to ask for.

Two things had to be decided first.

**Where the setting lives.** `boundaryOperatorTerms` moved onto
`RelaxedJacobiSolverConfig`, so the constant-density and divergence-free solves
can carry different operators — the split Part 9 wanted and could not express.
`IncompressibleSolverConfig.boundaryOperatorTerms` is kept as an *override*:
`None` (the new default) means "each solver's own", and any other value forces
both. Every probe in this document sets the bundle-level knob, so every A/B on
file keeps its meaning unchanged, which is what let the reproduction rows below
be checked at all.

**What the split should be.** §4 item 2 wanted `pressureSolver =
staticBoundary` with the divergence-free solver left `full`, on Part 9's
finding that the win belonged to the shifting solve. §10's closing list
objected that a split is exactly the mismatched-operator configuration behind
the unexplained
contraction collapse. Both were inferences from clamp-gauge measurements, and
Part 13's lesson is that those do not survive the gauge fix. So it was measured
— `minShift` throughout, published CFL, bounded `randomFlowIncompressible`,
nx=128, 900 steps, same band metric as the factorial:

| gauge | PS terms | DF terms | `rho` range | band, 2nd half | t_final | DF resid | PS resid | wall s |
|---|---|---|---|---|---|---|---|---|
| clamp | `full` | `full` | [0.902, 1.247] | 1.7782e-1 | 4.690 | 7.52e-2 | 2.51e-3 | 115.6 |
| `minShift` | `full` | `full` | [0.863, 1.154] | 1.4318e-1 | 6.458 | 1.19e-2 | 4.76e-4 | 61.8 |
| `minShift` | `staticBoundary` | `full` | [0.988, 1.014] | 6.4889e-3 | 6.231 | 1.71e-2 | 9.24e-4 | 123.9 |
| `minShift` | `full` | `staticBoundary` | [0.965, 1.091] | 7.1102e-2 | 6.301 | 9.20e-3 | 4.94e-4 | 74.0 |
| **`minShift`** | **`staticBoundary`** | **`staticBoundary`** | **[0.995, 1.007]** | **4.4810e-3** | 6.150 | 6.50e-3 | 9.59e-4 | 124.9 |

**Three of those five rows are configurations Part 13 published, and all three
reproduced to every recorded digit** (1.7782e-1, 1.4318e-1, 4.4810e-3) — the
fourth harness in this document validated that way, and here it also confirms
that moving the setting per-solver is bit-for-bit inert at the old values.

Three findings:

1. **The operator wants to be the same on both sides.** Splitting it costs
   1.45x with the divergence-free solve left historical (6.49e-3) and 16x the
   other way round (7.11e-2). So §4 item 2's proposal is rejected by
   measurement and §10 item 4's caution is upheld: `both` is what landed. Note
   the two halves are not two doses of one effect — `ps-only` has the *worst*
   divergence-free residual of any `minShift` row (1.71e-2, worse even than
   `full`/`full`'s 1.19e-2) while `df-only` has a good one (9.20e-3) and a bad
   density band. Each solve's operator shows up mostly in the *other* metric.
2. **Under the gauge, the divergence-free half-state does not diverge.**
   `df-only` is the configuration that turned a finite 901-step run into a NaN
   at t=1.65 (§2, §4 item 3) — under the clamp. Under `minShift` it runs the
   full 901 steps to t=6.30 with `rho` in [0.965, 1.091]. So the divergence
   that made "mismatched operators" look dangerous belongs to the clamp, and
   the risk §10 flagged for landing a per-solver default is not there.
   What is *not* settled: whether the halved per-sweep contraction that §4
   item 3 describes is also gone, or merely no longer fatal. That was measured
   under the clamp and has not been re-measured.
3. **It is free.** Two back-to-back 901-step comparisons disagree on the sign
   of an ~8% wall-time difference (124.9s vs 115.6s in the first, 123.7s vs
   123.9s in a later re-run), so the per-step cost is inside run-to-run
   variation on a shared GPU. Per unit of *physics* it is a clear win: the
   same 901 steps cover 31% more simulated time (t=6.150 against 4.690),
   because the adaptive `dt` is no longer being held down by a wall band that
   is 40x more compressed.

**What actually changes, case by case** (measured, not assumed —
`kinds != 0` and surface-flag counts taken from the running solves):

| case | non-fluid rows | surface-flagged | effect of the landing |
|---|---|---|---|
| `randomFlowIncompressible --bounded` | 2760 / 6856 | 0 | the whole change: 1.78e-1 → 4.48e-3 |
| `tgv`, `kolmogorovIncompressible`, `randomFlowIncompressible` (periodic) | 0 | 0 | strict no-op — `staticBoundary` has nothing to drop and `minShift` was already the default |
| `rotatingSquarePatch --scheme divergenceFree` | 0 | 96 / 100 | strict no-op — the guard's surviving free-surface half keeps the clamp, and there are no boundary rows to change the operator |

So the guard did not go away, it got scoped to the case it was right about.
`solveIncompressible` still downgrades `minShift` to `nonNegativeClamp` when
free-surface particles are present, where support genuinely is truncated and a
constant pressure genuinely is not forceless (§1.5); it no longer downgrades
merely because pressure rows are pinned. `forceShiftPressureGauge` survives as
the bypass for that remaining half.

**Verified:** full suite 241 passed / 1 skipped (the two known flakes did not
fire), `gradcheck_incompressible.py` passes both `includeBoundaryReaction`
branches, `run_sweep.py` 30/30, and the config round-trips with the new
`Optional` override and the two new per-solver fields. The two end rows were
then re-run *after* the defaults were flipped and reproduced again, which is
the check that the old configuration is still reachable and still identical —
the new defaults changed what you get by saying nothing, not what any explicit
setting means.

### The CFL question, re-run at the new defaults (Part 14)

Part 12's sweep found a knee at `cflFactor = 0.2` and made the case that the
published 0.4 was permissive on a bounded case. §10's list held that back on
the grounds that it had been measured where the near-wall band dominated the
error, and that the landing would remove 40x of that band. It does, and the
knee goes with it. Same probe, same case, same protocol — every run to t=3.0,
`randomFlowIncompressible --bounded`, nx=128 — before and after:

| `cflFactor` | steps | mean`\|rho-1\|` **old defaults** | mean`\|rho-1\|` **Part 14 defaults** | gain |
|---|---|---|---|---|
| **0.4** (published) | 457 / 472 | 5.17e-3 | **9.91e-4** | **5.2x** |
| 0.2 | 903 / 956 | 1.83e-3 | 8.50e-4 | 2.2x |
| 0.1 | 1830 / 1935 | 1.37e-3 | 7.32e-4 | 1.9x |

Read down the new column instead of across it: **halving the published
timestep now buys 1.17x, and halving it again 1.16x** — against 2.83x and
1.33x before. The knee at 0.2 was the boundary treatment, not the timestep.
And the new configuration at the published 0.4 is better than the old one at
0.1 (9.91e-4 against 1.37e-3) for a quarter of the steps.

So **item 2 closes: keep `cflFactor = 0.4`.** The case for departing from a
cited constant rested on a 2.8x that no longer exists. The tail agrees — the
sub-rest-density fraction at 0.4 falls 12.2% → 5.8%, and `rho_max` over the
whole run 1.202 → 1.006, so the worst particles are no longer strongly
timestep-limited either.

Two notes on reading the table. The step counts barely move (472 against 457
at 0.4), so the accuracy is not being bought by the adaptive `dt` quietly
shrinking — it is the same timestep on a better-conditioned near-wall state.
And the wall-clock columns are **not** comparable across the two sweeps: they
were run in different sessions with different GPU contention, which is enough
to swamp the per-step difference entirely (see the previous section's point 3).

### The stopping criterion — measured, and it was the wrong suspect (Part 15, `probe_stoppingCriterion.py`)

**No default changed; the finding did.** The full analysis is in the rewritten
§1.7. What landed in code is the machinery that made it measurable:

- **One configurable criterion across all three relaxed-Jacobi loops.** They
  used to spell two different tests inline and neither was reachable from the
  config, so "the criterion is broken" could be argued but not tested.
  `JacobiConvergenceCriterion` names the three forms — `flooredOneSided`
  (`solveIncompressible`'s historical test), `oneSided` ([BK] Alg. 3 / [I]
  §5.1) and `meanAbsolute` (both divergence-free loops' historical test) — and
  `modules/incompressible/convergence.py` computes them. The defaults are each
  solver's own historical statistic, so this is inert: the two end rows of
  Part 14's probe reproduce bit for bit.
- **`rtol` wired into the relaxed-Jacobi path as a disjunct** (§4 item 4's
  ask), with the same `mean|r| <= atol + rtol*mean|b|` contract the Krylov path
  states. It is inert at the shipped `rtol = 1e-5` and provably so: on the
  bounded case `mean|r| / mean|b|` is about 0.97 *at the last iteration*, which
  is the finding below in one number.

The measurement is `--mode trace`, which disables early exit
(`minIterations = maxIterations`, `rtol = 0`) so that every criterion is
evaluated **along the same iterate path** — same states, same iterates, three
readings, directly comparable rather than three different simulations.

Four results:

1. **The premise was half wrong.** `kolmogorovIncompressible` at nx=128
   terminates both solvers in 3 iterations on every step, statistic 2.26e-5
   against a 5e-4 tolerance. Non-termination is a bounded-case phenomenon and
   has been since `minShift` became the default in Part 4; nobody re-checked.
2. **The floor is not the defect.** On the constant-density solve the three
   statistics read 1.033e-3 / 1.020e-3 / 1.086e-3 — within 6%, all a factor of
   two above the tolerance. §1.7's "it is one line" is retracted.
3. **The constant-density solve does not converge in any norm.** 64 sweeps
   remove 2.7% of the residual and grow the pressure range 61x, linearly in the
   iteration count. It is an integrator; `maxIterations` is a gain. That is
   §1.1's unreachable setpoint observed *inside* one solve rather than across
   steps, and it explains why every criterion this document has tried failed
   in the same way.
4. **The published criterion's shape flatters published iteration counts, and
   the price is measured.** On the divergence-free solve the one-sided average
   is 8x smaller than the mean absolute (1.96e-3 vs 1.57e-2) purely by
   cancellation, and it is already under tolerance at the first iteration.
   Adopting it end to end takes that solve from 31.9 iterations to **3.0** —
   [BK]'s reported 4.5, near enough — **at 1.53x the density error** (6.88e-3
   against 4.48e-3). The same swap on the constant-density solve is
   **bit-identical**, which is the cleanest confirmation that its criterion
   never fires at all. Table in §1.7.

And one negative result worth its own line: **the shipped iteration budget is
on the accuracy/cost frontier** and should not change. Doubling the
constant-density budget buys 1.26x for 1.6x the wall time; halving the
divergence-free one loses 8%; the reallocation that the "one converges, one
does not" picture suggests — buy PS iterations with DF ones — measures worse
than the shipped pair at equal cost. Table in §1.7.

**What this changes downstream.** Item 9 (warm start) was gated on "fix the
stopping criterion first, because warm-starting a solver that is winding up
carries the wind-up across steps". That is now precise rather than
precautionary, and it splits the item: the divergence-free solve converges, so
warm-starting it is the ordinary optimisation [BK] describes; the
constant-density solve is an integrator, so warm-starting it would carry a
*linear ramp* across steps and is contraindicated outright. Item 11 (the real
`dfsph` scheme) inherits a sharper warning too: DFSPH proper puts the
constant-density solve into the *momentum* equation, where an amplitude set by
an iteration count becomes a force set by an iteration count.

### The shear-wave case — ported, and what it says (Part 16, `cases/shearWave.py`)

**Ported.** `shearWave` is registered, in the sweep, and carries four
assertions in `tests/test_physics.py`. It is the first incompressible case here
that grades this codebase against something other than itself.

**Why this case and not another.** A transverse sinusoidal shear wave,
`u_x = u0 sin(k_w y)`, `u_y = 0`, on a periodic box. Both nonlinear terms
vanish identically — `(u . grad) u = u_x d_x u_x e_x = 0` because `u_x` depends
only on `y`, and `div u = d_x u_x = 0` for the same reason — so

    u_x(y, t) = u0 sin(k_w y) exp(-nu k_w^2 t),   p = const

is exact for all time at any amplitude, **with zero pressure gradient**. `tgv`
is also an exact solution, but one in which a real pressure field balances a
real advection term, so a pressure error and a dissipation error are measured
together there. Here the exact pressure is constant, so every pressure the
solver produces is an artifact and every departure of the amplitude is
dissipation. At `nu = 0` the exact answer is that nothing happens.

**Four results.**

**1. The scheme is not very dissipative on this flow, and it converges to a
floor.** At `nu = 0`, `t = 1.0`, shipped defaults:

| nx | amplitude | deficit | max `rho` | disorder | max `\|v_y\|` |
|---|---|---|---|---|---|
| 32 | 0.992480 | 7.52e-3 | 1.00359 | 4.54e-2 | 1.33e-1 |
| 64 | 0.997221 | 2.78e-3 | 1.00324 | 2.45e-2 | 1.13e-1 |
| 128 | 0.998602 | 1.40e-3 | 1.00375 | 1.48e-2 | 7.23e-2 |
| 256 | 0.998942 | 1.06e-3 | 1.00361 | 9.64e-3 | 4.35e-2 |

The amplitude deficit falls 2.7x, 2.0x, then only 1.32x — converging onto a
floor near 1e-3 rather than to zero.

**2. The volume error does not converge at all.** `max rho` is 1.0036 ± 0.0003
across an **8x resolution range** — flat, while everything else improves.
That is §1.1 reproduced from a completely different direction: the summation
density's excess over `rho0` is set by how disordered the sampling is, not by
how fine it is, so refinement cannot remove it. §1.1 argued this spectrally and
demonstrated it with no dynamics at all (`probe_densityBiasVsDisorder.py`);
this is the same statement measured in a running simulation with an exactly
known answer.

**3. The physical viscosity is applied at about half its prescribed value —
independently confirming `tgv`'s 0.55x.** Graded against a *moving* target
(nx=128, t=2.0, amplitude reported relative to `exp(-nu k_w^2 t)`, so 1.0 is
exact at every `nu`):

| `nu` | analytic decay | measured/analytic | implied decay-rate ratio | disorder |
|---|---|---|---|---|
| 0 | 1.000000 | 0.995140 | — (pure artifact) | 2.07e-2 |
| 0.001 | 0.924012 | 1.040851 | **0.493** | 1.19e-3 |
| 0.01 | 0.453600 | 1.501154 | **0.486** | 7.79e-3 |

`tests/test_physics.py` asserts `tgv`'s kinetic energy decays at ~0.55x the
analytic rate and explains it as the Monaghan viscosity switch — viscosity is
deactivated for separating pairs, so roughly half the pairs dissipate at any
instant. **That explanation now has a second, independent measurement behind
it**: 0.49 on a flow whose exact pressure is constant, where it *cannot* be
pressure error, at two viscosities an order of magnitude apart. It is the first
number in this document that two unrelated cases agree on for a stated reason.

**4. The `ShiftApplication` question — the reason the case was ported — comes
back the other way.** §1.2 predicts the two velocity modes should dissipate
more, since they feed a permanent residual into momentum while the position
shift is momentum-neutral; on `tgv` that shows as 3.3x the analytic decay rate
against the shift's 0.55x. On this flow it does not happen. nx=128, t=2.0:

| mode | `nu = 0` amplitude | `nu = 0` max `rho` | `nu = 0.01` amplitude | `nu = 0.01` max `rho` | wall s (`nu = 0`) |
|---|---|---|---|---|---|
| `positionShift` *(default)* | 0.995100 | **1.00365** | 1.501436 | **1.00321** | 36.7 |
| `positionAndVelocity` | **0.998732** | 1.00268 | 1.499894 | 1.00177 | 15.4 |
| `inStepVelocity` | 0.997899 | 1.00312 | 1.501020 | 1.00202 | 19.2 |

**At `nu = 0.01` the three agree on dissipation to 0.1%** (1.5014 / 1.4999 /
1.5010 — all the same 0.49x half-viscosity), and at `nu = 0` the *default* is
the most dissipative of the three, by 4x in deficit. So the dissipation penalty
§1.2 attributes to the velocity modes is **not intrinsic to them**: on a flow
with no pressure gradient and no advection it disappears entirely. Whatever
produces `tgv`'s 3.3x needs one of those two things, and this case does not say
which.

What *does* separate the modes here, consistently at both viscosities, is the
other axis: **the position shift carries the largest volume error** (1.8x the
density excess of `positionAndVelocity` at both `nu`) and costs the most wall
time (2.4x at `nu = 0`). That is the opposite ranking to the one §1.2's
argument implies, on the axis this case can actually see.

**This is one case and it does not settle the default.** The velocity modes are
still the ones with a documented `tgv` dissipation problem and a documented
wall advantage (§6), and this flow has no wall and no pressure. What it does is
remove the *general* argument — "applied to velocity the residual is a
permanent unphysical forcing, applied to position it is momentum-neutral" —
from the list of things that can be asserted without qualification. §1.2 has
been narrowed accordingly.

**Still open: the comparison against [C]'s own curves.** The paper's Fig. 3 and
Fig. 4 are not in this repository, and neither the case nor this section
hard-codes numbers read off them. Everything above is this codebase measured
against an analytic solution, which is a real reference but not the published
one. Grading against [C] needs the paper in hand.

### `ShiftApplication`, re-measured at the current defaults (Part 17, `probe_shiftApplication.py`)

**The quantitative case for the shipped default does not survive re-measurement,
and the mechanism §1.2 attributes it to is the wrong one. The default has not
been changed, because what replaces that case is a judgement call rather than a
number.**

The evidence this question rested on had rotted in two different ways. The
bounded-case table in §6 was taken at the **legacy CFL** — it says so, "so
`positionShift` is at its death state" — which Part 13 later showed has no
viable configuration at all, and it predates both Part 14 defaults. And the
`tgv` decay ratios turn out not to be a stable statistic.

**`tgv`, artificial viscosity.** `decay/analytic`, fitted log-linearly over the
run, at three configurations:

| configuration | `positionShift` | `positionAndVelocity` | `inStepVelocity` |
|---|---|---|---|
| nx=128, 200 steps | 0.580 | 1.266 | 1.219 |
| nx=128, 500 steps | 0.585 | 1.423 | 1.306 |
| **nx=256** *(tgv's own default)*, 200 steps | 0.615 | **0.693** | **0.674** |

**At `tgv`'s own default resolution the three modes agree to within 12%.** The
recorded 3.2x / 3.4x does not reproduce anywhere, and the ratio for the two
velocity modes moves by 2x with resolution and 12% with duration, while
`positionShift` sits at 0.58-0.62 in every configuration — stable under
refinement, exactly as `tests/test_physics.py` claims for it.

That resolution dependence is the substantive point, not the disagreement.
**§1.2 attributes the velocity modes' dissipation to the permanent residual of
an unreachable setpoint** — and §1.1 and Part 16 both establish that that
residual is *resolution-independent* (Part 16 measured the volume error flat to
0.0003 across an 8x range). A penalty that halves from nx=128 to nx=256 cannot
be that residual. It is discretisation error. Combined with Part 16 — where the
three modes dissipate identically on a flow with no pressure gradient — the
mechanism §1.2 states is not what produces the effect §6 rejects the modes for.

**What does survive on `tgv`: the velocity modes are non-monotone at every
resolution**, including nx=256 where their net rate matches. A decaying viscous
flow with no forcing whose kinetic energy rises on some steps is having energy
injected, and that is a real defect regardless of what a fitted slope says
about the average. `positionShift` is monotone everywhere.

**The bounded case, at the published CFL and the Part 14 defaults.** nx=128,
900 steps — the run §6's table should have been:

| mode | band, 2nd half | sustained max `rho` | worst-ever `rho` | t_final | wall s |
|---|---|---|---|---|---|
| `positionShift` *(default)* | 4.4810e-3 | 1.0044 | [0.9949, 1.0071] | 6.150 | 124.7 |
| `positionAndVelocity` | **3.0074e-3** | **1.0033** | [0.9865, 1.0367] | **7.566** | 124.6 |
| `inStepVelocity` | 1.2753e-2 | 1.012 | [0.9866, 1.0202] | 7.449 | 202.8 |

> **Superseded in part by Part 18.** Every row here is at the case's *adaptive*
> `dt`, so the three modes ran at three different timesteps — the velocity modes
> damp the flow, and the CFL condition then hands them a larger `dt`. The
> excursion column below is a consequence of that, not of the modes. Part 18
> redoes this at a pinned `dt`.

"Sustained" is `max rho` sampled at deciles of the run; it is quoted because the
two summaries disagree for `positionAndVelocity`, and the disagreement is the
information. Its worst-ever `rho` is 1.0367 against the default's 1.0071 — 5x
the excursion — but at every decile it sits at 1.0026-1.0038, *below* the
default's steady 1.004. So its excursions are transient spikes on an otherwise
better-behaved run, where the default is uniformly mediocre.

**Read together, at the current defaults:**

| | `positionShift` | `positionAndVelocity` | `inStepVelocity` |
|---|---|---|---|
| `tgv` rate (nx=256) | 0.615 | 0.693 | 0.674 |
| `tgv` monotone | **yes** | no | no |
| `shearWave` amplitude | 0.9952 | **0.9987** | 0.9979 |
| `shearWave` max `rho` | 1.00335 | **1.00291** | 1.00315 |
| bounded band | 4.48e-3 | **3.01e-3** | 1.28e-2 |
| bounded excursion | **1.0071** | 1.0367 | 1.0202 |
| bounded wall s | 124.7 | **124.6** | 202.8 |

`inStepVelocity` is out: 2.8x the bounded band and 63% more wall time, with
nothing it wins. Between the other two, **`positionAndVelocity` is better on
every sustained metric across all three cases, at identical cost**, and worse
on exactly two things: energy monotonicity on `tgv`, and transient density
excursions on the bounded case.

**So the default stays, for now, and the reason is stated rather than
measured.** Both of the things `positionAndVelocity` loses on are *bounded-worst-
case* properties and both of the things it wins on are *averages*, and this
document has been wrong before by ranking on an average (§4's warning about the
legacy-CFL half of Part 13's factorial). Flipping a default on a 1.5x mean
improvement that comes with a 5x worse tail is not the trade this project has
been making. What would settle it is a metric nobody here has: whether those
transient excursions are the beginning of the wall accumulation that killed the
legacy-CFL runs, or noise on a run that is otherwise fine. `probe_boundedIncompressibleBlowup.py`
measures exactly that (penetration count and worst depth per step) and has not
been pointed at these three modes at the current defaults.

**What has changed is the argument, not the setting.** §1.2's mechanism claim
and §6's 3.2x are both withdrawn; the case for `positionShift` now rests on
energy monotonicity and tail behaviour, which is narrower and honest.

### The tail, measured — and `ShiftApplication` settles (Part 18, `probe_boundedIncompressibleBlowup.py`)

**The default is right, and for the first time the reason survives scrutiny.**
Part 17 left it resting on two worst-case properties, one of which turns out to
be an artifact of Part 17's own protocol. Both are now measured, at a **pinned
`dt`** — which the probe's own docstring has always required for an A/B whose
variants change the velocity field, and which Part 17's bounded table did not
use.

nx=128, `dt = 5e-3` fixed, to t=6.0 (1201 steps), published CFL configuration,
Part 14 defaults. `randomFlowIncompressible` has `nu = 0` and this scheme has no
artificial viscosity term, so **every joule lost below is numerical**.

| mode | `KE(6)/KE(0)` | mean `\|rho-1\|` near wall | in bulk | max `rho` | **particles past the wall** |
|---|---|---|---|---|---|
| `positionShift` *(default)* | **0.807** | 1.186e-3 | 8.95e-4 | 1.004 | **0** |
| `positionAndVelocity` | 0.409 | **5.89e-4** | **5.33e-4** | **1.003** | **0** |
| `inStepVelocity` | 0.412 | 7.17e-4 | 6.02e-4 | 1.009 | **0** |

**1. Nobody penetrates.** `nOutside` is 0 at every one of 1201 steps for all
three modes. §6's legacy table — 4506 / 239 / 63 particles inside the wall — was
the whole case for the velocity modes, and it was taken at the legacy CFL where
`positionShift` was in the act of dying. At the published CFL with the Part 14
defaults **there is no penetration for any mode to prevent**, so the argument
that motivated them is not weakened but void.

**2. The velocity modes cost half the flow's energy.** They retain 41% of the
initial kinetic energy at t=6 against the default's 81% — **2.1x the loss**, on
an inviscid case where the exact answer is that energy is conserved. This is
§1.2's claim, confirmed on the case that can show it, and it is large: not a
3.3x factor on a fitted decay rate but 59% of the flow simply gone.

**3. What they buy is 2x lower density error** — 5.9e-4 against 1.2e-3 near the
wall, 5.3e-4 against 9.0e-4 in the bulk. Real, and small in absolute terms:
both are a tenth of a percent.

**So the trade is stated exactly: half the flow's kinetic energy for half the
density error, with the wall behaviour identical.** That is not a trade worth
taking. A scheme that dissipates 59% of an inviscid flow in six time units is
not better than one that dissipates 19% and carries 0.1% density error instead
of 0.06%. `positionShift` stays, and now on a measurement rather than on §6's
withdrawn 3.2x.

**Correction to Part 17.** Its bounded table reported `positionAndVelocity`
reaching `rho = 1.0367` against the default's 1.0071 — a "5x larger worst-ever
excursion", which was half the stated reason for keeping the default. **That was
the adaptive timestep, not the mode.** At pinned `dt` its max `rho` over the
whole run is 1.003, the *lowest* of the three. The mechanism is visible in the
`vMax` column: the velocity correction damps the flow, the CFL condition hands a
slower flow a *larger* `dt`, and the excursion came from that larger timestep.
Part 17 compared three modes at three different timesteps and read the
difference as a property of the modes. The probe's docstring warned about
exactly this; the warning was not heeded.

### A dam break — the incompressible scheme's first working free surface (Part 19, `probe_dambreakIncompressible.py`)

Every incompressible case in this document is periodic or wall-bounded except
`rotatingSquarePatch`, which is broken in a way [BK] §5 documents as a method
limitation and which is a hard free-surface test (four convex corners, and the
arms it grows are surface-tension-sensitive). A dam break is the easier one:
gravity-driven, one mostly-flat free surface. `dambreak` is a weakly-compressible
case and takes `--scheme divergenceFree` with no wiring, so this costs nothing
to ask.

**It works.** nx=64, 3000 steps to t=1.5, no divergence, fluid density in
[0.907, 1.004] for the whole run, and a recognisable dam break: the column
falls, spreads, and runs out along the floor. That is the first free surface
this scheme has done without breaking.

**But it is a different dam break from the validated one.** `deltaSPH` is the
control — same geometry, same resolution, same `dt`:

| t | DF KE | DF front | deltaSPH KE | deltaSPH front |
|---|---|---|---|---|
| 0.3 | 4.15 | -0.767 | 2.12 | -0.586 |
| 0.5 | **7.42** | -0.270 | 3.61 | 0.353 |
| 0.7 | 3.16 | 0.137 | 4.23 | **1.462** |
| 0.8 | 1.66 | 0.344 | 4.35 | 1.984 *(far wall)* |
| 1.0 | 1.11 | 0.809 | 3.69 | — |
| 1.5 | 0.84 | 1.676 | 1.32 | — |

Two things, and they are the same thing seen twice:

1. **The run-out is roughly half speed.** From its start at x=-1.359 the front
   has travelled 1.50 by t=0.7 against `deltaSPH`'s 2.82. `deltaSPH` reaches
   the far wall at t≈0.8; the incompressible run has not by t=1.5.
2. **The kinetic energy peaks 2x higher and then collapses.** 7.42 at t=0.5
   against 3.61, then **88% of it is gone by t=0.8** while `deltaSPH` is still
   gaining. The peak is the column *falling*; what does not happen is the
   turn — the vertical momentum that should become horizontal run-out is
   dissipated instead.

**It is not a timestep artifact.** `dambreak` has no incompressible `timestep`
hook, so it runs at the weakly-compressible `dt = 5e-4`. With `dx = 0.03125`
and `|v|` peaking near 5, [BK]'s condition would permit `2.5e-3` — the run is
**5x finer in time than the CFL requires**, not coarser. (It is also why it
costs 294s against `deltaSPH`'s 61s for the same 3000 steps; an incompressible
`timestep` hook would recover most of that, and is worth adding.)

**The free-surface density deficit disappears, and that is worth explaining.**
Under a summation density a particle at a flat free surface reads about
`0.5 rho0`, because half its kernel support is empty. It does, at first: the
minimum fluid density is **0.518 at t=0.02**. By t=0.2 it is 0.979, and it
stays near `rho0` for the rest of the run — while the fluid still has a free
surface. Measuring the geometry directly
(`probe_dambreakIncompressible.py --mode surface`: neighbour counts within the
support for the topmost particle in each `dx`-wide column, a definition that
uses no density at all and so cannot be circular):

| | surface nbrs | bulk nbrs | ratio | surface `rho` | min surface `rho` |
|---|---|---|---|---|---|
| `divergenceFree`, t=0.02 | 28.6 | 42.3 | 0.677 | 0.766 | **0.518** |
| `divergenceFree`, t=0.5 | 33.2 | 44.3 | 0.749 | **1.0009** | **0.9968** |
| `deltaSPH`, t=0.02 | 33.9 | 42.6 | 0.797 | 1.0001 | 1.0000 |
| `deltaSPH`, t=0.5 | 29.4 | 43.4 | 0.677 | 1.0005 | 0.9994 |

**The solid statement is the within-scheme one.** Over that interval
`divergenceFree`'s surface density goes from 0.766 mean / 0.518 worst to
1.0009 mean / 0.9968 worst — a full recovery to `rho0` — while its surface
neighbour count stays at 0.68-0.75 of the bulk's, i.e. while the surface is
still geometrically a surface. The neighbours it has must therefore be closer
together than the bulk's. Surface compaction by the constant-density solve is
the natural reading of that, and it is what §4 records the same solve doing at
the rotating patch's corners — here succeeding rather than breaking.

**The cross-scheme comparison does not support it, and should not be quoted as
if it did.** The `deltaSPH` ratio is 0.797 at t=0.02 and 0.677 at t=0.5 — it
*crosses* `divergenceFree`'s 0.677 and 0.749, so the ordering reverses between
the two times. The ratio is evidently dominated by how thin and spread the
sheet is at that instant, not by the scheme, which is unsurprising given the
two runs are at visibly different flow states by any fixed time (fronts 0.6
apart at t=0.5). It discriminates nothing here. (`deltaSPH`'s density column
says nothing either way — it integrates rather than sums.)

So the compaction reading rests entirely on the within-`divergenceFree`
recovery, which is real but is one observation. The clean test is DF against
itself with the shift disabled at surface particles: `divergenceFree.py` has
`pressureB[surfaceIndicators == 1] = 0.0` commented out for exactly this, and
§4's "Known-open" entry explains why it is untestable as written.

**Where this leaves the scheme.** It does free surfaces, on the easy case,
without breaking — but it delivers half the run-out and dissipates most of the
flow's energy at the moment the dam break is supposed to convert it. That is
the same over-dissipation Part 18 measured on the bounded case (19% of KE in 6
time units there, 88% in 0.3 here), in the regime that exposes it.

### Open items, ranked

1. ~~**Run the 2x2x2x2.**~~ ~~**Land `minShift`-on-bounded + `staticBoundary`
   as the defaults.**~~ **Both done — Part 13 and Part 14 above.** The
   defaults shipped are `minShift` wherever there is no free surface and
   `staticBoundary` on *both* solvers. `consistent` and `akinciBoundaryVolume`
   did not land, as planned: the first is inert against `static` and the
   second diverges.
2. ~~**Move `BoundaryOperatorTerms` to `RelaxedJacobiSolverConfig`**~~ —
   **done, and the split it proposed is rejected.** The setting is per-solver
   now, but `pressureSolver = staticBoundary` with `divergenceFreeSolver =
   full` measures 1.45x *worse* than both (6.49e-3 vs 4.48e-3), so both is the
   default. The reasoning that endorsed the split ("the win belongs to
   `solveIncompressible`") was a clamp-gauge result, like every other boundary
   ranking this document has had to invert.
3. **The divergence-free half-state's contraction collapse is unexplained —
   and, under the gauge fix, no longer fatal.** Under `staticBoundary` applied
   to the divergence-free solve alone *with the clamp*, each solve removes
   ~20% of its incoming residual against ~50% under `full`; the incoming
   residual creeps 2.4e-2 → 3.8e-2 over 250 steps and then detonates (max
   `|a_p|` 19.8 → 1.04e4 at step 276, NaN at 282). Each solve still converges
   internally. Three mechanisms tested and eliminated (§2). **Part 14 removed
   the observable**: the same half-state under `minShift` runs all 901 steps to
   t=6.30. What is unmeasured is whether the per-sweep contraction is still
   halved there — if it is, the mechanism is still live and merely survivable,
   and it is worth understanding before any third solve is added; if it is not,
   this item closes. `probe_boundaryOperatorTerms.py --mode diag` measures it.
4. ~~**Wire `rtol` into the relaxed-Jacobi path**~~ ~~**Then the
   one-sided-average vs floored-average criterion behind a flag.**~~ **Both
   done — Part 15.** Both are in and both are inert, and the measurement they
   enabled says neither was the defect: the constant-density solve does not
   converge in any norm, so no residual criterion can end it (new §1.7). What
   is left of this item is *documentation*, not code — `maxIterations` on
   `pressureSolver` should be named and described as the shift gain it is.
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
8. ~~**Port [C]'s shear-wave decay case**~~ **Done — Part 16.** `shearWave` is
   registered, swept and tested. It confirmed the half-viscosity explanation
   behind `tgv`'s 0.55x independently (0.49x, at two viscosities, on a flow
   with no pressure error to confuse it), reproduced §1.1's structural density
   bias as a **resolution-independent** volume error, and narrowed §1.2 — the
   three `ShiftApplication` modes dissipate identically here. What is left of
   this item is the comparison against [C]'s Fig. 3/Fig. 4 themselves, which
   needs the paper: nothing in this repository has its curves, and nothing here
   hard-codes numbers read off them.
9. **Warm start — on the divergence-free solve only.** [BK] does a full one
   (worth ~3x in iteration count), [I] and [C] do `0.5 p(t-dt)`, [B] does none;
   this codebase does none (`incompressible.py:167`). Part 15 splits the item:
   the divergence-free solve genuinely converges, so warm-starting it is the
   ordinary optimisation and is now unblocked. The constant-density solve is an
   integrator whose pressure grows linearly in the iteration count, so
   warm-starting *it* would carry that ramp across steps — the cold start is
   load-bearing there, and this is no longer a "wait and see" but a "do not".
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
  The commented-out `pressureB[surfaceIndicators == 1] = 0.0` in
  `divergenceFree.py` is untestable as-is on this case (`detectFreeSurface`
  flags 96/100 particles at nx=32 on this thin patch) and, per Part 21, is not
  actually [BK]'s remedy anyway — it sits in the divergence-free solve, not the
  constant-density one, and it zeros the pressure outright rather than only
  its negative part. **Part 21 measured the real remedy's boundary on
  `dambreak` instead**, via the already-wired `forceShiftPressureGauge`: taking
  the clamp *away* (the opposite direction from strengthening it) NaNs the run
  in 4 steps, so the clamp is necessary, not merely damping. What is left of
  [BK]'s own remedy — Schechter & Bridson's ghost particles, now in
  `literature/` (`schechter2012`) — is the one structural option neither case
  has tried.
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

**Nothing is unavailable any more.** As of 08-29 all five above, plus the four
previously listed here as unavailable — Adami et al. 2012 (wall BC), Akinci et
al. 2012 (rigid-fluid coupling), Ihmsen et al. 2010 (adaptive timestep), Adami
et al. 2013 (transport velocity, the background-pressure question) — are in
`literature/`, along with 27 others. `literature/MANIFEST.md` maps the
shorthands above to bib keys and filenames and says what each newly-present
paper unblocks; `literature/ABSTRACTS.md` is the searchable index. Being
*present* is not being *read*: the claims below still carry whatever provenance
they carried before, and re-checking them against the documents is now possible
rather than done.

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
then removed 40x of that band, and **the re-run at the landed defaults says the
knee was the boundary treatment, not the timestep** (§4, Part 14): halving 0.4
now buys 1.17x instead of 2.83x. The published constant is no longer
permissive on this case, because the error it was permitting is gone.

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
| `boundaryPressureMode` | `mdbcDensity` | `plain` / `mdbcDensity` / `mdbcMlsPressure` / `consistent`. `consistent` was the best measured under the clamp gauge and is **inert** (0.007%) against `staticBoundary` once the gauge is fixed (Part 13), so it did not land; `mdbcMlsPressure` is the worst and should be deprecated. |
| `boundaryOperatorTerms` (per solver, on `RelaxedJacobiSolverConfig`) | **`staticBoundary`** on both | The published formulation, and one of the two defaults Part 14 landed: with `minShift` it is 40x the old default, and setting it on only one solver is 1.45x (PS only) or 16x (DF only) worse than on both. `diagonalOnly`/`operatorOnly` are deliberately-mismatched diagnostics — `diagonalOnly` runs the wall's Jacobi step 1.6x too large and NaNs in 47 steps. |
| `boundaryOperatorTerms` (on `IncompressibleSolverConfig`) | `None` | Bundle-level override: `None` means "each solver's own", any value forces both. Every A/B in this document sets it, so every recorded row still means what it says. |
| `akinciBoundaryVolume` | `False` | `consistent` only. Measured `m~/m_nominal` mean 1.102, max 1.456 on the five-layer band. Best row in the table *inside the operator*; fatal as actual mass (§2). |
| `shiftPressureGauge` | **`minShift`** | The Part 4 fix. `nonNegativeClamp` is the historical clamp and stays selectable. Scoped to solves with **no free surface** — Part 14 dropped the pinned-row half of that scoping, which is what makes it reach the bounded case at all. |
| `forceShiftPressureGauge` | `False` | Bypasses what is left of that scoping, i.e. the free-surface half. Its original target, the pinned-row half, is gone: half that guard's justification measured false (§1.5) and the other half's evidence was taken at 3x the CFL (§3.6). **The free-surface half is now measured (Part 21) and stays off**: forcing it on `dambreak` NaNs in 4 steps. |
| `shiftApplication` | `positionShift` | The paper-faithful default, and **settled in Part 18**: at a pinned `dt` the velocity modes buy 2x lower density error for **2.1x the kinetic-energy loss** on an inviscid case (41% retained against 81%), and the wall behaviour that used to justify them is identical — zero particles past the wall for all three modes. Its old justification (3.2x `tgv` dissipation) was withdrawn in Part 17 as unreproducible and resolution-dependent. |
| `densityEvolution` | `summation` | `continuity` (WCSPH standard) fails everywhere but `tgv`; `hybrid` matches `summation` exactly where support is complete, for ~21% less wall time on `tgv`, and dies at 286 steps at an mDBC wall (§4 item 7). |
| `mdbcPressureRelaxation` | `0.3` | Load-bearing for `mdbcMlsPressure` — at 1.0 it NaNs in 7-8 steps. Never swept; chosen to match the solver's own `relaxationFactor`. |
| `convergenceCriterion` (per solver) | `flooredOneSided` (PS) / `meanAbsolute` (DF) | Each solver's historical statistic, now one setting instead of two inline tests. `oneSided` is the published form. On the constant-density solve the swap is **bit-identical** (its criterion never fires); on the divergence-free solve it collapses the solve to 3.0 iterations for 1.53x the density error (§1.7). |
| `rtol` (relaxed-Jacobi path) | `1e-5` | Now a *disjunct* alongside the absolute test, same contract as the Krylov path. Inert at the default: `mean\|r\|/mean\|b\|` is ~0.97 at the last iteration on the bounded case. |
| `maxIterations` (`pressureSolver`) | `64` | **A gain, not a budget** (§1.7). The constant-density solve does not converge; its pressure grows linearly in the iteration count, so this sets the shift amplitude. Measured on the frontier: 128 buys 1.26x for 1.6x the time, 32 costs 1.46x. |
| `mdbcNoPenetrationShift` | `True` | Removing it is worse (§2). |
| `integrateRho` | `False` | Legacy alias; `resolveDensityEvolution` maps `True` → `continuity`. |
| `cflFactor` (incompressible cases) | **`0.4`** | Working tree only; multiplies `dx`. See §7. |

`ShiftApplication` comparison, bounded case, nx=128. **Superseded by Part 17 —
kept for the history, not for the ranking.** Every row is at the legacy CFL,
which Part 13 showed has no viable configuration at all, and predates both
Part 14 defaults. Part 17's table is the one to read:

| mode | near-wall `\|rho-1\|` | bulk | penetrating | `rho` range | outcome | `tgv` decay/analytic *(not reproducible — Part 17)* |
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
- `probe_perSolverBoundaryTerms.py` *(new, Part 14)* — the two solvers'
  `boundaryOperatorTerms` crossed under `minShift`, which is what settled the
  landed default. Three of its five rows are Part 13 cells and it prints an
  explicit reproduction check against them, so it is the cheapest regression
  test for "did anything under the incompressible path move". `--rows` subsets
  it.
- `probe_stoppingCriterion.py` *(new, Part 15)* — `--mode trace` evaluates all
  three criteria **along one fixed iterate path** (early exit disabled, so the
  runs are bit-identical and only the reading changes) and prints each
  statistic and the pressure range per iteration; `--mode budget` sweeps the
  two solvers' `maxIterations` (`--budgets 64:32 128:16`); `--mode ab` crosses the two
  solvers' criteria end to end at the shipped tolerances, which is what "adopt
  [BK] Alg. 3" actually means. Point it at
  `--case kolmogorovIncompressible --extra` for the periodic contrast, which is
  where the non-termination claim breaks.
- `probe_dambreakIncompressible.py` *(new, Part 19)* — the dam break under
  both schemes, reporting surge front, column height, kinetic energy and the
  size of the population that reads as free surface under a summation density.
  `--schemes divergenceFree` for one of them.
- `probe_dambreakEnergyBudget.py` *(new, Part 22)* — the per-step
  kinetic-energy budget of `dambreak --scheme divergenceFree`, closing `dKE`
  exactly and splitting it the two ways Part 22 reports: the work form
  (first-order works + quadratic remainder, binned in x for the *where*) and
  the sequential form (exact KE change of adding each force in step order, for
  the unambiguous *which channel*). The `chain` column is the completeness
  check (captured forces vs the integrator's real update). `--tLimit/--bins/
  --interval/--fixedDt` subset it; the dissipation window runs from the KE
  peak to t=0.8 by default.
- `probe_shiftApplication.py` *(new, Part 17)* — the three modes across `tgv`,
  the bounded case and `shearWave` in one table. `--nx/--tgvSteps` matter more
  than usual here: the `tgv` ratio is resolution-dependent for two of the three
  modes, which is itself the finding.
- `probe_shearWave.py` *(new, Part 16)* — the shear-wave case's three
  questions: `--mode shift` crosses the `ShiftApplication` modes (add `--nu` to
  do it against a real analytic decay instead of a stationary one),
  `--mode resolution` checks what converges and what does not, `--mode viscous`
  grades the applied viscosity against the prescribed one.
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
  now with kinetic energy (`ke`) alongside `vMax`, since `vMax` is one particle
  and the `ShiftApplication` question is whether the *flow* is damped.
  **`--fixedDt` is mandatory for any A/B that changes the velocity field** —
  the CFL hands a damped flow a larger `dt`, and Part 17 read that as a
  property of the modes.
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
| 19 | 08-29 | A dam break under `--scheme divergenceFree` (§4). It works — 3000 steps, no divergence, a recognisable collapse, the first free surface this scheme has not broken. But the run-out is half `deltaSPH`'s speed and 88% of the kinetic energy is dissipated at the moment the fall should become run-out. The free-surface density deficit (0.518 at t=0.02) disappears while the surface stays geometrically a surface. |
| 20 | 08-29 | `dambreak`'s incompressible `timestep` hook, landed. The published CFL (0.4) diverges here, unlike every other incompressible case measured — bisected to a safe `cflFactor = 0.2`, which buys 1.7x fewer steps (not the ~5x guessed) at the cost of a worse `rho_max` (1.105 vs 1.004). First hint that the impact itself is the sharp event Part 19's dissipation traces back to. |
| 21 | 08-29 | The free-surface clamp ruled out as the dissipation's cause. [BK]'s own text (read from the PDF, now that `literature/` exists) says clamping negative pressure at the surface *is* their published remedy, not a gap; forcing it off (`forceShiftPressureGauge`, previously untested at the free surface) NaNs `dambreak` in 4 steps rather than reducing the loss. Narrows item 1 to the impact itself. |
| 22 | 08-29 | The energy budget ran and named the channel (§4). The per-step KE budget closes exactly (chain 3.3e-6, gap 0.0); the loss is the incompressibility cycle — the DF projection (−35.8) and the Eq. 17 resample (+27.3) net to −8.5 (85% of the loss), with Monaghan viscosity secondary (−6.4, 64%) and the no-pen shift negligible. Both channels are `divergenceFree`-only, which carries the cross-scheme gap. Item 1 now reduces to the mechanism question: an `nx` convergence of the cycle's net. |
| 18 | 08-29 | The tail measured, and `ShiftApplication` settles (§4). At a pinned `dt`: **zero** wall penetration for all three modes, so §6's whole case for the velocity modes is void; and the velocity modes cost **2.1x** the kinetic energy of an inviscid flow for 2x lower density error. The default stays, on a measurement. Part 17's excursion claim was my own adaptive-`dt` artifact and is retracted. |
| 17 | 08-29 | `ShiftApplication` re-measured at the current defaults (§4). The 3.2x that justified the shipped default does not reproduce — at `tgv`'s own nx=256 the three modes agree to 12%, and the ratio moves 2x with resolution, so it cannot be the resolution-independent residual §1.2 blames. `positionAndVelocity` is better on every sustained metric at equal cost; the default stays on tail behaviour and energy monotonicity. |
| 16 | 08-29 | [C]'s shear-wave case ported (§4). An exact solution with a constant pressure, so dissipation and volume error separate. Confirms `tgv`'s half-viscosity at 0.49x independently; the volume error is resolution-independent (§1.1 from a new direction); the three `ShiftApplication` modes dissipate identically, which narrows §1.2. |
| 15 | 08-29 | The stopping criterion (§1.7). It was the wrong suspect: the periodic cases terminate in 3 iterations, the floor changes nothing, and the constant-density solve does not converge in any norm — it integrates, so `maxIterations` is a gain. The criterion is now one configurable setting across all three loops; no default changed. |
| 14 | 08-29 | The landing (§4). `boundaryOperatorTerms` moved per-solver; the two solvers crossed under `minShift` says **both**, not the PS-only split Part 9 implied. `minShift` on bounded and `staticBoundary` on both are now the defaults, and the bounded case ships at 4.48e-3. The half-state's divergence turns out to belong to the clamp. Three prior rows reproduced exactly. |

---

## 10. Overview — where things stand

### What is shipped and stable

The incompressible path is **VD+PS** (Cornelis et al.), faithfully implemented,
registered as `divergenceFree`. Three of this work's changes altered a shipped
default, all three measured before landing; every other new switch is opt-in
and default-inert, and the bug fixes are behaviour-preserving on every case
that existed before them.

- **`cflFactor = 0.4` against the particle diameter** — [BK]'s published
  condition in [BK]'s units (Part 12).
- **`ShiftPressureGauge.minShift` is the default** (Part 4) **and now reaches
  wall-bounded solves** (Part 14). It turns `kolmogorovIncompressible` at
  nx=128 from a NaN at step 574 into a stable 1000-step run with density in
  [0.980, 1.015], and it leaves `tgv`'s analytic decay rate alone to 0.4%. It
  still falls back to the historical clamp where there is a free surface.
- **`BoundaryOperatorTerms.staticBoundary` on both solvers** (Part 14), the
  formulation [BK] §3.2 states and SPlisHSPlasH implements. It is a strict
  no-op on every case with no `kind != 0` particles, which is every case here
  except the bounded one.
- Those last two are one fix at two points: together they take the bounded
  `randomFlowIncompressible` at nx=128 from a density band of 1.78e-1 to
  **4.48e-3**, 40x, and 5.4x better than they compose independently.
- **Several real bugs are fixed** (§7), the largest being the Eq. 17 resample,
  the boundary-row masking, and the `drhodt` pre-projection evaluation.
- Full suite passes (241 passed, 1 skipped), `gradcheck_incompressible.py`
  passes, and `run_sweep.py` is 30/30. Two known intermittent flakes, both
  pre-existing (§4).

**Case status at the shipped defaults:** `tgv`, `kolmogorovIncompressible` and
`shearWave` (all periodic) are healthy. `randomFlowIncompressible --bounded` is the case that
exercises everything; it is now the best-behaved it has ever been, and is still
where all the remaining error lives.
`rotatingSquarePatch --scheme divergenceFree` (free surface) is broken and is a
known method limitation, not an implementation bug — and is untouched by the
Part 14 defaults, which are both no-ops there. **`dambreak --scheme
divergenceFree` runs** (Part 19) — the scheme's only working free surface — but
at half `deltaSPH`'s run-out speed and with most of the flow's energy
dissipated on impact. It also needs its own, tighter CFL: `--cflFactor 0.2`
(Part 20), not the `0.4` every other incompressible case in this document
ships — the published constant diverges on this case.

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
3692 steps, i.e. a further 1.22x. Both runs also found the `--nSteps` defect
above independently.)*

**Superseded by Part 14, and worth reading as a warning.** Every number above
still reproduces, but every *conclusion* drawn from them was about the
boundary treatment wearing a timestep's clothes. Re-run at the landed defaults
the same sweep gives 9.91e-4 at 0.4, and halving buys 1.17x rather than 2.83x:
the knee at 0.2 was not firm, it was the near-wall band, and 0.4 turns out to
sit in the middle of a plateau after all (§4, "The CFL question, re-run at the
new defaults"). The one claim that survives unchanged is the negative one — 3x
the constant is broken in every configuration.

### Part 13, the factorial — run, and it settles the default question

**Done; Part 14 then landed what it endorsed.** Full tables and analysis
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

### Part 14, the landing — done

**The defaults are in.** `minShift` on wall-bounded solves, `staticBoundary`
on both solvers, and `randomFlowIncompressible --bounded` at nx=128 now ships
the 4.48e-3 configuration with no flags. Full tables in §4; the short version:

- **The bundle-level knob moved per-solver**, and the split it was moved for
  is rejected by measurement: `staticBoundary` on both (4.48e-3) beats the
  constant-density solve alone (6.49e-3) and the divergence-free solve alone
  (7.11e-2). §4 item 2 had endorsed the PS-only split on clamp-gauge evidence.
- **Three of the five rows are prior published configurations and reproduced
  to every digit**, which also proves the per-solver refactor inert at the old
  values.
- **The unexplained half-state divergence belongs to the clamp** — under
  `minShift` that configuration runs to completion. The mechanism behind it is
  still not explained, but it no longer gates anything.
- **The CFL question closes with the published constant intact.** Re-running
  Part 12's sweep at the new defaults, halving 0.4 buys 1.17x rather than
  2.83x: the knee was the boundary treatment. The new defaults at 0.4 beat the
  old ones at 0.1 for a quarter of the steps.
- Suite 241/1 skipped, gradcheck passes, `run_sweep.py` 30/30, config
  round-trips.

### Part 15, the stopping criterion — measured, and retired as a mystery

**No default changed.** §1.7 has been rewritten around what the measurement
actually found; the short version:

- **The periodic cases converge in 3 iterations** at the shipped defaults, both
  solvers. "Never terminates under every gauge" was a clamp-gauge observation
  that outlived the clamp. Non-termination is a near-wall phenomenon, like
  everything else here.
- **The floor was the wrong suspect.** All three criteria read within 6% of
  each other on the constant-density solve and all sit a factor of two above
  the tolerance.
- **That solve does not converge in any norm.** 64 sweeps remove 2.7% of the
  residual and grow the pressure field 61x, linearly in the iteration count.
  It is an integrator, and `maxIterations` is a shift gain wearing a
  convergence budget's name. No criterion can end it, which is why every one
  tried has failed identically.
- **The published criterion buys the published iteration count, and it costs
  accuracy**: adopting it takes the divergence-free solve from 31.9 iterations
  to 3.0 — [BK]'s 4.5, near enough — for 1.53x the density error. On the
  constant-density solve the same swap is bit-identical.
- **The shipped iteration budget is on the frontier** and should stay.
- Landed in code: one configurable `JacobiConvergenceCriterion` across all
  three relaxed-Jacobi loops (they spelled two different tests inline), and
  `rtol` as a relative disjunct. Both inert — Part 14's end rows reproduce bit
  for bit.

### Part 16, the shear-wave case — ported

**The first incompressible case here that grades this codebase against
something other than itself.** Full analysis in §4; the short version:

- **An exact solution with a constant pressure.** `u_x = u0 sin(k_w y)`,
  `u_y = 0` makes both nonlinear terms vanish identically, so
  `u_x = u0 sin(k_w y) exp(-nu k_w^2 t)` holds for all time with `p = const`.
  Every pressure the solver produces is therefore an artifact, and dissipation
  and volume error separate cleanly — which is why [C] reports two figures on
  this case and why `tgv`, whose exact solution carries a real pressure field,
  cannot make that separation.
- **`tgv`'s 0.55x now has independent corroboration.** The viscosity is applied
  at **0.49x** its prescribed value here, at two viscosities an order of
  magnitude apart, on a flow where it cannot be pressure error — which is the
  half-of-the-pairs Monaghan-switch explanation `tests/test_physics.py` states.
- **The volume error does not converge**: `max rho` is 1.0036 ± 0.0003 across
  nx = 32…256, while the amplitude error falls 7x. §1.1's structural density
  bias, measured in a running simulation with a known answer.
- **§1.2 is narrowed.** All three `ShiftApplication` modes dissipate
  identically here (0.1% apart at `nu = 0.01`); at `nu = 0` the *default* is
  the most dissipative. The velocity modes' dissipation penalty is not
  intrinsic to them — on a flow with no pressure gradient it vanishes. The
  position shift instead carries 1.8x the volume error and 2.4x the wall time.

### Part 17, `ShiftApplication` — the argument changed, the default did not

Full tables in §4. The short version:

- **§6's 3.2x is withdrawn.** At `tgv`'s own default nx=256 the three modes'
  decay ratios are 0.615 / 0.693 / 0.674 — within 12%. The velocity modes'
  ratio moves 2x between nx=128 and nx=256; `positionShift` holds 0.58-0.62
  everywhere.
- **§1.2's mechanism is withdrawn with it.** A penalty that halves under
  refinement cannot be the permanent residual of an unreachable setpoint —
  that residual is resolution-*independent*, which §1.1 argues and Part 16
  measured flat across an 8x range.
- **What survives is narrower and real**: the velocity modes are non-monotone
  at every resolution, i.e. a decaying unforced flow gains energy on some
  steps.
- **The bounded comparison, redone at the published CFL and the Part 14
  defaults**, has `positionAndVelocity` 1.5x better on the sustained band at
  identical wall time and 23% more simulated time — but with a 5x larger
  worst-ever density excursion. `inStepVelocity` is out on both cost and error.
- **The default stays**, on tail behaviour and energy monotonicity rather than
  on the number that used to justify it. Flipping it on a 1.5x mean improvement
  that carries a 5x worse tail is not the trade this project has been making.

### Part 18, the tail — `ShiftApplication` settles

Full tables in §4. At a pinned `dt` (which Part 17 failed to use, and which the
probe's docstring has always required for this comparison):

- **Zero wall penetration for all three modes**, at every one of 1201 steps.
  §6's legacy table — 4506 / 239 / 63 particles inside the wall — was the whole
  case for the velocity modes and was taken at the legacy CFL. At the published
  CFL with the Part 14 defaults there is nothing for them to prevent.
- **The velocity modes cost 2.1x the kinetic energy**: 41% retained at t=6
  against the default's 81%, on an inviscid case with no artificial viscosity,
  so all of it is numerical.
- **What they buy is 2x lower density error**, 5.9e-4 against 1.2e-3 near the
  wall — real, and a tenth of a percent either way.
- **So `positionShift` stays**, and the reason is now a measurement rather than
  §6's withdrawn 3.2x: half the flow's energy is not worth half the density
  error.
- **Part 17's "5x worse excursion" is retracted as my own protocol error** —
  three modes compared at three different timesteps, because the velocity modes
  damp the flow and the CFL then hands them a larger `dt`.

### Part 19, a dam break — the scheme's first working free surface

Full tables in §4. `dambreak --scheme divergenceFree`, nx=64, 3000 steps:

- **It works.** No divergence, fluid density in [0.907, 1.004], a recognisable
  collapse and run-out. The only other free-surface incompressible case,
  `rotatingSquarePatch`, is broken.
- **The run-out is half speed.** The front travels 1.50 by t=0.7 against
  `deltaSPH`'s 2.82 on identical geometry, resolution and `dt`.
- **88% of the kinetic energy is dissipated between t=0.5 and t=0.8** — the
  moment the falling column should be turning into horizontal run-out — while
  `deltaSPH` is still gaining. Same over-dissipation Part 18 measured on the
  bounded case, in the regime that exposes it.
- **Not a timestep artifact**: the run is 5x *finer* in time than [BK]'s CFL
  requires, because `dambreak` has no incompressible `timestep` hook. Adding
  one is worth ~5x wall time.
- **The free-surface density deficit vanishes** (0.518 at t=0.02, ~0.98
  thereafter) while the surface keeps only three quarters of the bulk neighbour
  count, i.e. while it is still geometrically a surface. Surface compaction is
  the natural reading and is what §4 records at the rotating patch's corners,
  but this measurement does not isolate it.

### Part 20, `dambreak`'s timestep hook — landed, and the "~5x cheaper" guess corrected

Part 19 guessed that giving `dambreak` an incompressible `timestep` hook would
be "cheap" and buy roughly 5x. `dambreakTimestep` (`cases/dambreak.py`) now
does this — active only under `--scheme divergenceFree`, reusing
`kolmogorovIncompressibleTimestep` exactly as `randomFlowIncompressible` does,
and a strict no-op for `deltaSPH` (`Case.timestep` is one hook shared by every
scheme a case might run under; it returns `config.dt` unchanged when
`ctx.scheme` is not `divergenceFree`). Measuring it found two things the guess
got wrong.

**The published CFL constant is not safe on this case.** `randomFlowIncompressible
--bounded` ships `cflFactor = 0.4` under the Part 14 defaults; `dambreak` does
not have the option. Bisected on `dambreak --nx 64 --scheme divergenceFree`,
full run to t=1.5 (nx=64, `--integrationScheme semiImplicitEuler`):

| `cflFactor` | outcome | steps | `rho` range |
|---|---|---|---|
| 0.4 (published) | **NaN at step 30** (t≈0.2) | — | — |
| 0.3 | **NaN at step 76** (t≈0.3) | — | — |
| 0.25 | survives | 1960 | [0.507, 1.231] |
| **0.2** | survives | **1769** | **[0.507, 1.105]** |
| fixed `dt=5e-4` (Part 19 baseline) | survives | 3000 | [0.907, 1.004] |

0.2 is the recommended value: 0.25 is technically stable but its `rho_max`
(1.231) is markedly worse, so there is no reason to run it. The mechanism is
presumably §1.6 again — the falling column's impact is a sharper, more
localised event than `randomFlowIncompressible`'s gentle bounded shear, and the
CFL condition's `vMax` is read from the *previous* step, so a fast-developing
local spike at the point of impact is exactly what a lagged advective
condition sees latest. **Unmeasured**: whether this is the same mechanism
behind Part 19's over-dissipation, since both are about what happens at the
moment of impact — worth keeping in mind while doing item 1.

**The step-count win is real but far smaller than guessed, and it is not free.**
At the recommended `cflFactor = 0.2`, the full run to t=1.5 takes 1769 steps
against the fixed-`dt` baseline's 3000 — **1.7x fewer**, not ~5x. (The 5x
figure in Part 19 was `dt_adv` at the *initial*, near-rest state compared
against the shipped fixed `dt`; it was never a measurement of the adaptive
run, which spends much of its time at higher `vMax` once the column falls,
where the CFL condition hands back a smaller `dt` than that initial estimate.)
And it is not a win on every axis: `rho_max` over the whole run is 1.105
against the baseline's 1.004, i.e. adaptive stepping here is trading some
density accuracy for fewer steps, not dominating the fixed `dt` on both. Wall
time was not compared cleanly — this session's GPU had unrelated processes
resident throughout (an `nvidia-smi` check found four other python/llama.cpp
processes holding device memory), which the rest of this document has already
found is enough to swamp a per-step difference (§4, Part 14 point 3), so no
wall-time number is reported here.

**What shipped:** `dambreakTimestep` in `cases/dambreak.py`, and the case
docstring now says `--cflFactor 0.2`, not the published 0.4.
`scripts/probe_dambreakIncompressible.py`'s `runScheme` passes it for
`divergenceFree` runs so every number that probe reports from here on is at
the stable value. No default in `Case.defaults` changed — `cflFactor` is one
config field shared by every scheme a case can run under, `deltaSPH` still
gets its own `0.3` unchanged, and a `divergenceFree` run requires passing
`--cflFactor 0.2` explicitly, the same way it already requires
`--integrationScheme semiImplicitEuler`.

### Part 21, the dissipation is not the free-surface clamp — the literature and a measurement agree

Item 1's leading candidate was §1.10's compaction story: the constant-density
solve drives free-surface particles back toward `rho0` against what the
geometry allows, and that was flagged as the plausible (but unestablished)
cause of Part 19's 88% kinetic-energy loss at impact. Two things now weigh
against it, one from the literature and one from a measurement — read
together with `literature/` before touching any code, per this session's
brief.

**The mechanism the codebase already runs at the free surface is [BK]'s own
published remedy, not a gap.** `bender2015`'s discussion section (extracted
from the PDF, p.9): "In SPH simulations the density near a free surface is
underestimated which causes unnatural particle clustering artifacts. In our
implementation this problem is solved by clamping negative pressures to
zero." That is exactly `ShiftPressureGauge.nonNegativeClamp`, which
`solveIncompressible` already falls back to on any solve with free-surface
particles (§1.5) — so the shipped configuration already implements the
paper's fix, it does not omit it. The paper's own "better solution" is
`schechter2012`'s ghost particles, a structural addition (a sampled layer in
the surrounding air), not a one-line pressure edit — so the commented-out
`pressureB[surfaceIndicators == 1] = 0.0` in `divergenceFree.py` that §1.10
and the old Known-open entry pointed at is neither this codebase's own
workaround nor the paper's remedy; it is a third, unpublished idea that
happens to be sitting in the file, in the wrong solver besides (the
divergence-free solve, which does not target density, rather than
`solveIncompressible`, which does).

**Removing the clamp does not slow the dissipation down — it kills the run in
four steps.** `scripts/probe_dambreakSurfaceGauge.py` forces
`forceShiftPressureGauge = True`, which keeps `ShiftPressureGauge.minShift`
active at the free surface instead of falling back to the clamp — the
free-surface half of that guard, explicitly flagged as untested in
`solver.py`'s own field description. Same case, same `cflFactor = 0.2`, nx=64:

| gauge at the free surface | outcome |
|---|---|
| shipped (clamp) | 1327 steps to t=1.0, no divergence, KE peaks 11.72 at t≈0.46 then falls to 1.18 by t=1.0 — Part 19's dissipation, reproduced |
| forced `minShift` (no clamp) | **NaN at step 4** (t≈0.03); `nLow` (surface population) roughly doubles in the first 3 steps and `rhoMin` collapses to 0.19 |

So the clamp is load-bearing, not a source of drag to relax: without it the
surface does not merely stay under-dense, it destabilises immediately — a
signed shifting potential can pull particles together at a boundary with
genuinely truncated support (§1.5's own reasoning, and exactly the caveat §4
item 12 already carried for MINRES-without-clamp: "a shifting potential that
may go negative can pull particles together... needs a free-surface test").
This is that test, run for the first time because `dambreak` is the first
working free surface, and it settles `forceShiftPressureGauge`'s free-surface
half as unsafe rather than merely unmeasured.

**Consequence: item 1's search moves off the free-surface treatment and onto
the impact itself.** The compaction is real (§1.10 measured it) and the clamp
that produces it is necessary for the run to survive at all, but forcing it
off does not reduce the dissipation — there is no dissipation to observe once
it is off, because the run is already dead. That rules out "relax the
free-surface pressure handling" as a fix and leaves the moment of impact — the
falling column striking the floor — as the remaining candidate, which is also
where Part 19's own timing (loss concentrated at t=0.5-0.8) and Part 20's CFL
finding (this case cannot survive the published constant, unlike every other
bounded case measured) both point.

Landed: `scripts/probe_dambreakSurfaceGauge.py`, the free-surface A/B above.
No config or default changed — `forceShiftPressureGauge` stays `False`.

### Part 22, the energy budget — the dissipation is the incompressibility cycle, not the walls or the viscosity

Item 1's instrument, run. `scripts/probe_dambreakEnergyBudget.py` closes the
kinetic-energy budget exactly, per step, and decomposes it two ways. The
**work form** splits `dKE` into each channel's first-order work plus a
quadratic remainder; it localizes work to an x-bin, so it answers *where*. The
**sequential form** takes the exact KE change of adding each force to the
running velocity, in the order the step applies them (gravity, no-penetration,
viscosity, the projection last, then the resample); those five values telescope
to `dKE` by construction and are unambiguous about *which step* removes KE and
which returns it. Same run as Part 19/21: `dambreak --scheme divergenceFree`,
nx=64, `cflFactor = 0.2`, 1327 steps to t=1.0.

**The closure is exact.** The captured forces reproduce the integrator's real
update to `max |u4 − v*| = 3.3e-6` (float32 round-off at |v|~10), the work-form
closure to `max |dKE − ΣW| = 1.15e-7`, and the decomposition gap (the captured
sum of the five accelerations against the integrator's) to exactly `0.0`. Every
velocity-changing term is accounted for; nothing is lost to the probe.

**The loss is the incompressibility cycle.** The dissipation window (KE peak
t=0.444, KE=12.13, to t=0.801) loses 10.04 KE, 82.8% of the peak — the same
event as Part 19's "88% between t=0.5 and t=0.8", measured from the peak instead
of a fixed start. The sequential channels:

| channel | KE change | share of the loss |
|---|---|---|
| divergence-free projection (`d_DF`) | −35.80 | 357% |
| Eq. 17 position-shift resample (`d_resample`) | +27.30 | −272% (returns) |
| **incompressibility cycle (the two summed)** | **−8.50** | **85%** |
| Monaghan viscosity (`d_visc`) | −6.37 | 64% |
| mDBC no-penetration shift (`d_nopen`) | −0.009 | 0.1% |
| gravity (`d_grav`) | +4.84 | source, opposes the loss |

The projection removes 35.8 of KE at impact and the resample gives 27.3 of it
back; the 8.5 the cycle keeps is the dominant single contribution to the loss.
Monaghan viscosity is the secondary channel (6.4). The no-penetration wall
shift — the other wall-side suspect — is negligible (0.009), so Part 21's
ruling out of the free-surface clamp is now joined by a ruling out of the
no-pen shift: **neither wall-side treatment is the dissipator.**

**The work form's `W_DF` alone overstates the projection's cost.** Its
first-order projection work is −44.3 in [0.4,0.5] — read naively, "the
projection dissipates 44". The sequential form shows +25.6 is returned by the
resample in that same window, so the cycle's *net* there is +0.1, essentially
energy-conserving. The cycle's true dissipation is front-loaded one window
later, at [0.5,0.6] (net −3.8), and decays after:

| window | `d_grav` | `d_nopen` | `d_visc` | `d_DF` | `d_resamp` | cycle net | `dKE` |
|---|---|---|---|---|---|---|---|
| [0.3,0.4] | +2.52 | −0.002 | −1.58 | −7.77 | +13.10 | +5.33 | +6.30 |
| [0.4,0.5] | +2.57 | −0.002 | −3.50 | −25.50 | +25.60 | +0.10 | −0.84 |
| [0.5,0.6] | +1.60 | −0.003 | −2.26 | −13.10 | +9.31 | −3.79 | −4.48 |
| [0.6,0.7] | +1.05 | −0.002 | −1.23 | −4.70 | +2.95 | −1.75 | −1.93 |
| [0.7,0.8] | +0.83 | −0.003 | −0.87 | −2.71 | +1.72 | −0.99 | −1.04 |

So the loss is not one channel throughout: in [0.4,0.5] the cycle is
net-neutral and the KE drop is viscosity (−3.5) outrunning gravity (+2.6); in
[0.5,0.6] the cycle itself turns net-dissipative and leads. Over the whole
window the cycle (−8.5) still outweighs viscosity (−6.4). The net is the
residual of two large, nearly-opposing terms (−36 and +27), so the answer is a
seesaw balance, not a single gross loss — and both terms are exact, given the
closure.

**It is where the impact is.** The per-bin table concentrates the loss in the
left bins, x ∈ [−2.0,−1.0] — the falling column's footprint — where the jet
strikes the floor, and the surviving KE moves right into the run-out. The
projection's work is most negative in exactly those bins (W_DF −9.0 in
x∈[−1.73,−1.51] at [0.4,0.5]).

**Why this closes the cross-scheme gap.** Both incompressibility channels are
specific to the `divergenceFree` scheme: the DF velocity projection and the
Eq. 17 position-shift resample are both part of its `finalize`, and neither
exists in the weakly-compressible `deltaSPH` control (Part 19). The Monaghan
viscosity (−6.4) is common to both schemes. So the 88%-vs-less gap between the
two schemes is carried by the incompressibility cycle — the thing the two
schemes differ on. What the budget does *not* by itself settle is why the
discrete cycle is net-dissipative at all: that −8.5 is the residual of two
~30-magnitude terms, and whether it is a discretization error that vanishes as
`nx` grows or a structural cost of the incompressible constraint is the natural
next instrument (an `nx` convergence of the cycle's net on this same case).

Landed: `scripts/probe_dambreakEnergyBudget.py` (the per-step / per-window /
per-bin budget above). No config or default changed.

### What is left, in order

1. **Explain the dam break's dissipation** (Part 19). **The channel is
   identified (Part 22)**: it is not the walls — Part 21 ruled out the
   free-surface clamp and Part 22's budget measures the no-pen shift at
   negligible — and it is not viscosity alone; it is the incompressibility
   cycle, the DF projection plus the Eq. 17 position-shift resample, net −8.5
   over the loss window (85% of it) against Monaghan viscosity's −6.4 (64%),
   localized at the column's impact footprint. Both channels exist only in the
   `divergenceFree` scheme, which is what carries the cross-scheme gap against
   the `deltaSPH` control. **What remains is the mechanism**: that −8.5 is the
   residual of two ~30-magnitude terms, and whether it is a discretization
   error that vanishes as `nx` grows or a structural cost of the incompressible
   constraint is open. An `nx` convergence of the cycle's net on this same case
   is the natural next instrument.
2. ~~Give `dambreak` an incompressible `timestep` hook.~~ **Done (Part 20)** —
   landed as `dambreakTimestep`, active only under `--scheme divergenceFree`.
   Worth ~1.7x fewer steps at the case's own safe `cflFactor = 0.2`, not the
   ~5x guessed, and not free (`rho_max` 1.105 against the fixed-`dt`
   baseline's 1.004) — see Part 20 for why the published 0.4 diverges here.
3. **Grade `shearWave` against [C]'s Fig. 3 and Fig. 4** (§4 item 8's
   remainder). Blocked on the paper — `literature/MANIFEST.md`.
4. **Warm-start the divergence-free solve** (§4 item 9, split by Part 15).
   Unblocked; do **not** warm-start the constant-density solve.
5. **Re-measure the divergence-free half-state's contraction** under `minShift`
   (§4 item 3) — still the one mechanism observed and never explained.
6. **Then** the rename and the scheme split.

### What is next, concretely

Item 1's instrument ran (Part 22) and it names the channel: the incompressibility
cycle, at the impact. What is left of item 1 is the mechanism question — the
cycle's net is the residual of two ~30-magnitude terms, so the next measurement
is an `nx` convergence of that net (does the −8.5 fall with resolution, i.e. is
it a discretization error, or is it flat, i.e. a structural cost of the
incompressible constraint?). That single run decides whether the fix to look for
is a better projection/resample or an acceptance that the incompressible scheme
dissipates impact flows and should be reserved for them accordingly. Items 3-5
stand as ranked; the rename and the scheme split stay last.
