# Weakly compressible examples: notebook-migration plan

**In progress.** Slots 01, 02, 03, 04, 08, 09 and 10 are done (six notebooks,
seven files); 05, 06, 07, 11, 12 and 13 are open. The `.py` wrappers were already
in the current style and all 10 case modules were already registered in
`CASE_MODULES` — what this plan closes is the notebooks, which were all on the
pre-`warpSPHBootstrap` shape: a 43-line boilerplate cell copied verbatim, the
config built by hand, `visualize(...)` spelled out inline per notebook, and (in
06–13) a 67-line viscosity-sweep cell copy-pasted between files that have no
use for it.

Goal is the same as `../compressible/MIGRATION_STATUS.md`: the case is the
single source of the physics, the `.py` is a thin runner wrapper, and the
notebook is the *fat*, editable form with a visible step loop.
`../../PORTING_EXAMPLES.md` is the procedure and the *why*;
`../../CLEANUP_PLAN.md` §"Notebook simplification, the rest" is the item this
closes (the `examples/incompressible/` half stays open after this).

## Target end-state

Every case is:

- `src/warpSPH/cases/<name>.py` — the `Case` (geometry/IC, diagnostics, plot
  hooks, defaults). **Exists for all 10.**
- `examples/weaklyCompressible/<slot>-<name>.py` (or
  `<slot>-<name>/<name>_<variant>.py` for a multi-variant case) — a thin
  `caseMain()` wrapper. **Exists for all 13.**
- `examples/weaklyCompressible/<slot>-<name>.ipynb` (or the per-variant
  equivalent) — the *fat* notebook: intro markdown with an `outputs/` animation
  **and a table documenting every knob**, parameters cell, IC built through the
  real case code, region preview, an **unrolled step loop** with a
  `# <-- hook point` comment, plots called directly rather than through
  `case.setupPlot`/`updatePlot`, then whatever analysis is genuinely
  notebook-shaped for that case.

A case gets its own `<slot>-<name>/` directory only if it has more than one
variant worth showing side by side; a single-variant case stays a flat numbered
file. Numbering gaps left by a merge are fine — the compressible family has them
at 07 and 15.

## Widen the case before porting the notebook

Added 2026-08-14, and it is the reason this pass is not purely mechanical.
Several of these cases were *one point* of a family they could just as well
cover — the two impact notebooks differed only in the shape of the colliding
bodies — and a notebook that documents every knob is worth much more when the
knobs span the family. So: **expand the case first, then write the notebook
against the expanded parameter set**, in the style of the (now `dambreak`)
`datagen/weaklyCompressible/generator.py` parameterisation, where the geometry,
the forcing and the perturbations are all flags rather than edits.

The shared machinery for that lives in `cases/weaklyCompressible.py`:

| helper | what it gives a case |
|---|---|
| `SHAPE_PRESETS` | 17 closed 2D primitives (`circle`, `box`, `roundedBox`, `rhombus`, `trapezoid`, `parallelogram`, `equilateralTriangle`, `triangleIsosceles`, `pentagon`, `hexagon`, `octogon`, `hexagram`, `star5`, `vesica`, `cutDisk`, `unevenCapsule`, `moon`), keyed by name |
| `shapeArgs(name, size, aspect)` | one characteristic half-size and one aspect ratio → that primitive's own argument list, so a case never has to know that `sdBox` takes half-extents while `sdHexagon` takes a radius |
| `shapeSdf(name, ..., args=, rotation=)` | the SDF, rotated (degrees CCW) and translated |
| `sdfBounds(sdf, domain)` | `(centre, halfExtent)` **measured** on a grid — the primitives are not all centred on their own origin and a rotated one never is |
| `centredShapeSdf(...)` | the SDF placed so its *measured* centre lands where asked, plus that half-extent |
| `OBSTACLE_PARAMS` / `paramShapeSdf(ctx)` | the five `obstacle*` knobs (`Shape`, `Size`, `Aspect`, `Rotation`, `Offset`) and the SDF they describe, so `--obstacleShape star5` means the same thing in every case that has an obstacle |

Two library fixes fell out of using them (both in
`geometry/sdfFunctionality/implicitFunctions.py`): `sdPentagon`, `sdOctogon`,
`sdHexagram`, `sdParallelogram` and `sdMoon` updated `p` **in place**, which
made `sampleSDF`'s autograd gradient fail a version-counter check — every one
of those five shapes was unusable as a region. `sdMoon` additionally wrote
through to the caller's tensor. All five are out-of-place now, and two leftover
debug `print`s went with them.

Expanded so far:

- **`impact`** — was `--shape circle|box` with a fixed velocity. Now: any
  `SHAPE_PRESETS` shape (`size`, `aspectRatio`, `rotation`); `arrangement`
  `pair` or `ring` (`nBodies`, `ringPhase`); placement by `separation` or by
  `touching`+`gap` (measured from the shape, which is what the squares notebook
  hard-coded as `H/2 + dx`); `impactAngle` for oblique and `lateralOffset` for
  off-centre collisions; `spin` per body; and a `bodies` list that bypasses the
  arrangement entirely. Each body's velocity is now a **per-region
  `initialConditions` callable** rather than a post-hoc test on the sign of a
  coordinate, which is what makes rings, overlapping bounding boxes and spin
  work at all. Centres are snapped to the particle lattice, so a mirrored pair
  starts with exactly zero momentum.
- **`squarePatch`** — `shape`/`size`/`aspectRatio`/`rotation` instead of
  `halfExtent`. The corners are the benchmark, so `--shape circle` is the null
  experiment and `star5`/`hexagram` are sharper versions of the same one.
- **`movingObstacle`**, **`randomFlow`**, **`kolmogorov`** — all three now take
  the shared `OBSTACLE_PARAMS`. `obstacleRadius` became `obstacleSize`.
- **`droplet`** — no new geometry (the analytic solution is for a circle), but
  `DROPLET_STRETCH`, `DROPLET_PERIOD` and `analyticEnvelope()` are exported so
  the notebook's envelope overlay is not three bare literals. The old notebook's
  wide ellipse was drawn `2R x R`, which is not the area of the circle it came
  from; `analyticEnvelope` is area-conserving.

Still worth expanding, before their slots are ported: `tgvWeaklyCompressible`
(05 — the `nu` sweep is the case's own `effectiveViscosity`, so the sweep should
be a loop over `run()`), `channelFlow` (11/13 — the inlet/outlet and the
obstacle are hard-coded where `dambreak`'s are flags).

## The notebook shape here

One shape, not two. Every case in this family is 2D and plots particle fields,
so all the notebooks are `particlePlot`-shaped and follow
`../compressible/08-Hydrostatic.ipynb` cell for cell: `buildFieldPlotter` /
`refreshFieldPlotter` from `cases/plotting.py` called directly (they are
`setupPlot`/`updatePlot` minus `openWindow`/`pumpEvents`, which do not
live-update in a Jupyter cell here), with the case's own `Field` list passed in.
No `profilePlot` case in this family.

Three differences from the compressible template, all of which will bite if the
`08` notebook is copied unread — every ported notebook states them in its intro
cell:

1. **The IC cell needs a fourth call.** Every weakly compressible case has an
   `initialConditions` hook (only `linearWave` does on the compressible side),
   and it is where `setupTimestep` picks the sound speed and `config.dt`
   *together* from `targetDt`. The cell is
   `buildContext` → `configureScheme` → `buildSystem` →
   **`case.initialConditions(ctx, system)`** → `initializeNewState`. Skip it and
   `config.dt` is `None` (the runner raises for exactly this, `runner.py:238`),
   and the case's own IC — the rigid rotation, the strain field, the lid
   Dirichlet condition, the obstacle's spin — is never applied.
2. **The loop is always `range(nSteps)`.** No case in this family defines a
   `timestep` hook, so `timeLimited` is false and `dt` is fixed for the whole run
   after step 1 above. `while t < tLimit` is wrong here.
3. **`markerSize` is a case param** and varies (4 or 8); pass it through rather
   than hardcoding, `buildFieldPlotter` already reads `ctx.param('markerSize', 2)`.

`VELOCITY_DENSITY_FIELDS` (`cases/weaklyCompressible.py`) is this family's
equivalent of the compressible cases' per-case `*_FIELDS` lists and 8 of the 10
cases use it; `impact` has its own `IMPACT_FIELDS` (same two panels, `flare`
rather than `RdBu`), exported for the notebook to import rather than re-derive.

**Every notebook opens with a full options table** — one row per `CaseSpec`
field and per case parameter, with the value that notebook uses and what it
does. That is the deliverable, not a nicety: the parameters cell is only
editable if the reader knows what each knob means.

## Slot status

| slot | notebook | case | state |
|---|---|---|---|
| 01 | `01-impact/impact_spheres.ipynb` | `impact` | **done** — merged with 02 into `01-impact/`, case expanded, `IMPACT_FIELDS` exported |
| 02 | `01-impact/impact_squares.ipynb` | `impact` | **done** — `--touching` replaces the hand-written `H/2 + dx` |
| 03 | `03-rotating-square-patch.ipynb` | `squarePatch` | **done** — pilot; angular-momentum drift recorded at the hook point |
| 04 | `04-oscillating-droplet.ipynb` | `droplet` | **done** — analytic envelope kept, constants exported from the case |
| 05 | `05-taylor-green-vortex.ipynb` | `tgv-wc` | open — the viscosity-sweep notebook, **the one place the `nu` sweep belongs** |
| 06 | `06-periodic-random-flow.ipynb` | `randomFlow` (`--no-bounded --obstacle`) | open — merge into `06-randomFlow/` |
| 07 | `07-bounded-random-flow.ipynb` | `randomFlow` (`--bounded`) | open — becomes `06-randomFlow/randomFlow_bounded.ipynb` |
| 08 | `08-kolmogorov-flow.ipynb` | `kolmogorov` | **done** — dead forcing-prototype cells dropped; mean-profile-vs-forcing panel added |
| 09 | `09-lid-driven-cavity.ipynb` | `ldc` | **done** — renamed from `09-LDC.ipynb`; centreline profiles kept |
| 10 | `10-moving-obstacle.ipynb` | `movingObstacle` | **done** — mean-velocity-vs-target panel at the hook point |
| 11 | `11-driven-square.ipynb` | `channelFlow.drivenSquareCase` | open — unblocked, `DAMBREAK_FIELDS` is importable now |
| 12 | `12-dambreak.ipynb` | `dambreak` | open — unblocked; keep the density-field cell |
| 13 | `13-openFlow.ipynb` | `channelFlow.openFlowCase` | open — 1704 lines, 57 cells, the worst one; do it last |

`naca.ipynb` stays as it is: a standalone SDF-visualisation scratchpad with no
case, already recorded as won't-fix in `CLEANUP_PLAN.md`. (It is also the only
notebook here shipping stored outputs, 216 KB of them — worth stripping while
passing by, separately from this migration.)

`utils.py` is a re-export shim for `warpSPH.caseUtils`, imported only by the 12
and 13 notebooks. It should have no importers left once those two are ported;
delete it in the same commit as slot 13, not before.

## The dam-break plot path — closed 2026-08-14

Slots 11, 12 and 13 did not go through `particlePlot`: `dambreak.py`'s
`setupPlot`/`updatePlot` called `caseUtils/weaklyCompressiblePlot.setupPlotter`,
which built its panels inline and titled them with `buildPlotText`, and
`channelFlow.py` imports those same two hooks. There was no window-free core to
call from a notebook — exactly the position `particlePlot` was in before
`08-Hydrostatic` split out `buildFieldPlotter`/`refreshFieldPlotter`.

Done, in that shape:

- `DAMBREAK_FIELDS` — **two panels**, velocity and the cyclic-coloured particle
  IDs, which is what `plotDensity=False` (the shipped default) always rendered.
  `DAMBREAK_FIELDS_DENSITY` is the three-panel version, and `dambreakFields(ctx)`
  picks between them, so `--plotDensity` still works. `Field` already supported
  the cyclic family, so the panels needed no new plotting code.
- The title is now `figureTitle` (case/t/dt/ptcls plus the diagnostics row).
  The obstacle description, the fluid/boundary split and `c0` that
  `buildPlotText` also printed are **dropped deliberately** — they are in the
  spec, the export and the notebook's own cells.
- `setupPlot`/`updatePlot` are five lines each: `buildFieldPlotter` with the
  case's own `plotWidth`/`plotHeight` figsize, then `openWindow`/`pumpEvents`.
  The duplicated try/except fallback went with them (`visualizeWithFallback`
  does that already), as did `caseUtils/weaklyCompressiblePlot.py` and its
  `datagen/weaklyCompressible/plot.py` re-export shim.

Two things surfaced while looking at the result, both fixed:

- **Boundary particles were invisible.** `PlottingOptions.boundaryVisualization`
  defaults to `Hide`, and nothing in `cases/plotting.py` set it — so the
  lid-driven cavity plotted as a square of fluid with no walls, and the moving
  obstacle's spinning body did not appear at all. `Field` now carries
  `boundary='Passive'` (grey crosses, excluded from the colour normalisation)
  and `fluid='Visualize'`, so **every** case in every family shows its walls
  without letting them set the colour range. Cases with no boundary particles
  are unaffected.
- **The dam break's default column was wrong.** `fluidWidth` defaulted to
  `5/2 * 1/3`, i.e. a column `0.833 * W = 3.33` wide by `1/3 * L = 0.67` tall in
  a 4 x 2 tank — a shallow slab over 83% of the floor, not a dam break. Every
  line of `datagen/weaklyCompressible/cases/dambreak.sh` passes an explicit
  `--fluidWidth` (5/12, 1/4 or 1/12), so the default was never exercised — the
  same class of stale default as `obstacleType='circle'`. It is now
  `fillRatio=2/3, fluidWidth=1/6`: a 0.667 x 1.333 column, the canonical
  Koshizuka & Oka 1:2 proportions with six column-widths of run-out.
  `examples/sweeps/dambreak_obstacle.yaml` sets `fillRatio` but not
  `fluidWidth`, so it inherits the fix.

Verified with `warpsph-run dambreak --plot`, `run_sweep.py --cases dambreak
drivenSquare openFlow` and the dam-break physics tests.

## What is genuinely notebook-shaped, and what to delete

The sort from `PORTING_EXAMPLES.md` §2.1 — geometry/IC, diagnostics, plots, and
everything else is runner — has already been done once for these cases (that is
what produced the case modules). So for this pass the question per cell is only:
*does this exist in the case?* If yes, delete the cell and call the case. If no,
it is either analysis worth keeping or scratch worth deleting.

The ported notebooks settled on one more shared cell than the compressible
template has: a **region preview** (`plotRegions` over `ctx.scratch['regions']`)
between the IC cell and the plot setup. Every old notebook here had one, and it
earns its place now that the geometry is parameterised — it is where you see
that `--shape moon --rotation 40` did what you meant.

**Keep, as its own cells:**

- **05, viscosity calibration.** The `alphaToNu`/`nuToAlpha` markdown, the
  Reynolds number, the `nu_limit` from `alpha >= 0.01`, the KE-decay fit against
  `KE(0) exp(-4 nu k^2 t)`, and the `nu` sweep. `tgvWeaklyCompressible.py`
  already exports `analyticDecayRate` and `effectiveViscosity(result)` for
  exactly this — the sweep becomes a loop over `run()` calls plus
  `effectiveViscosity`, not a re-implementation of the step loop.
- **12, the reconstructed density field.** `computeDensities(...)` re-evaluated
  after the run and plotted — a free-surface check the live panels do not show.
- **13, one datastructure-inspection cell** (hash map / Verlet list sizes) if it
  still runs; it is the only cell in that notebook that teaches something the
  case cannot.

**Delete:**

- The 43-line boilerplate cell → 4 lines of `bootstrap()` + imports.
- Every inline `visualize(...)`/`PlottingOptions` block (~50 lines each) →
  `buildFieldPlotter(ctx, runningState, FIELDS)`.
- The `nu_tests` sweep cell where it was copy-pasted without a purpose: 06, 07,
  11, 12, 13 (67 lines each, identical, all ending in a
  `plotter.updateQuantities` call on a plotter from a different cell). It stays
  in 05 only. (08, 09 and 10 are done; theirs are gone.)
- The trailing run of empty cells every notebook ends with (4–5 each).
- 13's `torch.profiler` cells, its commented-out alternate system builds, and
  the half-finished `# def ldcDirichlet` block pasted into 11/12/13.
- 08's dead forcing-prototype cells (`# forcingRegion = ...` and the two empty
  neighbours) — `kolmogorov.py` has the working forcing.

## Order of the remaining work

1. **05** — port plus the viscosity calibration that stays, and the `run()`-loop
   form of the sweep.
2. **06+07 → `06-randomFlow/`**, the multi-variant pattern, following
   `01-impact/`. The obstacle is now `OBSTACLE_PARAMS`, so both notebooks'
   options tables gain those five rows.
3. **11, 12** — first users of the converted dam-break plot path
   (`from warpSPH.cases.dambreak import DAMBREAK_FIELDS`, or `dambreakFields(ctx)`
   if the notebook should honour `--plotDensity`).
4. **13** — last. Expect a rewrite rather than a port: 57 cells, most of them
   scratch, and `openFlowCase`'s defaults already encode the setup the notebook
   builds by hand.
5. **Sweep-up:** delete `utils.py`, add `EXAMPLES_SUMMARY.md` for this directory
   (the compressible one is the template), and flip the `CLEANUP_PLAN.md` item
   to note this family done and only `examples/incompressible/` remaining.

## Outputs and animations

`examples/weaklyCompressible/outputs/` (and `01-impact/outputs/` for the
multi-variant case) holds one `<slot>-<Case>.gif` / `.mp4` / `.png` per case,
referenced from the intro cell as `![](outputs/<name>.gif)`, rendered at the
notebook's own shipped settings. Notebooks ship with **no stored outputs**.

`scripts/render_examples.py` does the rendering, for every family rather than
just this one. It runs the example **wrappers** (so the settings are the ones
that example ships with, not the bare case defaults), reads each artefact's name
out of the sibling notebook's `![](outputs/<name>.gif)` reference so the file
that lands is the file the notebook asks for, and copies the gif, the mp4 and
the final frame into the right `outputs/`:

    scripts/render_examples.py --list                     # what runs, and where it lands
    scripts/render_examples.py --only weaklyCompressible  # this family
    scripts/render_examples.py --only 01-impact --trace   # plus a per-particle trajectory.h5
    scripts/render_examples.py --only sod -- --nx 64      # forwarded flags

These are full-length runs at the shipped resolution -- `09-lidDrivenCavity` is
60k steps and 6k frames -- so expect tens of minutes each. `--trace` adds
`--store --storeMode trajectory`; those files stay in the export tree (they are
large, and they are not example artefacts) and their paths are printed.

## Verification

Per case, in this order:

1. `python scripts/check_imports.py --static` — AST-scans notebook cells, which
   a runtime check never reaches. (Two failures pre-date this work:
   `datagen/weaklyCompressible/quartzTest.ipynb` imports `warpSPH.io.io`, and
   `examples/compressible/14-triplePoint/triplePoint_equalMass.ipynb` has an
   unterminated string in a cell.)
2. Execute the notebook on a small-`nx`/short-`tLimit` copy. `nbconvert` is not
   installed in the `warp` env; exec'ing the code cells in order is the stand-in
   that was used here. Either way it proves the notebook runs and nothing about
   the numbers (`PORTING_EXAMPLES.md` §5).
3. `python scripts/run_sweep.py --cases <name>` — unchanged case behaviour.
4. Look at one field plot deliberately, against what the case is supposed to do:
   density inside ±1% of `rho0` (that is what `weaklyCompressibleDiagnostics`'s
   `maxDensity`/`minDensity` are for), the vortex still a vortex, the dam break
   not exploding. Every ported notebook now ends with a cell plotting those
   density bounds against the ±1% band, so this is a cell to read rather than a
   judgement to make.

Whole-suite once the family is done: `bash scripts/run_tests.sh` and
`python scripts/run_sweep.py` with no `--cases`.

Note that `tests/test_physics.py` covers only `dambreak` out of this family's 10
cases. The migration does not change what the cases *compute*, so no new tests
are strictly required — but the case expansions above do change how the geometry
is built, so `run_sweep.py --cases impact squarePatch droplet tgv-wc randomFlow
kolmogorov ldc movingObstacle dambreak drivenSquare openFlow` is the check that
matters, and it passed 11/11 after them. If the dam-break plot conversion is
taken, run the dam-break tests before and after and confirm they are untouched.
