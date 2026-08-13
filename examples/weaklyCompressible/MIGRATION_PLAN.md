# Weakly compressible examples: notebook-migration plan

**Open.** The 13 `.py` wrappers in this directory are already in the current
style (thin `caseMain(<case>, PRESET + sys.argv[1:])`, 28 lines each) and all
10 case modules are registered in `CASE_MODULES`. What is *not* done is the
notebooks: all 13 are still on the pre-`warpSPHBootstrap` shape — a 43-line
boilerplate cell copied verbatim, the config built by hand, `visualize(...)`
spelled out inline per notebook, and (in 06–13) a 67-line viscosity-sweep cell
copy-pasted between files that have no use for it.

Goal is the same as `../compressible/MIGRATION_STATUS.md`: the case is the
single source of the physics, the `.py` is a thin runner wrapper, and the
notebook is the *fat*, editable form with a visible step loop.
`../../PORTING_EXAMPLES.md` is the procedure and the *why*;
`../../CLEANUP_PLAN.md` §"Notebook simplification, the rest" is the item this
closes (the `examples/incompressible/` half stays open after this).

## Target end-state

Every case is:

- `src/warpSPH/cases/<name>.py` — the `Case` (geometry/IC, diagnostics, plot
  hooks, defaults). **Already exists for all 10.**
- `examples/weaklyCompressible/<slot>-<name>.py` (or
  `<slot>-<name>/<name>_<variant>.py` for a multi-variant case) — a thin
  `caseMain()` wrapper. **Already exists for all 13.**
- `examples/weaklyCompressible/<slot>-<name>.ipynb` (or the per-variant
  equivalent) — the *fat* notebook: intro markdown with an `outputs/` animation,
  parameters cell, IC built through the real case code, an **unrolled step loop**
  with a `# <-- hook point` comment, plots called directly rather than through
  `case.setupPlot`/`updatePlot`. **This is the work.**

A case gets its own `<slot>-<name>/` directory only if it has more than one
variant worth showing side by side; a single-variant case stays a flat numbered
file. Numbering gaps left by a merge are fine — the compressible family has them
at 07 and 15.

## The notebook shape here

One shape, not two. Every case in this family is 2D and plots particle fields,
so all 13 notebooks are `particlePlot`-shaped and follow
`../compressible/08-Hydrostatic.ipynb` cell for cell: `buildFieldPlotter` /
`refreshFieldPlotter` from `cases/plotting.py` called directly (they are
`setupPlot`/`updatePlot` minus `openWindow`/`pumpEvents`, which do not
live-update in a Jupyter cell here), with the case's own `Field` list passed in.
No `profilePlot` case in this family.

Three differences from the compressible template, all of which will bite if the
`08` notebook is copied unread:

1. **The IC cell needs a fourth call.** Every weakly compressible case has an
   `initialConditions` hook (only `linearWave` does on the compressible side),
   and it is where `setupTimestep` picks the sound speed and `config.dt`
   *together* from `targetDt`. The cell is
   `buildContext` → `configureScheme` → `buildSystem` →
   **`case.initialConditions(ctx, system)`** → `initializeNewState`. Skip it and
   `config.dt` is `None` (the runner raises for exactly this, `runner.py:238`),
   the noise/freestream/Kolmogorov forcing is never stamped on, and the run is
   silently a different case.
2. **The loop is always `range(nSteps)`.** No case in this family defines a
   `timestep` hook, so `timeLimited` is false and `dt` is fixed for the whole run
   after step 1 above. `while t < tLimit` is wrong here.
3. **`markerSize` is a case param** and varies (4 or 8); pass it through rather
   than hardcoding, `buildFieldPlotter` already reads `ctx.param('markerSize', 2)`.

`VELOCITY_DENSITY_FIELDS` (`cases/weaklyCompressible.py`) is this family's
equivalent of the compressible cases' per-case `*_FIELDS` lists and 8 of the 10
cases already use it — the notebook imports it from there rather than
re-deriving the two panels.

## Full slot list

| slot | notebook | case | fields | work |
|---|---|---|---|---|
| 01 | `01-impact_spheres.ipynb` | `impact` (`--shape circle --nx 256`) | own list, **not exported** | merge into `01-impact/`; export `IMPACT_FIELDS` |
| 02 | `02-impact-squares.ipynb` | `impact` (`--shape box --nx 128 --tLimit 1.5`) | same | becomes `01-impact/impact_squares.ipynb` |
| 03 | `03-rotating-square-patch.ipynb` | `rotatingSquarePatch` | `VELOCITY_DENSITY_FIELDS` | straight port |
| 04 | `04-oscillating-droplet.ipynb` | `oscillatingDroplet` | `VELOCITY_DENSITY_FIELDS` | straight port + keep the analytic-envelope panel (below) |
| 05 | `05-taylor-green-vortex.ipynb` | `tgvWeaklyCompressible` | `VELOCITY_DENSITY_FIELDS` | the viscosity-sweep notebook — **the one place the `nu` sweep belongs** |
| 06 | `06-periodic-random-flow.ipynb` | `randomFlow` (`--no-bounded --obstacle`) | `VELOCITY_DENSITY_FIELDS` | merge into `06-randomFlow/` |
| 07 | `07-bounded-random-flow.ipynb` | `randomFlow` (`--bounded`) | same | becomes `06-randomFlow/randomFlow_bounded.ipynb` |
| 08 | `08-kolmogorov-flow.ipynb` | `kolmogorov` | `VELOCITY_DENSITY_FIELDS` | straight port; drop the four dead forcing-prototype cells |
| 09 | `09-LDC.ipynb` | `lidDrivenCavity` | `VELOCITY_DENSITY_FIELDS` | rename to `09-lid-driven-cavity.ipynb` to match its `.py` |
| 10 | `10-moving-obstacle.ipynb` | `movingObstacle` | `VELOCITY_DENSITY_FIELDS` | straight port |
| 11 | `11-driven-square.ipynb` | `channelFlow.drivenSquareCase` | **none — dambreak plot path** | needs the plot gap below |
| 12 | `12-dambreak.ipynb` | `dambreak` | **none — dambreak plot path** | needs the plot gap below; keep the density-field cell |
| 13 | `13-openFlow.ipynb` | `channelFlow.openFlowCase` | **none — dambreak plot path** | 1704 lines, 57 cells, the worst one — do it last |

`naca.ipynb` stays as it is: it is a standalone SDF-visualisation scratchpad with
no case, already recorded as won't-fix in `CLEANUP_PLAN.md`. (It is also the only
notebook here shipping stored outputs, 216 KB of them — worth stripping while
passing by, separately from this migration.)

`utils.py` is a re-export shim for `warpSPH.caseUtils`, imported only by the 12
and 13 notebooks. It should have no importers left once those two are ported;
delete it in the same commit as slot 13, not before.

## The one real code gap: the dam-break plot path

Slots 11, 12 and 13 do not go through `particlePlot`. `dambreak.py`'s
`setupPlot`/`updatePlot` call `caseUtils/weaklyCompressiblePlot.setupPlotter`,
which builds its three panels (`velocities`, `densities`, `UIDs`) inline and
whose title is `buildPlotText`, reading `ctx.scratch['args']`/`['simSetup']` for
the obstacle and domain description. `channelFlow.py` imports those same two
hooks. There is no window-free core to call from a notebook — which is exactly
the position `particlePlot` was in before `08-Hydrostatic` split out
`buildFieldPlotter`/`refreshFieldPlotter`, and the fix is the same shape.

Convert `dambreak.py` to `particlePlot(DAMBREAK_FIELDS)`:

- `DAMBREAK_FIELDS = VELOCITY_DENSITY_FIELDS + [Field('UIDs', 'particle IDs',
  colorMap='twilight', colorMapKind='cyclic')]` — `Field` already supports the
  cyclic family, so no new plotting code is needed for the panels themselves.
- The `plotDensity` flag that switched between the 2- and 3-panel mosaic becomes
  a choice of `Field` list; keep both exported (`DAMBREAK_FIELDS`,
  `DAMBREAK_FIELDS_NO_DENSITY`) if the flag is still wanted.
- The title is the loss to weigh. `figureTitle` prints case/t/dt/ptcls plus any
  diagnostics row, `buildPlotText` also printed the obstacle description, the
  fluid/boundary split and `c0`. Either accept `figureTitle` (it already gains
  `maxDensity`/`minDensity`/`maxVelocity` from `dambreak.diagnostics` when a row
  is passed) or give `buildFieldPlotter` an optional `title=` callable — one
  parameter, default `figureTitle`, no caller changes.
- `caseUtils/weaklyCompressiblePlot.py` then has no importer inside `warpSPH`
  and can go, together with `resolvePlotBackend`'s duplicated try/except fallback
  in `dambreak.setupPlot` (`visualizeWithFallback` does that already).

Do this **before** slot 11, and verify it with `warpsph-run dambreak --plot` and
`python scripts/run_sweep.py --cases dambreak drivenSquare openFlow`, because it
is the only step in this migration that changes what a `.py` example renders.

## What is genuinely notebook-shaped, and what to delete

The sort from `PORTING_EXAMPLES.md` §2.1 — geometry/IC, diagnostics, plots, and
everything else is runner — has already been done once for these cases (that is
what produced the case modules). So for this pass the question per cell is only:
*does this exist in the case?* If yes, delete the cell and call the case. If no,
it is either analysis worth keeping or scratch worth deleting.

**Keep, as its own cells:**

- **05, viscosity calibration.** The `alphaToNu`/`nuToAlpha` markdown, the
  Reynolds number, the `nu_limit` from `alpha >= 0.01`, the KE-decay fit against
  `KE(0) exp(-4 nu k^2 t)`, and the `nu` sweep. `tgvWeaklyCompressible.py`
  already exports `analyticDecayRate` and `effectiveViscosity(result)` for
  exactly this — the sweep becomes a loop over `run()` calls plus
  `effectiveViscosity`, not a re-implementation of the step loop.
- **04, the analytic envelope.** The initial circle and the max-width/max-height
  ellipses overlaid on the droplet. Constants (`R`, and the `1.931843` aspect)
  belong next to the case's `R`/`A`/`B` params; export them from
  `oscillatingDroplet.py` rather than leaving a bare literal in a cell.
- **12, the reconstructed density field.** `computeDensities(...)` re-evaluated
  after the run and plotted — a free-surface check the live panels do not show.
- **13, one datastructure-inspection cell** (hash map / Verlet list sizes) if it
  still runs; it is the only cell in that notebook that teaches something the
  case cannot.

**Delete:**

- The 43-line boilerplate cell in all 13 → 4 lines of `bootstrap()` + imports.
- Every inline `visualize(...)`/`PlottingOptions` block (~50 lines each, 13 of
  them) → `buildFieldPlotter(ctx, runningState, FIELDS)`.
- The `nu_tests` sweep cell where it was copy-pasted without a purpose: 06, 07,
  08, 09, 10, 11, 12, 13 (67 lines each, identical, all ending in a
  `plotter.updateQuantities` call on a plotter from a different cell). It stays
  in 05 only.
- The trailing run of empty cells every notebook ends with (4–5 each).
- 13's `torch.profiler` cells, its commented-out alternate system builds, and
  the half-finished `# def ldcDirichlet` block pasted into 09/11/12/13.
- 08's dead forcing-prototype cells (`# forcingRegion = ...` and the two empty
  neighbours) — `kolmogorov.py` has the working forcing.

## Order of work

1. **The dam-break plot gap** (above). Unblocks 11/12/13 and is the only
   library change in the plan.
2. **Export `IMPACT_FIELDS`** from `impact.py` — one line, mirrors
   `HYDROSTATIC_FIELDS`.
3. **Slot 03 as the pilot.** Smallest single-variant case with a shared field
   list; it establishes this family's cell template (the `initialConditions`
   line, `range(nSteps)`, `VELOCITY_DENSITY_FIELDS`) that 04/05/08/09/10 then
   copy.
4. **01+02 → `01-impact/`**, the multi-variant pattern, following `06-sedov/`.
5. **04, 09, 10** — straight ports off the pilot. Rename `09-LDC.ipynb` to
   `09-lid-driven-cavity.ipynb`.
6. **05 and 08** — ports plus the analysis that stays (viscosity calibration;
   forcing).
7. **06+07 → `06-randomFlow/`**.
8. **11, 12** — first users of the converted dam-break plot path.
9. **13** — last. Expect it to be a rewrite rather than a port: 57 cells, most
   of them scratch, and `openFlowCase`'s defaults already encode the setup the
   notebook builds by hand.
10. **Sweep-up:** delete `utils.py`, add `EXAMPLES_SUMMARY.md` for this
    directory (the compressible one is the template), and flip the
    `CLEANUP_PLAN.md` item to note this family done and only
    `examples/incompressible/` remaining.

## Outputs and animations

There is no `examples/weaklyCompressible/outputs/` yet — the compressible family
keeps one `<slot>-<Case>.gif` / `.mp4` / `.png` per case there, referenced from
the intro cell as `![](outputs/<name>.gif)`. Create it and produce each
animation from the notebook's own `encodeFrames` call at the end of the run, at
the shipped default resolution, once the notebook is otherwise final. Ship the
notebooks with **no stored outputs**.

## Verification

Per case, in this order:

1. `python scripts/check_imports.py --static` — AST-scans notebook cells, which
   a runtime check never reaches.
2. `jupyter nbconvert --to notebook --execute` on a small-`nx`/short-`tLimit`
   copy. This proves the notebook runs; it proves nothing about the numbers
   (`PORTING_EXAMPLES.md` §5).
3. `python scripts/run_sweep.py --cases <name>` — unchanged case behaviour.
4. Look at one field plot deliberately, against what the case is supposed to do:
   density inside ±1% of `rho0` (that is what `weaklyCompressibleDiagnostics`'s
   `maxDensity`/`minDensity` are for), the vortex still a vortex, the dam break
   not exploding.

Whole-suite once the family is done: `bash scripts/run_tests.sh` and
`python scripts/run_sweep.py` with no `--cases`.

Note that `tests/test_physics.py` covers only `dambreak` out of this family's 10
cases. This migration does not change the cases, so no new tests are strictly
required — but if the dam-break plot conversion above is taken, run the
dam-break tests before and after and confirm they are untouched by it.
