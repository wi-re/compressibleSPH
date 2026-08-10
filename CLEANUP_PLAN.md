# warpSPH — Cleanup Plan

Working document for the cleanup sweep preceding forward-mode AD work.
Core and Integrators have already been overhauled; this repo (the former frontend,
`~/dev/warpSPHFrontend` → now `~/dev/warpSPH`) was the lagging piece.

**Status:** Phases 0, 1 and 2 **complete**. Phase 3 is **partly done**
(2026-08-10): `SchemeBundle` and the `compParams`/`schemeConfig` unification have
landed together with the notebook sweep they gated; **namespace hygiene and
`__all__` remain open**. Repo weight remains **deliberately deferred** — see
"Deferred: repo weight" — though its separable piece, the `nbstripout` filter, is
now installed.

---

## Decisions already made

- **Naming target — DONE as of 2026-08-07:** local dir == GitHub repo == import ==
  PyPI dist, for all four packages. The original plan deferred the PyPI axis
  (project names there can't be un-registered once taken), but it was renamed too
  for full consistency — the old mangled dist names below are retired/abandoned,
  not in use:

  | role | dir / GitHub / import / PyPI (unified) | old PyPI dist (retired) |
  |---|---|---|
  | core | `warpSPHCore` | — |
  | integrators | `warpSPHIntegrators` | `sphWarpIntegrators` |
  | plotting | `warpSPHPlotting` | `sphwarpSPHPlottingting` |
  | frontend (this repo) | `warpSPH` | — |

- **GitHub rename semantics:** renaming *does* free the old name immediately.
  The redirect it leaves behind dies the moment anything reuses that name.
  The core→frontend swap necessarily consumed it, so that redirect was lost by design.
  All four renames (`wi-re/warpSPH`→`wi-re/warpSPHCore`, `wi-re/compressibleSPH`→
  `wi-re/warpSPH`, `wi-re/integrators`→`wi-re/warpSPHIntegrators`,
  `wi-re/sphPlotting`→`wi-re/warpSPHPlotting`) are done and confirmed via
  `git remote -v` in all four local clones.

- **Accepted risk:** renames break the dill-encoded callables in existing `.h5`
  datasets (244 local files). Fine — they are regenerable and no large dataset has
  been built yet. This unblocks deleting the compat shims outright.

---

## Phase 0 — Manual (yours, in progress)

### 0.1 GitHub renames — DONE

- [x] Rename `wi-re/warpSPH` → `wi-re/warpSPHCore`
- [x] Rename `wi-re/compressibleSPH` → `wi-re/warpSPH` (kills the redirect from above)
- [x] Rename `wi-re/integrators` → `wi-re/warpSPHIntegrators`
- [x] Rename `wi-re/sphPlotting` → `wi-re/warpSPHPlotting`
- [x] Re-point local remotes in all four `~/dev/warpSPH*` clones
- [x] Note added to the `warpSPH` README calling out the former repo name/location
      (see README.md, top of file) — the redirect couldn't say this for you, and
      stale links would otherwise land on the wrong repo.

### 0.2 Import rename — `integrators` — DONE

`integrators` on PyPI is a **third-party package** (v0.1.1, Chaoming Wang,
*"backend-free numerical integration library for differential equations"*) — same
problem domain. Our top-level `integrators` module shadowed it. If anything ever pulled
it in transitively, imports would resolve by `sys.path` order. Silent wrong-package
import in a numerics stack, right before AD work.

- [x] Rename the top-level module in `~/dev/warpSPHIntegrators`
- [x] Update `from warpSPHIntegrators.integration import *` across this repo
      (verified: no remaining bare `from integrators` / `import integrators` imports
      anywhere in `src/`, `examples/`, or `datagen/`)

### 0.3 Deletions — DONE (2026-08-10)

Dead / duplicated, verified:

- [x] `src/warpSPH/configurations/presets.py/` — a **directory** named `presets.py`,
      containing two zero-byte files. Unreferenced.
- [x] `src/warpSPH/ml/` — deleted. The "revive or delete" question was moot: all four
      files are **100% commented out** (0 live non-comment, non-blank lines out of 2,108).
      There was no code to revive; git history holds the text if it's ever wanted.
- [x] `src/warpSPH/caseUtils/rayleighTaylor/` — removed the **whole directory**, not just
      `bcs.py`: it contained only that one file, byte-identical to the copy under
      `caseUtils/compressible/rayleighTaylor/`, with no `__init__.py` and no importers.
      (This also removes it from the Phase 1 missing-`__init__.py` list.)
- [x] `examples/13-openFlow.ipynb` — stale dup of `examples/weaklyCompressible/13-openFlow.ipynb`
- [x] `examples/weaklyCompressible/utils.py` — **reduced to a re-export shim**, not deleted:
      `12-dambreak.ipynb` and `13-openFlow.ipynb` still do `from utils import ...`.
      Now mirrors the datagen shim, pointing at `warpSPH.caseUtils`.
- [x] `examples/incompressible/dfsph.py`, `dfsph_step.py` — both zero bytes
- [x] `bak/` — 6 old notebooks, that's what git is for
- [x] `src/warpSPH/legacy_utils.py` + the `sys.modules.setdefault("utils", ...)` hack
      (plus the now-unused `import sys as _sys`).
      **The hack was actively harmful, not just dead:** it ran at `import warpSPH` time and
      registered `sys.modules["utils"]`, so every later `from utils import ...` in a notebook
      resolved to `legacy_utils` and *shadowed* the local `datagen/` and `examples/`
      `utils.py`. Removing it restores the local shims.
      Phase 3's warning that this was load-bearing for import ordering did **not**
      materialize — verified, all 49 packages still import (see Phase 1 notes).
- [x] Root clutter: `profile.json` (4 MB), `nu_eff_vs_nu.png` (196 KB),
      `examples/weaklyCompressible/warpSPH_trace.json` (3.5 MB),
      `datagen/weaklyCompressible/log.txt`, `dist/` (was in fact **untracked**, not
      committed — removed from disk only)

### 0.4 Doc links — DONE (2026-08-07)

- [x] `README.md` — clone block, all four URLs; title; package-layout section
      (was still `src/compressibleSPH/*`, imports, video link) — all updated to
      the unified `warpSPH` naming.
- [x] `boilerplate.md` — codebase/backend/timestepper URLs updated to the current
      repo names. This text gets pasted into published video descriptions.

---

## Phase 1 — Mechanical fixes — DONE except repo weight (2026-08-10)

Small, independent, unblocks Phase 2.

- [x] **`warpSPHPlotting` is not editable-installed.** — **was already fixed** by commit
      `bbdb8ad`; the plan's data predated it. All four packages are now editable-installed
      from `~/dev/*` at 0.5.0.
      Separately cleaned up three **stale distributions** left over from the rename:
      a real `sphWarpPlotting` 0.4.0 wheel that shipped a top-level `warpPlot` package
      (itself broken — it did `from sphWarpCore import *`, a name that no longer exists),
      plus stale `sphWarpCore` / `sphWarpIntegrators` egg-infos.
- [x] **Pin backend deps.** — **was already done**; `pyproject.toml:26-28` pins all three
      at `>=0.5.0`. No version skew remains.
- [x] **`configurationToDict` silently drops `nx` and `dx`.** Fixed — both
      `configurationToDict` and `dictToConfig` are now generated from
      `dataclasses.fields()`, so a new field can't be silently dropped again.
      **Two bugs, not one:** `dictToConfig` also *hard-crashed* on any config with
      `dt=None` (the dataclass default) via `float(None)` at the old line 174 — the
      round-trip was broken outright, not merely lossy.
      Two traps the rewrite had to avoid:
      - `targetNeighbors` is annotated `int` but legitimately holds a **float**
        (`n_h_to_nH(4,2) == 50.265…`), so decoding must **not** coerce by annotation.
        Only structural types (enum / device / dtype / domain) are converted.
      - `DomainDescription` was **two different classes** at the time — since resolved,
        see the Phase 3 note below.
      `buildConfig` now also accepts `nx`/`dx` directly, so the notebooks'
      post-hoc `config.nx = nx` is no longer required.
      Verified: all fields round-trip exactly, through both JSON and the HDF5 path;
      dicts written without `nx`/`dx` still load (fall back to dataclass defaults).
- [x] **`prepExport` hardcodes the output path.** Now takes an `exportRoot` argument,
      defaulting to `$WARPSPH_EXPORT_ROOT` and then to `export` — the previous
      behaviour is unchanged when neither is set. Also switched to `os.path.join`
      and a `with` block for the config write.
- [x] **Add the missing `__init__.py`** — 4, not 6: `ml/` and `caseUtils/rayleighTaylor/`
      were deleted in 0.3. Added to `utils/noiseFunctions/`, `utils/sdfFunctionality/`,
      `modules/sps/`, `caseUtils/waveEquation/`. Behaviour-neutral: all four are consumed
      via explicit submodule paths (`from .noiseFunctions.generator import …`), never
      `import *`. Verified all 4 are picked up by `find_packages` (49 packages total).
- [ ] **Repo weight — DEFERRED until after Phase 2 (decided 2026-08-10).** Not worth
      doing now; see "Deferred: repo weight" below for the reasoning and the measured
      numbers, which correct what this item originally claimed.

---

## Deferred: repo weight (revisit after Phase 2)

**Decision (2026-08-10): do not do this now. Do it once the scripts are runnable as
`.py`, the physics is verified, and there is a polished final set of files to publish
as release assets.**

Rationale: a history rewrite is a one-shot, disruptive operation, and the media it
would move is *output of code that is about to be rewritten*. Doing it now means
rewriting history around renders that Phase 2 will regenerate anyway — paying the
disruption twice and preserving artifacts that aren't the final ones. Once Phase 2
lands and the physics is confirmed, the polished outputs become release assets and the
rewrite happens once, against files worth keeping.

Measured 2026-08-10 — this **corrects** the original claim of "~380 MB of
*repeatedly-recommitted* `.mp4`/`.gif`":

| | measured |
|---|---|
| `.git` | 543 MB |
| `.mp4` + `.gif` | 30 files, 338 MB, **30 blob versions** — i.e. each committed exactly **once** |
| `.ipynb` | 42 notebooks, **312 blob versions**, 289 MB |

So the media is *not* recommitted churn; there is no duplication there to reclaim.
The actual history bloat is the **notebooks** (312 versions of 42 files), which is what
`nbstripout` addresses and which LFS-on-media would not touch at all.

Two further constraints found while scoping it:

- `git-lfs` is **not installed** (apt candidate 3.7.1-1); `git lfs install` is a
  prerequisite, as is a clean working tree.
- LFS *relocates* bytes rather than removing them. GitHub's free tier is 1 GB storage
  **and 1 GB/month bandwidth**; at 338 MB, roughly three fresh clones per month
  exhaust the bandwidth quota. A data pack would likely be needed.

Because `examples/compressible/outputs/` is regenerable render output, the stronger
option at that point is to **purge rather than migrate** — drop it from history, add a
gitignore rule, and attach the polished renders to a GitHub release. That reclaims the
338 MB outright and costs no LFS quota. Same rewrite mechanics, with
`git filter-repo --path examples/compressible/outputs --invert-paths` instead of
`git lfs migrate import`.

The `nbstripout` pre-commit hook is **separable and non-destructive** — it can be added
at any time without a history rewrite, and doing so early stops the notebook history
from growing further. **Installed 2026-08-10** (`nbstripout --install --attributes
.gitattributes`, so `.gitattributes` is committed but the `filter.nbstripout.*` git
config is local and each clone must re-run it).

Note what the clean filter actually does: the **working tree keeps its rendered
figures** (11.5 MB of the 12.7 MB of notebooks), while what git *stores* is stripped —
e.g. `02-Linear_Wave.ipynb` is 141 KB on disk and 13 KB as a blob. So the next commit
touching a notebook removes its outputs from version control without removing them
from the working copy. Only the 18 stale error tracebacks were deleted outright, by
decision; the figures were kept.

---

## Phase 2 — Examples → runnable scripts + first tests — DONE (2026-08-10)

- [x] Extract `warpSPH.runner`:
  - `bootstrap(precision, dim)` — **could not live in `warpSPH.runner`.**
    `warpSPHCore.type_config` resolves precision at *its own import*, and any
    `warpSPH.*` import pulls `warpSPHCore` in transitively, so by the time
    `warpSPH.runner` is importable the choice is already locked. It is therefore a
    top-level module, `src/warpSPHBootstrap.py`, mirroring how core already ships
    `warpSPHCore_config` via `py-modules`. It is the one thing a script may import
    first. Verified end to end: `warpsph-run sod --precision float64` really does
    run at `torch.float64`.
  - `CaseSpec` (`runner/caseSpec.py`) — dataclass over the union of the argparse
    surfaces from `parser.py` and `01-sod-shock-tube-1d.py`. Round-trips through
    JSON and YAML; case-specific knobs live in `params`. Precedence is
    CaseSpec defaults → case defaults → `--config` file → explicit CLI flags,
    which works because every generated flag defaults to `None`, so "not passed"
    stays distinguishable from "passed the default". Booleans get both `--x` and
    `--no-x` so a `true` in a config file is still overridable. Unknown keys raise
    rather than being silently dropped.
  - `run(case, spec)` (`runner/runner.py`) — step loop, CUDA-event timing,
    diagnostics accumulation, plot hook, both export modes (`states` = one file per
    stored step, the examples' pattern; `trajectory` = one growing `trajectory.h5`,
    the datagen pattern), NaN bail-out, ffmpeg encode. Returns a `RunResult` whose
    `series(key)` gives one diagnostic across the run.
  - each case is a `Case` (`runner/case.py`) of hooks over a `RunContext`:
    `configureScheme` / `buildSystem` / `initialConditions` / `diagnostics`
    (+ optional `setupPlot` / `updatePlot` / `extraData`).
  - `runner/media.py` — the ffmpeg block; degrades to a no-op when ffmpeg is absent
    rather than failing a run that already produced its frames.
  - `runner/cli.py` + `src/warpSPHRun.py` — `warpsph-run <case>` console script.

  **Found while extracting: the three step functions disagree on the name of their
  own config argument.** `compSPH_step`/`crkSPH_step` take `compParams`;
  `deltaSPH_step`/`dfsph_step` take `schemeConfig`. The integrator forwards
  `**kwargs` verbatim, so a caller that guesses wrong gets a `TypeError`. The runner
  introspects `signature(stepFunction)` instead of assuming — that one detail is what
  lets a single loop drive all three schemes. Worth unifying alongside `SchemeBundle`
  in Phase 3; the introspection is a bridge, not the destination.

- [x] Convert 2–3 cases as proof — all three done, one per family:
  - `warpSPH/cases/sod.py` (compressible, CompSPH)
  - `warpSPH/cases/tgv.py` (incompressible, DFSPH)
  - `warpSPH/cases/dambreak.py` (weakly compressible, deltaSPH)

  The three former entry points are now thin wrappers over the same case objects:
  `examples/compressible/01-sod-shock-tube-1d.py` (264 → 21 lines),
  `examples/incompressible/01-tgv-incomp.py` (259 → 21 lines), and
  `datagen/weaklyCompressible/generator.py` (256 → 76 lines, keeping only its
  dataset-specific timestamped naming and `compressed/` archival step).
  `datagen/weaklyCompressible/parser.py` was **deleted** — fully superseded by
  `dambreakCase.params`, and verified to have no remaining importers (`.py` or
  `.ipynb`). `datagen/weaklyCompressible/plot.py` moved into the package as
  `warpSPH/caseUtils/weaklyCompressiblePlot.py` with a re-export shim left behind,
  matching the `utils.py` shims from Phase 0.3. It is deliberately *not* in
  `caseUtils/__init__.py`'s star imports, so `from warpSPH.caseUtils import *` still
  does not drag in `warpSPHPlotting`.

  **Three latent breakages surfaced by the conversion**, all pre-existing:
  - `datagen/weaklyCompressible/generator.py` **crashed on its own defaults**:
    `parser.py` defaulted `--obstacleType` to `circle`, which is not a key
    `buildPresetObstacles` returns any more (the presets are now
    `circleBottom`/`circleMiddle`/`circleTop`, …). `presets.get(...)` returned
    `None` and the next line indexed it. The case defaults to `circleMiddle` and
    validates the key with a message listing the valid ones.
  - `examples/incompressible/01-tgv-incomp.py` called `solveIncompressible(...)`
    without the `dt` argument it has since grown — a `TypeError` on any run.
  - the same script imported the local zero-byte `dfsph.py` / `dfsph_step.py`
    deleted in Phase 0.3; the real step function comes from `buildScheme`. It also
    built a `regions` list that was never passed anywhere. Both dropped.

- [x] First tests — `tests/`, 20 tests, ~7 s warm (**previously zero tests in this
      repo**). `tests/conftest.py` bootstraps in `conftest` because that is the last
      point before pytest imports the test modules — the same ordering constraint as
      above.
  - `test_physics.py` — 20 steps per case at coarse resolution, asserting properties
    rather than golden numbers: Sod total-energy conservation (measured drift is
    **exactly 0** — CompSPH is energy-conserving by construction), Sod
    thermal→kinetic conversion, TGV monotone decay and decay rate vs the analytic
    `KE(t) = KE(0) e^(-4 ν k² t)`, dam-break density bounds and gravitational work.
  - `test_caseSpec.py` — serialization and the override precedence above. No GPU.
  - `test_runner.py` — case registry, enum resolution, and the `compParams` vs
    `schemeConfig` introspection, parameterized over all three schemes.

  **Physics note worth following up separately:** the measured TGV decay rate sits at
  **0.55–0.60× the analytic rate** and is *stable under refinement* — 0.605 at
  nx=32/20 steps, 0.564 at nx=32/50, 0.550 at nx=64/200. So it is not a
  discretisation error that refines away; the effective viscosity of this DFSPH
  viscous operator is roughly half the prescribed ν. The test therefore asserts a
  wide band (0.6 ± 45%), which catches viscosity being disconnected (rate → 0) or
  mis-scaled without pretending the 0.55 factor is understood. (The deleted
  `nu_eff_vs_nu.png` suggests this has been looked at before.)

- [~] Notebooks stay for exploration but import the same runner — **mechanism
      delivered, bulk conversion still not attempted.** `bootstrap()` and `run()` are
      both usable from a notebook and the README documents the pattern. The separate
      **notebook sweep** (2026-08-10, see Phase 3) brought all 41 notebooks onto the
      current APIs, but they still carry their own hand-rolled step loops rather than
      calling `run()`. Converting them to the runner remains open.

End state, as delivered:

```bash
warpsph-run tgv --config examples/sweeps/tgv_nu.yaml --nx 128
python -m warpSPH.cases.sod --plot --store      # same case, precision fixed at import
```

```python
from warpSPHBootstrap import bootstrap; bootstrap(precision='float64')
from warpSPH.runner import run
from warpSPH.cases.sod import sodCase
result = run(sodCase, nx=400, nSteps=100)
```

Example sweep files live in `examples/sweeps/`; `pyproject.toml` gained the
`warpsph-run` console script, the `py-modules` entry for the two top-level
modules, and a `[tool.pytest.ini_options]` block.

## Phase 3 — Structural (do before AD, since AD touches every scheme)

- [x] **`buildScheme` returned a bare 7-tuple — DONE 2026-08-10.** It now returns a
      frozen `SchemeBundle` dataclass (`schemes/builder.py`) with named fields, and the
      if/elif chain became a `{enum: factory}` table with case-insensitive string
      aliases. `SchemeBundle.__iter__` still yields the legacy 7 in order, pinned to
      `_LEGACY_TUPLE_ORDER` rather than to `dataclasses.fields()` — **that pinning is
      the point**: an 8th field (the AD tangent propagator) can be appended without
      shifting what any surviving positional unpack binds.
      `RunContext` now holds the bundle and exposes the seven names as read-through
      properties, so there is one source of truth.

- [x] **The `compParams` / `schemeConfig` split — DONE 2026-08-10.** All five step
      functions now name their config `schemeConfig`; `runner._schemeConfigKeyword`
      and its `signature()` introspection are deleted, and the loop passes
      `schemeConfig=` unconditionally. Two tests pin the invariant.
      **The rename had a second site the plan did not predict:** the integrator
      forwards the same `**kwargs` to `system.finalize`, and `CompSPHSystem.finalize`
      named the argument `compParams` in its signature — renaming only the step
      functions left every CompSPH run raising `TypeError: finalize() missing 1
      required positional argument: 'compParams'`. `modules/` keeps `compParams` as a
      local parameter name; those are called positionally or by their own keyword and
      are not reachable through the integrator's kwargs forwarding, so they were left
      alone deliberately.
- [x] **Two colliding `DomainDescription` classes — found and FIXED 2026-08-10.**
      `warpSPH.utils.domain` defined its own `DomainDescription` alongside
      `warpSPHCore.dataTypes.domain_t.DomainDescription`. In
      `configurations/simulationConfig.py` the core one won purely by import order
      (`from ..utils import *` line 6, then `from warpSPHCore import *` line 8), so
      `SimulationConfig.domain: DomainDescription` annotated a class the field never held
      and `isinstance` against the visible name silently failed.
      **Resolved by deleting the local definition** — `utils/domain.py` now imports the
      core type, which was the intended owner all along (the two were identical).
      `buildDomainDescription` returns it, and serialization binds it explicitly rather
      than depending on star-import order. This also fixes the annotations in
      `systems/baseState.py`, `systems/waveSystem.py`, and four `modules/` files, which
      now name the class they actually receive.
- [ ] **Namespace hygiene.** 342 `import *` statements. `__init__.py` re-exports
      everything from 10 subpackages into one flat namespace, and several `caseUtils`
      modules do `from warpSPH import *` — an absolute self-import of a
      partially-initialized package. It works *only* because `legacy_utils` is loaded
      at the very end of `__init__.py`. Deleting that shim in Phase 0.3 removes the
      fuse but also the thing currently holding the ordering together — **re-verify
      imports after that deletion.**
- [ ] Only ~30% of modules define `__all__`, so star imports drag in every
      transitively-imported name. Add `__all__` as modules get touched.

### Notebook sweep — DONE 2026-08-10

Run together with `SchemeBundle`, because the rename made it **mandatory rather than
optional**: once the step functions stopped accepting `compParams=`, every notebook
that passed it would have raised `TypeError` on its first step.

- [x] **35 positional `buildScheme` unpacks → named bundle access**, across 33
      notebooks plus `examples/compressible/01-sod-shock-tube-resume.py`. Downstream
      names (`fn`, `export_fn`, `import_fn`, `SimulationSystem`, …) are preserved, so
      only the binding line changed.
- [x] **17 `compParams=` → `schemeConfig=`** at the integrator call sites.
- [x] **A name-shadowing trap removed in 33 notebooks.** The old unpack bound the
      scheme's config class to the name `SimulationConfig` — the *same name* as the
      global simulation config that `from warpSPH import *` provides. Every
      `schemeConfig = SimulationConfig()` was silently reading the shadowed name.
      They now say `bundle.SimulationConfig()`, and `SimulationConfig` keeps its one
      meaning. `datagen/weaklyCompressible/obstacle_init.ipynb` is the deliberate
      exception: its helper *returns* `SimulationConfig` in a tuple, so the local
      binding stays.
- [x] **`from dfsph import *` deleted** from `01-taylor-green-vortex.ipynb` and
      `periodic-random-flow.ipynb`. It resolved to the zero-byte local `dfsph.py`
      removed in Phase 0.3, so it had **never** provided anything — `dfsph_step` was
      always coming from `warpSPH`'s star export. Those calls now use `fn` off the
      bundle, like every other notebook.
- [x] **18 stored error tracebacks removed** from 14 notebooks (several still quoting
      the pre-rename `~/dev/compressibleSPH/...` paths).
- [x] All 41 notebooks re-validated: JSON parses, every code cell compiles.

Deliberately **not** done, and still open:

- The notebooks keep their own hand-rolled step loops; none call `runner.run()` yet.
- `datagen/weaklyCompressible/bak/` — 5 tracked `.py` backups (114 KB) superseded by
  the Phase 2 runner conversion. Left untouched rather than migrated; by the same
  reasoning that retired the other `bak/` in Phase 0.3, these are **deletion
  candidates**, not sweep targets.

---

## Phase 4 — AD readiness audit

First-pass scan, not yet verified case by case:

- **97** `.item()` / `.cpu()` / `.numpy()` calls inside `schemes/`, `modules/`, `systems/`
- **106** `no_grad` / `detach` sites across `src/`

Each is a break in the tangent chain for forward-mode AD. Concentrated in:
`modules/timestep/compressible.py`, `modules/shifting/`, `modules/mdbc/`.

The timestep one matters most: adaptive `dt` depends on state, so `dt` itself carries a
tangent. Dropping it there is a **silent correctness bug, not a crash.**

- [ ] Dedicated audit of all 97 + 106 sites, classified as
      (a) genuinely non-differentiable / fine, (b) needs a differentiable path,
      (c) needs an explicit `stop_gradient` with a comment saying why

---

## Suggested order

Phase 0 ✅ → 1 ✅ → 2 ✅ → **3 (in progress — `SchemeBundle` ✅, notebook sweep ✅,
namespace hygiene + `__all__` remain)** → 4 → repo weight (deferred; its Phase 2
precondition is now met).

The original "load-bearing" shortlist is now **fully done**: editable plotting install,
`integrators` import rename, `nx`/`dx` round-trip, the duplicate `DomainDescription`,
and — as of 2026-08-10 — `SchemeBundle` together with the `compParams`/`schemeConfig`
unification.

What is left in Phase 3 is the namespace work (342 `import *`, `__all__` coverage). It
is the one remaining item that is *cheaper before* AD than after, but unlike
`SchemeBundle` it has no forcing function, so it can be sequenced against Phase 4 on
its merits. The notebook sweep removed one concrete instance of it — the
`SimulationConfig` shadowing — which is a reasonable model for the rest: fix the
shadowing at the binding site, leave the star imports alone until a module is touched.

Deliberately last: the repo-weight rewrite, so it operates on the polished Phase 2
files that are actually worth publishing rather than on soon-to-be-regenerated output.
Its one separable piece, the `nbstripout` hook, can be added at any time.
