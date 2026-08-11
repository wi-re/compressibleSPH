# warpSPH — Cleanup Plan

Working document for the cleanup sweep preceding forward-mode AD work.
Core and Integrators have already been overhauled; this repo (the former frontend,
`~/dev/warpSPHFrontend` → now `~/dev/warpSPH`) was the lagging piece.

**Status:** Phases 0, 1 and 2 **complete**; Phase 2's one remaining `[~]` item,
converting the examples to runnable scripts, is now **done in full** (2026-08-10)
— see "Phase 2b" below. Phase 3 is **partly done**: `SchemeBundle` and the
`compParams`/`schemeConfig` unification landed together with the notebook sweep
they gated, and the namespace work is now **measured and done for `schemes/`**
(see Phase 3) — what remains there is `__all__` coverage across the other 180
modules, reclassified as legibility rather than correctness. Repo weight remains
**deliberately deferred** — see "Deferred: repo weight" — though its separable
piece, the `nbstripout` filter, is now installed.

**Phase 3b** (2026-08-10) records the repair of the manual package reshuffle
(`io/`, `math/`, `sampling/`) and the backlog it left behind. The repair itself is
done and verified — including one *silent* tensor/scalar bug the move introduced —
and the repo now carries `scripts/check_imports.py`, `run_tests.sh` and
`run_sweep.py` to make that verification one command each. Its naming/structure
backlog is now **mostly cleared** (2026-08-11): the `sample`/`sampling` hazard,
the `domain.py` collision, and all three stutter modules (`math/math.py`,
`utils/util.py`, `io/io.py`) are fixed — see "Open: naming and structure" below for
what landed and what's deliberately still open. What remains overall is the dead-code
sweep and the notebook simplification, neither load-bearing.

**Phase 4** (2026-08-11, ongoing across two sessions) is the AD-readiness gradcheck
rollout. Tiers 0 and 1 are done; Tier 2 is partly done (`adaptiveSupport`, `deltaSPH`,
`shockCapturing`, `mdbc` gradchecked and wired in, 5 modules remain). The `.detach()`
audit (4.2) is done. A warp-lang-version regression found and **fixed same day**
(2026-08-11) — three previously-clean scripts (`wp_surfaceAware`, `compSPH`,
`dissipation`) failed under the currently-installed warp-lang 1.15.0; the same-array
ternary pattern Tier 0 had flagged as a risk was confirmed real and broader than
diagnosed (non-array-read ternaries were affected too), and is now closed with a new
`access_optional` helper (added to warpSPHCore) replacing every affected inline
ternary — including `modules/crk/{accel,dudt}.py`'s separate, older
`explicitPressure`/`individual_cs` gap, re-threaded the same day now that the helper
made it mechanical. Full suite: 53/53 pass. `pyproject.toml` still pins no warp-lang
version **by decision**: pin to 1.17 once it ships (fixes the bug at the source);
any version is fine until then as long as kernel-code ternaries go through
`access_optional`. See Phase 4 below, and the "Regression" section under Tier 2 for
the full writeup.

**Phase 4** (2026-08-11) is now **in progress**: a gradient-check plan for this repo's
25 custom warp-kernel files, modeled directly on warpSPHCore's own
`gradcheck_*_native.py` methodology, plus a completed first pass of the `.detach()`
audit. Its headline finding — adaptive `dt` losing its tangent in
`modules/timestep/compressible.py` — is now **fully fixed and verified**, including the
kernel-level half that needed a new warpSPHCore capability (`asScalarArg`) and, en route,
uncovered and fixed a real, unrelated, scheme-independent segfault in
`computeCompSPHBalanceTermWarp` (a missing query→reference fallback for energies/
pressures). **Tiers 0 and 1 of the gradcheck infrastructure are now built and checked**
(2026-08-11): Tier 0's `modules/pressure/wp_surfaceAware.py` gradchecks clean (caveat:
only verified against the currently-installed warp-lang 1.17.0.dev3, not the
pyproject-implied 1.12.0/1.16.0 where the analogous Interpolate bug is confirmed present);
`geometry/sdf.py` / `regions/domainSDF.py` gradchecked into finding and fixing a **real,
previously-latent crash** (`sampleSDF`/`sampleDomainSDF` raised `RuntimeError` the instant
a caller passed a position tensor that already required grad). Tier 1's
`modules/compSPH/*` and `modules/dissipation/*` gradcheck clean; `modules/crk/*` turned up
**two real bugs, both now fixed** — a self-interaction 0/0 in `modules/crk/limiter.py`'s
`computeVanLeer` (fixed here) and a deeper bug in warpSPHCore's `correctGradientCRK` (a
mis-contracted axis inside a hand-written accumulation loop, fixed upstream same day) that
had made CRKSPH, the production default scheme, not AD-correct with respect to position
until now. See Phase 4 below.

**Continued 2026-08-11 (second session):** three previously-written but
undocumented Tier 2 scripts (`adaptiveSupport`, `deltaSPH`, `shockCapturing` —
already gradchecking clean, or, for `shockCapturing`, correctly failing on a tracked
known bug) were wired into `tests/test_gradcheck_scripts.py` (was 45 tests, is now
49). A new `gradcheck_mdbc.py` found and fixed **three real bugs — two in
warpSPHCore's own math/autograd layer, one here** — just to get
`modules/mdbc/wp_nopenshift.py` running under gradient tracking at all (a missing
`vec1i` type that broke `zero_like`'s codegen, an unguarded float32 literal in a
float64 kernel, and the autograd bridge unconditionally setting `requires_grad` on
non-float kernel outputs), then gradchecks clean once the test fixture supplied real
densities instead of `None` (a fourth, adjacent NaN-adjoint hazard, documented not
fixed). **Running the full suite also surfaced a regression**: three previously
"clean" Tier 0/1 scripts (`wp_surfaceAware`, `compSPH`, `dissipation`) now fail
under the currently-installed warp-lang **1.15.0** — confirmed reproducible,
narrowed to the pressure-handling same-array ternary pattern, but **not yet
root-caused or fixed**; see "Regression" under Tier 2 below for the full writeup and
recommended next step. One adjacent real bug (an unguarded `referencePressures[j]`
read in `compSPH/{accel,dudt}.py`) was found and fixed along the way but does not
explain the regression. Tier 2 itself is otherwise partly done — see Tier 2 below
for what's left (`incompressible/wp_alpha.py`, `liu/wp_mat.py`,
`surfaceDetection/*`, `util/*`, `sample/wp_deltaShift.py`).

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

- **warp-lang version pin — deliberately deferred (decided 2026-08-11).**
  `pyproject.toml` will be pinned to **1.17** once that version ships — it fixes
  several issues relevant here (the Interpolate same-array-adjoint bug Tier 0 and
  the "Regression" section under Phase 4.1 both discuss is confirmed fixed there).
  Until then, **any warp-lang version is fine to develop against**; the constraint is
  to be careful with ternary (`X if cond else Y`) expressions inside `@wp.func`/
  `@wp.kernel` bodies, since some versions before 1.17 miscompile them (confirmed for
  1.15.0, see Phase 4.1's "Regression" writeup). The `access_optional` helper
  (warpSPHCore `util/stateUtil.py`) is the established workaround — use it instead of
  an inline ternary for any `arr[index] if cond else default`-shaped expression, and
  prefer explicit `if`/`else` blocks over inline ternaries generally in kernel code
  until the pin lands. Don't propose pinning to 1.15.0/1.16.0/1.12.0 as a "fix" for
  the version-drift risk — this is an intentional wait, not an oversight.

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

- [x] Notebooks stay for exploration but import the same runner — **mechanism
      delivered; the bulk conversion is Phase 2b below, done 2026-08-10.**
      (Text below is as written before that; it records what was true then.) `bootstrap()` and `run()` are
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

## Phase 2b — Every example as a runnable case — DONE (2026-08-10)

Phase 2 converted three cases as proof and left the rest as notebooks. This
finishes the job: **every notebook under `examples/` that runs a simulation is
now a case**, 25 in total, each with a thin `.py` wrapper next to its notebook.
`warpsph-run` with no case name lists them.

### Runner additions the conversion forced

- **`Case.postStep(ctx, state, step)`.** Kidder drives its two boundary bands
  from the analytic solution *after* each integrator step; there was no hook for
  "re-impose something the step function does not know about".
- **`Case.timestep(ctx, state) -> dt`.** Kidder, Woodward-Colella and the
  equal-mass triple point recompute the acoustic-CFL `dt` every step. The shared
  implementation is `cases.compressible.compressibleTimestep`.
- **The loop is time-bounded when a `timestep` hook is present.** `nSteps =
  tLimit / dt0` is wrong the moment `dt` moves, which is exactly what the hook
  is for, so those runs loop `while t < tLimit` as the notebooks did. Cases
  without the hook keep the old fixed-step behaviour, so nothing changed for
  them. `tests/test_runner.py` pins both halves.
- **Setup may replace the spec.** `run()` re-reads `ctx.spec` after
  `buildSystem`/`initialConditions`, because Kidder and Sedov only learn their
  time limit from the analytic solution (the collapse time; the time to reach
  `goalRadius`).
- **List/dict case params no longer generate CLI flags.** argparse inferred
  `float` from them; Woodward-Colella's shock regions and the dam break's
  gravity vector are `--config`-only now.

### Three shared modules, not fifteen copies

The notebooks were near-identical within each family, so the shared part is
factored out rather than duplicated:

- `cases/compressible.py` — CRKSPH + B7 + `gamma`/`rho0` + viscosity switch +
  Owen adaptive support, and the kinetic/thermal/total-energy diagnostics.
- `cases/weaklyCompressible.py` — the band-widened domain, the SDF region
  helpers, and `setupTimestep` (sound speed and `dt` chosen *together* from
  `targetDt`), plus density-bound diagnostics.
- `cases/plotting.py` — `particlePlot(fields)` for the `warpSPHPlotting`
  mosaics every 2D notebook built by hand, and `profilePlot(axes)` for the 1D
  scatter panels. **Backend default changed from `vispy` to `matplotlib`**:
  vispy needs a GL context, which a script over ssh or in CI does not have.
  `--plotBackend vispy` restores it.

### Live plots, and the backend that makes them affordable

The notebooks never had to display anything -- the notebook frontend does that
for you. A console script does, so a straight port built the figure, wrote its
PNGs and **opened no window at all**. `runner/display.py` is that missing piece:
`openWindow` (matplotlib interactive mode + `show`), `pumpEvents` (a `draw_idle`
alone never repaints inside a tight step loop), `holdWindow` and `closeWindow`.
It lives in `runner/` rather than `cases/` because the runner has to tear a
figure down at the end of a run and must not import from `warpSPH.cases`;
`cases/plotting.py` re-exports the names.

**Backend is chosen by dimension: vispy for 2D, matplotlib for 1D.** Measured on
a 16k-particle Kelvin-Helmholtz run, 6 frames: matplotlib spent **19.97 s** in
plotting against 1.65 s of physics -- the observer costing an order of magnitude
more than the thing observed. vispy brings that to **3.97 s**, of which the
redraw itself is 0.08 s and the rest is the 300-dpi PNG export. `--plotBackend`
overrides; a vispy canvas that cannot start falls back to matplotlib with a
message rather than killing the run (verified with `DISPLAY` unset).

**Both backends leak without an explicit close**, and this was found the hard
way -- a sweep left 20+ windows open. matplotlib figures are registered with
pyplot's global manager by `subplots`/`subplot_mosaic`; a vispy `SceneCanvas`
holds a live GL window. `run()` now tears the handle down in `_teardownPlot`.
`caseMain` sets `holdPlot=True` so a person at a terminal keeps the final frame
on screen, while programmatic `run()` leaves it off so a sweep never stalls.

`warpSPH.cases.tgv` gave up its hand-rolled matplotlib scatter for the shared
`particlePlot` in the process, so it benefits like every other 2D case, and
`caseUtils.weaklyCompressiblePlot.setupPlotter` took a `backend` argument in
place of its hardcoded `'vispy'` (default unchanged, so notebooks are unaffected).

### What a run prints

Added after the live-plot work, for the case that motivates scripts over
notebooks in the first place: a long run nobody is sitting in front of.

- **Banner**, printed once setup is done and `dt` is final, so it shows what
  will run rather than what was asked for — resolved dt, derived step count,
  particle count, domain, and the output paths.
- **Report**, printed at the end: finished-or-DIVERGED, wall time, step-time
  statistics, every diagnostic as initial/final/min/max, and the files actually
  written (counted from disk, not inferred from the configured intervals — a
  run that stopped early would otherwise be credited with output it never
  produced). A diverged run also exits non-zero, so a shell script can tell.
- **`--quiet` / `-q`** suppresses all three of banner, report and progress bar,
  and drops warp's per-module load logging to `LOG_WARNING` for the duration
  (restored afterwards — it is a third-party global).
- **`progress` became tri-state** (`None` = auto). A tqdm bar redirected to a
  file writes a carriage-return smear that buries the report the run exists to
  produce, so it is on only when `sys.stderr.isatty()`.

Lives in `runner/report.py`, and replaces the old `_describe` that only ran
under `--verbose`.

### README rewritten

The old README was organised as a reference dump -- seven sections of bare enum
listings before anything about running a simulation -- and had gone stale in two
ways that mattered: it described the package as compressible-only (it is three
families now) and its Quick Start showed the **pre-`SchemeBundle` 7-tuple
unpack**, which is both wrong and the exact `SimulationConfig` shadowing trap the
notebook sweep removed.

Rewritten around what someone actually does with it: install, run a case, the
runner (CLI, config precedence, output layout, plots, the banner/report), the
case tables linking each `.py` script, **how to write a new case** (hook table
and a worked skeleton), the Python/notebook API, then the enum reference, layout,
tests and gallery. Every relative link is checked, every documented case name is
checked against the registry (25/25), every documented hook against
`dataclasses.fields(Case)`, every flag against `CaseSpec`, and both Python
snippets and the case-listing shell snippet were executed.

**One bug fell out of writing it.** The README claims `--quiet` makes a run
silent; it did not. `systems/incompressible.py` carried two leftover debug
`print`s in `finalize` -- the same lines are commented out at the two sibling
sites -- that fired on *every step of every DFSPH run*. They are now gated on
`verbose`, and the runner forwards `spec.verbose` to the integrator instead of a
hardcoded `False`, so `--verbose` reaches a scheme's own reporting and `--quiet`
is finally true.

### Cases that cover more than one notebook

Where two notebooks differed only in a flag they became one case, and the
example scripts pin the flag:

| case | notebooks | what differed |
|---|---|---|
| `sedov` | 06, 07 | `--dim 1` vs `--dim 2` |
| `triplePoint` | 14, 15 | `--equalMass` (equal particle mass) vs `--no-equalMass` |
| `impact` | wc 01, 02 | `--shape circle` vs `--shape box` |
| `squarePatch` | wc 03, incomp 03 | `--scheme deltaSPH` vs `--scheme divergenceFree` |
| `randomFlow` | wc 06, 07, incomp periodic | `--bounded`, `--scheme` |
| `openFlow` / `drivenSquare` | wc 13, 11 | dam-break hooks under channel defaults |

`openFlow` and `drivenSquare` deserve a note: both notebooks hand-wrote the
obstacle SDF, the Dirichlet band and the inflow ramp that
`caseUtils.weaklyCompressible` already implements for the dam break — 13 had
even started importing `buildObstacleSDF` from there. They are the dam break's
hooks under different defaults rather than a third copy.

### Latent breakages surfaced, as in Phase 2

- **Both Sedov notebooks crashed on their own default.** They ask for
  `initialization='hat'`, and `buildSedov` raises `NotImplementedError` for it.
  The dead code below that raise calls `warpKernelToDiffSPHKernel` /
  `diffSPHKernel`, names left over from the pre-warp diffSPH stack that exist
  nowhere in the tree. The case defaults to `'singular'`, which runs; reviving
  `'hat'` is a separate job and is **not** done.
- **`--store` had never worked for the incompressible family.**
  `io.exportSimulationSystem` name-checked only `CompressibleSPHScheme` and
  `WeaklyCompressibleSPHScheme`, so an `IncompressibleSPHScheme` member reached
  `attrs` as a Python object and h5py rejected it with "Object dtype has no
  native HDF5 equivalent". Three sites disagreed about this — one already had
  the third enum. Unified into `io.schemeAttribute`, and
  `schemeNameToSimulationScheme` learned to read the names back.
- **`RigidBody.toDict` assumed tensors.** The moving-obstacle notebook sets
  `body.angularVelocity = 1.0`, a plain float, so exporting the config threw
  `'float' object has no attribute 'detach'`. `bodyID` was a second instance:
  annotated `int`, populated from a tensor, so it arrived as a numpy `int32`
  that `json.dump` refused. Both now go through one converter.

### Verification

- All 25 cases run 3 steps at coarse resolution, and again with `--plot
  --store`, writing frames and HDF5. No failures. The plot pass is run **one
  case per process, sequentially**, so each window is gone before the next
  starts and a leak cannot hide behind another case's; every case is checked
  for leftover matplotlib figures and live vispy canvases, and all report zero.
- The headless path is verified with `DISPLAY`/`WAYLAND_DISPLAY` unset: vispy
  falls back to matplotlib, frames are still written, nothing leaks.
- `pytest` is 32 tests (was 20); the new ones pin the registry/`CASE_MODULES`
  agreement, that every case names a resolvable scheme, that params stay
  serialisable, the two new hooks, and the banner/report/quiet behaviour.

### Still open

- The notebooks keep their own hand-rolled step loops; none call `runner.run()`
  yet. That was true after Phase 2 and is still true — the *scripts* are the
  supported path now, and the notebooks are for exploration.
- `examples/weaklyCompressible/naca.ipynb` has no case: it is a standalone
  NACA-airfoil SDF visualisation with no simulation in it.
- `examples/incompressible/1d-test.ipynb` has no case: an exploratory 1D DFSPH
  scratchpad, not a published example.
- `datagen/weaklyCompressible/bak/` — still the deletion candidates flagged in
  the notebook sweep.

### The TGV effective-viscosity note is resolved, not open

Phase 2 flagged the measured TGV decay rate sitting at 0.55–0.60x the analytic
rate and stable under refinement, and left it as "worth following up". It is
**expected SPH behaviour**: the diffusion operator carries a **Monaghan switch**
that deactivates viscosity for particle pairs that are *separating*, so only the
approaching half of the pairs dissipates at any instant and the effective
viscosity is roughly half the prescribed `nu`. Disabling the switch does recover
the analytic decay rate, but causes problems in other aspects of the simulation,
so it stays on. The wide band in `tests/test_physics.py` is therefore correct as
written — it catches viscosity being disconnected or mis-scaled without treating
the ~0.55 factor as an error to drive out. Recorded in the test's docstring and
in `cases/tgvWeaklyCompressible.py`.

---

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
- [~] **Namespace hygiene — measured, re-scoped, and `schemes/` done (2026-08-10).**
      The original framing here was **wrong in its risk assessment**, and measuring it
      is what showed that:

      - **The `legacy_utils` fuse never blew.** All 274 modules import cleanly after the
        Phase 0.3 deletion, including the `caseUtils` modules that do `from warpSPH
        import *`. The re-verification this item demanded is done; nothing was holding
        the ordering together but itself.
      - **The top-level namespace is already clean.** `warpSPH` exposes 233 public
        names, `__all__` covers **all** of them (zero leakage), zero shadow builtins,
        and of the 9 names exported by more than one subpackage
        (`DomainDescription`, `DiffusionParameters`, `getPeriodicPositions`, …)
        **all resolve to the same object**. There is no collision left to fix — the
        `SimulationConfig` shadowing the notebook sweep removed was the only real one.
      - **So this is a legibility job, not a correctness one**, and the "cheaper before
        AD" argument is weaker than it looked. What remains is *interior*: modules that
        star-import and define no `__all__`, so they re-export everything they imported.

      **`schemes/` converted to explicit imports (2026-08-10)** — the subtree AD
      actually lands on. All 7 files (`compSPH`, `crkSPH`, `deltaSPH`, `dfsph`,
      `monaghan`, `waveEquation`, `builder`) plus `schemes/__init__.py`: every
      `import *` replaced by explicit per-subpackage imports, and each file given an
      `__all__`. Each scheme turned out to use only **10–17** of the 68 names
      `warpSPH.modules` exports, so the blocks are 8–14 lines.

      Names were bound **by object identity against the live module**, not by guessing
      an owner, so star-import resolution order is preserved exactly. Verified: the
      `warpSPH` public API is **byte-identical** before and after (0 names added, 0
      removed), 32/32 tests pass, and all 25 cases still run.

      Effect: `warpSPH.schemes` went from **334 public names to 14**. The
      `modules/` interior flattening was deliberately **left alone** — its leaf
      `__init__.py`s all declare real `__all__`s, it exports only 68 names for 21
      physics subpackages, and its names already carry their subpackage
      (`computeMdbcDensity`, `computeCompSPHAccelWarp`, …), so hierarchical *call
      sites* would stutter without adding information. The hierarchy belongs in the
      import block, not at the call site.

### Latent breakages surfaced by the `schemes/` conversion

Same pattern as Phases 2 and 2b: making the implicit explicit is what exposes them.

- [x] **`buildScheme(divergenceFree)` was one star import away from breaking.**
      `builder._divergenceFree` imported `incompressibleConfigToDict` and
      `dictToIncompressibleSPHConfig` **from `.dfsph`** — but those are defined in
      `..configurations`, and `dfsph` only relayed them as a side effect of
      `from warpSPH.configurations import *`. Removing that star import would have
      broken every DFSPH run. Now imported from `..configurations`, where they live.
      **Fixed**, since the conversion required it.

- [x] **The Monaghan scheme was broken and could not run — FIXED 2026-08-10.**
      **Two separate rots, not one**, both from signature changes that never reached
      `monaghan.py`:

      1. All three boundary-condition helpers grew a `t` parameter —
         `enforceDirichlet(system, t, dt, …)`, `computeForcing(system, dt, t, …)`,
         `enforceUpdates(updates, system, dt, t, …)` — and `monaghan.py` still called
         all three with the pre-`t` argument list. Hard `TypeError` on the first step.
         It now binds `t = currentSystem.t`, exactly as `compSPH.py` does.
      2. Behind that, `computeMomentumConsistent` was called with
         `supportScheme=SupportScheme.Gather`, a keyword the function **no longer
         takes** — its `supportMode` is now hardcoded to `SuperSymmetric` internally,
         so the argument was not renamed, it was removed. The surviving slot is
         `schemeConfig` (unused in the body, so this is arity-only).

      **The same dead `supportScheme=` keyword was fixed at two more sites**:
      `compSPH.py:75` and `crkSPH.py:86`. Both sit behind `if currentState.divergence
      is None:` — the "computing for the first time" branch — so they were latent
      rather than firing, but they would have raised the identical `TypeError` the
      moment that branch was taken.

- [x] **`crkSPH.py:338` passed `t` and `dt` to `computeForcing` in the wrong order —
      FIXED 2026-08-10.** The signature is `computeForcing(system, dt, t, config,
      compParams)`; `compSPH`, `deltaSPH` and `dfsph` all passed `dt` then `t`,
      `crkSPH` passed `t` then `dt`. Both are floats, so it never crashed — the
      swapped values reached every user forcing function as `(…, t, dt)`. Silently
      wrong, and only when a boundary condition defines `forcingFunctions`, which is
      why the CRKSPH cases always smoke-tested clean. All five schemes now agree.

### Scheme comparison from the command line — DONE (2026-08-10)

The point of fixing Monaghan was to be able to compare the three compressible
solvers on the same case.

- [x] **All three compressible solvers run every compressible case.** Verified as a
      13-case × 3-solver matrix (`sod`, `linearWave`, `noh`, `sedov`, `gresho`, `yee`,
      `kidder`, `woodwardColella`, `triplePoint`, `kelvinHelmholtz`, `rayleighTaylor`,
      `shearingNoh`, `hydrostatic` × `CompSPH`, `CRKSPH`, `Monaghan`). `--scheme`
      already reached `buildScheme`; what was missing was a working Monaghan.
- [x] **The banner names the solver, not just the scheme enum.** It printed
      `scheme  Monaghan (CompressibleSPHScheme)`; a comparison run is judged on which
      *step function* ran, so it now reads
      `scheme  Monaghan (CompressibleSPHScheme) | solver compressibleSPH_Monaghan`.
      Suppressed by `--quiet` along with the rest of the banner, as before.
- [x] **`--help` stopped lying about defaults.** `buildArgumentParser` reported the
      generic `CaseSpec` value for every flag — it told you Sod ran a `Wendland2`
      kernel when the case sets `B7`, and that `--scheme` defaulted to `None`. It now
      takes the case's resolved defaults for the help text (the flags still default to
      `None`, which is what keeps the override precedence working), and `--scheme`
      lists the valid solver names.
- [x] **Tests: 42, was 32.** Sod now runs under all three solvers, asserting
      non-divergence, thermal→kinetic conversion, and a per-solver energy-drift bound
      (CompSPH measures **exactly 0**, CRKSPH 4.5e-6, Monaghan 9.4e-4 — Monaghan is
      not an energy-conserving discretisation, so its bound is loose on purpose). One
      more test pins that `--scheme` actually reaches `buildScheme`, so a comparison
      run cannot silently compare a scheme against itself. **This is the coverage gap
      that let Monaghan rot unnoticed.**

- [ ] **Remaining `__all__` coverage.** 94/274 files (34%, was 88/274); **180 modules**
      still define none and star-export everything they imported. Worst offenders are
      all in `caseUtils/compressible/`: `linearWave.wave` (447 public names),
      `rayleighTaylor.sample` (383), `kidder.sample` (375), `yeeVortex.sample` (374),
      both `triplePoint` modules (374). 300 `import *` remain in `src/` (was 336; the
      plan's original "342" counted the whole tree). Add `__all__` as modules get
      touched — the `schemes/` conversion above is the worked pattern.

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

## Phase 3b — Post-reshuffle repair and backlog (2026-08-10)

Follows the manual package reshuffle that split `util.py`/`io.py`/`dataset.py` and
`utils/{math,naca,noise,sampling,scatter,sdf}` into the new `io/`, `math/` and
`sampling/` packages.

### Repaired — DONE (2026-08-10)

Eight breaks, found by the new `scripts/check_imports.py`:

| break | fix |
|---|---|
| `caseUtils/waveEquation/casefile.py` imported `n_h_to_nH` from `...sampling` | merged into the existing `...utils` import |
| 3 notebooks importing `warpSPH.utils.naca` | → `warpSPH.sampling.naca` |
| 2 notebooks importing `warpSPH.caseUtils.sedov` | → `warpSPH.caseUtils.compressible.sedov` |
| 2 docstrings citing pre-move paths | updated |

**One was a silent runtime bug, not an import error.** The reshuffle replaced
`volumeToSupportHelper` (tensor-aware) with warpSPHCore's `volumeToSupport`, which
uses `math.sqrt` and raises on a multi-element tensor. It killed `yee` outright and
left `modules/adaptiveSupport/optimalSupportMonaghan.py:13` **latently broken for 2D
and 3D** — `computeH` passes a per-particle `V = m / rho`, and only the `dim == 1`
branch avoids the sqrt. Nothing caught it because `Owen` is the default scheme.

`utils/support.py` now wraps core with a tensor-aware dispatch; tensor and scalar
paths verified identical in all three dimensions.

- [ ] Upstream the tensor support into `warpSPHCore.util.support.volumeToSupport`,
      then collapse the wrapper back to a plain re-export. Deliberately **not** done
      here — it is a separate repo, and this is the kind of cross-repo change that
      should be made once, on purpose.

### Export folders are timestamped — DONE (2026-08-10)

`export/<caseName>_<YYYY-MM-DD_HH-MM-SS>/`, applied at the single choke point in
`prepExport`, which already computed the timestamp and simply was not using it.
Same-second collisions take a `-1` suffix. Added `latestExportPath()` /
`findExportRuns()` so readers need not know a run's exact name; both fall back to the
old flat layout, so existing trees still resolve. `WARPSPH_EXPORT_TIMESTAMP=0` opts
out. The Sod resume notebook and script now resolve the newest run instead of
hardcoding `export/01-sodShockTube`.

- [ ] `datagen/weaklyCompressible/loader.ipynb` still hardcodes read paths
      (`export/semiPeriodic/`, …). Existing data loads; **newly generated data will
      not be found there.** Left alone pending a decision on the dataset workflow.

### Three scripts — DONE (2026-08-10)

```bash
scripts/check_imports.py     # 273 modules + 1535 first-party imports
scripts/run_tests.sh         # 42 tests
scripts/run_sweep.py         # 25/25 cases, ~3.5 min smoke
```

`check_imports.py` runs two passes: a real import of every module under `warpSPH`,
then an AST scan of every `.py` and notebook code cell that checks module *and*
symbol. The second pass is what caught the notebook and function-level imports — the
runtime pass alone never executes them. `run_sweep.py` runs one case per process,
sequentially (a case must tear down before the next starts, and one crash must not
take the sweep with it); anything after `--` forwards to every case, so the configs
in `examples/sweeps/` compose with it.

This replaces the hand-rolled `for case in $(warpsph-run …)` loop previously
documented in the README.

### Open: naming and structure

None of these are correctness issues — they are the legibility cost of the reshuffle,
and worth clearing before AD makes tracebacks harder to read.

- [x] **`sample/` vs `sampling/` hazard — FIXED 2026-08-11.** `sampling/` (what a
      sampler is defined in terms of: SDF, NACA, `ParticleSet`/`PointCloud`,
      `SamplingScheme`) is renamed to **`geometry/`**, so it no longer reads as a
      near-duplicate of `sample/` (the samplers themselves). `sampling/enumTypes.py`
      is renamed to **`geometry/types.py`** as part of the same move — it holds
      NamedTuples plus one enum, not the 9 solver enums that live in the unrelated
      top-level `warpSPH/enumTypes.py`, and `types.py` says that. `sample/sampling.py`
      — the file whose own name collided with the package it now no longer sits next
      to — is renamed to **`sample/bySamplingScheme.py`**, which is what it actually
      does: dispatch `sampleParticles` over `config.samplingScheme`
      (regular/jittered/random/optimal/glass). 18 importers across `src/`, 3 notebooks
      and the README's package-layout table were updated to match; `git mv` preserved
      history on all three renames.
- [x] **`regions/domain.py` vs `utils/domain.py` — FIXED 2026-08-11**, as the one
      instance of the 20-duplicate-basenames item with an unambiguous fix: the two
      files don't share a purpose (one builds a `DomainDescription`, the other treats
      the domain as an SDF for region carving), so `regions/domain.py` is renamed to
      **`regions/domainSDF.py`**, matching the two functions it actually defines
      (`domainSDF`, `sampleDomainSDF`). The remaining basename collisions
      (`sod.py`, `noh.py`, `region.py`, `sdf.py`, the `caseUtils/compressible/*/sample.py`
      family) are each case- or package-namespaced on purpose — same pattern as
      `caseUtils/compressible/<case>/<case>.py` — and reads fine in context, so they're
      **not** being churned; only the two pairs above were actually ambiguous.
- [x] **Stutter modules — FIXED 2026-08-11.** `math/math.py` (9 lines) and
      `utils/util.py` (15 lines) are inlined into their package `__init__.py` and
      deleted; 6 external importers of `math.math` repointed to `math` directly. One
      latent bug fell out of the `math/math.py` inlining: `math/noiseFunctions/generator.py`
      imported `from ..math import getPeriodicPositions`, which — from two packages
      down — resolved to the *old* `math/math.py` submodule, not the `math` package;
      once that submodule was gone this became a `ModuleNotFoundError`, fixed to
      `from .. import getPeriodicPositions`. `io/io.py` (667 lines, was the largest
      stutter module and the one with the least self-evident split) is broken into
      **`io/hdf5.py`** (the dump/load primitives, dtype conversion, dict↔h5
      serialization — everything `.export` and `.importIO` share), **`io/export.py`**
      (`schemeAttribute`, `exportSimulationSystem`, `writeInitialData`, `writeFrame`,
      `prepExport`, the run-folder helpers), **`io/importIO.py`** (the inverse:
      `schemeNameToSimulationScheme`, `importSimulationSystem`, `importConfigs` — named
      `importIO` rather than `import.py` since the latter is a reserved word), and
      **`io/parsers.py`** (the seven CLI string→enum parsers, unrelated to HDF5 at all).
      `io/__init__.py`'s public surface — the actual contract other modules rely on —
      is unchanged: same names, same `__all__`, just re-pointed to the new submodules.
      6 external importers (`warpSPH/__init__.py`, `runner/runner.py`, `io/dataset.py`,
      2 `datagen/weaklyCompressible/*.py` files, 1 notebook) updated to the new paths.
      Verified beyond `check_imports.py`/`run_tests.sh`: a live `sod --store` run
      round-tripped through `exportSimulationSystem`/`writeInitialData`/`writeFrame`
      on the way out and `importSimulationSystem`/`importConfigs` on the way back in.
- [ ] **`math/` shadows a stdlib name.** Harmless under absolute imports, but it costs
      a beat of thought at every reading. Not renamed — no concrete replacement name
      was ever proposed, and inventing one now is scope the naming pass didn't ask for.
- [ ] **Dead code left by the move.** The `utils/__init__.py` commented-out imports
      *of the old `sampling`/`math.math` paths* were removed as part of the two fixes
      above (they'd have been actively misleading otherwise). The wider sweep is still
      open: `schemes/crkSPH.py` (119 commented code lines),
      `modules/mdbc/wp_nopenshift.py` (69), `shockCapturing/CullenDehnen2010.py` (63),
      `schemes/dfsph.py` (58), `caseUtils/compressible/sod/sod.py` (57),
      `schemes/deltaSPH.py` (55).

### Open: simplifying the examples

The `.py` examples are already thin wrappers — Phase 2b did that. **The notebooks are
the remaining fat: 13,112 lines of code across 34 notebooks, against 42 KB total for
33 scripts.**

- [ ] **The boilerplate cell is duplicated verbatim in 16 notebooks** — and it still
      uses the older `warpSPHCore_config.configure(...)` path rather than
      `warpSPHBootstrap.bootstrap()`. ~20 lines × 16 collapsing to two, and it puts
      notebooks and scripts on one bootstrap story. **Cheapest win; do this first.**
- [ ] **34 notebooks for 25 cases**, most duplicating a `.py` sibling. A notebook that
      only re-derives what `warpsph-run <case>` already does could become the same thin
      wrapper the scripts are, keeping bespoke cells only for the analysis and plotting
      that is genuinely notebook-shaped.
- [ ] **Worst offenders first**: `13-openFlow.ipynb` (1178 LOC, 3.6 MB),
      `12-dambreak.ipynb` (720), and the three incompressible notebooks (~1950 combined).
- Committed media under `examples/compressible/outputs/` is **not** listed here as a
  separate item — it is the same 338 MB already covered by "Deferred: repo weight",
  and the reasoning there still holds.

Suggested order: bootstrap cell (mechanical, 16 files, immediate) → collapse the
duplicate notebooks → media, with the repo-weight rewrite.

---

## Phase 4 — AD readiness audit

First-pass scan (2026-08-10), not yet verified case by case at the time:

- **97** `.item()` / `.cpu()` / `.numpy()` calls inside `schemes/`, `modules/`, `systems/`
- **106** `no_grad` / `detach` sites across `src/` (2026-08-11 recount: 89 literal
  `.detach(` sites, 0 `no_grad` — the original 106 likely double-counted chained
  `.detach().cpu().item()` calls or was scoped slightly differently; not worth
  reconciling exactly, see 4.2 below for the actual per-site count that matters)

Each is a break in the tangent chain for forward-mode AD. Concentrated in:
`modules/timestep/compressible.py`, `modules/shifting/`, `modules/mdbc/`.

The timestep one matters most: adaptive `dt` depends on state, so `dt` itself carries a
tangent. Dropping it there is a **silent correctness bug, not a crash.** — **confirmed
and refined in 4.2 below**, with one important nuance: the naive fix (just stop
detaching) is not sufficient on its own.

### 4.1 — Gradient-check plan for this repo's custom warp kernels (2026-08-11)

warpSPHCore (`~/dev/warpSPH`) already has a mature answer to "how do we know a custom
warp kernel's backward pass is right," built the hard way — three real, silent bugs
(reentrancy in the AD bridge, a ternary that zeroed an adjoint, a multi-output
`isinstance` check that missed tuples) were each invisible to forward-only checks and
only surfaced once `torch.autograd.gradcheck` was run directly against the kernel. None
of that machinery exists in this repo yet, and this repo has its own 25 files of custom
`@wp.kernel`/`@wp.func` code — the scheme-specific physics (compSPH, CRKSPH, deltaSPH,
dissipation, pressure, mdbc, shifting, surface detection, ...) built *on top of*
warpSPHCore's already-gradchecked operators (Density, Gradient, ...) via the same
structured-kernel ABI. That layer is exactly as capable of hiding the same bug classes,
and currently has zero coverage of its own.

**What warpSPHCore's methodology actually is** (`tests/operations/test_gradcheck_scripts.py`,
`scripts/gradcheck_*_native.py`, `scripts/_gradcheck_common.py`,
`.claude/skills/gradcheck/SKILL.md`):

- One standalone script per operator (`scripts/gradcheck_<op>_native.py`), each calling
  `torch.autograd.gradcheck` **directly against the real entry point** (no manual
  Jacobian, no per-call cloning workaround) under forced `warpSPHCore_PRECISION=float64`
  — gradcheck's numerical Jacobian needs the precision, and float32 isn't enough headroom.
- Scripts run as **subprocesses**, not in-process pytest functions, because precision is
  baked into every `@wp.kernel`/`@wp.func` at first `warpSPHCore` import and cannot
  change mid-process — importing two scripts into one pytest process would have the
  second one silently reuse the first's precision. `tests/operations/test_gradcheck_scripts.py`
  just shells out to each script and asserts exit code 0.
- Shared fixtures live in one `_gradcheck_common.py`: `make_domain`, `single_particle_case`
  (isolates the self-interaction term — a symmetric kernel's gradient at `r=0` must be
  exactly zero, so any nonzero value here is unambiguously a bug), `line_case` (small,
  regular, so self- vs. non-self gradient contributions are easy to separate by hand),
  `grid_case_2d`, `build_adjacency` / `build_grid_adjacency` (adjacency is treated as
  frozen/non-differentiable and built once from **detached** positions — this is the one
  place `.detach()` inside a gradcheck script is *correct by design*: rebuilding the
  neighbor list under finite-difference perturbation would introduce a genuine
  discontinuity right at the support-radius boundary, which is a modeling choice, not a
  bug being hidden).
- New scripts get registered in one `GRADCHECK_SCRIPTS` list; that's the only wiring
  needed for CI to pick them up.
- The one gotcha worth carrying over verbatim: **a ternary assigned to a local variable,
  where both branches index the same Warp array, can compile fine, run the correct
  branch at runtime, and still produce a silently-zero adjoint for that array read** —
  confirmed the exact cause of Interpolate's bug, confirmed fixed upstream in
  `warp-lang` 1.17.0.dev3, **not yet fixed in the 1.12.0 this repo (and warpSPHCore) is
  pinned to**. A ternary is only dangerous when both branches read the *same* array
  (different-array ternaries, e.g. `mj/rhoj if ... else referenceVolumes[j]`, are a
  confirmed non-issue per warpSPHCore's own lessons-learned).

**Recipe for this repo**, following that pattern rather than reinventing it:

1. Add `scripts/_gradcheck_common.py` here. Don't duplicate warpSPHCore's helpers —
   import `make_domain`/`line_case`/`grid_case_2d`/`build_adjacency`/`build_grid_adjacency`
   from `warpSPHCore` (or vendor the handful of lines if they're not exported publicly;
   check first) since this repo's `ParticleState`/`DomainDescription` *are* warpSPHCore's.
   Add repo-local helpers only for what core's fixtures don't cover: this repo's
   per-scheme state objects (`CompSPHState`, `CRKState`, ...) with density/pressure/
   internal-energy/support populated, and minimal `SimulationConfig`/`SchemeConfig`
   construction. **This is not optional** — confirmed the hard way while chasing the
   `computeCompSPHBalanceTermWarp` segfault below: a bare `ParticleState` (core's own
   fixture shape) passes through this repo's scheme-level kernels with no error and a
   silent memory-safety failure, because those kernels need `internalEnergies`/
   `pressures`/velocity-derived fields the bare struct doesn't carry and nothing
   validates are present. Force `warpSPHCore_PRECISION=float64` via `os.environ.setdefault` the
   same way, before any `warpSPH`/`warpSPHCore` import — and note this repo has its own
   precision-locking entry point, `warpSPHBootstrap.bootstrap()`; a gradcheck script
   should set the env var directly rather than going through `bootstrap()`, matching
   how warpSPHCore's own scripts bypass any higher-level config helper for the same
   "must happen before the first heavy import" reason.
2. One `scripts/gradcheck_<module>.py` per kernel-bearing module, or per group that
   always fires together in a real step (e.g. compSPH's accel+dudt+balance, since
   `compSPH_step` always calls all three) — call `torch.autograd.gradcheck` straight
   against the module's public `compute*Warp` entry point, `eps=1e-6, atol=1e-5`, float64,
   single-particle + line-of-N cases, every differentiable-flag combination the module
   exposes (grad-h on/off, CRK on/off, renormalization on/off — exactly the surface that
   already hid the `t`/`dt` argument-swap and `supportScheme=` keyword bugs Phase 3
   found by hand).
3. New file `tests/test_gradcheck_scripts.py`, subprocess-per-script against
   `GRADCHECK_SCRIPTS`, same rationale and same shape as warpSPHCore's — this repo's
   `check_imports.py`/`run_tests.sh` pattern already established `scripts/` as the
   right home for standalone verification tooling.
4. Once 2-3 scripts exist, write a repo-local `gradcheck` skill (or extend an existing
   one) mirroring warpSPHCore's `.claude/skills/gradcheck/SKILL.md`, so adding the next
   one follows a documented recipe instead of re-deriving it.

**Rollout order** — inventory of all 25 files, by priority:

*Tier 0 — start here, already-flagged risk (see 4.2's findings). DONE 2026-08-11:*
- [x] **Infrastructure built.** `scripts/_gradcheck_common.py` (vendored from
      warpSPHCore's own `scripts/_gradcheck_common.py` — not importable, since that file
      lives under warpSPHCore's `scripts/`, not its installed package — plus a
      `compute_densities` repo-local addition), `scripts/gradcheck_wp_surfaceAware.py`,
      `scripts/gradcheck_sdf.py`, and `tests/test_gradcheck_scripts.py` (subprocess-per-
      script, same rationale as warpSPHCore's: this repo's own `tests/conftest.py`
      already locks the main pytest process to `float32` via `bootstrap()` at collection
      time, before a gradcheck script could ever request `float64`, so in-process was
      never an option here either). `scripts/gradcheck_scalarArg_dt.py` (Phase 4.2) is
      now wired into the same test file — it existed since 2026-08-11 but had no
      automated coverage until now. 45 tests total (was 42).
- [x] **`modules/pressure/wp_surfaceAware.py` — gradchecks clean, with a caveat worth
      flagging rather than treating as closed.** The exact same-array ternary shape
      that broke Interpolate upstream is present (`P_j = referencePressures[j] if
      referencePressures.shape[0] > 1 else referencePressures[0]`, and its query-side
      and `mask_j`/`mask_i` twins) and confirmed live via `deltaSPH` and `monaghan`
      (`computePressureForceSurfaceAware`). `gradcheck_wp_surfaceAware.py` runs
      `torch.autograd.gradcheck` against `computePressureSurfaceAwareWarp` across all 6
      `PressureForceScheme` values, each with both a per-particle pressure array (safe
      branch) and a broadcast one-element array (the suspect branch, since that's what
      actually exercises `referencePressures[0]` via the ternary) — **all pass**.
      **But**: this environment's `warp` conda env currently has warp-lang **1.17.0.dev3**
      installed (from a local checkout, per Phase 4.2's own note that this drifted
      mid-session), not the pyproject-implied 1.12.0/1.16.0 — and 1.17.0.dev3 is the
      version the Interpolate bug is confirmed **fixed** in. So this result shows the
      ternary is safe on the dev build actually running here, not that it's safe on the
      version this repo is nominally pinned to. Re-running
      `python scripts/gradcheck_wp_surfaceAware.py` under an actual 1.12.0/1.16.0
      environment is the outstanding step before calling this file's risk fully closed —
      deliberately not done here, since swapping the shared conda env's warp-lang version
      is exactly the kind of cross-cutting environment change that shouldn't happen as a
      side effect of checking one file.
- [x] **`geometry/sdf.py` / `regions/domainSDF.py` — gradchecked, found and fixed a real
      bug, not just a hypothetical risk.** Both `sampleSDF` and `sampleDomainSDF` build a
      working tensor via `x_ = x.clone()` with no `.detach()`, then unconditionally do
      `x_.requires_grad = True`. When the caller's `x` already requires grad — exactly
      the case forward-mode AD needs, since it's how a position tensor carrying a tangent
      would reach an SDF-based boundary/region term — `x.clone()` is a **non-leaf**
      tensor (it inherits `requires_grad=True` already), and PyTorch raises
      `RuntimeError: you can only change requires_grad flags of leaf variables` the
      instant that flag is touched, even though the value isn't changing. No caller in
      this repo currently passes a `requires_grad=True` `x` (see
      `caseUtils/weaklyCompressible.py`'s call sites), so this was latent rather than
      live — same shape as the `computeCompSPHBalanceTermWarp` segfault Phase 4.2 found:
      invisible until AD actually threads a tangent through the path. **Fixed** in both
      files by guarding the assignment (`if not x_.requires_grad: x_.requires_grad =
      True`) — correct in both branches, since `x_` is only a fresh leaf when `x` itself
      didn't require grad; verified separately that the guarded version still correctly
      chains second-order gradients back to the original tensor. `gradcheck_sdf.py`
      covers both functions, both `invert` values, a hand-rolled circle SDF (isolating
      `sampleSDF`'s own logic) and the real `getSDF('circle')` `torch.vmap`-wrapped path
      — a crash-regression check, a detached-output check for the `requires_grad=False`
      branch, and a full `gradcheck` of the `(d, grad)` output pair against `x` for the
      `requires_grad=True` branch. All pass, no warp-lang version caveat here (pure
      PyTorch, no Warp kernel involved).

*Tier 1 — core scheme physics, exercised by every compressible/CRK run. DONE 2026-08-11,
mixed result — two real bugs found, one fixed in this repo, one open upstream:*
`modules/compSPH/{accel,dudt,balance}.py`, `modules/crk/{accel,dudt}.py` (largest and
most-corrected kernels — CRK, grad-h, renormalization, volume all compose here, and this
exact family already had two silent-wrong-argument bugs found by hand in Phase 3),
`modules/dissipation/{wp_diffusion,wp_conductivity,wp_dissipation}.py` (the viscosity/
conduction terms — also where the TGV effective-viscosity finding lives, so calibrating
that switch via AD later needs this to be right first).

- [x] **`_gradcheck_common.py` grew two repo-local fixtures.** `make_compressible_state`
      builds a real `CompressibleState` (`systems/compressibleMonaghan.py` — same required
      fields as `CompSPHState`, and already proven to work standalone by
      `scripts/troubleshoot_balanceTerm_segfault.py`) from independent differentiable
      leaves, with `pressures`/`soundspeeds`/`alphas` populated (every real compSPH/CRK/
      Monaghan run populates them). `compute_crk_state` wraps warpSPHCore's
      `computeCRKFactors` on detached positions, frozen/re-leafed exactly like
      `compute_densities` — that op's own backward is warpSPHCore's
      `gradcheck_crk_native.py`'s job, not this repo's.
- [x] **`modules/compSPH/{accel,dudt,balance}.py` — gradchecks clean.**
      `scripts/gradcheck_compSPH.py` checks `computeCompSPHAccelWarp` and
      `computeCompSPHdudtWarp` against positions/supports/masses/densities/velocities/
      internalEnergies/pressures/soundspeeds/alphas, and `computeCompSPHBalanceTermWarp`
      against the same plus `ap_ij`/`av_ij` (treated as independent leaves, not chained
      through accel's own output — operator-level testing, consistent with every other
      script here) across all 6 `EnergyScheme` values. All pass. `dt` stays a plain float
      throughout — its differentiability is `gradcheck_scalarArg_dt.py`'s job (Phase 4.2),
      not duplicated here.
- [x] **`modules/dissipation/{wp_diffusion,wp_conductivity,wp_dissipation}.py` —
      gradchecks clean.** `scripts/gradcheck_dissipation.py` checks
      `computeViscosityWarp`/`computeConductivityWarp`/`computeThermalDissipationWarp` at
      their own Monaghan-shaped entry points (not just indirectly via compSPH's shared
      `computePi_actual`). All pass.
- [x] **`modules/crk/{accel,dudt}.py` — found two real bugs, both now fixed (one here,
      one upstream in warpSPHCore).** `scripts/gradcheck_crk.py`:
      1. **Fixed here.** `modules/crk/limiter.py`'s `computeVanLeer` divided by
         `sgn(grad_j)*abs(grad_j)` (and the symmetric term) unconditionally, patching a
         resulting NaN *value* afterward (`if ri != ri: ri = 1.0`). Every adjacency list
         here includes the self-interaction pair (`x_ij == 0` — confirmed by direct
         inspection: 5 of 19 edges in the 5-particle test line are `i==j`), which drives
         `grad_i == grad_j == 0`, a genuine 0/0. Forward-safe (the NaN is overwritten) but
         not backward-safe: reverse-mode AD differentiates the expression that was
         evaluated, not the value it was replaced with, so the singular local derivative
         (`1/0` in the division's own backward formula) poisons the adjoint regardless of
         the later overwrite — the same underlying class of bug as the ternary-adjoint
         issue Tier 0 chased, in a different guise (a post-hoc NaN patch instead of a
         same-array ternary). Fixed by guarding the division itself with an `if/else`
         *before* it happens, preserving the exact "no flow → limiter is 1" intent.
      2. **Fixed upstream, in warpSPHCore, same day.** With the limiter bug fixed (and
         separately, with both limiters disabled to rule them out entirely), gradcheck
         still failed: a finite but *wrong* Jacobian, not a NaN. Isolated by elimination —
         zeroing viscosity (`C_l=C_q=0`, `alphas=0`, `velocities=0`) still failed,
         narrowing it to `pressureTerm_ij`'s `gradw_ij`, i.e. `modules/crk/accel.py`'s two
         `computeKernelGradientCRK` calls. Further isolated by swapping in an identity CRK
         correction (`A=1, B=0, gradA=0, gradB=0`, which reduces
         `warpSPHCore.crk.kernel.correctGradientCRK` to the plain kernel gradient): that
         passed, pointing at `correctGradientCRK`'s handling of a nonzero `B`/`gradB` — the
         one part of the formula this repo's isolation couldn't get further than without
         editing warpSPHCore directly. **Root cause, confirmed by the fix itself**:
         `term4` contracted `gradBi` against `x_ij` via an explicit
         `for row / for col: product[row] += x_ij[col] * gradBi[row, col]` loop that
         contracted the wrong axis — using a loop-accumulated value nonlinearly within the
         same function, a known Warp-AD bug shape (related to but distinct from the
         "reentrancy in the AD bridge" class already in warpSPHCore's lessons-learned: not
         reentrancy here, but reverse-mode AD through a hand-written index-accumulation
         loop silently producing a wrong adjoint). **Fixed upstream** by replacing the loop
         with a single `matmul(wp.transpose(gradBi), x_ij)` — confirmed
         `scripts/gradcheck_crk.py` now passes clean, both `computeCrkSPHAccelWarp` and
         `computeCrkSPHdudtWarp`. Before this landed, **CRKSPH — the production default
         scheme (`schemeConfig.energyScheme = EnergyScheme.CRK`) — was not AD-correct with
         respect to position**; it is now. Same cross-repo shape as Phase 3b's
         `volumeToSupport` and Phase 4.2's `asScalarArg` piece 2, both likewise resolved by
         a dedicated warpSPHCore-side follow-up rather than worked around here.
      `gradcheck_crk.py` is wired into `GRADCHECK_SCRIPTS` in
      `tests/test_gradcheck_scripts.py` alongside `gradcheck_compSPH.py` and
      `gradcheck_dissipation.py`. 48 tests total (was 45 before Tier 1).

*Tier 2 — scheme-specific correction terms, live but narrower blast radius. Partly
done (2026-08-11): `adaptiveSupport`, `deltaSPH`, `shockCapturing`, `mdbc` are
gradchecked and wired into the suite; `incompressible/wp_alpha.py`, `liu/wp_mat.py`,
`surfaceDetection/*`, `util/*`, `sample/wp_deltaShift.py` remain.*

- [x] **`modules/adaptiveSupport/{wp_omega,wp_psi0}.py` — gradchecks clean.**
      `scripts/gradcheck_adaptiveSupport.py` checks `computeOmegaWarp` (grad-h
      correction) and `computePsi0Warp` (Owen adaptive-support reference spacing) —
      both share the family's usual same-array-safe apparent-volume ternary and take
      only positions/supports/masses/densities. No bugs found.
- [x] **`modules/deltaSPH/wp_viscosityDelta.py`, `modules/shockCapturing/wp_computeM.py`
      — gradcheck clean; `modules/shockCapturing/wp_vsig.py` — one bug fixed, one
      confirmed and left open (upstream warp-lang, not fixable here).**
      `scripts/gradcheck_deltaSPH.py` and `scripts/gradcheck_shockCapturing.py`.
      `computeVsigWarp`'s signal-velocity computation surfaced:
      1. **Fixed.** `computeVsigWarp` had a `referenceVelocities = queryVelocities`
         fallback but no equivalent `referenceCs = queryCs` one, so a caller
         supplying `queryCs=` explicitly (any bare-`ParticleState` gradcheck case, and
         in practice every real call site too, just via a `.soundspeeds` attribute
         happening to live on the same object) fell through to a **size-1 dummy
         tensor read out of bounds** whenever `referenceParticles` lacked
         `.soundspeeds` — silently reading stray memory. Turned out **not unique to
         vsig**: the identical missing-fallback gap (`referenceCs`/`referenceAlphas`/
         `referencePressures` never defaulting to their `query*` counterparts) was
         present in 7 more files sharing this parameter family —
         `modules/{compSPH,crk}/{accel,dudt}.py` and
         `modules/dissipation/{wp_conductivity,wp_diffusion,wp_dissipation}.py` — all
         previously gradchecked "clean" in Tiers 0/1 only because those scripts always
         populate `.soundspeeds`/`.alphas`/`.pressures` on the *same* state object
         passed as both query and reference. Fixed identically in all 8 files, plus an
         unrelated `torch.scalar_t32` typo (not a real torch dtype — should have been
         `get_torch_precision()`) in the same dummy-tensor line, present in the same 8
         files.
      2. **Confirmed, not fixed — needs an upstream warp-lang fix.** With the memory
         bug out of the way, `computeVsigWarp`'s backward is *deterministically wrong*
         for 2 of 3 particles in a hand-picked, tie-free case (verified 3 independent
         ways: `gradcheck`, `torch.autograd.grad` cross-checked per output, and a
         from-scratch manual central-difference replay — the manual replay agrees with
         hand calculus and disagrees with `torch.autograd`'s result). Root cause:
         `out = wp.max(out, vsigs)`, a loop-carried variable reassigned via a
         nonlinear op inside the neighbor loop — same underlying bug *class* as Tier
         1's `correctGradientCRK` finding ("reverse-mode AD through a loop silently
         produces a wrong adjoint"), different shape: here the adjoint sometimes
         attributes the gradient to the wrong neighbor or drops it to zero, even
         though the forward value is correct. Swapping in the logically-equivalent
         `wp.where(vsigs > out, vsigs, out)` made no difference, confirming the bug is
         in how Warp differentiates the loop-carried reassignment itself, not in
         `wp.max`'s adjoint. `gradcheck_shockCapturing.py` runs gradcheck against this
         case and treats "fails with exactly this mismatch" as the expected, tracked
         state — a regression guard, not a script failure, so a future Warp upgrade
         that changes this either direction will show up as a script-behavior change
         rather than staying silently stale.
- [x] **`modules/mdbc/wp_nopenshift.py` — three real bugs found and fixed (getting
      the kernel to run under gradient tracking at all), then gradchecks clean.**
      `scripts/gradcheck_mdbc.py` is the first script in this family to exercise a
      *multi-output* kernel (a float correction array plus an int32 neighbor-count
      array) under `requires_grad`, and every one of the three bugs it found was in
      that combination specifically, not in this module's own physics:
      1. **Fixed upstream, in warpSPHCore.** `zero_like_warp` on the int32
         neighbor-count output crashed Warp's own compiler
         (`AttributeError: 'Var' object has no attribute 'is_builtin'`).
         `warpSPHCore/math/wp_zero.py`'s length-1 int32 `zero_like` overload was the
         **one dtype** still building its zero via the generic call form
         `vector(length=1, dtype=wp.int32)(0)` — every float16/32/64 sibling instead
         returns a concrete pre-declared class (`vec1f(0.0)`, etc., from
         `wp_vec1.py`), because no `vec1i` class existed for int32 to use the same
         pattern. Fixed by adding `vec1i` to `wp_vec1.py` and pointing the overload at
         it — the int32 case now follows the same pattern as every other dtype
         instead of being the one exception that used a form this warp-lang version
         (1.15.0) can't codegen.
      2. **Fixed here.** `norm_j = safe_sqrt(...) + 1e-12` — a bare Python float
         literal, which Warp infers as `wp.float32` regardless of the active
         precision, added to a `scalar_t` (float64 under gradcheck) value: a genuine
         type mismatch, caught only because this is the first time this file has run
         at float64. Every other `+ eps` site in the same file already wraps the
         literal in `scalar_t(...)`; this was the one straggler. Fixed the same way.
      3. **Fixed upstream, in warpSPHCore — the significant one.**
         `warpSPHCore/autograd/launcher.py`'s `launch_kernel` unconditionally set
         `output.requires_grad = requires_grad` on **every** output of a multi-output
         kernel the moment any input required grad, with no check that the output's
         own dtype can legally carry a gradient — so `wp.to_torch` on the int32
         neighbor-count output raised `RuntimeError: only Tensors of floating point
         and complex dtype can require gradients`. Every gradcheck script before this
         one only ever exercised single-output-float kernels, so this is the first
         time a mixed-dtype multi-output launch ran under `requires_grad` in this repo
         at all — a structural gap in the bridge itself, same shape as the plan's
         other cross-repo AD-bridge findings (`asScalarArg`, the CRK
         accumulation-axis bug). Fixed by gating `requires_grad` on the output's own
         dtype (new `_dtype_is_float` helper, checking `_wp_scalar_type_` for
         vector/matrix dtypes) — purely additive: every existing single-float-output
         kernel is unaffected, only a non-float output now correctly stays
         `requires_grad=False` instead of crashing.

      With the kernel finally running under gradients, one more real hazard turned
      up — not fixed in source, worked around in the test fixture instead:
      `computeMdbcNoPenShift_Func_i` computes `apparentVolume = mj / rhoj`
      unconditionally per neighbor even though it is dead for this call
      (`useVolume` is always False here, so the value is never read downstream). The
      script's first attempt passed `densities=None` on a bare `ParticleState`,
      which defaults to a zero-filled array, so `rhoj = 0` and
      `apparentVolume = inf` — forward-dead, but reverse-mode AD still built an
      adjoint through the division, and that NaN propagated back into every mass
      gradient for the early-returned boundary-particle threads. Same bug *class* as
      Tier 1's CRK-limiter self-interaction 0/0 (a computed-but-discarded expression
      whose singularity poisons the backward pass anyway) in a different guise
      (division by an absent field, not a same-array ternary or loop-carried max).
      Worked around by giving the case real, nonzero densities via
      `compute_densities` (this family's usual fixture) instead of `None`; left as a
      documented trap in the script for whoever next touches `useVolume` plumbing in
      this file, since any caller that omits densities on a bare `ParticleState` —
      legitimate per every other gradcheck script's own `hasattr`-fallback precedent
      — hits this same NaN the instant gradients are requested through this path.
- [ ] `modules/incompressible/wp_alpha.py`, `modules/liu/wp_mat.py`,
      `modules/surfaceDetection/{wp_barecasco,wp_dilate,wp_maronne}.py`,
      `modules/util/{wp_sum,wp_numNeighbors}.py`, `sample/wp_deltaShift.py` — not yet
      started.

*Not in scope — verified non-differentiable by construction, checked 2026-08-11:*
`modules/adaptiveSupport/wp_psi.py` is the **one file of the 25 using a raw `wp.launch`
instead of `warpWrapper2`** — i.e., it has no backward pass at all, the same shape as
the pre-fix `pinv2x2` gap warpSPHCore closed. But its only inputs are Python
`int`/`float` LUT parameters and a kernel enum, not simulation-state tensors, and it
runs once to build a lookup table rather than per-step — nothing differentiable ever
flows through it. Not a gap; worth a one-line comment at its `wp.launch` call saying so,
so a future reader doesn't assume it's an oversight matching its `wp_omega`/`wp_psi0`
siblings, which *do* go through the bridge.

### Regression, found and FIXED same day (2026-08-11): the same-array ternary bug
was real, broader than diagnosed, and is now closed with a proper helper

Running the full suite while wiring in Tier 2's new scripts (`scripts/run_tests.sh`)
first surfaced that **`gradcheck_wp_surfaceAware.py`, `gradcheck_compSPH.py`, and
`gradcheck_dissipation.py` — all three previously documented as "gradchecks clean"
in Tier 0/1 above — failed**, reproducibly, standalone, under the currently-installed
warp-lang **1.15.0** (`pyproject.toml` pins no version at all — bare `"warp-lang"` —
which is how the installed version has now silently drifted three times across this
plan's history: 1.12.0 → a 1.17.0.dev3 local dev checkout → 1.15.0 from PyPI). This
was exactly the risk Tier 0's own `wp_surfaceAware` caveat had flagged and left open.

**Root cause, confirmed by the fix: broader than the "same-array ternary" diagnosis.**
The failing cases all traced to `P_j = referencePressures[j] if
referencePressures.shape[0] > 1 else referencePressures[0]`-shaped code — but the
fix that actually closed it (see below) also had to touch several ternaries that are
**not** same-array reads at all (e.g. `xi = sphKernel_xi(...) if
viscosityParams.correctXi else scalar_t(1.0)`, a function-call-result-vs-literal
ternary in `dissipation/pi.py`) — so "different-array ternaries are a confirmed
non-issue," the distinction Tier 0 relied on to call `referenceCs[j] if individual_cs
else viscosityParams.c_s`-shaped code safe, does not fully hold under warp-lang
1.15.0's codegen. The safe conclusion is narrower: **any inline Python conditional
expression (`X if cond else Y`) compiled inside a `@wp.func`/`@wp.kernel` is suspect
on this warp-lang version**, not just the same-array-read subset previously
suspected.

**Fixed.** A generic helper, `access_optional(arr, index, condition, defaultValue)`
— `return arr[index] if condition else defaultValue`, but written as a real
`@wp.func` with an explicit `if`/`else` block rather than compiled from an inline
ternary at each call site — was added to **warpSPHCore** (`util/stateUtil.py`,
exported from `util/__init__.py`) since the pattern recurs across every module in
this family, not just this repo's. Every affected same-array/array-vs-default
ternary was rewritten to call it:
`modules/pressure/wp_surfaceAware.py` (`P_j`/`P_i`/`mask_j`/`mask_i`, the original
flagged risk), `modules/compSPH/{accel,dudt}.py`, `modules/dissipation/{pi,
wp_conductivity,wp_diffusion,wp_dissipation}.py`, and `modules/mdbc/
wp_nopenshift.py`'s`apparentVolume` line — all of the `X[j] if flag else Y`-shaped
sites this family shares. The handful of non-array-read ternaries in
`dissipation/pi.py` (`xi`, `C_l_`/`C_q_`, `viscosityTerm`) were rewritten as explicit
pre-declared-variable-plus-`if` blocks instead, since `access_optional` only fits the
array-read shape. One unrelated typo was fixed in the same pass:
`DiffusionParameters.thermalConducitiyTerm` → `thermalConductivityTerm` (4 files:
the dataclass definition and its to/from-dict round-trip), caught because it sat
right next to the ternary being rewritten.

**Verified, twice.** First, `gradcheck_wp_surfaceAware.py`/`gradcheck_compSPH.py`/
`gradcheck_dissipation.py` individually: all three now report `ALL PASSED` (`wp_
surfaceAware` explicitly re-checks the broadcast-pressure ternary branch and
confirms no adjoint-zeroing). Second, the full suite (`scripts/run_tests.sh`):
**53/53 pass**, zero failures anywhere — up from the 3 failures this section
originally reported.

**One inconsistency found and fixed while verifying:** `modules/compSPH/accel.py`'s
own `Pj = referencePressures[j]` (an `explicitPressure`-guard fix from earlier the
same day) had been overwritten back to an unconditional read somewhere in the sweep
above, while `dudt.py`'s equivalent line was correctly converted to
`access_optional`. Fixed to match: `Pj = access_optional(referencePressures, j,
explicitPressure, scalar_t(0.0))`.

**`modules/crk/{accel,dudt}.py`'s missing `explicitPressure`/`individual_cs` guards —
FIXED same day (2026-08-11), once `access_optional` existed to make it mechanical.**
Both files previously read `P_j = referencePressures[j]` and `cs_j =
referenceCs[j]` **unconditionally**, with no guard at all — the commented-out dead
code above the live `warpWrapper2` call (`#         explicitPressure, queryPressures_,
referencePressures_,`) showed the flag used to be threaded through and was dropped at
some point without the reads being re-guarded, structurally preventing CRK's accel/
dudt from supporting an implicit-pressure caller at all (unlike compSPH's version of
the same functions). Re-threaded `individual_cs`/`explicitPressure` through
`_Func_i` → `_Func_Adjacency` → `_Kernel` → the top-level wrapper in both files,
mirroring `compSPH/{accel,dudt}.py`'s exact pattern (flag derivation, parameter
placement, `access_optional` guards) and removing the two `raise ValueError` calls
that previously fired only when `.pressures`/`.soundspeeds` were an explicit `None`
attribute, not when they were simply absent (the actual gap). One dead line fell out
of the sweep: `dudt.py`'s `Pj = referencePressures[j]` was unconditional *and*
unused — immediately overwritten by `Pj = P_j` before its first read — deleted
rather than converted. Verified: `scripts/gradcheck_crk.py` still `ALL PASSED`, a
real 3-step CRKSPH run (`python -m warpSPH.cases.sod --scheme CRKSPH --nSteps 3`)
completes cleanly, and the full suite passes.

**Warp-lang version pin: deliberately deferred, not an open risk to chase.** Per
the "Decisions already made" section at the top of this file: pin to **1.17** once
it ships (fixes the underlying Interpolate/ternary-adjoint bug at the source); any
version is fine to develop against until then, so long as ternaries inside
`@wp.func`/`@wp.kernel` bodies go through `access_optional` or an explicit `if`/
`else` rather than an inline conditional expression.

### 4.2 — `.detach()` audit (started 2026-08-11)

Went through all 89 literal `.detach(` call sites in `src/warpSPH`, grouped by file,
classified against the (a)/(b)/(c) scheme this phase set out with. Result: the large
majority — `io/export.py` (26), `configurations/rigidBody.py` (13),
`configurations/{region,weaklyCompressible,simulationConfig,incompressible}.py`,
every `cases/*.py` diagnostics function (`kineticEnergy`/`maxVelocity`/... dicts),
`cases/plotting.py`, `regions/plot.py`, `runner/{runner,report}.py` — are **(a),
genuinely fine**: HDF5/JSON export, `matplotlib` plotting, and human-readable reporting
are real non-differentiable boundaries, and `.detach().cpu().item()`/`.numpy()` is the
correct way to cross them. No action needed on any of these.

Three sites are not:

- **`modules/timestep/compressible.py:61`, `return initial_dt.cpu().item()` — (b),
  needs a differentiable path, and the fix is not just "stop detaching."**
  `initial_dt = torch.clamp(torch.min(targetCFL * h / (c_s * xi)), ...)` is a real
  function of `system.state.densities` (through `idealGasEOS`'s sound speed) when
  `config.adaptiveDt` is set, so the CFL-derived timestep genuinely carries a tangent
  back to density and hence to position. `.cpu().item()` collapses it to a plain Python
  float, and every step function's signature (`compSPH_step(system, dt: float, ...)`)
  declares `dt` as a plain float throughout — so `d(dt)/d(density)` is dropped before
  `dt` is even passed in, not just at some later detach.

  Traced where that tensor would need to survive to matter: it flows into pure-PyTorch
  arithmetic inside the step functions (`v_halfstep = velocities + 0.5 * dt * dvdt`) and
  into the integrator's own position/velocity update (`warpSPHIntegrators`) — both would
  correctly backprop through a tensor `dt` with zero further changes, if one reached
  them. But `dt` *also* reaches warp kernels directly as a scalar argument — e.g.
  `computeCompSPHBalanceTermWarp` — and there a second, independent limitation applies:
  **verified against `warpSPHCore`'s autograd bridge (`autograd/wrapper.py`,
  `autograd/stateAwareWarpFunction.py`) that scalar (non-array) kernel arguments cannot
  carry a gradient through it at all**, regardless of whether the caller detaches them.
  `warpWrapper2` casts every scalar `additionalArguments` entry through `scalar_t(...)`
  before its tensor/non-tensor split ever runs, and the kernel signatures involved
  declare `dt: scalar_t` by value, not `wp.array(dtype=scalar_t)` — so the existing
  `dt.detach().cpu().item()` calls in `schemes/{compSPH,crkSPH}.py` (feeding that
  specific call) are **currently no-ops**, not the actual severing point.

  So a real fix is two independent pieces, only the first of which is in this repo's
  reach:
  1. Let `dt` survive as a 0-dim tensor through `computeTimestep` and every plain-PyTorch
     use of it in the step/integrator arithmetic (drop the `float` type annotations,
     stop coercing to `.item()` except at genuine reporting boundaries).
  2. Give warpSPHCore's autograd bridge a differentiable-scalar path (a `wp.array`-typed
     `dt` kernel parameter plus bridge support) for the kernels that consume `dt`
     internally — a warpSPHCore-side change, out of this repo's reach alone, and the
     same shape of gap as `pinv2x2`'s missing backward was before it got wired in.
  Not fixed as part of this pass — recorded here so the next AD work doesn't rediscover
  the "detaching dt does nothing" trap and stop looking, when the actual gap is one
  layer up and one repo over.

  **Update, same day — piece 2 landed and is now verified working end-to-end.**
  warpSPHCore grew `asScalarArg` (`autograd/scalar_arg.py`) plus a `wp.array(dtype=
  scalar_t)` / `param[0]` convention for opting a kernel parameter into differentiability,
  proven upstream by `scripts/gradcheck_scalar_arg_native.py` on a demo kernel. Applying
  it to `computeCompSPHBalanceTermWarp` (the one kernel `dt` currently reaches directly)
  went through three rounds:

  1. **First attempt looked like it crashed the interpreter** — a native segfault inside
     `wp.launch`, confirmed with `python -X faulthandler`, the moment a gradient through
     `dt` was requested. Reverted rather than shipped, since `dt` never reaches this
     kernel as a `requires_grad` tensor in any currently-exercised path anyway (piece 1,
     below, was and is still open) — full test suite passed either way, but the signature
     would have looked differentiable and silently crashed the moment someone actually
     exercised it as such.
  2. **The crash turned out to be a red herring — a real, unrelated, pre-existing bug**,
     found by systematically ruling out the `dt`-integration itself as the cause (isolated
     the `monotonic`/`hybrid` arithmetic, the full six-branch dispatch, the `ap_ij`/`av_ij`
     edge-count shape, domain size, `dim`, adjacency construction method, and — decisively
     — the warp-lang version: it crashed identically under both the pinned 1.12.0 release
     and the (accidentally-installed, since-restored) 1.17.0.dev3 dev build). The actual
     cause: `computeCompSPHBalanceTermWarp`'s `referenceEnergies`/`referencePressures`
     were missing the query→reference fallback that `referenceVelocities` already had
     (`if referenceVelocities is None: referenceVelocities = queryVelocities`, with no
     equivalent two lines for the other two) — so a caller passing a bare `ParticleState`
     plus explicit `queryEnergies=`/`queryPressures=` (exactly what every gradcheck-style
     standalone call does, and what every real step-function call does *not* do, since
     real call sites always pass a full state object the `hasattr` fallback quietly
     covers for) reached the kernel launch with `referenceEnergies=None`, which the
     compiled kernel then unconditionally read as `referenceEnergies[j]`. **Fixed**
     with the missing two-line fallback, next to the existing velocities guard.
     Full writeup, the exact fallback-chain trace, and the reusable repro harness are in
     `scripts/troubleshoot_balanceTerm_segfault.py`. Confirmed scheme-independent before
     the fix (`PdV`/`diminishing`/`monotonic`/`hybrid`/**`CRK`, the production default**,
     all crashed the same way; only the data-independent `equalWork` survived) and fixed
     for all six after.
  3. **Re-applied the `dt` integration on top of the fix** — same shape as attempt 1
     (`wp.array(dtype=scalar_t)` on the top-level `@wp.kernel`, `dt[0]` read once there
     and forwarded as a plain `scalar_t` into the nested `@wp.func` layers, `asScalarArg`
     at the Python call site) — and it now works cleanly. Verified three ways:
     `scripts/troubleshoot_balanceTerm_segfault.py --mode backward` for both `CRK`
     (`dt.grad == 0`, analytically correct — CRK's output depends only on the *sign* of
     the dt-carrying term, never its magnitude) and `monotonic` (`dt.grad` nonzero); a
     direct `torch.autograd.gradcheck` against `computeCompSPHBalanceTermWarp` under
     `monotonic` (the one scheme with a genuinely smooth, non-zero-a.e. dt-dependence —
     `term_2`'s division), passing for both 0-dim and 1-element `dt`; and the plain-float
     regression case still resolving to `requires_grad=False`. Full test suite: 42/42.

  The `EnergyScheme.monotonic`/`.hybrid` ternary → explicit-`if/else` rewrite from the
  first (mis-)diagnosis is kept — harmless, matches this codebase's established
  convention, full suite passes with it, even though it turned out not to be what fixed
  anything. `schemes/{compSPH,crkSPH}.py` no longer call the (always-was-a-no-op)
  `dt.detach().cpu().item()` before passing `dt` into `computeCompSPHBalanceTermWarp`;
  they now pass `dt` through directly, and it is a real gradient path as of this fix.

  **Piece 1 — also done, same day.** `computeTimestep` (`modules/timestep/compressible.py`)
  no longer collapses the CFL-derived `initial_dt` to a Python float; `return
  initial_dt.cpu().item()` is now `return initial_dt`, keeping it a tensor on
  whichever device the state already lives on. `compSPH_step`/`crkSPH_step`'s `dt`
  parameter and `compressibleTimestep`'s return type are annotated `float | torch.Tensor`
  to match (Python doesn't enforce it, but the old `dt: float` was actively wrong the
  moment this landed). Nothing else needed to change:

  - `runner.py`'s step loop already separated "plain float for bookkeeping" (`_scalar()`,
    used locally for `nSteps`/`storeSteps`/progress-bar math) from what's actually passed
    to the step function (`ctx.config.dt`, unconverted) — so a tensor `dt` flows straight
    through to `compSPH_step`/`crkSPH_step` without the runner needing any change.
  - Config serialization (`configurations/simulationConfig.py`'s `_encodeValue`) already
    dispatches on runtime type, not the static annotation, and already had a
    `torch.Tensor` branch — a tensor-valued `dt` round-trips through export/JSON exactly
    like the scalar it structurally is.
  - `adaptiveDt=True` is the shared default for every compressible case (`cases/
    compressible.py`'s `COMPRESSIBLE_DEFAULTS`), so `tests/test_physics.py`'s Sod/TGV/
    dam-break runs already exercise this path on every run, not just a targeted new test.
    Full suite: 42/42, unchanged.

  Verified directly, not just by absence of regressions: with `internalEnergies.
  requires_grad_(True)`, `computeTimestep` now returns a tensor with
  `grad_fn=<ClampBackward1>`, and `dt.backward()` reaches `internalEnergies.grad` with
  exactly one nonzero entry — correct, since `dt = torch.min(dt_cfl_left)` only has a
  subgradient through the argmin particle. (First attempt at this check used
  `densities.requires_grad_(True)` instead and found nothing — not a bug: `idealGasEOS`
  computes sound speed from `u` when `u` is provided, which is how `computeTimestep`
  always calls it, so `c_s` genuinely never touches `rho` in this configuration. Correct
  physics, not a gap — the test was wrong, not the fix.)

  **Phase 4.2's `dt` finding is now fully closed**, both pieces: the CFL-derived adaptive
  timestep is a real, verified tensor with a gradient back to the state that determines
  it, all the way through `computeCompSPHBalanceTermWarp`'s kernel-level consumption of
  it.

- **`caseUtils/weaklyCompressible.py:526-530` (`forcing()`, random-flow case) and
  `modules/noise/sampleDivergenceFree.py` — (c), needs an explicit comment, currently
  silent.** The turbulent forcing term is built by calling a `scipy`
  `RegularGridInterpolator` on `pos.detach().cpu()` — numpy has no autograd, so
  `d(forcing)/d(position)` is unconditionally zero, and this forcing term feeds directly
  into a state update in a case that's otherwise part of the differentiable rollout.
  Likely fine in intent (it's a fixed pre-baked turbulence field, not something anyone
  is meant to optimize through) but currently undocumented as a deliberate choice —
  needs the one-line comment Phase 4's own (c) bucket calls for, not a rewrite.

- **`geometry/sdf.py:69-78`, `regions/domainSDF.py:12-15` — (a), correct as designed,
  but worth calling out because the shape is more subtle than a plain detach.** Both
  compute SDF normals via `torch.autograd.grad(..., create_graph=True, retain_graph=True)`
  and then conditionally detach the result based on `x.requires_grad`, so a caller that
  doesn't need gradients doesn't pay for a retained graph, and a caller that does gets
  one. This is deliberate dual-mode design, not a bug — but it's exactly the kind of
  hand-rolled AD code that a plain forward-value check can't validate, which is why it's
  also listed in 4.1's Tier 0 rollout rather than assumed fine and left alone.

**Not yet done:** the `.item()`/`.cpu()`/`.numpy()` sites beyond `.detach()` (the
original 97-count scope) haven't been individually re-audited the same way — the three
findings above came from tracing `.detach()` specifically, per this session's ask.
`modules/shifting/` and `modules/mdbc/` (flagged in the original first-pass scan) turned
out on inspection to be either commented-out debug prints (`mdbc/wp_nopenshift.py`,
`velocity.py`) or scalar CFL/spacing bookkeeping (`shifting/delta.py`,
`shifting/wrapper.py`) that looks like genuine (a) — non-differentiable by nature,
reporting/scalar-config values — but wasn't traced as deeply as the timestep finding
above; worth a second pass once the gradcheck scripts from 4.1 exist to actually test
the claim rather than eyeball it.

- [x] **Root-caused and fixed the `computeCompSPHBalanceTermWarp` standalone-call
      segfault** — a real bug: `referenceEnergies`/`referencePressures` were missing the
      query→reference fallback that `referenceVelocities` already had, so a caller
      relying on that fallback (as every real call site's `hasattr` path effectively did,
      by luck) could reach the kernel launch with `referenceEnergies=None`. Two-line fix
      in `modules/compSPH/balance.py`, next to the existing velocities guard. Verified
      against the original bare-`ParticleState` repro (not just the `CompressibleState`
      workaround): all six `EnergyScheme` values now run clean, no crash. Full suite:
      42/42 still pass.
- [x] Re-verify `forward-grad` and `backward` modes in
      `scripts/troubleshoot_balanceTerm_segfault.py` — both confirmed: `backward` gives
      `dt.grad==0` for `CRK` (analytically correct) and nonzero for `monotonic`, and
      `torch.autograd.gradcheck` against `computeCompSPHBalanceTermWarp`/`monotonic`
      passes for both 0-dim and 1-element `dt`
- [x] Re-applied the `dt: wp.array(dtype=scalar_t)`/`asScalarArg` integration to
      `computeCompSPHBalanceTermWarp` now that the real bug is fixed — works cleanly,
      full suite still 42/42
- [x] Finished the `computeTimestep` fix (4.2) — piece 1: `computeTimestep` no longer
      coerces to `.item()`, step-function signatures updated, `runner.py`/config
      serialization needed no changes (already tensor-safe by design). Verified directly:
      `internalEnergies.requires_grad_(True)` → `computeTimestep` returns a tensor with
      `grad_fn`, backprop reaches `internalEnergies.grad` correctly (nonzero only at the
      argmin particle, matching `dt = torch.min(...)`). Full suite: 42/42. **Phase 4.2's
      `dt` finding is fully closed, both pieces.**
- [ ] Build 4.1's gradcheck infrastructure, starting with Tier 0: `wp_surfaceAware.py`,
      `geometry/sdf.py`/`regions/domainSDF.py`
- [x] Add the one-line non-differentiability comment to the noise-forcing sites (4.2) —
      done in `caseUtils/weaklyCompressible.py`'s `forcing()` (Kolmogorov/random-flow
      case); `modules/noise/sampleDivergenceFree.py` still open, same reasoning applies
- [ ] Second-pass audit of the remaining `.item()`/`.cpu()`/`.numpy()` sites once
      gradcheck coverage exists to verify claims rather than assume them

---

## Suggested order

Phase 0 ✅ → 1 ✅ → 2 ✅ → 2b ✅ → **3 (in progress — `SchemeBundle` ✅, notebook
sweep ✅, `schemes/` explicit imports ✅, `__all__` coverage elsewhere remains)** →
3b (repair ✅, backlog open) →
4 → repo weight (deferred; its
Phase 2 precondition is now fully met — the examples are runnable `.py`, so the
polished renders that a history rewrite should operate on can now be
regenerated unattended).

The original "load-bearing" shortlist is now **fully done**: editable plotting install,
`integrators` import rename, `nx`/`dx` round-trip, the duplicate `DomainDescription`,
and — as of 2026-08-10 — `SchemeBundle` together with the `compParams`/`schemeConfig`
unification.

What is left in Phase 3 is `__all__` coverage on the 180 modules that still define
none. **Measuring it changed the recommendation:** the flat namespace has no
collisions and no leakage left, so this is legibility, not correctness, and the
"cheaper before AD" argument no longer holds for the tree at large. `schemes/` was
converted anyway because that is where AD lands and the explicit import block doubles
as a per-scheme dependency manifest — to audit which `.item()`/`.cpu()` sites a scheme
can reach you now read 8 lines instead of resolving a star import by hand. Do the rest
opportunistically, as AD touches each module.

**Phase 4 is now in progress (2026-08-11), not just scoped.** The `.detach()` half of
the audit is done (4.2) and found one real, non-obvious bug-shaped gap
(`computeTimestep`'s adaptive `dt` losing its tangent, compounded by warpSPHCore's
autograd bridge not supporting differentiable scalar kernel arguments at all). 4.1's
Tiers 0 and 1 are also now done. Tier 0: the gradcheck infrastructure
(`scripts/_gradcheck_common.py`, `tests/test_gradcheck_scripts.py`) is built,
`modules/pressure/wp_surfaceAware.py`'s flagged same-array ternary gradchecks clean (open
caveat — only verified under the currently-installed warp-lang 1.17.0.dev3, not the
pyproject-implied 1.12.0/1.16.0 where the analogous Interpolate bug is confirmed present),
and `geometry/sdf.py` / `regions/domainSDF.py` turned up and fixed a real latent crash.
Tier 1: `modules/compSPH/*` and `modules/dissipation/*` gradcheck clean using two new
repo-local state fixtures (`make_compressible_state`, `compute_crk_state`); `modules/crk/*`
found two real bugs, **both now fixed** — `modules/crk/limiter.py`'s self-interaction 0/0
(fixed here, a forward-safe-but-backward-poisoning NaN-patch-after-the-fact, same
underlying class as Tier 0's ternary risk in a different guise) and a deeper bug in
warpSPHCore's `correctGradientCRK` (a hand-written accumulation loop contracting `gradBi`
against the wrong axis; fixed upstream, same day, by replacing it with a single `matmul`).
Before the second fix landed, **CRKSPH — the production default scheme — was not
AD-correct with respect to position**; `scripts/gradcheck_crk.py` now confirms it is, and
is wired into the pass/fail suite like every other Tier 0/1 script. 48 tests, up from 42
at the start of Phase 4.1.

**Tier 2 is partly done (2026-08-11, second session).** `adaptiveSupport`, `deltaSPH`,
`shockCapturing` (three scripts that already existed but weren't wired into the suite
or written up) and a new `mdbc` script are gradchecked and wired in (49 tests, up from
48). `mdbc` alone found and fixed three real bugs just to get a multi-output kernel
running under gradients at all — two in warpSPHCore's own math/autograd layer (a
missing `vec1i` type breaking `zero_like`'s codegen; the autograd bridge setting
`requires_grad` on non-float kernel outputs unconditionally) and one here (an
unguarded float32 literal in a float64 kernel) — see Tier 2 above for the full
writeup. Remaining: `incompressible/wp_alpha.py`, `liu/wp_mat.py`,
`surfaceDetection/*`, `util/*`, `sample/wp_deltaShift.py`.

**The warp-lang-version caveat Tier 0 flagged and left open has now bitten — and been
fixed, same day.** Re-running the full suite while wiring in Tier 2 found that
`wp_surfaceAware`, `compSPH`, and `dissipation` — all three previously "gradchecks
clean" — failed under the currently-installed warp-lang **1.15.0**, reproducibly and
standalone. This was exactly the risk the original Tier 0 writeup warned about
("this result shows the ternary is safe on the dev build actually running here, not
that it's safe on the version this repo is nominally pinned to"). Root cause turned
out **broader** than the original "same-array ternary" diagnosis — some of the
ternaries that needed rewriting weren't same-array reads at all, so the safe
conclusion is narrower: any inline `X if cond else Y` compiled inside a `@wp.func`
is suspect on this warp-lang version, not just the same-array-read subset. **Fixed**
by adding a proper `access_optional(arr, index, condition, defaultValue)` helper to
warpSPHCore (an explicit `if`/`else` `@wp.func`, not an inline ternary) and rewriting
every affected site across `wp_surfaceAware.py`, `compSPH/{accel,dudt}.py`,
`dissipation/{pi,wp_conductivity,wp_diffusion,wp_dissipation}.py`, and `mdbc/
wp_nopenshift.py` to use it. One adjacent real bug fixed along the way (an unguarded
`referencePressures[j]` read in `compSPH/{accel,dudt}.py`, plus a stray typo,
`thermalConducitiyTerm` → `thermalConductivityTerm`, in `DiffusionParameters`).
**Verified: full suite 53/53 pass.** See the "Regression" writeup under Tier 2 above
for the complete story. `pyproject.toml` still pins no warp-lang version — **by
decision, not oversight**: the plan pins to 1.17 once it ships (fixes the underlying
bug at the source), and until then any version is fine as long as ternaries in
kernel code go through `access_optional` rather than an inline conditional
expression (see "Decisions already made" at the top of this file). **Also fixed the
same day:** `modules/crk/{accel,dudt}.py` never had `explicitPressure`/
`individual_cs` guards at all (a separate, older gap) — re-threaded through both
files now that `access_optional` made it mechanical, following `compSPH`'s exact
pattern. Verified via `gradcheck_crk.py` (still clean) and a real 3-step CRKSPH run.

By now `_gradcheck_common.py` carries enough of the fixture vocabulary (state
builders, CRK factors, densities) that step 4 of the 4.1 recipe — a repo-local
`gradcheck` skill mirroring warpSPHCore's — is worth doing once Tier 2's remaining 5
modules are closed out, rather than waiting further. Phase 3b's repair is already
done, and everything still open there is legibility. The one exception worth folding
into Phase 4 rather than leaving in 3b: upstreaming the tensor-aware
`volumeToSupport` into warpSPHCore, since AD will care whether that path is
differentiable.

Deliberately last: the repo-weight rewrite, so it operates on the polished Phase 2
files that are actually worth publishing rather than on soon-to-be-regenerated output.
Its one separable piece, the `nbstripout` hook, can be added at any time.
