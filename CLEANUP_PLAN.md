# warpSPH — Cleanup Plan

Working document for the cleanup sweep preceding forward-mode AD work.
Core and Integrators have already been overhauled; this repo (the former frontend,
`~/dev/warpSPHFrontend` → now `~/dev/warpSPH`) was the lagging piece.

**Status:** Phase 0 **complete** (0.3 deletions done 2026-08-10). Phase 1 complete
except repo weight, which is **deliberately deferred until after Phase 2** — see
"Deferred: repo weight". Phase 2 is next.

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
from growing further.

---

## Phase 2 — Examples → runnable scripts + first tests

The conversion is already half-designed; the pattern just hasn't been extracted.

**The duplication:** all 42 notebooks open with a near-identical ~35-line boilerplate
cell (precision config → `wp.init()` → `TORCH_CUDA_ARCH_LIST` → warning filters → star
imports). Four drifting copies exist:
`examples/compressible/01-sod-shock-tube-1d.py:1-42`,
`examples/incompressible/01-tgv-incomp.py:1-40`,
`datagen/weaklyCompressible/generator.py:1-47`, and every notebook's cell 0.
Same story for the step loop, plotter setup, and the ffmpeg export cell.

**Existing files that are already the target form** — use as templates:
`examples/compressible/01-sod-shock-tube-1d.py` (argparse, 264 lines),
`examples/incompressible/01-tgv-incomp.py`,
and `datagen/weaklyCompressible/{parser,generator}.py` — the most mature pair, already
does config-sweep-as-data-generator.

- [ ] Extract `warpSPH.runner`:
  - `bootstrap(precision, dim)` — the boilerplate cell, once
  - `CaseSpec` — dataclass over the argparse surface currently duplicated across
    `parser.py` / `01-sod-shock-tube-1d.py`; serializable to/from JSON or YAML so
    sweeps are config files, not shell strings
  - `run(case_spec)` — step loop + export/plot hooks
  - each case reduces to a `build_regions` / `initial_conditions` / `diagnostics` triple
- [ ] Convert 2–3 cases as proof (suggest: TGV, Sod, dambreak — one per family)
- [ ] First tests: run 20 steps at nx=32, assert a physical invariant
      (e.g. TGV kinetic-energy decay slope within tolerance). **There are currently
      zero tests in this repo** — Core and Integrators both have suites.
- [ ] Notebooks stay for exploration but import the same runner, so they can't drift

End state: `python -m warpSPH.cases.tgv --config sweep/tgv_re1000.yaml`,
and ML data generation is a sweep over config files.

---

## Phase 3 — Structural (do before AD, since AD touches every scheme)

- [ ] **`buildScheme` returns a bare 7-tuple** (`schemes/builder.py:12-36`) that every
      notebook and script unpacks positionally. Adding an 8th element — likely a
      tangent-propagation fn during AD — is a breaking change to 42 notebooks.
      Make it a `SchemeBundle` dataclass with named fields.
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

Phase 0 ✅ → 1 ✅ → **2 (next)** → 3 → 4 → repo weight (deferred; needs Phase 2 done first).

Items 1-3 of the original "load-bearing" shortlist are all done (editable plotting
install, `integrators` import rename, `nx`/`dx` round-trip). What remains of it:

1. `SchemeBundle` (Phase 3) — cheap now, expensive after AD lands

(The duplicate `DomainDescription` was also on this list and is now fixed.)

Deliberately last: the repo-weight rewrite, so it operates on the polished Phase 2
files that are actually worth publishing rather than on soon-to-be-regenerated output.
Its one separable piece, the `nbstripout` hook, can be added at any time.
