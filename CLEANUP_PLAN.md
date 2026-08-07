# warpSPH — Cleanup Plan

Working document for the cleanup sweep preceding forward-mode AD work.
Core and Integrators have already been overhauled; this repo (the former frontend,
`~/dev/warpSPHFrontend` → now `~/dev/warpSPH`) was the lagging piece.

**Status:** Phase 0 mostly done — 0.1 (GitHub renames) and 0.2 (`integrators` import
rename) are complete and verified; 0.4 (doc links) fixed 2026-08-07. 0.3 (deletions)
is still open. Phase 1 not yet started.

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

### 0.3 Deletions

Dead / duplicated, verified:

- [ ] `src/warpSPH/configurations/presets.py/` — a **directory** named `presets.py`,
      containing two zero-byte files. Unreferenced.
- [ ] `src/warpSPH/ml/` — 2,108 lines, no `__init__.py`, **zero imports anywhere**.
      `ml/dataset.py` is a stale ancestor of the top-level `dataset.py`.
      Decide: revive as the ML entry point (ties to Phase 2) or delete.
- [ ] `src/warpSPH/caseUtils/rayleighTaylor/bcs.py` — byte-identical to
      `caseUtils/compressible/rayleighTaylor/bcs.py` (`diff` = 0 lines)
- [ ] `examples/13-openFlow.ipynb` — stale dup of `examples/weaklyCompressible/13-openFlow.ipynb`
- [ ] `examples/weaklyCompressible/utils.py` (304 lines) — duplicates what was already
      promoted into `src/warpSPH/caseUtils/weaklyCompressible.py`. The datagen copy was
      already reduced to a 40-line shim; this one wasn't.
- [ ] `examples/incompressible/dfsph.py`, `dfsph_step.py` — both zero bytes
- [ ] `bak/` — 6 old notebooks, that's what git is for
- [ ] `src/warpSPH/legacy_utils.py` + the `sys.modules.setdefault("utils", ...)` hack
      at `__init__.py:55-57` — only exists for old dill payloads, which we've accepted losing
- [ ] Root clutter: `profile.json` (4 MB), `nu_eff_vs_nu.png` (196 KB),
      `examples/weaklyCompressible/warpSPH_trace.json` (3.5 MB),
      `datagen/weaklyCompressible/log.txt`, `dist/` (committed despite being gitignored)

### 0.4 Doc links — DONE (2026-08-07)

- [x] `README.md` — clone block, all four URLs; title; package-layout section
      (was still `src/compressibleSPH/*`, imports, video link) — all updated to
      the unified `warpSPH` naming.
- [x] `boilerplate.md` — codebase/backend/timestepper URLs updated to the current
      repo names. This text gets pasted into published video descriptions.

---

## Phase 1 — Mechanical fixes (resume here)

Small, independent, unblocks Phase 2.

- [ ] **`warpSPHPlotting` is not editable-installed.** It resolves to
      `site-packages/warpSPHPlotting` @ 0.4.0 while `~/dev/warpSPHPlotting` is at 0.4.5.
      Every plotting edit in that sibling repo is currently invisible here.
      `pip install -e ~/dev/warpSPHPlotting`
- [ ] **Pin backend deps.** `pyproject.toml:22-32` lists the three backends with no
      version constraints. Current skew: core 0.4.1 installed / 0.4.5 in repo;
      integrators 0.4.2 / 0.5.0; plotting 0.4.0 / 0.4.5. This is the mechanism behind
      the last five "fix upstream breakage" commits. Lower bounds turn silent
      breakage into an install-time error.
- [ ] **`configurationToDict` silently drops `nx` and `dx`.**
      `configurations/simulationConfig.py:125-155` is a hand-written field list;
      `nx`/`dx` are declared at lines 35-36 but never exported. Every notebook sets
      `config.nx = nx` after `buildConfig`, so **resolution does not survive a config
      round-trip** — any resumed or reloaded run loses it.
      Verified: `set(fields) - set(configurationToDict(...).keys()) == {'dx', 'nx'}`.
      Fix by generating from `dataclasses.fields()` so it can't recur.
- [ ] **`prepExport` hardcodes the output path.** `io.py:299-302` writes to
      `export/{caseName}` relative to CWD, not overridable. Parallel sweeps collide.
- [ ] **Add the 6 missing `__init__.py`** (currently PEP 420 implicit namespace pkgs):
      `ml/`, `utils/noiseFunctions/`, `utils/sdfFunctionality/`, `modules/sps/`,
      `caseUtils/waveEquation/`, `caseUtils/rayleighTaylor/`.
      They land in the wheel today, but namespace-package handling in
      `[tool.setuptools.packages.find]` is version-sensitive. Cheap insurance.
- [ ] **Repo weight:** 527 MB `.git` vs 395 MB tracked. `examples/compressible/outputs/`
      is ~380 MB of repeatedly-recommitted `.mp4`/`.gif`, and 39 of 42 notebooks are
      committed *with cell outputs*. Consider `git lfs migrate` or release assets,
      plus an `nbstripout` pre-commit hook — otherwise every rerun inflates history
      and produces unreviewable diffs.

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

Phase 0 (manual) → 1 → 2 → 3 → 4.

If short on time, the load-bearing items are:
1. Editable plotting install — you're developing against a stale backend right now
2. The `integrators` import rename — only item that's an actual correctness risk
3. `nx`/`dx` round-trip — silently corrupts reproducibility of every resumed run
4. `SchemeBundle` — cheap now, expensive after AD lands
