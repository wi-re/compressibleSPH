# warpSPH — Cleanup Plan

Status tracker for the cleanup sweep preceding forward-mode AD work. Core and
Integrators (separate repos) are already overhauled; this repo was the lagging
piece. For reusable lessons (bug classes, gotchas, process notes) rather than
current status, see `LESSONS_LEARNED.md` and, for AD/gradcheck specifically,
`.claude/skills/gradcheck/SKILL.md`.

## Status

| Phase | Status |
|---|---|
| 0 — GitHub/import renames, dead-code deletions, doc links | ✅ Done |
| 1 — Mechanical fixes (editable installs, config round-trip, `__init__.py`s) | ✅ Done except repo weight (deferred, see below) |
| 2 — Examples → runnable scripts + first tests | ✅ Done |
| 2b — Every example as a runnable case (27/27) | ✅ Done |
| 3 — Structural (`SchemeBundle`, `compParams`→`schemeConfig`, `DomainDescription`, namespace) | ✅ Done except `__all__` coverage (legibility, open) |
| 3b — Post-reshuffle repair (`io/`, `math/`, `geometry/` packages) | ✅ Repair done; backlog open (below) |
| 4 — AD-readiness (gradcheck Tiers 0-2, `.detach()`/`.item()`/`.cpu()`/`.numpy()` audits) | ✅ **Done (2026-08-12)** — zero open findings; the Phase 4.2 `balanceTerm` segfault no longer reproduces either (2026-08-15, see below) |
| Repo weight (git-history rewrite) | ⏸ Deferred by decision |

Naming is fully unified: local dir == GitHub repo == import == PyPI dist for all
four packages (`warpSPH`, `warpSPHCore`, `warpSPHIntegrators`, `warpSPHPlotting`) —
see README for the current layout, not reconstructed here.

## Remaining work

### Legibility (Phase 3 / 3b backlog — no correctness stakes)

- [ ] **Module documentation.** Added as a tracked item 2026-08-15 — it had
      never been one, so the plans read as though the only remaining legibility
      work was `__all__` and dead comments. Measured 2026-08-15:

      | scope | documented | share |
      |---|---|---|
      | modules | 53/277 | 19% |
      | public functions | 113/797 | 14% |
      | classes | 11/82 | 13% |

      The distribution matters more than the total, because it is not uniform
      neglect — it tracks exactly what the cleanup sweep covered:

      | directory | module docstrings | share |
      |---|---|---|
      | `cases/` | 30/30 | 100% |
      | `runner/` | 8/8 | 100% |
      | `schemes/` | 1/8 | 12% |
      | `math/` | 1/12 | 8% |
      | `caseUtils/` | 3/42 | 7% |
      | `configurations/` | 1/20 | 5% |
      | `regions/` | 0/9 | 0% |
      | `modules/` | **1/104** | **1%** |

      `cases/` and `runner/` are at 100% because Phases 2/2b/3 worked through
      them; the physics layer was never in scope for those phases. So this is
      the *last* layer, not a neglected one — and `cases/`+`runner/` are the
      worked pattern to copy: say what the module is for and what bites, not
      what the code already says.

      **Do this together with the `__all__` item below.** They are the same pass
      over the same files — `modules/`, `configurations/`, `regions/`,
      `schemes/`, `caseUtils/` — and working out a module's real exports is most
      of the work of describing it. Splitting them means reading `modules/`
      twice. Unlike `__all__`, this one does not decompose well into
      "opportunistically as modules get touched": a directory at 1% needs a
      deliberate pass to become navigable at all.

      Scope honestly: this is a project, comparable to the notebook migration
      below, not a chore to slot into another change.
- [ ] **`__all__` coverage.** 106/277 files (38%) define one (re-measured
      2026-08-15; was 94/274 on 2026-08-12); the rest still star-export
      everything they import. `schemes/` (converted to explicit imports,
      `__all__` added) is the worked pattern for the rest. Do opportunistically
      as modules get touched — *except* when the module-documentation pass above
      is picked up, which should sweep both at once over the same files.
- [ ] **Dead commented-out code** in `schemes/crkSPH.py`,
      `modules/mdbc/wp_nopenshift.py`, `shockCapturing/CullenDehnen2010.py`,
      `schemes/dfsph.py`, `caseUtils/compressible/sod/sod.py`,
      `schemes/deltaSPH.py`. The "~421 lines" figure recorded here on 2026-08-12
      did not reproduce on 2026-08-15: a re-count over the same six files gave
      598 by one definition of "commented-out code" and 522 by a looser one, and
      every per-file number came out higher than listed. The counting rule was
      never written down, so **fix the definition first, then re-measure** —
      per LESSONS_LEARNED's "re-measure a stale plan's numbers" rule, which was
      itself derived from this kind of drift.
- [x] **Notebook simplification, pilot (2026-08-12): Sod.** Started with
      `01-sod-shock-tube-1d.py`/`01-Sod_Shock_Tube_1D(_resume).ipynb`, moved into
      `examples/compressible/01-sod/` (`sod_1d.py`/`.ipynb`,
      `sod_resume.py`/`.ipynb`) — the new per-case-directory convention for a case
      that needs more than one variant (numbered case dir, unnumbered contents).
      **Corrects the plan's original framing below**: a notebook that only
      re-derives its `.py` sibling should not become the same thin
      `caseMain()`-wrapped script — the user wants notebooks *fatter*, not
      thinner, because they're the place to prototype new physics/hooks. Sod's
      notebooks now build ICs via the real case code and call the same generic
      helpers the runner does (`buildContext`, `Case.setupPlot`/`updatePlot`,
      `encodeFrames`), but keep **the step loop itself unrolled and visible in the
      cell** instead of hidden behind `run()`, plus an explicit "parameters" cell
      surfacing every knob a CLI flag would (`nx`, `tLimit`, ...). Don't re-propose
      collapsing them into a thin wrapper.
      - Also switched Sod's examples (not `sodCase.defaults` — `run_sweep`/tests/
        `sod_highres.yaml` are untouched) to the datagen trajectory export scheme
        (`storeMode='trajectory'`, one growing `trajectory.h5`) instead of
        one-file-per-stored-step. This required real fixes, not a rename:
        `writeInitialData`/`writeFrame` (`warpSPH/io/export.py`) were hard-coded to
        the weakly-compressible/dam-break state shape and crashed on a compressible
        state (`schemeConfig.rigidBodies`, `state.ghostOffsets` don't exist on
        `CompSPHConfig`/`CompSPHState`) — now guarded with `getattr(..., default)`.
        Added an `extraFields` parameter (default `()`, every existing caller
        unaffected) so a case can declare additional per-frame fields; `Case` grew
        an `extraFields` attribute so `run()` threads it through automatically.
        Sod passes `('internalEnergies', 'supports')` — per the user, only
        `internalEnergies` is the core EOS state variable worth writing every frame
        (`pressures`/`soundspeeds`/`entropies` are recomputed from it via
        `idealGasEOS` on load, to keep file size down); `supports` also needs
        per-frame export because Sod's default adaptive support scheme drifts from
        the t=0 snapshot, unlike `masses`/`kinds`/`materials`/`UIDs` which are
        genuinely IC-only. Added `warpSPH.io.loadTrajectory`/`loadTrajectoryFrame`
        — a real generic reader (promoting the shape of `compressedLoader.ipynb`'s
        inline, plot-only logic into a library function that reconstructs a live,
        resumable `SimulationState`, which nothing did before).
      - **2D and 3D Sod: done (2026-08-12)**, `warpSPH/cases/sodND.py`
        (`sod2d`/`sod3d`, plus `examples/compressible/01-sod/sod_2d.py`/`_3d.py`
        and matching notebooks following `sod_1d.ipynb`'s pattern — parameters
        cell, real case code for the IC, unrolled step loop — with two cells
        that only make sense here: the sampler's integer choices inspected
        without building anything, and the density field drawn in the slab
        itself (2D) or a thin z-slice of it (3D)),
        sampled at equal particle *mass* rather than equal spacing --
        `caseUtils/compressible/sod/sodND.py`. The 3D plotting gap this entry
        used to cite turned out not to exist: the solution is 1D along the tube,
        so the existing six Sod profile panels work in any dimension as a
        scatter (`plotSod_` now indexes the x column explicitly, a no-op in 1D).
        Running actual 3D physics for the first time found two real bugs, both
        fixed and both regression-tested: warpSPHCore's B7 kernel had a **3D
        normalisation constant 16x too small** (1D and 2D correct), and Owen's
        adaptive-support psi LUT was cached in an **unkeyed module global**, so
        the first dimension a process touched won and everything after it
        relaxed supports against the wrong table.
      - **Backprop-over-trajectory demo: done (2026-08-12)**,
        `examples/compressible/01-sod/sod_backprop.ipynb`, both stages the user
        asked for — self-consistency recovery (perturbed `masses`, then `gamma`)
        and the fit to `sodSolution.solve`'s analytic Riemann solution. Not in
        CI (a 500-step BPTT is too expensive per commit); the logic is written
        as importable functions so a future
        `scripts/gradcheck_sod_trajectory.py` can lift it. Design record and
        measured results in that directory's `BACKPROP_PLAN.md`. Found and fixed
        a real severed-gradient bug on `balance.py`'s `gamma` kernel argument
        (the `dt` treatment of Phase 4.2, applied to the argument next to it),
        guarded by a new `run_balance_gamma_gradcheck` in
        `scripts/gradcheck_compSPH.py`.
- [x] **Notebook simplification, compressible family (2026-08-13).** All 13
      `examples/compressible/` slots (14 cases) converted; see
      `examples/compressible/MIGRATION_STATUS.md` for the full case list and
      the two notebook shapes (`profilePlot` 1D, `particlePlot` 2D field —
      the latter's window/event-loop-free core, `buildFieldPlotter`/
      `refreshFieldPlotter`, is new in `cases/plotting.py`, piloted on `08`).
      `14`/`15` merged into one `14-triplePoint/` directory the way `06`/`07`
      merged into `06-sedov/` earlier in this item.
- [ ] **Notebook simplification, the rest.** `examples/weaklyCompressible/` is
      12 of 13 slots done (see its own `MIGRATION_PLAN.md` for the per-slot
      table, which is the authority here). Only slot **13**
      (`channelFlow.openFlowCase`) remains: slot 11 was finished 2026-08-14 with
      its case redesigned, and is no longer a `channelFlow` hook — this entry
      claimed "11 and 13 remain, both `channelFlow`" until 2026-08-15, which
      contradicted `MIGRATION_PLAN.md` on both the count and the case.
      `examples/incompressible/` is untouched — and two of its three notebooks are the weakly compressible
      cases under `--scheme divergenceFree`, so they follow 03 and 06 rather
      than needing cases of their own.
      **Read `PORTING_EXAMPLES.md` first** — the procedure,
      the notebook conventions that have a reason behind them, and (separately)
      how to take a case to 2D/3D, all written from doing Sod; the compressible
      pass above is a second worked example, particularly for any case that
      turns out to be 2D/field-plot shaped.
      - Boilerplate cell duplicated verbatim in the remaining notebooks, still
        on the pre-`warpSPHBootstrap` config path — cheapest win, do first.
      - Apply the Sod pilot's pattern (real case code + generic helpers, visible
        step loop, explicit parameters cell) rather than collapsing to thin
        wrappers; keep genuinely notebook-shaped analysis/plotting as its own cells.
      - Worst offender left: `13-open-flow.ipynb` — 57 cells, 1199 code LOC
        (re-measured 2026-08-15; this read "1704 LOC" before). The
        incompressible notebooks are the other half: 692 (`01-taylor-green-vortex`),
        697 (`periodic-random-flow`), 434 (`03-rotating-square-patch`), and
        608 in the `1d-test.ipynb` scratchpad that is deliberately not ported.
- [x] `datagen/weaklyCompressible/bak/` — gone; the directory no longer exists.

### Documentation and usability pass (2026-08-15)

A survey of the repo from a reader's/new-user's point of view, deliberately run
*without* reading these planning documents first, so the findings are what the
tree itself says rather than what the plans claim. Verified throughout with the
standing commands (89 tests, `check_imports.py`, 27/27 sweep).

- [x] **`scripts/check_imports.py` was silently under-checking.** It assumed a
      notebook cell's `source` is always a list of lines; nbformat also permits
      a single string, and iterating that walks it character by character. Two
      notebooks (`01-sod/sod_resume.ipynb`, `14-triplePoint/triplePoint_equalMass.ipynb`)
      were reported as bogus `SyntaxError`s and their imports never actually
      checked. Fixed; scanned first-party imports went 1596 → 1612. Worth noting
      given this file is listed above as a standing verification command.
- [x] **504 dead commands in `datagen/weaklyCompressible/cases/`.** Seven of the
      eight `.sh` batches still passed `--timeLimit`, which no longer exists —
      every one of them failed instantly with `unrecognized arguments`. Only
      `examples.sh` had been updated. All 28 flags used across the batches were
      diffed against `generator.py`'s parser; `--timeLimit` was the only stale
      one. Renamed to `--tLimit` throughout and smoke-tested. The typo'd
      near-duplicate `exmples.sh` (an older copy of `examples.sh`, missing the
      two kolmogorov lines) was deleted.
- [x] **Example naming brought in line with the documented convention.** Both
      `MIGRATION_STATUS.md` and `MIGRATION_PLAN.md` specify
      `<slot>-<name>.ipynb`, but the compressible family shipped
      `08-Hydrostatic.ipynb` next to `08-hydrostatic.py`. 12 files renamed (git
      records all as renames) and 108 references updated across 53 files;
      `render_examples.py --list` verified byte-identical afterwards, so no
      output artefact changed name. `siblingNotebook`'s case-insensitive/slot
      fallback stays — it is what makes a *new* example work before anyone has
      thought about naming.
- [x] **New documentation.** `CONTRIBUTING.md` (install order, the per-clone
      `nbstripout --install` step, what to run before committing — the repo has
      no hosted CI **by decision**, since the suite needs a GPU),
      `datagen/README.md`, and `examples/weaklyCompressible/EXAMPLES_SUMMARY.md`
      (the weakly-compressible gallery; the compressible one had no counterpart,
      so ~40 tracked GIF/MP4/PNG were reachable from nothing).
- [x] **`--help` made usable.** Every `CaseSpec` flag read `CaseSpec.<field>`;
      the enum-valued ones listed no values, and the `--no-x` twins were
      `argparse.SUPPRESS`ed, so `--no-periodic` was undiscoverable while the
      visible `--periodic` was a no-op. Added a per-field help table, enum
      choices read off the enums themselves, and `BooleanOptionalAction` so both
      polarities show on one line. `choices=` deliberately *not* used — the
      `parse*` helpers match case-insensitively and it would start rejecting
      `--kernel b7`.
- [x] **Packaging metadata.** `pyproject.toml` had no `authors` and no
      `[project.urls]`, and its `description` was copy-pasted from `warpSPHCore`
      ("neighbor search and operators") — i.e. PyPI advertised this package as
      the thing it depends on. Identity copied from `warpSPHIntegrators` for
      consistency across the four repos.
- [x] **The Phase 4.2 `computeCompSPHBalanceTermWarp` segfault is resolved.**
      `scripts/troubleshoot_balanceTerm_segfault.py` — which still described a
      live crash — is clean in all three modes and all six energy schemes; its
      "only `equalWork` survives" finding no longer holds. Cause, per the user,
      was **not** the out-of-bounds read the investigation was chasing:
      - The harness never built the reference states properly, so the minimal
        repro was an *invalid* input rather than a stripped-down valid one. Its
        "cold process / fresh allocations" lead was a dead end.
      - The real bug: `referenceVolumes` left as `None` instead of falling back
        to `queryVolumes`, so a reference state with no volume member passed the
        kernel a null array. Several entry points were affected, not just this
        one. Fixed upstream in `warpSPHCore` on 2026-08-06 by `120c4bf` ("make
        the state path the primary one") — the fallback now at
        `warpSPHCore/operations.py:52`. CRK specifically was fixed by passing a
        fully set-up state.

      Recorded in the script's docstring too. The elimination record there is
      kept, but its conclusions have to be read knowing the input was faulty.
      **Deleting the script is a call for the user, not cleanup.**
- [x] Smaller: `log.txt` (a committed pytest dump) removed and gitignored; an
      invalid escape sequence in `caseUtils/compressible/sod/sodSolution.py`
      made a raw docstring; `tests/test_gradcheck_scripts.py` now discovers the
      gradcheck scripts by glob instead of a hardcoded list that could silently
      stop covering one; six notebooks declaring nbformat 4.5 without cell `id`
      fields normalized; the wave-equation subsystem documented (below).

### Cross-repo (owned by warpSPHCore) — DONE (2026-08-12)

- [x] `warpSPHCore.util.support.volumeToSupport` now dispatches on
      `isinstance(volume, torch.Tensor)` internally. `warpSPH/utils/support.py`'s
      wrapper collapsed to a plain re-export; verified scalar and tensor paths agree
      across all three dimensions, full suite still green.
- [x] `datagen/weaklyCompressible/loader.ipynb` — removed. `compressedLoader.ipynb`
      already reads the current (compressed) export layout, making the hardcoded-path
      loader redundant rather than something to fix.
- [x] **`B7_C_d`'s 3D normalisation constant was 16x too small**
      (`warpSPHCore/kernels/kernelFunctions/B7.py`): `1024/(105*pi)` where the
      integral of `k` over the unit ball gives `16384/(105*pi)`. An SPH density
      sum over a uniform 3D lattice returned 1/16 of the mass it was built from;
      1D and 2D were already correct, and B7 is the compressible examples'
      default kernel. Found by `sod3d`, the repo's first 3D case. Every other
      kernel checked at the same time and correct in all three dimensions
      (`Poly6`/`Spiky` are off in 1D/2D, but those are the graphics kernels,
      normalised for 3D only — left alone).

### Won't fix / rejected (recorded so they aren't re-proposed)

- `math/` shadows the stdlib name — harmless under absolute imports, no replacement
  name was ever proposed, out of scope for the naming pass.
- Remaining basename collisions (`sod.py`, `noh.py`, `region.py`, `sdf.py`, the
  `caseUtils/compressible/*/sample.py` family) — each is case/package-namespaced on
  purpose and reads fine in context; not churned.
- `examples/weaklyCompressible/naca.ipynb`, `examples/incompressible/1d-test.ipynb`
  — deliberately have no case (standalone SDF viz / exploratory scratchpad, not
  published examples).
- **The wave-equation subsystem stays** (`schemes/waveEquation.py`,
  `systems/waveSystem.py`, `configurations/waveEquationConfig.py`,
  `sample/waveSystem.py`, `caseUtils/waveEquation/`). It has no registered case,
  no test and no example, so a survey reads it as dead — it is not. Per the user
  (2026-08-15) it is kept as a compact demo of the SPH operators over
  *unstructured, mesh-like data*: a moving-neighbourhood Laplacian on scattered
  points with a heterogeneous wave speed, which is the shape of problem
  graph/point-cloud ML models are posed on. Module docstrings and a README
  section now say so. Note it is **not runnable as it stands**: the path is
  `build_configs_from_casefile` → `sampleParticles` → `genInitial` →
  `finalizeWaveSystemSetup` → integrate with `f_wave_equation`, and nothing runs
  those five stages; no TOML casefile ships either, and `genInitial` allocates
  `nx**2` so it is 2D-only. Making it a usable demo is unscoped work, not cleanup.
- **No hosted CI.** Per the user (2026-08-15): the test suite and the case sweep
  both need a CUDA device, so GitHub runners would be slow and burn compute
  budget for little return. The checks are run locally before commits and bigger
  changes instead — see `CONTRIBUTING.md`. Don't propose a GitHub Actions
  workflow for tests, sweeps or `check_imports.py`.
- ~~Sedov's `initialization='hat'` — both notebooks crashed on this default; the case
  now defaults to `'singular'`. Reviving `'hat'` is a separate, unscoped job.~~ Done
  2026-08-13: `'hat'` now deposits E0 on the central particle and smooths it with one
  SPH interpolation pass over the finalized adaptive supports; it's the case default
  again. Sedov also moved into its own directory (`examples/compressible/06-sedov/`,
  1D/2D/3D) in the same pass, which surfaced a real bug: `B7_C_d(dim=3)` in the sibling
  `warpSPHCore` repo was 16x too small (a regression of the "B7's 3D kernel
  normalisation" bug noted in `PORTING_EXAMPLES.md` section 4.7 — apparently never
  fixed for the `sampleRegularParticles`/plain-`Density` code path Sod's own 3D builder
  doesn't use). Fixed there (uncommitted, separate repo); see
  `tests/test_physics.py::test_uniformLatticeDensityMatchesBuiltDensity` for the
  regression test.

## Deferred by decision

- **Repo weight / git-history rewrite.** Its precondition (examples runnable as
  `.py`, physics verified) is now met, but still deliberately last — the media in
  question should be rewritten once, against the final polished assets, not against
  output Phase 2 was about to regenerate anyway. `nbstripout` (the separable,
  non-destructive piece) is already installed. When this is picked up: the media
  (`.mp4`/`.gif`) was measured at 338 MB across 30 files, each committed exactly
  once — no recommit churn to reclaim there; the actual bloat is 312 blob versions
  of 42 notebooks (289 MB), which `nbstripout` addresses going forward. `git-lfs` is
  not installed on this machine.
- **warp-lang version pin.** `pyproject.toml` pins no version. Pin to **1.17** once
  it ships (fixes the same-array-ternary/Interpolate adjoint bug at the source).
  Until then any installed version is fine, provided kernel-code ternaries route
  through warpSPHCore's `access_optional` (or an explicit `if`/`else`) rather than
  an inline conditional expression. Don't pin to 1.15.0/1.16.0/1.12.0 as a
  workaround — that's been explicitly ruled out.

## Notes

- Renames break the dill-encoded callables in the ~244 pre-existing local `.h5`
  datasets — accepted, since they're regenerable and no large dataset had been
  built yet.
- `scripts/check_imports.py`, `scripts/run_tests.sh` (89 tests, ~2 min), `scripts/
  run_sweep.py` (27/27 cases, ~4 min) are the standing verification commands; prefer
  them over hand-rolled loops. Counts re-measured 2026-08-15 (they read 58 and 25/25
  here until then, and the README said 42 — all three were stale). The `gradcheck` skill (`.claude/skills/gradcheck/
  SKILL.md`) covers the 15 gradcheck scripts specifically.
