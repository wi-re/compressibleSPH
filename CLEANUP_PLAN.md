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
| 2b — Every example as a runnable case (25/25) | ✅ Done |
| 3 — Structural (`SchemeBundle`, `compParams`→`schemeConfig`, `DomainDescription`, namespace) | ✅ Done except `__all__` coverage (legibility, open) |
| 3b — Post-reshuffle repair (`io/`, `math/`, `geometry/` packages) | ✅ Repair done; backlog open (below) |
| 4 — AD-readiness (gradcheck Tiers 0-2, `.detach()`/`.item()`/`.cpu()`/`.numpy()` audits) | ✅ **Done (2026-08-12)** — zero open findings |
| Repo weight (git-history rewrite) | ⏸ Deferred by decision |

Naming is fully unified: local dir == GitHub repo == import == PyPI dist for all
four packages (`warpSPH`, `warpSPHCore`, `warpSPHIntegrators`, `warpSPHPlotting`) —
see README for the current layout, not reconstructed here.

## Remaining work

### Legibility (Phase 3 / 3b backlog — no correctness stakes)

- [ ] **`__all__` coverage.** 94/274 files (34%) define one; the other 180 still
      star-export everything they import. `schemes/` (converted to explicit imports,
      `__all__` added) is the worked pattern for the rest — do opportunistically as
      modules get touched, not as a dedicated pass.
- [ ] **Dead commented-out code**, ~421 lines: `schemes/crkSPH.py` (119),
      `modules/mdbc/wp_nopenshift.py` (69), `shockCapturing/CullenDehnen2010.py` (63),
      `schemes/dfsph.py` (58), `caseUtils/compressible/sod/sod.py` (57),
      `schemes/deltaSPH.py` (55).
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
      - **Deferred, not built in this pass**: 2D and 3D variants of Sod (3D also
        needs a general plotting solution that doesn't exist yet), and a
        backprop-over-trajectory demo — two-stage per the user: (1) self-consistency
        recovery (perturb a known parameter, recover it via BPTT) first, as the
        straightforward case, then (2) fitting the SPH trajectory to
        `sodSolution.solve`'s analytic Riemann solution, which is more involved.
- [ ] **Notebook simplification, the rest.** 13,112 LOC across the other 33
      notebooks vs. 42 KB total for their equivalent `.py` scripts (both figures
      predate the Sod pilot above).
      - Boilerplate cell duplicated verbatim in 16 notebooks, still on the
        pre-`warpSPHBootstrap` config path — cheapest win, do first.
      - Apply the Sod pilot's pattern (real case code + generic helpers, visible
        step loop, explicit parameters cell) rather than collapsing to thin
        wrappers; keep genuinely notebook-shaped analysis/plotting as its own cells.
      - Worst offenders: `13-openFlow.ipynb` (1178 LOC, 3.6 MB), `12-dambreak.ipynb`
        (720 LOC), the three incompressible notebooks (~1950 combined).
- [ ] `datagen/weaklyCompressible/bak/` — 5 stale `.py` backups (114 KB), superseded
      by the Phase 2 runner conversion. Deletion candidate, not yet actioned.

### Cross-repo (owned by warpSPHCore) — DONE (2026-08-12)

- [x] `warpSPHCore.util.support.volumeToSupport` now dispatches on
      `isinstance(volume, torch.Tensor)` internally. `warpSPH/utils/support.py`'s
      wrapper collapsed to a plain re-export; verified scalar and tensor paths agree
      across all three dimensions, full suite still green.
- [x] `datagen/weaklyCompressible/loader.ipynb` — removed. `compressedLoader.ipynb`
      already reads the current (compressed) export layout, making the hardcoded-path
      loader redundant rather than something to fix.

### Won't fix / rejected (recorded so they aren't re-proposed)

- `math/` shadows the stdlib name — harmless under absolute imports, no replacement
  name was ever proposed, out of scope for the naming pass.
- Remaining basename collisions (`sod.py`, `noh.py`, `region.py`, `sdf.py`, the
  `caseUtils/compressible/*/sample.py` family) — each is case/package-namespaced on
  purpose and reads fine in context; not churned.
- `examples/weaklyCompressible/naca.ipynb`, `examples/incompressible/1d-test.ipynb`
  — deliberately have no case (standalone SDF viz / exploratory scratchpad, not
  published examples).
- Sedov's `initialization='hat'` — both notebooks crashed on this default; the case
  now defaults to `'singular'`. Reviving `'hat'` is a separate, unscoped job.

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
- `scripts/check_imports.py`, `scripts/run_tests.sh` (58 tests), `scripts/
  run_sweep.py` (25/25 cases) are the standing verification commands; prefer them
  over hand-rolled loops. The `gradcheck` skill (`.claude/skills/gradcheck/
  SKILL.md`) covers the 15 gradcheck scripts specifically.
