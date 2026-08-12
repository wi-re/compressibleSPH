# warpSPH — Lessons Learned (cleanup sweep, 2026-08)

Reusable, still-relevant lessons from the pre-AD cleanup sweep tracked in
`CLEANUP_PLAN.md`. That file is a status tracker now; this file is the "why" and
"watch out for" that's worth keeping around. Not a chronological log — if something
here is superseded by a later fix, it's been removed rather than marked stale.

For the AD/gradcheck bug-class catalog specifically (nine bug classes, with fix
recipes, found rolling gradcheck out across 25 kernel files), see
**`.claude/skills/gradcheck/SKILL.md`** — not duplicated here.

## AD-readiness / autograd bridge

- **Scalar (non-array) kernel arguments cannot carry a gradient through
  warpSPHCore's autograd bridge at all**, regardless of whether the caller detaches
  them first. `warpWrapper2` casts every scalar `additionalArguments` entry through
  `scalar_t(...)` before its tensor/non-tensor split runs, and a kernel declaring
  `x: scalar_t` by value (not `wp.array(dtype=scalar_t)`) severs the tangent right
  there — "stop calling `.item()`" on the Python side is a no-op fix, since the
  severing point is the kernel signature, one layer up. The working pattern (added to
  warpSPHCore, proven on `computeCompSPHBalanceTermWarp`'s `dt`): declare the
  parameter as `wp.array(dtype=scalar_t)`, read `param[0]` once inside the kernel,
  wrap the Python-side call in `asScalarArg`. Check whether a target kernel receives
  a value as a plain scalar type before assuming a differentiability fix is just
  "stop detaching."
- **Adaptive `dt` genuinely carries a tangent back to state** (via sound speed →
  density) whenever `config.adaptiveDt` is set. This was the one confirmed
  tangent-dropping bug found across ~200 `.detach()`/`.item()`/`.cpu()`/`.numpy()`
  call sites, in two full audit passes. Most such calls are genuinely fine — loop-
  control scalars in fixed-point solvers, print/reporting boundaries, dead code whose
  only consumer is commented out. Don't assume a bare `.item()` sighting is a bug;
  trace it to its actual consumer before flagging it.
- **`x.clone()` followed by `x_.requires_grad = True` crashes if the caller's `x`
  already required grad** — `clone()` of a tensor that requires grad is a non-leaf,
  and PyTorch refuses to set `requires_grad` on a non-leaf. Guard with `if not
  x_.requires_grad: x_.requires_grad = True`. Hit in `geometry/sdf.py` and
  `regions/domainSDF.py`; was latent until a caller actually passes a
  `requires_grad=True` position tensor, which is exactly what forward-mode AD needs
  to do.
- **A bare `ParticleState` (warpSPHCore's own minimal fixture) is not a safe
  stand-in for this repo's scheme-level state objects.** Several kernels here read
  `internalEnergies`/`pressures`/velocity-derived fields that nothing validates are
  present, so a bare fixture can reach a kernel launch with `None` for one of them
  and segfault rather than raise. `scripts/_gradcheck_common.py`'s
  `make_compressible_state`/`compute_crk_state` exist because of this — use them (or
  an equally real state object) instead of the bare fixture for new gradcheck
  coverage of this repo's scheme layer.

## Testing / coverage gaps

- **A scheme or branch only reachable via a flag rots unnoticed if no test passes
  that flag.** Monaghan (`--scheme Monaghan`) had two independent breaking signature
  drifts — a missing `t` parameter on three boundary-condition helpers, a removed
  `supportScheme=` keyword — invisible because every default case runs CRKSPH or
  CompSPH. Fixed by adding a Sod-under-all-three-solvers test. When a code path is
  flag-gated, the coverage gap *is* the flag; add a test that passes it, not just a
  test of the default path.
- **Runtime-only import checking misses notebook/function-level imports.** A plain
  `python -c "import warpSPH"` never executes a notebook cell's own `from x import
  y`. `scripts/check_imports.py` needed a second pass — an AST scan of every `.py`
  file and notebook code cell, checking module *and* symbol — to catch what the
  runtime pass alone couldn't. Don't treat a clean top-level import as sufficient
  verification in a repo with notebooks.
- **A planning doc can lag reality in both directions.** Two items were already
  fixed by an earlier commit while still listed open; separately, three gradcheck
  scripts existed, worked, and were committed but were never wired into the test
  suite or written up. Before trusting a status claim in a planning doc, diff it
  against the actual code and test list rather than re-deriving or re-trusting the
  prose.

## Python/Warp gotchas

- **A keyword argument in a call doesn't rebind the caller's local of the same
  name.** `someWarpCall(..., c_max = c_max.cpu().item(), ...)` reads as if it
  detaches the outer `c_max`, but it only binds the callee's parameter — the
  caller's own `c_max` local is untouched, and any later use of it in the same scope
  still carries a gradient. Verify by variable *identity*, not name, whenever a
  `.item()`/`.detach()` shows up inside a call's keyword arguments.
- **The installed warp-lang version can drift silently.** It moved three times
  across this sweep (1.12.0 → a local 1.17.0.dev3 dev checkout → 1.15.0 from PyPI)
  because `pyproject.toml` deliberately pins nothing (see "Decisions in force" in
  CLEANUP_PLAN.md). A script that passed last session can fail this one with zero
  code changes on either side — check `python -c "import warp;
  print(warp.__version__)"` against what a script last passed under before assuming
  a new failure is a real code regression.
- **A kernel argument can be computed, threaded through several call layers, and
  never actually read inside the kernel.** `shifting/delta.py` computes `c_max`/`dx`
  and passes them into `computeDeltaShiftWarp`'s scalar args; the kernel uses them to
  compute a `shiftScaling` value it then never applies (the next line, `out +=
  shiftAmount`, ignores it). Harmless, but worth a grep for the parameter name
  *inside* the kernel body — not just at the call site — before trusting that a
  value passed to a kernel actually matters.

## Process / cross-repo

- **Cross-repo fixes belong in the repo that owns the bug, made once, on purpose —
  not worked around locally as a side effect of an unrelated task.**
  `correctGradientCRK`'s wrong-axis contraction, the autograd bridge's
  `requires_grad` dtype gate, `access_optional`, and `asScalarArg` were all found
  while working in this repo but fixed in `warpSPHCore`. `volumeToSupport`'s
  scalar-only limitation is the same shape and now resolved end to end: `warpSPH/
  utils/support.py` carried a tensor-aware local wrapper until core grew the same
  `isinstance(volume, torch.Tensor)` dispatch internally (2026-08-12), at which point
  the wrapper collapsed to a plain re-export — collapsing it *before* the upstream
  fix landed would have silently reintroduced the 2D/3D Monaghan crash the wrapper
  existed to prevent. General rule: a local wrapper papering over an upstream gap is
  a marker of unfinished work, not dead weight — don't remove it until the upstream
  fix is actually in place, and check the marker's precondition, not just its
  presence, before removing it.
- **Measure before acting on an inherited assumption.** The original repo-weight
  plan assumed ~380 MB of repeatedly-recommitted media; measuring first found the
  `.mp4`/`.gif` files were each committed exactly once (nothing to reclaim there) and
  the real bloat was 312 blob versions of 42 notebooks — a different fix entirely
  (`nbstripout`, not LFS). Re-measure a stale plan's numbers before executing its
  prescribed fix.
- **A rename/restructure that touches star-imported names can create an invisible
  shadowing trap.** The pre-`SchemeBundle` 7-tuple unpack bound a scheme's config
  class to the name `SimulationConfig` in 33 notebooks — the same name `from
  warpSPH import *` also provides for the *global* simulation config — so
  `schemeConfig = SimulationConfig()` was silently reading the shadowed local
  instead of the intended global class. When flattening a star-import surface or
  introducing a named-bundle API, grep the outgoing names against the incoming ones,
  not just for import breakage.
