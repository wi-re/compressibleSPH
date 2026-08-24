#!/usr/bin/env python3
"""Stability limit + internal-solver loop limits on the `waveEquation` case --
the systematized version of the notebook's "payoff" section (RK4 diverges at
32x its CFL dt; SDIRK2 + JFNK stays bounded), extended to every scheme and to
the internal solver settings that decide whether an implicit run at large dt
is actually solving anything.

For each scheme and each `dt = mult * dt_CFL`:

* explicit schemes show where they stop: the first multiplier at which
  max|u| goes non-finite within `--steps` steps (`diverged`), the damped
  border term's `dt < ~2.8/dampingStrength` limit being the binding one for
  the high-order ones (the case's own `cflFactor=0.1` comment made concrete);
* implicit schemes report two things the state alone cannot:
  (a) whether the run stayed finite and bounded (`bounded`), and
  (b) whether the **internal solver converged** -- `converged/solves` and the
  per-solve iteration count. A `JFNKSolver(max_iterations=5)` at 64x dt that
  stays finite but hits its budget every stage is a different (worse) result
  than one that converges in 2 iterations, and only this column tells them
  apart; Picard's fixed count has no such signal at all, which is precisely
  why the notebook (and NOTES.md S3.4) move to JFNK past the Picard limit.

Usage (repo root, `warp` conda env)::

    python benchmarks/wave/bench_stability.py
    python benchmarks/wave/bench_stability.py --multipliers 1 4 16 64 --steps 20
    python benchmarks/wave/bench_stability.py --implicit-schemes sdirk2_jfnk_jvp_1e-8
    python benchmarks/wave/bench_stability.py --explicit-schemes none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from warpSPHBootstrap import bootstrap  # noqa: E402  (must precede warpSPH imports)


def parseArgs(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, default=32,
                    help='lattice resolution (kept small: the point is dt, not cost)')
    ap.add_argument('--multipliers', type=float, nargs='+', default=[1, 2, 4, 8, 16, 32, 64],
                    help='dt multipliers of the case CFL dt to sweep')
    ap.add_argument('--steps', type=int, default=10, help='steps per (scheme, dt) run')
    ap.add_argument('--explicit-schemes', nargs='*', default=None, metavar='KEY',
                    help='registry keys (or "all" / "none"); default: the curated set')
    ap.add_argument('--implicit-schemes', nargs='*', default=None, metavar='KEY',
                    help='registry keys (or "all" / "none"); default: the loop-limit matrix')
    ap.add_argument('--bounded-factor', type=float, default=10.0,
                    help='"bounded" = finite and max|u| <= factor x initial max|u| throughout')
    ap.add_argument('--device', default=None, help="torch device, e.g. 'cuda:0' or 'cpu'")
    ap.add_argument('--out', default=None, help='results directory (default: timestamped)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--param', action='append', default=[], metavar='NAME=VALUE',
                    help="extra case param override, e.g. --param obstacleEnabled=False")
    return ap.parse_args(argv)


def parseParams(pairs: List[str]) -> dict:
    """`--param NAME=VALUE` pairs -> dict, with true/false/int/float parsing."""
    out = {}
    for pair in pairs:
        if '=' not in pair:
            raise SystemExit(f'--param expects NAME=VALUE, got {pair!r}')
        name, value = pair.split('=', 1)
        low = value.lower()
        if low in ('true', 'false'):
            out[name] = (low == 'true')
        else:
            try:
                out[name] = int(value)
            except ValueError:
                try:
                    out[name] = float(value)
                except ValueError:
                    out[name] = value
    return out


def resolveList(keys: List[str], default: List[str]) -> List[str]:
    """`--explicit-schemes`/`--implicit-schemes`: omitted -> the suite default,
    "none" -> empty (skip that family), "all" is handled by `getSchemes`,
    otherwise the named keys."""
    if not keys:
        return default
    if len(keys) == 1 and keys[0].lower() == 'none':
        return []
    return keys


def main(argv: List[str] = None) -> int:
    args = parseArgs(argv if argv is not None else sys.argv[1:])
    rt = bootstrap(precision='float32')

    from benchmarks.common import (STABILITY_EXPLICIT_DEFAULT, STABILITY_IMPLICIT_DEFAULT,
                                   getSchemes, runScheme, buildWaveCase, report, fmt)

    ctx, system, buildSeconds = buildWaveCase(nx=args.nx, device=args.device,
                                              **parseParams(args.param))
    dtCFL = float(ctx.config.dt)
    explicitKeys = resolveList(args.explicit_schemes, STABILITY_EXPLICIT_DEFAULT)
    implicitKeys = resolveList(args.implicit_schemes, STABILITY_IMPLICIT_DEFAULT)
    schemes = getSchemes(explicitKeys, []) + getSchemes(implicitKeys, [])
    print(f'nx={args.nx}  ({system.state.u.shape[0]} particles)  dt_CFL={dtCFL:.5f}  '
          f'steps={args.steps}  device={rt.device}')
    print(f'explicit ({len(explicitKeys)}): {", ".join(explicitKeys) or "-"}')
    print(f'implicit ({len(implicitKeys)}): {", ".join(implicitKeys) or "-"}')

    records = []
    for spec in schemes:
        for mult in args.multipliers:
            dt = mult * dtCFL
            # warmup=0 on purpose: the stability behaviour *is* the early steps.
            rec = runScheme(ctx, system, spec, args.steps, dt, warmup=0,
                            trackU=True, keepFields=False)
            rec.buildSeconds = buildSeconds
            uMax0 = rec.uMax0
            peak = rec.uMaxPeak
            bounded = (not rec.diverged and peak is not None
                       and peak <= args.bounded_factor * uMax0)
            rec.extra = dict(mult=mult, dt=dt, bounded=bounded)
            tag = 'DIVERGED' if rec.diverged else ('bounded' if bounded else 'UNBOUNDED')
            conv = f" conv={rec.convergedSolves}/{rec.solves}" if rec.solves else ''
            iters = (f" iters={round(rec.itersMean, 1) if rec.itersMean is not None else '-'}/max{rec.itersMax}"
                     if rec.solves else '')
            print(f'  {spec.key:28s} x{mult:<4g} peak|u|={fmt(peak, ".3g")} {tag}{conv}{iters}')
            records.append(rec.toDict())


    # --- report ---------------------------------------------------------------
    outDir = report.outDirFor('stability', args.out)
    meta = report.environmentMeta(precision=rt.precision, extra=dict(
        nx=args.nx, dtCFL=dtCFL, multipliers=args.multipliers, steps=args.steps,
        boundedFactor=args.bounded_factor))
    report.writeResults(outDir, 'stability', meta, records)

    rows = []
    for r in records:
        ex = r['extra']
        iters = '-'
        if r['solves']:
            iters = (f"{round(r['itersMean'], 1) if r['itersMean'] is not None else '-'}/"
                     f"{r['itersMax']}")
        status = 'diverged' if r['diverged'] else ('bounded' if ex['bounded'] else 'unbounded')
        rows.append([
            r['key'], f"x{ex['mult']:g}", ex['dt'],
            fmt(r['uMaxPeak'], '.3g'), status,
            (f"{r['convergedSolves']}/{r['solves']}" if r['solves'] else '-'),
            iters, fmt(r['msPerStep'], '.1f'),
        ])
    table = report.mdTable(
        ['scheme', 'dt mult', 'dt', 'peak max|u|', 'status', 'solver conv',
         'iters mean/max', 'ms/step'], rows)
    text = (
        f"dt pushed past the explicit stability limit (the case's own comment: "
        f"`cflFactor=0.3` already blows up; the binding limit is the absorbing "
        f"border's linear damping, `dt < ~2.8/dampingStrength`). `bounded` = "
        f"finite and peak max|u| <= {args.bounded_factor:g}x the initial value for "
        f"the whole run. `solver conv` = converged/stage-solves: a JFNK that "
        f"used up `max_iterations` without converging shows < 1.0 here even "
        f"when the state stayed finite; Picard (fixed count) has no such "
        f"signal, which is the point of the JFNK rung. Caveat under float32: "
        f"the DIRK driver checks convergence with its own weighted norm "
        f"(rtol=1e-3), whose residual floor sits above the JFNK `tol`, so the "
        f"flag typically reads 0/N (at large dt the stage residual is far above "
        f"the floor) and the budget-exhausted `iters` column (== max_iterations) "
        f"is the honest signal; under float64 the "
        f"flag itself discriminates. `iters` is the per-solve iteration count "
        f"of the internal solver (mean/max): for JFNK it grows with dt (GMRES "
        f"work against a stiffer stage operator) even though Newton itself "
        f"converges in one correction on this linear problem.\n\n"
        f"{table}"
    )
    plots = []
    if not args.no_plot:
        plots.append(report.plotStability(
            outDir, records,
            f'Wave case stability, nx={args.nx}, {args.steps} steps per run',
            args.bounded_factor))
    report.writeSummary(outDir, 'Wave case -- stability past the explicit limit, solver loop limits',
                        [('Method and results', text, plots)])
    print(f'\nwrote {outDir}/summary.md', end='')
    if plots:
        print(f' and {outDir}/stability.png')
    else:
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

