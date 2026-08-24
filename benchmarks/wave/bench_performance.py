#!/usr/bin/env python3
"""Runtime and memory of the time integrators on the `waveEquation` case,
vs. particle count -- the suite version of the question the notebook leaves
implicit ("how much does the implicit solve cost, and does it scale like the
explicit step?").

For each resolution in `--nxs` (particle count = nx^2) and each scheme:

* `buildSeconds` -- case sampling + Verlet-list + IC cost (measured once per
  resolution; the build is scheme-independent);
* `msPerStep` / `msPerRhs` -- steady-state step cost (untimed warmup first,
  so warp kernel compilation never enters the number) and the per-RHS cost,
  which is the fair comparison between a 1-eval Euler step and a ~70-eval
  JFNK step;
* `fEvalsPerStep` -- the measured right-hand-side count, i.e. the price a
  scheme actually pays (multistep schemes are run with `history` threaded,
  so they show their nominal 1, not their 5-6 evaluation starter);
* memory -- peak allocated / reserved GPU memory for the run (allocator
  stats reset per run) plus the static state+adjacency footprint in MB and
  KB per particle.

The summary adds a log-log scaling exponent per scheme (slope ~1: linear in
particle count, the expectation for a fixed-support-radius SPH step with a
cell list; ~2: the naive all-pairs regime).

Usage (repo root, `warp` conda env)::

    python benchmarks/wave/bench_performance.py
    python benchmarks/wave/bench_performance.py --nxs 32 64 128 --steps 40
    python benchmarks/wave/bench_performance.py --schemes rk4 sdirk2_jfnk_jvp_1e-6
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
    ap.add_argument('--nxs', type=int, nargs='+', default=[32, 64, 128, 256],
                    help='lattice resolutions to sweep (particle count = nx^2)')
    ap.add_argument('--steps', type=int, default=25, help='timed steps per run')
    ap.add_argument('--warmup', type=int, default=50,
                    help='untimed steps before the timed region (absorbs warp '
                         'kernel loads AND the GPU clock ramp: a cold-GPU first '
                         'run otherwise biases the smallest-N point -- and the '
                         'fitted slope -- upward)')
    ap.add_argument('--schemes', nargs='*', default=None,
                    help='registry keys (or "all"); default: the suite curated set')
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


def main(argv: List[str] = None) -> int:
    args = parseArgs(argv if argv is not None else sys.argv[1:])
    rt = bootstrap(precision='float32')

    from benchmarks.common import (PERFORMANCE_DEFAULT, getSchemes, runScheme,
                                   buildWaveCase, loglogFit, report, fmt)

    schemes = getSchemes(args.schemes, PERFORMANCE_DEFAULT)
    overrides = parseParams(args.param)
    print(f'nx sweep: {args.nxs}   steps={args.steps} (after {args.warmup} warmup)   '
          f'device={rt.device}')
    print(f'schemes: {", ".join(s.key for s in schemes)}')

    # Prime the warp kernel cache once (a throwaway tiny build + step) so the
    # one-time compilation/loads never land inside a measured build or step.
    primeNx = min(args.nxs) // 2 or 8
    pCtx, pSystem, _ = buildWaveCase(nx=primeNx, device=args.device, **overrides)
    pDt = float(pCtx.config.dt)
    runScheme(pCtx, pSystem, getSchemes(['rk4'], None)[0], 1, pDt, warmup=0,
              keepFields=False)
    print(f'primed warp kernels with nx={primeNx} (cost excluded)')

    records = []
    for nx in args.nxs:
        ctx, system, buildSeconds = buildWaveCase(nx=nx, device=args.device, **overrides)
        dtCFL = float(ctx.config.dt)
        nParticles = int(system.state.u.shape[0])
        print(f'\nnx={nx}  ({nParticles} particles)  dt_CFL={dtCFL:.5f}  '
              f'build={buildSeconds:.2f}s')
        for spec in schemes:
            rec = runScheme(ctx, system, spec, args.steps, dtCFL,
                            warmup=args.warmup, keepFields=False)
            rec.buildSeconds = buildSeconds
            rec.extra = dict(nx=nx, nParticles=nParticles, dt=dtCFL)
            kbPart = (rec.peakAllocatedMB * 1024.0 / nParticles
                      if rec.peakAllocatedMB > 0 else None)
            print(f'  {spec.key:28s} {rec.msPerStep:9.2f} ms/step  '
                  f'{rec.msPerRhs:8.3f} ms/RHS  {rec.fEvalsPerStep:5.1f} f/step  '
                  f'peak {rec.peakAllocatedMB:8.1f} MB ({fmt(kbPart, ".0f")} KB/particle)')
            records.append(rec.toDict())


    # --- scaling exponents ----------------------------------------------------
    slopes: dict = {}
    for spec in schemes:
        pts = [(r['nParticles'], r['msPerStep']) for r in records if r['key'] == spec.key]
        fit = loglogFit([x for x, _ in pts], [y for _, y in pts])
        if fit is not None:
            slopes[spec.key] = fit[0]

    # --- report ---------------------------------------------------------------
    outDir = report.outDirFor('performance', args.out)
    meta = report.environmentMeta(precision=rt.precision, extra=dict(
        nxs=args.nxs, steps=args.steps, warmup=args.warmup))
    report.writeResults(outDir, 'performance', meta, records)

    rows = []
    for r in records:
        kbPart = (r['peakAllocatedMB'] * 1024.0 / r['nParticles']
                  if r['peakAllocatedMB'] > 0 else None)
        rows.append([
            r['key'], r['extra']['nx'], f"{r['nParticles']:,}",
            fmt(r['buildSeconds'], '.2f'),
            fmt(r['msPerStep'], '.2f'), fmt(r['msPerRhs'], '.3f'),
            fmt(r['fEvalsPerStep'], '.1f'),
            fmt(r['peakAllocatedMB'], '.1f'), fmt(r['peakReservedMB'], '.1f'),
            fmt(r['staticStateMB'], '.2f'), fmt(kbPart, '.0f'),
        ])
    table = report.mdTable(
        ['scheme', 'nx', 'particles', 'build s', 'ms/step', 'ms/RHS', 'f/step',
         'peak alloc MB', 'peak reserved MB', 'static MB', 'KB/particle'], rows)

    slopeRows = [[k, fmt(s, '.2f'), 'linear (cell-list SPH step)' if s < 1.4
                  else 'super-linear (check neighbour search)' if s < 2.0
                  else '~quadratic (all-pairs regime)']
                 for k, s in sorted(slopes.items(), key=lambda kv: -kv[1])]
    slopeTable = report.mdTable(['scheme', 'log-log slope (ms/step vs N)', 'reading'],
                                slopeRows)
    text = (
        f"Steady-state step cost and memory per resolution. `ms/RHS` is the fair "
        f"cross-scheme number: a JFNK step is one stage solve worth of Newton "
        f"corrections x GMRES iterations of right-hand-side evaluations, and this "
        f"column is what one of those costs. `f/step` is the measured count "
        f"(multistep schemes are run with `history` threaded, so they show their "
        f"nominal 1, not the 5-6 evaluations of the un-threaded Dormand-Prince "
        f"starter). `build s` is scheme-independent (case sampling + Verlet list + "
        f"IC) and is identical within a row group. `peak alloc MB` is the "
        f"allocator high-water mark for the whole run (reset per run); "
        f"`static MB` is the state+adjacency footprint before any integration. "
        f"Memory numbers are CUDA allocator stats on GPU; on CPU they fall back "
        f"to the process RSS high-water mark.\n\n"
        f"{table}\n\n"
        f"## Scaling\n\n"
        f"{slopeTable}"
    )
    plots = []
    if not args.no_plot:
        plots.append(report.plotPerformance(
            outDir, records, f'Wave case integrator cost, steps={args.steps}', slopes))
    report.writeSummary(outDir, 'Wave case -- integrator runtime and memory vs. particle count',
                        [('Method and results', text, plots)])
    print(f'\nwrote {outDir}/summary.md', end='')
    if plots:
        print(f' and {outDir}/performance.png')
    else:
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

