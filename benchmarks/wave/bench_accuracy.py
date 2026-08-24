#!/usr/bin/env python3
"""Temporal accuracy + measured convergence order of the time integrators on
the registered `waveEquation` case -- the suite version of the notebook's
"correctness check" cells (same-dt RK4 vs. SDIRK2+JFNK, and the
backwardEuler dissipation comparison).

Method (see `benchmarks/README.md` for the full rationale):

* one resolution (`--nx`), one simulated horizon (`--tEnd`), a `dt` grid
  `dt_CFL, dt_CFL/2, ...` where `dt_CFL` is the case's own CFL-derived dt
  (the notebook's `DT_DEFAULT`);
* the **reference** is a converged run of `--reference-scheme` (RK4 by
  default) one refinement level finer than the finest tested `dt` -- same
  spatial discretization as everything else, so the shared spatial error
  cancels and the measured error is temporal;
* every scheme runs every `dt` level; error is the relative L2 of the final
  `u` (and `v`) field against the reference; the effective order is the
  log-ratio between successive `dt` levels, which a correctly implemented
  order-p scheme tracks to its nominal `p` (and a 1st-order one like
  backward Euler at the notebook's dt shows as order ~1, i.e. its own
  numerical dissipation, exactly the notebook's cell 11 observation).

Usage (repo root, `warp` conda env)::

    python benchmarks/wave/bench_accuracy.py
    python benchmarks/wave/bench_accuracy.py --nx 64 --refinements 3
    python benchmarks/wave/bench_accuracy.py --schemes sdirk2_jfnk_jvp_1e-6 sdirk2_jfnk_fd_1e-6
    python benchmarks/wave/bench_accuracy.py --schemes all --no-plot
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from warpSPHBootstrap import bootstrap  # noqa: E402  (must precede warpSPH imports)


def parseArgs(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, default=64, help='lattice resolution (nx x nx particles)')
    ap.add_argument('--tEnd', type=float, default=0.5, help='simulated horizon for every run')
    ap.add_argument('--refinements', type=int, default=3,
                    help='number of dt levels: dt_CFL, dt_CFL/2, ..., dt_CFL/2^(N-1)')
    ap.add_argument('--reference-scheme', default='rk4',
                    help='registry key for the reference run (default: rk4)')
    ap.add_argument('--reference-refinements', type=int, default=1,
                    help='extra refinements the reference runs below the finest tested dt')
    ap.add_argument('--schemes', nargs='*', default=None,
                    help='registry keys (or "all"); default: the suite curated set')
    ap.add_argument('--device', default=None, help="torch device, e.g. 'cuda:0' or 'cpu'")
    ap.add_argument('--warmup', type=int, default=3, help='untimed steps before the timed region')
    ap.add_argument('--out', default=None, help='results directory (default: timestamped)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--param', action='append', default=[], metavar='NAME=VALUE',
                    help="extra case param override, e.g. --param obstacleEnabled=False "
                         "(true/false/int/float parsing)")
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

    from benchmarks.common import (ACCURACY_DEFAULT, getSchemes, runScheme,
                                   buildWaveCase, relL2, effectiveOrder, report, fmt)

    ctx, system, buildSeconds = buildWaveCase(nx=args.nx, device=args.device,
                                              **parseParams(args.param))
    dtCFL = float(ctx.config.dt)
    schemes = getSchemes(args.schemes, ACCURACY_DEFAULT)
    print(f'nx={args.nx}  ({system.state.u.shape[0]} particles)  dt_CFL={dtCFL:.5f}  '
          f'tEnd={args.tEnd}  device={rt.device}  build={buildSeconds:.2f}s')
    print(f'schemes: {", ".join(s.key for s in schemes)}')

    dts = [dtCFL / 2**i for i in range(args.refinements)]

    # --- reference: the notebook's "what is correct" anchor ------------------
    refSpec = getSchemes([args.reference_scheme], None)[0]
    dtRef = dtCFL / 2 ** (args.refinements + args.reference_refinements)
    nRef = max(1, round(args.tEnd / dtRef))
    dtRefA = args.tEnd / nRef
    print(f'\n[reference] {refSpec.label}: {nRef} steps of dt={dtRefA:.5f}')
    refRec = runScheme(ctx, system, refSpec, nRef, dtRefA, warmup=args.warmup)
    if refRec.diverged:
        raise SystemExit('reference run diverged; the whole comparison is void')
    refU, refV = refRec.uFinal, refRec.vFinal
    print(f'  reference: uMax={refRec.uMaxFinal:.4f}  '
          f'totalEnergy={refRec.energyFinal:.4f}  {refRec.fEvals} f-evals')

    # --- the sweep -----------------------------------------------------------
    records = []
    for spec in schemes:
        prev = None  # (errU, dtActual) of the coarser level, for the order
        for i, dt in enumerate(dts):
            n = max(1, round(args.tEnd / dt))
            dtA = args.tEnd / n
            rec = runScheme(ctx, system, spec, n, dtA, warmup=args.warmup)
            rec.buildSeconds = buildSeconds
            if rec.diverged:
                rec.extra = dict(dt=dtA, nSteps=n, errU=float('nan'), errV=float('nan'),
                                 estOrder=None, energyDrift=None)
                print(f'  {spec.key:28s} dt={dtA:.5f}: DIVERGED at step {rec.stepsDone}')
            else:
                errU = relL2(rec.uFinal, refU)
                errV = relL2(rec.vFinal, refV)
                order = (effectiveOrder(prev[0], errU, ratio=prev[1] / dtA)
                         if prev is not None else None)
                e0, eF = rec.energy0, rec.energyFinal
                drift = ((eF - e0) / abs(e0)) if (e0 is not None and eF is not None
                                                  and math.isfinite(e0) and e0 != 0
                                                  and math.isfinite(eF)) else None
                rec.extra = dict(dt=dtA, nSteps=n, errU=errU, errV=errV,
                                 estOrder=order, energyDrift=drift)
                if prev is not None:
                    print(f'  {spec.key:28s} dt={dtA:.5f}: errU={errU:.3e} errV={errV:.3e} '
                          f'order={order if order is None else round(order, 2)} '
                          f'{rec.msPerStep:.1f} ms/step ({rec.fEvalsPerStep:.1f} f/step)')
                else:
                    print(f'  {spec.key:28s} dt={dtA:.5f}: errU={errU:.3e} errV={errV:.3e} '
                          f'{rec.msPerStep:.1f} ms/step ({rec.fEvalsPerStep:.1f} f/step)')
            prev = (rec.extra['errU'], dtA) if not rec.diverged else None
            records.append(rec.toDict())


    # --- report ---------------------------------------------------------------
    outDir = report.outDirFor('accuracy', args.out)
    meta = report.environmentMeta(precision=rt.precision, extra=dict(
        nx=args.nx, tEnd=args.tEnd, dtCFL=dtCFL, refinements=args.refinements,
        reference=f'{refSpec.key} @ dt={dtRefA:.5f} ({nRef} steps)',
        referenceUmax=refRec.uMaxFinal, referenceEnergy=refRec.energyFinal,
        buildSeconds=buildSeconds))
    report.writeResults(outDir, 'accuracy', meta, records)

    rows = []
    for r in records:
        ex = r['extra']
        iters = '-'
        if r['solves']:
            iters = (f"{round(r['itersMean'], 1) if r['itersMean'] is not None else '-'}/"
                     f"{r['itersMax']}")
        rows.append([
            r['key'], fmt(ex.get('dt')), ex.get('nSteps', '-'),
            fmt(ex.get('errU'), '.3e'), fmt(ex.get('errV'), '.3e'),
            fmt(ex.get('estOrder'), '.2f'), fmt(ex.get('energyDrift'), '.3e'),
            fmt(r['msPerStep'], '.2f'), fmt(r['fEvalsPerStep'], '.1f'),
            (f"{r['convergedSolves']}/{r['solves']}" if r['solves'] else '-'),
            iters,
        ])
    table = report.mdTable(
        ['scheme', 'dt', 'steps', 'err u', 'err v', 'order', 'dE/E',
         'ms/step', 'f/step', 'solver conv', 'iters mean/max'], rows)
    text = (
        f"Reference: **{refSpec.label}** at dt={dtRefA:.5f} ({nRef} steps) on the "
        f"full case (sources + obstacle + absorbing border), nx={args.nx}.\n\n"
        f"`err u/v`: relative L2 of the final field vs. the reference (shared spatial "
        f"discretization, so temporal error only). `order`: measured convergence order "
        f"between successive dt levels (`-` at the coarsest). Case-specific caveat: at "
        f"this horizon the field still carries time-under-resolved high-wavenumber "
        f"content, so the measured order sits below the nominal one for every scheme "
        f"(~1, rising with refinement) -- the cross-scheme agreement at equal dt is "
        f"the correctness check. `solver conv`: "
        f"converged/stage-solves for implicit schemes; `iters` the per-solve iteration "
        f"count of the internal solver (mean/max).\n\n"
        f"{table}"
    )
    plots = []
    if not args.no_plot:
        plots.append(report.plotAccuracy(
            outDir, records,
            f'Wave case temporal error, nx={args.nx}, tEnd={args.tEnd}',
            refSpec.key, f'{refSpec.label} (ref)'))
    report.writeSummary(outDir, 'Wave case -- temporal accuracy of time integrators',
                        [('Method and results', text, plots)])
    print(f'\nwrote {outDir}/summary.md', end='')
    if plots:
        print(f' and {outDir}/accuracy_error_vs_dt.png')
    else:
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

