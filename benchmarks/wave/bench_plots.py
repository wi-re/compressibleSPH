#!/usr/bin/env python3
"""Turn stored suite run outputs into a set of scaling graphs.

Post-processor for the three wave suites: it reads one or more run outputs
(a results directory, i.e. whatever `--out` wrote, or a `results.json` file
directly) and renders the full scaling graph set for that suite -- one
figure per quantity (cost, error, memory, solver activity, ...), a combined
overview grid, and a `scaling.md` index with the provenance and the fitted
log-log scaling exponents (`benchmarks/common/scaling.py`). Nothing is
re-run: this only reads the JSON, so it is cheap, needs no GPU, and can
overlay several runs of the same suite (each run's records get tagged and
appear as separate legend entries).

Usage (repo root, `warp` conda env -- any env with matplotlib works, the
suite code is not imported)::

    python -m benchmarks.wave.bench_plots /tmp/full_perf
    python -m benchmarks.wave.bench_plots results/accuracy_2026-08-24_10-00-00
    python -m benchmarks.wave.bench_plots runA runB --out /tmp/plots   # overlay
    python -m benchmarks.wave.bench_plots /tmp/full_stab --no-overview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.common import scaling  # noqa: E402


def parseArgs(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', metavar='RESULTS_DIR_OR_JSON',
                    help="a suite run output (its results directory or the "
                         "results.json file); all inputs must be the same suite")
    ap.add_argument('--out', default=None,
                    help='where to write the graph set (default: timestamped '
                         'results/plots_<stamp>)')
    ap.add_argument('--title', default=None,
                    help='figure title (default: derived from the suite)')
    ap.add_argument('--no-overview', action='store_true',
                    help='skip the combined overview grid')
    return ap.parse_args(argv)


_SUITE_TITLES = {
    'performance': 'Wave case -- integrator cost and memory scaling',
    'accuracy': 'Wave case -- temporal accuracy scaling',
    'stability': 'Wave case -- stability envelope and solver-loop scaling',
}


def main(argv: List[str] = None) -> int:
    args = parseArgs(argv if argv is not None else sys.argv[1:])

    inputs = [scaling.loadInput(p) for p in args.inputs]
    suites = {payload.get('suite') for _, payload in inputs}
    if len(suites) != 1 or None in suites:
        raise SystemExit(f'inputs must all be the same suite, got {sorted(map(str, suites))}')
    suite = inputs[0][1]['suite']
    meta = inputs[0][1].get('meta', {})
    records: List[dict] = []
    multi = len(inputs) > 1
    for label, payload in inputs:
        for r in payload['records']:
            if multi:
                r = dict(r)
                r['run'] = label
            records.append(r)

    boundedFactor = meta.get('boundedFactor') if suite == 'stability' else None
    title = args.title or _SUITE_TITLES.get(suite, f'waveEquation case -- {suite}')

    from benchmarks.common import report  # for the timestamped out-dir convention
    outDir = report.outDirFor('plots', args.out)
    names = scaling.plotScalingSet(records, suite, outDir, title,
                                   boundedFactor=boundedFactor,
                                   overview=not args.no_overview, inputs=inputs)

    print(f'suite: {suite}  ({len(records)} records from {len(inputs)} run(s))')
    for n in names:
        print(f'  {outDir / n}')
    print(f'  {outDir / "scaling.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
