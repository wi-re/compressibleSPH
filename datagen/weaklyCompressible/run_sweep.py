#!/usr/bin/env python
"""Run sweep configs written by `obstacle_init.ipynb` -- one, many, or all.

    python run_sweep.py sweeps/dambreak/some_config.json      # one config
    python run_sweep.py sweeps/impact/some_config.json --plot # one, live
    python run_sweep.py --family dambreak                     # a whole family
    python run_sweep.py --family dambreak --family impact     # a few families
    python run_sweep.py --all                                 # every family
    python run_sweep.py --all --dry-run                       # list, don't run

`generator.py` only knows how to run `dambreakCase` -- it predates the
`impact` family. This dispatches on the family encoded in each config's own
parent directory (`sweeps/<family>/...`) via `utils.FAMILY_CASE`, so it runs
any sweep this directory generates, singly or in bulk, and archives every
result into `compressed/` the same way `generator.py` does.

`--family`/`--all` are anchored to this script's own location, not the
shell's current directory, so `cd datasets && python ../run_sweep.py --family
dambreak` still finds `datagen/weaklyCompressible/sweeps/dambreak/`. Output
(`compressed/`, `export/`) is *not* anchored the same way -- it follows the
current directory, exactly like `generator.py` and every other
`warpSPH.runner` case, so `cd datasets && python ../run_sweep.py --family
dambreak` writes `datasets/compressed/`, letting you sort different batches
into different output folders by choice of working directory. An explicit
config path/glob/directory you type is likewise resolved against the current
directory, the way any other CLI tool's path argument would be.

Meant to be left running unattended over a whole dataset:

- a config that already succeeded (its `<config>.done` sentinel exists) is
  skipped on the next invocation, so a batch can be killed and restarted
  without re-running everything;
- a config that raises, or whose run diverges, is logged and skipped rather
  than aborting the rest of the batch (`--stop-on-error` to abort instead);
- batches of more than one config run quiet (no per-run banner/progress bar
  -- otherwise hundreds of runs bury the one line per config that actually
  matters) and print one line per config instead. A single config keeps the
  normal, verbose `warpSPH.runner` output.
"""

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

import argparse                                              # noqa: E402
import glob                                                   # noqa: E402
import os                                                     # noqa: E402
import sys                                                    # noqa: E402
import time                                                   # noqa: E402
import traceback                                              # noqa: E402

from utils import FAMILY_CASE, archive                        # noqa: E402
from warpSPH.runner import CaseSpec, run                      # noqa: E402

#: This script's own directory -- what `--family`/`--all` anchor to, so they
#: still find the real sweeps regardless of the caller's current directory.
#: Output (compressed/export) deliberately does NOT use this -- see the
#: module docstring.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEPS_DIR = os.path.join(SCRIPT_DIR, 'sweeps')


def familyOf(configPath: str) -> str:
    family = os.path.basename(os.path.dirname(os.path.abspath(configPath)))
    if family not in FAMILY_CASE:
        raise ValueError(
            f'{configPath!r} is not under a known family directory. Expected '
            f'sweeps/{{{", ".join(sorted(FAMILY_CASE))}}}/...')
    return family


def _configGlob(pattern):
    """`glob.glob`, excluding `index.json` -- `writeSweep`'s metadata file for
    `sweep_browser.ipynb`, not a `CaseSpec`, but it matches every `*.json`
    glob a sweep directory would otherwise be expanded with."""
    return sorted(p for p in glob.glob(pattern) if os.path.basename(p) != 'index.json')


def expandConfigs(items, families, allFamilies):
    """Positional paths (files, directories, or shell-unexpanded globs),
    plus every config under each `--family`/`--all`, as one de-duplicated,
    order-preserving list."""
    paths = []
    for item in items:
        if os.path.isdir(item):
            paths += _configGlob(os.path.join(item, '*.json'))
        elif any(ch in item for ch in '*?['):
            paths += _configGlob(item)
        else:
            paths.append(item)

    wanted = sorted(FAMILY_CASE) if allFamilies else list(families)
    for family in wanted:
        if family not in FAMILY_CASE:
            raise ValueError(f'Unknown family {family!r}. Known: {sorted(FAMILY_CASE)}')
        paths += _configGlob(os.path.join(SWEEPS_DIR, family, '*.json'))

    seen, unique = set(), []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def runOne(path, *, plot, store, video, quiet):
    family = familyOf(path)
    case = FAMILY_CASE[family]
    spec = CaseSpec.load(path).merged(plot=plot, store=store, video=video, quiet=quiet)
    result = run(case, spec)
    archive(result, family)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='*',
                        help='sweeps/<family>/*.json paths, globs, or directories')
    parser.add_argument('--family', action='append', default=[], metavar='NAME',
                        help='run every config under sweeps/NAME/ (repeatable)')
    parser.add_argument('--all', action='store_true', help='run every config in every family')
    parser.add_argument('--plot', action='store_true', help='draw frames while each run proceeds')
    parser.add_argument('--no-store', dest='store', action='store_false', default=True,
                        help="don't write trajectory.h5 (on by default)")
    parser.add_argument('--video', action='store_true', help='encode each run to a video too')
    parser.add_argument('--force', action='store_true',
                        help='re-run configs that already have a .done sentinel')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='abort the batch on the first failed/diverged config')
    parser.add_argument('--limit', type=int, default=None,
                        help='run at most this many configs (after skipping done ones)')
    parser.add_argument('--dry-run', action='store_true',
                        help='list what would run, without running it')
    args = parser.parse_args()

    configs = expandConfigs(args.configs, args.family, args.all)
    if not configs:
        parser.error('nothing to run -- pass a config path, --family NAME, or --all')

    pending = [c for c in configs if args.force or not os.path.exists(c + '.done')]
    skipped = len(configs) - len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f'{len(configs)} config(s) selected'
         + (f', {skipped} already done (skipping)' if skipped else '')
         + f', {len(pending)} to run.')
    if args.dry_run:
        for path in pending:
            print(' ', path)
        return

    batch = len(pending) > 1
    failures = []
    diverged = []
    startedAll = time.time()
    for i, path in enumerate(pending, 1):
        print(f'[{i}/{len(pending)}] {path}', flush=True)
        startedOne = time.time()
        try:
            result = runOne(path, plot=args.plot, store=args.store, video=args.video, quiet=False)
        except Exception:
            traceback.print_exc()
            failures.append(path)
            if args.stop_on_error:
                break
            continue

        if result.diverged:
            print(f'  diverged at step {result.nSteps} ({time.time() - startedOne:.1f}s)')
            diverged.append(path)
            if args.stop_on_error:
                break
            continue

        if batch:
            print(f'  ok, {result.nSteps} steps ({time.time() - startedOne:.1f}s)')
        with open(path + '.done', 'w') as f:
            f.write(time.strftime('%Y-%m-%dT%H:%M:%S\n'))

    elapsed = time.time() - startedAll
    bad = failures + diverged
    print(f'\n{len(pending) - len(bad)}/{len(pending)} succeeded in {elapsed:.1f}s.')
    if failures:
        print(f'{len(failures)} failed:')
        for path in failures:
            print(' ', path)
    if diverged:
        print(f'{len(diverged)} diverged:')
        for path in diverged:
            print(' ', path)
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
