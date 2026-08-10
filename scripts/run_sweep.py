#!/usr/bin/env python3
"""Run every registered case and report which ones survive.

Each case runs in its **own process**, sequentially -- a case must be able to
tear down and be analysed before the next one starts, and one crashing case
must not take the sweep with it.

By default this is a *smoke* sweep: every case runs for a handful of steps just
to prove it builds, samples, and steps without blowing up. That is the mode you
want after refactoring. ``--full`` instead lets every case run to its own
``tLimit``, which is a real (long) production sweep.

Usage::

    scripts/run_sweep.py                     # smoke sweep, 5 steps per case
    scripts/run_sweep.py --nSteps 25         # a bit deeper
    scripts/run_sweep.py --full              # run every case to completion
    scripts/run_sweep.py --cases sod noh     # only these
    scripts/run_sweep.py --skip dambreak     # everything but these
    scripts/run_sweep.py --list              # show the registry and exit
    scripts/run_sweep.py -- --scheme CRKSPH  # pass extra flags to every case

Everything after ``--`` is forwarded verbatim to every case, so the per-case
sweep configs in ``examples/sweeps/`` compose with this::

    scripts/run_sweep.py --cases sod -- --config examples/sweeps/sod_highres.yaml

Output (logs + per-case export trees + ``summary.json``) lands in a timestamped
directory under ``sweeps/`` so repeat sweeps never overwrite each other.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Cases that are expected to be slow even in smoke mode, purely informational.
SLOW_HINT = {"dambreak", "openFlow", "movingObstacle", "triplePoint"}


@dataclass
class Result:
    case: str
    status: str          # ok | fail | timeout
    seconds: float
    returncode: int | None
    logPath: str


def discoverCases() -> list[dict]:
    """Ask a child process for the case registry.

    Done out-of-process so the sweep driver never imports warpSPH itself --
    importing it would lock the precision for the whole run.
    """
    code = (
        "from warpSPH.cases import importAll; importAll()\n"
        "from warpSPH.runner import listCases, getCase\n"
        "import json\n"
        "print('__CASES__' + json.dumps("
        "[{'name': n, 'description': getCase(n).description} for n in listCases()]))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__CASES__"):
            return json.loads(line[len("__CASES__"):])
    sys.stderr.write(proc.stdout + proc.stderr)
    raise SystemExit("could not enumerate cases -- see output above")


def displayPath(path: Path) -> str:
    """Repo-relative when it lives under the repo, absolute otherwise."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def runCase(case: str, sweepDir: Path, args, passthrough: list[str]) -> Result:
    logPath = sweepDir / "logs" / f"{case}.log"
    logPath.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "warpSPHRun", case,
           "--precision", args.precision,
           "--exportRoot", str(sweepDir / "export")]
    if not args.full:
        cmd += ["--nSteps", str(args.nSteps)]
    if args.store:
        cmd += ["--store"]
    if not args.verbose:
        cmd += ["--quiet"]
    cmd += passthrough

    start = time.monotonic()
    with logPath.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
            returncode = proc.returncode
            status = "ok" if returncode == 0 else "fail"
        except subprocess.TimeoutExpired:
            returncode = None
            status = "timeout"
            log.write(f"\n\n== killed after {args.timeout}s ==\n")
    seconds = time.monotonic() - start
    return Result(case, status, seconds, returncode, displayPath(logPath))


def tail(path: Path, n: int = 12) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "<no log>"
    return "\n".join(f"      | {line}" for line in lines[-n:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", metavar="CASE", help="only run these cases")
    parser.add_argument("--skip", nargs="+", metavar="CASE", default=[], help="skip these cases")
    parser.add_argument("--nSteps", type=int, default=5, help="steps per case in smoke mode (default: 5)")
    parser.add_argument("--full", action="store_true", help="run each case to its own tLimit instead")
    parser.add_argument("--precision", default="float32", help="scalar precision (default: float32)")
    parser.add_argument("--timeout", type=float, default=900, help="per-case timeout in seconds (default: 900)")
    parser.add_argument("--store", action="store_true", help="also write trajectories, not just run")
    parser.add_argument("--outRoot", default=None, help="parent for sweep dirs (default: <repo>/sweeps)")
    parser.add_argument("--list", action="store_true", help="list the registered cases and exit")
    parser.add_argument("--verbose", action="store_true", help="do not pass --quiet to the cases")
    parser.add_argument("rest", nargs="*", help="extra flags forwarded to every case (after --)")
    args = parser.parse_args()

    registry = discoverCases()
    names = [c["name"] for c in registry]

    if args.list:
        for entry in registry:
            print(f"  {entry['name']:<18} {entry['description']}")
        return 0

    selected = names if not args.cases else list(args.cases)
    unknown = [c for c in selected if c not in names]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(names)}", file=sys.stderr)
        return 2
    selected = [c for c in selected if c not in args.skip]

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outRoot = Path(args.outRoot) if args.outRoot else REPO / "sweeps"
    sweepDir = outRoot / f"sweep_{stamp}"
    sweepDir.mkdir(parents=True, exist_ok=True)

    mode = "full" if args.full else f"smoke ({args.nSteps} steps)"
    print(f"== warpSPH sweep -- {mode}, {len(selected)} cases, {args.precision} ==")
    print(f"   output: {sweepDir}")
    print()

    results: list[Result] = []
    for i, case in enumerate(selected, 1):
        hint = "  (slow)" if case in SLOW_HINT and args.full else ""
        print(f"[{i:>2}/{len(selected)}] {case:<18}{hint}", end="", flush=True)
        result = runCase(case, sweepDir, args, args.rest)
        results.append(result)
        mark = {"ok": "ok", "fail": "FAIL", "timeout": "TIMEOUT"}[result.status]
        print(f" {mark:>8}  {result.seconds:6.1f}s")

    summary = {
        "timestamp": stamp,
        "mode": mode,
        "precision": args.precision,
        "extraArgs": args.rest,
        "results": [asdict(r) for r in results],
    }
    (sweepDir / "summary.json").write_text(json.dumps(summary, indent=2))

    failed = [r for r in results if r.status != "ok"]
    print()
    print(f"== {len(results) - len(failed)}/{len(results)} passed ==")
    if failed:
        print()
        for result in failed:
            print(f"  {result.case} [{result.status}] -> {result.logPath}")
            print(tail(Path(result.logPath) if os.path.isabs(result.logPath) else REPO / result.logPath))
            print()
    print(f"summary: {sweepDir / 'summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
