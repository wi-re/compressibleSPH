"""The `if __name__ == '__main__'` block shared by every case module."""

from __future__ import annotations

import sys
from typing import List, Optional

from .case import Case
from .caseSpec import buildArgumentParser, specFromArgs
from .runner import RunResult, run

__all__ = ['caseMain']


def caseMain(case: Case, argv: Optional[List[str]] = None) -> RunResult:
    """Parse `argv` against `case`'s knobs and run it.

    Note that ``--precision`` can only take effect when the process was
    bootstrapped before ``warpSPH`` was imported -- see :mod:`warpSPHBootstrap`.
    Running a case module directly (``python -m warpSPH.cases.sod``) imports
    ``warpSPH`` first, so use ``python -m warpSPHRun sod`` or set
    ``warpSPHCore_PRECISION`` when a non-default precision is wanted.
    """
    parser = buildArgumentParser(
        description=case.description or f'Run the {case.name} case.',
        caseParams=case.params,
    )
    args = parser.parse_args(argv)

    defaults = dict(case.defaults)
    defaults.setdefault('caseName', case.name)
    defaults.setdefault('scheme', case.scheme)
    spec = specFromArgs(args, caseParams=case.params, defaults=defaults)

    from warpSPHBootstrap import activePrecision, bootstrap
    active = activePrecision()
    if active is not None and active != spec.precision:
        print(f'warning: requested precision {spec.precision!r} but warpSPHCore is '
              f'already running at {active!r}; continuing at {active!r}. Use '
              f'`python -m warpSPHRun {case.name} --precision {spec.precision}` to '
              f'select it before import.', file=sys.stderr)
        spec = spec.merged(precision=active)
    bootstrap(precision=spec.precision, verbose=spec.verbose)

    return run(case, spec)
