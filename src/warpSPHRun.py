"""CLI entry point: ``warpsph-run <case> [options]``.

Lives at top level, alongside :mod:`warpSPHBootstrap`, so that it can bootstrap
the precision *before* importing ``warpSPH``. ``python -m warpSPH.cases.sod``
works too, but by then ``warpSPHCore`` is already imported and ``--precision``
can no longer take effect.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from warpSPHBootstrap import PRECISIONS, bootstrap

__all__ = ['main']

_CASES = ('sod', 'tgv', 'dambreak')


def _splitArgv(argv: List[str]):
    """Peel off the case name and the pre-import flags; leave the rest for the case."""
    parser = argparse.ArgumentParser(
        prog='warpsph-run',
        description='Run a warpSPH case.',
        add_help=False,
    )
    parser.add_argument('case', nargs='?', choices=_CASES, help='which case to run')
    parser.add_argument('--precision', choices=PRECISIONS, default='float32',
                        help='scalar precision; must be chosen before import (default: float32)')
    parser.add_argument('--dim', default=None,
                        help="fixed dimension for warp types, or 'Any' (default: Any)")
    parser.add_argument('-h', '--help', action='store_true', dest='help')
    known, rest = parser.parse_known_args(argv)

    if known.case is None or known.help:
        parser.print_help()
        if known.case is None:
            print(f'\ncases: {", ".join(_CASES)}', file=sys.stderr)
            raise SystemExit(2 if not known.help else 0)
        rest = rest + ['--help']

    return known, rest


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known, rest = _splitArgv(argv)

    dim = known.dim
    if dim is None or str(dim).lower() in ('any', 'dynamic'):
        from typing import Any
        dim = Any
    else:
        dim = int(dim)

    bootstrap(precision=known.precision, dim=dim)

    import importlib
    module = importlib.import_module(f'warpSPH.cases.{known.case}')

    from warpSPH.runner import caseMain, getCase
    # Keep the precision the case sees consistent with what was bootstrapped.
    result = caseMain(getCase(known.case), rest + ['--precision', known.precision])
    return 1 if result.diverged else 0


if __name__ == '__main__':
    raise SystemExit(main())
