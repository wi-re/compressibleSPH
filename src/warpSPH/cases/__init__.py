"""Runnable cases.

Each module here declares one :class:`~warpSPH.runner.case.Case` and nothing
else -- the boilerplate, the config plumbing and the step loop all live in
:mod:`warpSPH.runner`. Import a case module to register it; ``warpSPHRun``
imports them all so it can dispatch by name.
"""

from __future__ import annotations

CASE_MODULES = ('sod', 'tgv', 'dambreak')


def importAll():
    """Import every case module so the registry is fully populated."""
    import importlib
    for name in CASE_MODULES:
        importlib.import_module(f'{__name__}.{name}')


__all__ = ['CASE_MODULES', 'importAll']
