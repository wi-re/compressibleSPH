"""Bootstrap the runtime before anything imports ``warpSPH``.

pytest imports ``conftest.py`` before collecting test modules, which is the only
place precision can still be chosen -- see :mod:`warpSPHBootstrap`.
"""

import pytest

from warpSPHBootstrap import bootstrap

RUNTIME = bootstrap(precision='float32')


@pytest.fixture(scope='session')
def runtime():
    return RUNTIME


@pytest.fixture(scope='session')
def exportRoot(tmp_path_factory):
    """Keep any test that stores output out of the repo's `export/` tree."""
    return str(tmp_path_factory.mktemp('export'))
