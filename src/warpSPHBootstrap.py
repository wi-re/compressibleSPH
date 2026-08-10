"""Pre-import runtime setup for warpSPH scripts and notebooks.

``warpSPHCore.type_config`` resolves its precision and dimension settings *at
import time*, and every ``warpSPH`` import transitively pulls in
``warpSPHCore``. The choice therefore has to be made before the first heavy
import, which is why this lives in a standalone top-level module rather than in
``warpSPH.runner``: importing ``warpSPH.runner`` would already be too late.

Keep this module free of heavy imports -- it is the one thing a script is
allowed to import first::

    from warpSPHBootstrap import bootstrap
    rt = bootstrap(precision='float32')

    from warpSPH import *
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

__all__ = ['Runtime', 'bootstrap', 'activePrecision', 'PRECISIONS']

# Mirrors warpSPHCore.type_config._PRECISION_ALIASES; kept here so the value can
# be validated (and offered on a CLI) without importing the core package first.
PRECISIONS = ('float16', 'half', 'float32', 'single', 'float64', 'double')

_CANONICAL = {
    'float16': 'float16', 'half': 'float16',
    'float32': 'float32', 'single': 'float32',
    'float64': 'float64', 'double': 'float64',
}


@dataclass
class Runtime:
    """What a bootstrapped process ended up running on."""

    device: Any
    dtype: Any
    precision: str
    dim: Any
    cuda: bool

    def __str__(self):
        return f'Runtime(device={self.device}, dtype={self.dtype}, precision={self.precision}, dim={self.dim})'


def activePrecision() -> Optional[str]:
    """Canonical name of the precision already locked in, or ``None``.

    Returns ``None`` when ``warpSPHCore`` has not been imported yet, i.e. while
    the precision is still free to be chosen.
    """
    module = sys.modules.get('warpSPHCore.type_config')
    if module is None:
        return None
    return _CANONICAL.get(getattr(module.get_precision(), '__name__', ''))


def bootstrap(
    precision: str = 'float32',
    dim: Any = Any,
    *,
    initWarp: bool = True,
    setArchList: bool = True,
    filterWarnings: bool = True,
    verbose: bool = False,
) -> Runtime:
    """Configure precision, initialize warp, and report the active runtime.

    This is the ~35-line boilerplate cell that opened every notebook and script,
    in one place. Calling it twice is safe as long as the precision matches what
    is already active; a mismatch raises rather than silently running at the
    wrong precision.

    ``setArchList`` pins ``TORCH_CUDA_ARCH_LIST`` to the local GPU, which cuts
    warp kernel compile times substantially.
    """
    if precision not in _CANONICAL:
        raise ValueError(
            f"Invalid precision {precision!r}. Expected one of: {', '.join(PRECISIONS)}."
        )
    requested = _CANONICAL[precision]

    active = activePrecision()
    if active is not None and active != requested:
        raise RuntimeError(
            f'warpSPHCore is already imported at {active} precision, so {requested} '
            f'can no longer be selected -- precision is fixed at first import. '
            f'Call bootstrap() before importing warpSPH, or set '
            f'warpSPHCore_PRECISION={requested} in the environment.'
        )

    import warpSPHCore_config as swc
    swc.configure(precision=requested, dim=dim)

    from warpSPHCore.type_config import get_type_config, get_torch_precision

    if initWarp:
        import warp as wp
        wp.init()

    import torch
    if setArchList and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{props.major}.{props.minor}'

    if filterWarnings:
        import warnings
        try:
            from tqdm import TqdmExperimentalWarning
            warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)
        except ImportError:
            pass

    cuda = torch.cuda.is_available()
    runtime = Runtime(
        device=torch.device('cuda:0') if cuda else torch.device('cpu'),
        dtype=get_torch_precision(),
        precision=requested,
        dim=dim,
        cuda=cuda,
    )

    if verbose:
        print(get_type_config())
        print(runtime)

    return runtime
