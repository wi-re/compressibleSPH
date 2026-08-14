"""The argparse surface that used to be copy-pasted per script, as data.

``CaseSpec`` is the union of the knobs that ``examples/*/01-*.py`` and
``datagen/weaklyCompressible/parser.py`` each declared separately. Because it is
a dataclass of plain scalars it round-trips through JSON and YAML, so a sweep is
a directory of config files rather than a shell script full of flag strings.
Case-specific knobs live in :attr:`CaseSpec.params` and are declared by the
:class:`~warpSPH.runner.case.Case` itself.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional

__all__ = ['CaseSpec', 'buildArgumentParser', 'schemeNames', 'specFromArgs']


def _readMapping(path: str) -> Dict[str, Any]:
    """Load a ``.json``/``.yaml``/``.yml`` file, requiring a top-level mapping."""
    with open(path, 'r') as f:
        if os.path.splitext(path)[1].lower() in ('.yaml', '.yml'):
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'{path} must contain a mapping, got {type(data).__name__}.')
    return data


@dataclass
class CaseSpec:
    """Everything needed to run a case, minus the case's own physics."""

    caseName: str = 'case'

    # --- discretisation -----------------------------------------------------
    nx: int = 128
    dim: int = 2
    L: float = 2.0
    n_h: float = 4.0
    periodic: bool = True

    # --- scheme selection ---------------------------------------------------
    # Held as strings so a spec stays serialisable; resolved to enums in
    # `warpSPH.runner.runner` via the parse* helpers in `warpSPH.io`.
    scheme: Optional[str] = None
    kernel: str = 'Wendland2'
    integrationScheme: str = 'rungeKutta2'
    supportMode: str = 'SuperSymmetric'
    gradientMode: str = 'Difference'
    laplacianMode: str = 'Brookshaw'
    samplingScheme: str = 'regular'
    verletScale: Optional[float] = None

    # --- time stepping ------------------------------------------------------
    tLimit: float = 1.0
    dt: Optional[float] = None
    adaptiveDt: bool = True
    cflFactor: float = 0.3
    minDt: Optional[float] = 1e-8
    maxDt: Optional[float] = 1e-2
    # When set, overrides the tLimit-derived step count. Tests use this to run a
    # fixed, short trajectory without having to reason about the adaptive dt.
    nSteps: Optional[int] = None

    # --- runtime ------------------------------------------------------------
    precision: str = 'float32'
    device: Optional[str] = None

    # --- output -------------------------------------------------------------
    plot: bool = False
    plotInterval: int = 10
    #: Open a live, updating window alongside the exported frames. Ignored when
    #: matplotlib has no interactive backend, so leaving it on is safe headless.
    show: bool = True
    #: Block on the final figure when the run ends, instead of closing it.
    #: `caseMain` turns this on -- a person at a console wants to look at the
    #: result -- while a programmatic `run()` leaves it off so nothing stalls.
    holdPlot: bool = False
    #: 'matplotlib', 'vispy' or 'pyVista'. `None` picks by dimension: a 2D
    #: particle plot goes to vispy because a matplotlib scatter of ~10^5
    #: points costs more per frame than the physics step it is drawing.
    plotBackend: Optional[str] = None
    #: Forwarded verbatim to `warpSPHPlotting.visualize`'s `backendOptions`.
    #: E.g. `{'jupyter_backend': 'image', 'app_backend': 'egl'}` to render
    #: vispy fully offscreen (headless GL, no window/widget) and push frames
    #: as plain Jupyter image updates -- for notebooks over a remote-SSH
    #: connection where the `jupyter_rfb` widget's comm channel to the
    #: browser does not come up, even though the kernel itself renders fine.
    plotBackendOptions: Optional[Dict[str, Any]] = None
    store: bool = False
    #: 'states' writes one HDF5 file per stored step (the examples' pattern);
    #: 'trajectory' writes a single growing trajectory.h5 (the datagen pattern).
    storeMode: str = 'states'
    storeInterval: int = 50
    #: Simulated-time export interval, used when storeMode == 'trajectory'.
    exportInterval: float = 0.002
    exportRoot: Optional[str] = None
    video: bool = False
    #: `None` means "when a terminal is watching". Redirected to a file, a tqdm
    #: bar writes a carriage-return smear that buries the report an unattended
    #: run exists to produce, so it is off unless asked for.
    progress: Optional[bool] = None
    verbose: bool = False
    #: Suppress the setup banner, the progress bar and the completion report.
    #: Everything a run says about itself goes through these three, so this is
    #: the one switch that makes a run silent.
    quiet: bool = False

    # --- case-specific knobs ------------------------------------------------
    params: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ---------------------------------------------------------

    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def merged(self, **overrides) -> 'CaseSpec':
        """A copy with `overrides` applied; `params` is merged, not replaced."""
        params = dict(self.params)
        params.update(overrides.pop('params', {}) or {})
        return replace(self, params=params, **overrides)

    # -- serialisation -------------------------------------------------------

    def toDict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def fromDict(cls, data: Dict[str, Any], *, strict: bool = True) -> 'CaseSpec':
        """Build a spec from a dict, routing unknown keys into `params`.

        Unknown keys are only tolerated when they were declared by a case (the
        caller pre-seeds `params`); with `strict` they otherwise raise, so a
        typo in a sweep file fails loudly instead of being silently ignored.
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        params = dict(kwargs.pop('params', {}) or {})
        if extra:
            if strict:
                raise ValueError(
                    f"Unknown CaseSpec fields: {sorted(extra)}. Case-specific knobs "
                    f"belong under 'params'."
                )
            params.update(extra)
        return cls(params=params, **kwargs)

    @classmethod
    def load(cls, path: str, *, strict: bool = True) -> 'CaseSpec':
        """Read a spec from a ``.json``, ``.yaml`` or ``.yml`` file."""
        return cls.fromDict(_readMapping(path), strict=strict)

    def save(self, path: str) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as f:
            if os.path.splitext(path)[1].lower() in ('.yaml', '.yml'):
                import yaml
                yaml.safe_dump(self.toDict(), f, sort_keys=False)
            else:
                json.dump(self.toDict(), f, indent=4)
        return path


# -- argparse generation -----------------------------------------------------

#: Short options for the flags reached for often enough to earn one.
_SHORT_FLAGS = {'quiet': '-q', 'verbose': '-v'}


def _addField(parser: argparse.ArgumentParser, name: str, default: Any, annotation: Any, help: str):
    """Declare one flag, inferring its type from the dataclass default."""
    short = _SHORT_FLAGS.get(name)
    if isinstance(default, bool):
        # store_true would make `plot: true` in a config file un-overridable from
        # the CLI, so both polarities get a flag.
        names = [f'--{name}'] + ([short] if short else [])
        parser.add_argument(*names, dest=name, action='store_true', default=None, help=help)
        parser.add_argument(f'--no-{name}', dest=name, action='store_false', default=None,
                            help=argparse.SUPPRESS)
        return

    kind = float
    if isinstance(default, int):
        kind = int
    elif isinstance(default, str):
        kind = str
    elif default is None:
        # Optional[...] fields: fall back to the annotation.
        text = str(annotation)
        kind = int if 'int' in text and 'float' not in text else (str if 'str' in text else float)
    parser.add_argument(f'--{name}', dest=name, type=kind, default=None, help=help)


def schemeNames() -> List[str]:
    """Every solver name `--scheme` accepts, across the three scheme families."""
    from ..enumTypes import (CompressibleSPHScheme, IncompressibleSPHScheme,
                             WeaklyCompressibleSPHScheme)
    return [member.name
            for enumClass in (CompressibleSPHScheme, WeaklyCompressibleSPHScheme,
                              IncompressibleSPHScheme)
            for member in enumClass]


def buildArgumentParser(description: str = 'Run a warpSPH case.',
                        caseParams: Optional[Dict[str, Any]] = None,
                        defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    """An argparse parser covering every :class:`CaseSpec` field.

    `caseParams` are the case's own knobs and their defaults; each becomes a
    flag that lands in ``CaseSpec.params`` instead of a top-level field.

    `defaults` are the *case's* resolved defaults, used for the help text only.
    Without them ``--help`` reports the generic :class:`CaseSpec` value -- it
    would tell you Sod runs a Wendland2 kernel when the case actually sets B7.
    The flags themselves still default to ``None``, which is what keeps
    "not passed" distinguishable from "passed the default".
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default=None,
                        help='JSON/YAML file of CaseSpec fields. CLI flags override it.')
    parser.add_argument('--saveConfig', type=str, default=None,
                        help='Write the fully resolved spec to this path and continue.')

    for f in fields(CaseSpec):
        if f.name in ('params', 'plotBackendOptions'):
            continue
        shown = (defaults or {}).get(f.name, f.default)
        if f.name == 'scheme':
            helpText = (f'solver to run (default: {shown!r}). '
                        f'One of: {", ".join(schemeNames())}')
        else:
            helpText = f'CaseSpec.{f.name} (default: {shown!r})'
        # Type inference stays on the dataclass default/annotation: a case
        # default of a different type must not change the flag's type.
        _addField(parser, f.name, f.default, f.type, helpText)

    for name, value in (caseParams or {}).items():
        # A list/dict-valued parameter (Woodward-Colella's shock regions, the
        # dam break's gravity vector) has no sensible flag form -- argparse
        # would infer `float` from it. Those stay settable via --config only.
        if isinstance(value, (list, dict)):
            continue
        _addField(parser, name, value, type(value), f'case parameter (default: {value!r})')

    return parser


def specFromArgs(args: argparse.Namespace,
                 caseParams: Optional[Dict[str, Any]] = None,
                 defaults: Optional[Dict[str, Any]] = None) -> CaseSpec:
    """Resolve a spec from parsed args.

    Precedence, lowest to highest: :class:`CaseSpec` defaults, the case's own
    `defaults`, the ``--config`` file, then explicitly passed CLI flags. Flags
    default to ``None`` so "not passed" is distinguishable from "passed the
    default value" -- which is what makes a config file overridable at all.
    """
    caseParams = dict(caseParams or {})
    known = {f.name for f in fields(CaseSpec)}

    spec = CaseSpec(params=dict(caseParams))
    if defaults:
        spec = spec.merged(**{k: v for k, v in defaults.items() if k in known},
                           params={k: v for k, v in defaults.items() if k not in known})

    config = getattr(args, 'config', None)
    if config:
        raw = _readMapping(config)
        # Only keys actually present in the file participate; a file that omits
        # `nx` must not clobber a case default with CaseSpec's generic one.
        present = {k: v for k, v in raw.items() if k in known and k != 'params'}
        extra = {k: v for k, v in raw.items() if k not in known}
        spec = spec.merged(**present, params={**(raw.get('params') or {}), **extra})

    overrides = {}
    params = dict(spec.params)
    for name, value in vars(args).items():
        if value is None or name in ('config', 'saveConfig'):
            continue
        if name in known:
            overrides[name] = value
        else:
            params[name] = value

    spec = spec.merged(**overrides, params=params)

    if getattr(args, 'saveConfig', None):
        spec.save(args.saveConfig)

    return spec
