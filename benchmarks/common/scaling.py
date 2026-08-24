"""Post-hoc scaling graphs for suite run outputs.

Consumes the `results.json` files the three wave suites write
(`bench_accuracy` / `bench_performance` / `bench_stability`) and renders a
set of scaling graphs: one figure per quantity panel, a combined overview
grid, and a `scaling.md` index carrying the provenance and the fitted
log-log scaling exponents. Pure post-processing -- no case build, no GPU,
re-runnable against any stored output, and able to overlay several runs of
the same suite (each record carries a `run` tag that shows up in the
legends).

The x-axis is the axis each suite sweeps: particle count (performance),
dt (accuracy), dt multiplier (stability). One curve per scheme, so a panel
answers "how does this quantity scale, and how do the schemes compare on
it".
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from cycler import cycler  # noqa: E402

from .metrics import fmt, loglogFit

# (x accessor, x label) per suite -- the axis that suite sweeps.
AXES: Dict[str, Tuple[str, str]] = {
    'performance': ('nParticles', 'particles (N)'),
    'accuracy': ('extra.dt', 'dt'),
    'stability': ('extra.mult', 'dt / dt_CFL (the case CFL dt)'),
}

# A panel: (id, y label, y accessor, log-scale y). The accessor is a dotted
# path into the record ('extra.errU') or a callable record -> value.
Panel = Tuple[str, str, Union[str, Callable[[Dict[str, Any]], Any]], bool]


def _convFrac(r: Dict[str, Any]) -> Optional[float]:
    solves = r.get('solves') or 0
    return (r.get('convergedSolves', 0) / solves) if solves > 0 else None


PANELS: Dict[str, List[Panel]] = {
    'performance': [
        ('ms_per_step', 'ms / step', 'msPerStep', True),
        ('ms_per_rhs', 'ms / RHS evaluation', 'msPerRhs', True),
        ('f_per_step', 'RHS evaluations / step', 'fEvalsPerStep', False),
        ('peak_memory', 'peak allocated (MB)', 'peakAllocatedMB', True),
        ('static_memory', 'static state + adjacency (MB)', 'staticStateMB', True),
        ('build_time', 'case build (s)', 'buildSeconds', True),
    ],
    'accuracy': [
        ('error_u', 'relative L2 error, u', 'extra.errU', True),
        ('error_v', 'relative L2 error, v', 'extra.errV', True),
        ('measured_order', 'measured convergence order', 'extra.estOrder', False),
        ('energy_drift', 'energy drift |dE/E|',
         lambda r: (abs(v) if (v := (r.get('extra') or {}).get('energyDrift'))
                    is not None and math.isfinite(v) and v != 0 else None), True),
        ('cost_per_step', 'ms / step', 'msPerStep', True),
    ],
    'stability': [
        ('envelope', 'peak max|u| over the run', 'uMaxPeak', True),
        ('cost_per_step', 'ms / step', 'msPerStep', True),
        ('solver_iters', 'internal-solver iterations (mean)', 'itersMean', False),
        ('convergence', 'converged stage-solves (fraction)', _convFrac, False),
    ],
}

_COLORS = [colormaps['tab20'](i / 20) for i in range(20)]
_MARKER_CYCLE = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+', 'x',
                 '8', '.', ',']


def _resolve(record: Dict[str, Any], accessor: Union[str, Callable]) -> Any:
    if callable(accessor):
        try:
            return accessor(record)
        except Exception:
            return None
    cur: Any = record
    for part in accessor.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _ok(value: Any, logY: bool) -> bool:
    if value is None:
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value):
        return False
    return value > 0.0 if logY else True


def _byScheme(records: Sequence[Dict[str, Any]],
              xAccessor: str) -> "List[Tuple[str, List[Dict[str, Any]]]]":
    """Records grouped by scheme key, insertion order kept, rows sorted by x."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r['key'], []).append(r)
    return [(key, sorted(rows, key=lambda r: float(_resolve(r, xAccessor) or 0.0)))
            for key, rows in groups.items()]


def _labelOf(r: Dict[str, Any]) -> str:
    run = r.get('run')
    return f"{r['label']} ({run})" if run else r['label']


def _drawPanel(ax: plt.Axes, records: Sequence[Dict[str, Any]], suite: str,
               panel: Panel, boundedFactor: Optional[float] = None) -> List:
    """One quantity vs. the suite's x-axis, one line per scheme.

    Returns the line artists (for a shared legend). Diverged stability runs
    are marked with a grey `x` at their last finite peak; the stability
    envelope also draws the bounded threshold when `boundedFactor` is given.
    """
    pid, yLabel, yAcc, logY = panel
    xAcc, xLabel = AXES[suite]
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xscale('log')
    if logY:
        ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)

    lines = []
    for i, (key, rows) in enumerate(_byScheme(records, xAcc)):
        pts = []
        for r in rows:
            x, y = _resolve(r, xAcc), _resolve(r, yAcc)
            if _ok(x, True) and _ok(y, logY):
                pts.append((float(x), float(y)))
        if not pts:
            continue
        color = _COLORS[i % len(_COLORS)]
        marker = _MARKER_CYCLE[i % len(_MARKER_CYCLE)]
        if logY:
            (line,) = ax.loglog([p[0] for p in pts], [p[1] for p in pts],
                                color=color, marker=marker, markersize=4,
                                linewidth=1.2, label=_labelOf(rows[0]))
        else:
            (line,) = ax.plot([p[0] for p in pts], [p[1] for p in pts],
                              color=color, marker=marker, markersize=4,
                              linewidth=1.2, label=_labelOf(rows[0]))
        lines.append(line)

    if pid == 'envelope':
        for r in records:
            if r.get('diverged') and _ok(r.get('uMaxPeak'), True):
                ax.plot([float(_resolve(r, xAcc))], [float(r['uMaxPeak'])],
                        'x', color='0.3', markersize=7)
        if boundedFactor:
            base = next((float(r['uMax0']) for r in records if r.get('uMax0')), None)
            if base:
                ax.axhline(base * boundedFactor, color='0.5', linestyle=':',
                           label=f'bounded threshold ({boundedFactor:g}x initial max|u|)')
    return lines


def _save(fig: plt.Figure, outDir: Path, name: str) -> str:
    fig.savefig(Path(outDir) / name, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return name


def _dedup(lines: List) -> Tuple[List, List]:
    handles, labels = [], []
    for ln in lines:
        if ln.get_label() not in labels:
            handles.append(ln)
            labels.append(ln.get_label())
    return handles, labels


def plotPanel(records: Sequence[Dict[str, Any]], suite: str, panel: Panel,
              outDir: Path, title: str,
              boundedFactor: Optional[float] = None) -> str:
    """Standalone single-panel figure: `scaling_<panel_id>.png`."""
    pid, yLabel, _, _ = panel
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_prop_cycle(cycler(color=_COLORS))
    handles, labels = _dedup(_drawPanel(ax, records, suite, panel, boundedFactor))
    if handles:
        ax.legend(handles=handles, labels=labels, fontsize=6, loc='best',
                  ncol=2 if len(handles) > 8 else 1)
    ax.set_title(f'{title} -- {yLabel}')
    return _save(fig, outDir, f'scaling_{pid}.png')


def plotOverview(records: Sequence[Dict[str, Any]], suite: str, outDir: Path,
                 title: str, boundedFactor: Optional[float] = None) -> str:
    """All panels of the suite in one grid: `scaling_overview.png`."""
    panels = PANELS[suite]
    ncols = 3 if len(panels) > 4 else 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    elif ncols == 1:
        axes = [a for row in axes for a in (row,)]
    else:
        axes = [a for row in axes for a in row]
    handles, labels = [], []
    for ax, panel in zip(axes, panels):
        ax.set_prop_cycle(cycler(color=_COLORS))
        for ln in _drawPanel(ax, records, suite, panel, boundedFactor):
            if ln.get_label() not in labels:
                handles.append(ln)
                labels.append(ln.get_label())
    for ax in axes[len(panels):]:
        ax.axis('off')
    if handles:
        fig.legend(handles=handles, labels=labels, fontsize=6, loc='lower center',
                   ncol=min(6, max(1, len(handles) // 2)),
                   bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, y=1.0)
    return _save(fig, outDir, 'scaling_overview.png')


def exponentTable(records: Sequence[Dict[str, Any]], suite: str) -> str:
    """Log-log scaling exponent (slope of y vs. the suite x-axis) per scheme
    and log-scaled panel -- the numbers behind the curves. `-` where fewer
    than two points exist to fit."""
    xAcc, _ = AXES[suite]
    cols = [p[0] for p in PANELS[suite] if p[3]]
    head = ['scheme'] + cols
    rows = []
    for key, group in _byScheme(records, xAcc):
        row = [group[0]['label']]
        for pid in cols:
            panel = next(p for p in PANELS[suite] if p[0] == pid)
            pts = [(float(_resolve(r, xAcc)), float(_resolve(r, panel[2])))
                   for r in group
                   if _ok(_resolve(r, xAcc), True) and _ok(_resolve(r, panel[2]), True)]
            fit = loglogFit([p[0] for p in pts], [p[1] for p in pts])
            row.append(fmt(fit[0], '.2f') if fit else '-')
        rows.append(row)
    table = ['| ' + ' | '.join(head) + ' |',
             '|' + '|'.join(['---'] * len(head)) + '|']
    table += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(table)


def loadInput(path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
    """A results directory (its `results.json`) or a JSON file directly.

    Returns `(runLabel, payload)`; `runLabel` is the directory name, with a
    leading `<suite>_` stripped so legend labels stay short."""
    p = Path(path)
    j = p / 'results.json' if p.is_dir() else p
    if not j.is_file():
        raise SystemExit(f'no results.json at {path}')
    payload = json.loads(j.read_text())
    label = p.name if p.is_dir() else (p.parent.name if p.parent != Path('.')
                                       else p.stem)
    suite = payload.get('suite', '')
    if suite and label.startswith(suite + '_'):
        label = label[len(suite) + 1:]
    return label, payload


def plotScalingSet(records: Sequence[Dict[str, Any]], suite: str, outDir: Path,
                   title: str, boundedFactor: Optional[float] = None,
                   overview: bool = True,
                   inputs: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None
                   ) -> List[str]:
    """The full set for one suite: one PNG per panel, the overview grid, and
    a `scaling.md` index (provenance + figures + exponent table)."""
    if suite not in PANELS:
        raise SystemExit(f'unknown suite {suite!r} (expected one of {sorted(PANELS)})')
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    for panel in PANELS[suite]:
        names.append(plotPanel(records, suite, panel, outDir, title, boundedFactor))
    if overview:
        names.append(plotOverview(records, suite, outDir, title, boundedFactor))

    lines = [f'# {title} -- scaling graphs', '']
    if inputs:
        lines += ['## Inputs', '']
        for label, payload in inputs:
            meta = payload.get('meta', {})
            extra = {k: v for k, v in meta.items()
                     if k in ('nx', 'nxs', 'tEnd', 'dtCFL', 'multipliers',
                              'steps', 'warmup', 'refinements', 'boundedFactor')}
            lines.append(f'- `{label}`: {meta.get("timestamp", "?")} '
                         f'({meta.get("device", "?")}), {extra}')
        lines.append('')
    lines += ['## Graphs', '']
    for panel in PANELS[suite]:
        lines += [f'### {panel[1]}', '', f'![{panel[1]}](scaling_{panel[0]}.png)', '']
    if overview:
        lines += ['## Overview', '', '![overview](scaling_overview.png)', '']
    lines += ['## Scaling exponents', '',
              'Log-log slope of each quantity against the suite x-axis '
              '(particles: ~1 is linear; error panels: the nominal temporal '
              'order once the asymptotic regime is reached).', '',
              exponentTable(records, suite), '']
    (outDir / 'scaling.md').write_text('\n'.join(lines))
    return names