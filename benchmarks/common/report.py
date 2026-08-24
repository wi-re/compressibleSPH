"""Results output: `results.json`, `summary.md`, and PNG plots.

Plots are always rendered with the non-interactive Agg backend so the suites
run headless (remote SSH, CI) -- the notebook's `%matplotlib widget` is for
a human at a kernel, not for a benchmark.
"""

from __future__ import annotations

import datetime
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .metrics import fmt


def environmentMeta(precision: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What this run ran on -- version, device, precision, and the git
    revisions of the three repos whose code the numbers depend on."""
    meta: Dict[str, Any] = {
        'timestamp': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'warp': _attr(lambda: __import__('warp').__version__),
        'torch': torch.__version__,
        'device': (torch.cuda.get_device_name(torch.cuda.current_device())
                   if torch.cuda.is_available() else 'cpu'),
        'cudaAvailable': torch.cuda.is_available(),
        'precision': precision or 'float32',
        'python': _attr(lambda: __import__('sys').version.split()[0]),
    }
    for repo in ('warpSPH', 'warpSPHIntegrators', 'warpSPHCore'):
        meta[f'{repo}@git'] = _gitSha(repo)
    if extra:
        meta.update(extra)
    return meta


def _attr(fn):
    try:
        return fn()
    except Exception:
        return None


def _gitSha(repo: str) -> Optional[str]:
    """Best-effort `git rev-parse --short HEAD` for a sibling/local checkout.

    The repos live in developer checkouts (`~/dev/...`), not in the
    installed site-packages, so the location is not knowable from the
    import system alone -- try the obvious neighbours of this file and
    fall back to `None` rather than failing a benchmark over bookkeeping.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / repo,            # <repo>/benchmarks/common/report.py -> <dev>/warpSPH/...
        Path.home() / 'dev' / repo,
    ]
    for root in candidates:
        if (root / '.git').exists():
            try:
                out = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'],
                                     capture_output=True, text=True, timeout=5)
                if out.returncode == 0:
                    return out.stdout.strip()
            except Exception:
                pass
    return None


def outDirFor(name: str, out: Optional[str] = None) -> Path:
    """Timestamped results dir under the suite's `results/` by default."""
    if out:
        path = Path(out)
    else:
        stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = Path(__file__).resolve().parents[2] / 'results' / f'{name}_{stamp}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def writeResults(outDir: Path, suite: str, meta: Dict[str, Any],
                 records: Sequence[Dict[str, Any]]) -> Path:
    path = Path(outDir) / 'results.json'
    payload = {'suite': suite, 'meta': meta, 'records': list(records)}
    path.write_text(json.dumps(payload, indent=2, default=_jsonDefault))
    return path


def _jsonDefault(obj):
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def mdTable(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A GitHub-flavoured markdown table; cells are pre-formatted strings."""
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def writeSummary(outDir: Path, title: str, sections: Sequence[tuple]) -> Path:
    """`sections` is a list of `(heading, markdownBody, [plotFileName, ...])`."""
    lines = [f'# {title}', '']
    for heading, body, plots in sections:
        lines += [f'## {heading}', '', body.strip(), '']
        for p in plots:
            lines += [f'![{p}]({p})', '']
    path = Path(outDir) / 'summary.md'
    path.write_text('\n'.join(lines))
    return path


def saveFig(fig, outDir: Path, name: str) -> str:
    fig.savefig(Path(outDir) / name, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return name


def plotAccuracy(outDir: Path, records: Sequence[Dict[str, Any]],
                 title: str, refKey: str, refLabel: str) -> str:
    """Relative error vs. dt (log-log), one line per scheme, with dashed
    reference slope lines for orders 1-4 anchored on the reference scheme's
    coarsest-dt point -- the classic convergence plot.

    Expects each record's `extra` to carry `errU` and `dt`.
    """
    byScheme: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r['key'] == refKey:
            continue
        err = r.get('extra', {}).get('errU')
        if err is None or not (math.isfinite(err) and err > 0):
            continue
        byScheme.setdefault(r['key'], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    marker = 'o'
    for key, rows in byScheme.items():
        rows.sort(key=lambda r: r['dt'])
        xs = [r['dt'] for r in rows]
        ys = [r['extra']['errU'] for r in rows]
        ax.loglog(xs, ys, marker, label=rows[0]['label'], markersize=4)
        marker = cycle_marker(marker)

    # Reference order lines anchored on the reference scheme's coarsest point.
    refRows = sorted((r for r in records if r['key'] == refKey
                      and r.get('extra', {}).get('errU', 0) > 0), key=lambda r: r['dt'])
    if refRows:
        anchor = refRows[-1]
        for order in (1, 2, 3, 4):
            scale = anchor['extra']['errU']
            dts = [anchor['dt'] * 10 ** e for e in (-2.0, -1.0, 0.0)]
            ax.loglog(dts, [scale * (d / anchor['dt']) ** -order for d in dts],
                      '--', color='0.75', alpha=0.6)
            ax.text(dts[-1], scale * (10 ** 0) ** -order * 1.15, f'O(dt^{order})',
                    fontsize=7, color='0.5', ha='right')
        ax.plot([anchor['dt']], [anchor['extra']['errU']], 's', color='0.4',
                label=f'{refLabel} (reference, dt/...)')

    ax.set_xlabel('dt')
    ax.set_ylabel('relative L2 error, final u field (vs. reference)')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    return saveFig(fig, outDir, 'accuracy_error_vs_dt.png')


_MARKER_CYCLE = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+', 'x']


def cycle_marker(current: str) -> str:
    i = _MARKER_CYCLE.index(current) if current in _MARKER_CYCLE else 0
    return _MARKER_CYCLE[(i + 1) % len(_MARKER_CYCLE)]


def plotPerformance(outDir: Path, records: Sequence[Dict[str, Any]],
                    title: str, slopes: Optional[Dict[str, float]] = None) -> str:
    """ms/step and peak allocated memory vs. particle count (log-log),
    one line per scheme. `slopes` (key -> fitted exponent) is annotated on
    the time panel."""
    byScheme: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        byScheme.setdefault(r['key'], []).append(r)

    fig, (axT, axM) = plt.subplots(1, 2, figsize=(11, 5))
    marker = 'o'
    for key, rows in byScheme.items():
        rows.sort(key=lambda r: r['nParticles'])
        xs = [r['nParticles'] for r in rows if r['msPerStep'] > 0]
        ysT = [r['msPerStep'] for r in rows if r['msPerStep'] > 0]
        if xs:
            axT.loglog(xs, ysT, marker, label=rows[0]['label'], markersize=4)
        ysM = [r['peakAllocatedMB'] for r in rows if r['peakAllocatedMB'] > 0]
        xsM = [r['nParticles'] for r in rows if r['peakAllocatedMB'] > 0]
        if xsM:
            axM.loglog(xsM, ysM, marker, label=rows[0]['label'], markersize=4)
        if slopes and key in slopes and len(xs) >= 2:
            axT.annotate(f'{slopes[key]:.2f}', (xs[-1], ysT[-1]),
                         textcoords='offset points', xytext=(4, 4), fontsize=7)
        marker = cycle_marker(marker)

    axT.set_xlabel('particles')
    axT.set_ylabel('ms / step')
    axT.set_title(f'{title} -- time per step (annotated: log-log slope)')
    axT.grid(True, which='both', alpha=0.3)
    axM.set_xlabel('particles')
    axM.set_ylabel('peak allocated memory (MB)')
    axM.set_title('GPU memory, peak allocated (CUDA events / reset per run)')
    axM.grid(True, which='both', alpha=0.3)
    axT.legend(fontsize=6, loc='best')
    fig.savefig(Path(outDir) / 'performance.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return 'performance.png'



def plotStability(outDir: Path, records: Sequence[Dict[str, Any]],
                  title: str, boundedFactor: float) -> str:
    """Peak max|u| vs. dt multiplier (log y), one line per scheme.

    A run that went non-finite is drawn with an open marker and a `x` at the
    last finite point -- "bounded" means finite and within `boundedFactor`
    of the initial max|u| for the whole run (recorded in `extra['bounded']`).
    """
    byScheme: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        byScheme.setdefault(r['key'], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    marker = 'o'
    for key, rows in byScheme.items():
        rows.sort(key=lambda r: r['extra'].get('mult', r['dt']))
        finite = [(r['extra']['mult'], r['uMaxPeak']) for r in rows
                  if r.get('uMaxPeak') is not None and math.isfinite(r['uMaxPeak'])]
        if not finite:
            continue
        xs = [m for m, _ in finite]
        ys = [v for _, v in finite]
        ax.loglog(xs, ys, marker, label=rows[0]['label'], markersize=4)
        for r in rows:
            if r.get('diverged'):
                m = r['extra'].get('mult', r['dt'])
                peak = r.get('uMaxPeak')
                if peak is not None and math.isfinite(peak):
                    ax.plot([m], [peak], 'x', color='0.3', markersize=6)
        marker = cycle_marker(marker)

    base = None
    for r in records:
        if r.get('uMax0'):
            base = r['uMax0']
            break
    if base:
        ax.axhline(base * boundedFactor, color='0.5', linestyle=':',
                   label=f'bounded threshold ({boundedFactor:g} x initial max|u|)')

    ax.set_xlabel('dt / dt_CFL (the case cflFactor=0.1 dt)')
    ax.set_ylabel('peak max|u| over the run')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=6, loc='best')
    return saveFig(fig, outDir, 'stability.png')

