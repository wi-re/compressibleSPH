"""The step loop, once.

`examples/compressible/01-sod-shock-tube-1d.py`,
`examples/incompressible/01-tgv-incomp.py` and
`datagen/weaklyCompressible/generator.py` each carried their own copy of: build
config, unpack ``buildScheme``, initialize state, loop the integrator, time it,
accumulate diagnostics, plot every N, export every M, encode a video. This
module is that code, parameterised by a :class:`~warpSPH.runner.case.Case`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..configurations import buildConfig
from ..enumTypes import (CompressibleSPHScheme, IncompressibleSPHScheme,
                         WeaklyCompressibleSPHScheme)
from ..io import (createOutFile, exportSimulationSystem, prepExport, writeFrame,
                  writeInitialData)
from ..schemes import buildScheme
from ..utils import buildDomainDescription
from .case import Case, RunContext
from .caseSpec import CaseSpec
from .media import encodeFrames

__all__ = ['RunResult', 'run', 'buildContext', 'resolveEnum']


@dataclass
class RunResult:
    """What a run produced. `trajectory` is one dict of diagnostics per step."""

    ctx: RunContext
    state: Any
    trajectory: List[Dict[str, float]] = field(default_factory=list)
    exportPath: Optional[str] = None
    videoPath: Optional[str] = None
    nSteps: int = 0
    diverged: bool = False

    def series(self, key: str) -> np.ndarray:
        """One diagnostic across the whole run, as an array."""
        return np.array([row[key] for row in self.trajectory if key in row])


def resolveEnum(enumClass, value):
    """Case-insensitive name lookup, passing through values already resolved."""
    if value is None or isinstance(value, enumClass):
        return value
    for member in enumClass:
        if member.name.lower() == str(value).lower():
            return member
    raise ValueError(
        f'Invalid {enumClass.__name__} {value!r}. Valid options are: '
        f'{[m.name for m in enumClass]}'
    )


def _resolveScheme(name: str):
    """Map a scheme name onto whichever of the three scheme enums owns it."""
    for enumClass in (CompressibleSPHScheme, WeaklyCompressibleSPHScheme, IncompressibleSPHScheme):
        for member in enumClass:
            if member.name.lower() == str(name).lower():
                return member
    raise ValueError(f'Unknown scheme {name!r}.')


def buildContext(case: Case, spec: CaseSpec) -> RunContext:
    """Resolve a spec into a config, a scheme, and a populated context."""
    # Idempotent, and cheap once done. Running a case module directly imports
    # warpSPH without going through warpSPHBootstrap, so warp would otherwise
    # still be uninitialized here and every kernel launch would fail.
    import warp as wp
    wp.init()

    from warpSPHCore.type_config import get_torch_precision
    from warpSPHIntegrators import IntegrationSchemeType
    from warpSPHCore import (GradientScheme, KernelFunctions, LaplacianScheme,
                             SupportScheme)
    from ..utils.sampling import SamplingScheme

    device = torch.device(spec.device) if spec.device else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    dtype = get_torch_precision()

    domain = buildDomainDescription(spec.L, spec.dim, spec.periodic, device, dtype)

    config, integrator = buildConfig(
        domain=domain,
        dim=spec.dim,
        kernel=resolveEnum(KernelFunctions, spec.kernel),
        targetNeighbors=_nHtoNH(spec.n_h, spec.dim),
        supportMode=resolveEnum(SupportScheme, spec.supportMode),
        gradientMode=resolveEnum(GradientScheme, spec.gradientMode),
        laplacianMode=resolveEnum(LaplacianScheme, spec.laplacianMode),
        integrationScheme=resolveEnum(IntegrationSchemeType, spec.integrationScheme),
        samplingScheme=resolveEnum(SamplingScheme, spec.samplingScheme),
        verletScale=spec.verletScale,
        device=device,
        dtype=dtype,
        dt=spec.dt,
        minDt=spec.minDt,
        maxDt=spec.maxDt,
        adaptiveDt=spec.adaptiveDt,
        cflFactor=spec.cflFactor,
        nx=spec.nx,
        dx=spec.L / spec.nx,
    )

    scheme = _resolveScheme(spec.scheme or case.scheme)
    bundle = buildScheme(scheme)

    return RunContext(
        spec=spec,
        case=case,
        config=config,
        integrator=integrator,
        schemeConfig=bundle.SimulationConfig(),
        scheme=scheme,
        device=device,
        dtype=dtype,
        bundle=bundle,
    )


def _nHtoNH(n_h, dim):
    from warpSPHCore import n_h_to_nH
    return n_h_to_nH(n_h, dim)


def _scalar(value) -> float:
    return value.detach().cpu().item() if isinstance(value, torch.Tensor) else float(value)


class _Timer:
    """CUDA-event timing where available, wall clock otherwise."""

    def __init__(self, device):
        self.cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __enter__(self):
        if self.cuda:
            self._begin = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._begin.record()
        else:
            self._begin = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._begin.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._begin) * 1000.0
        return False


def run(case: Case, spec: Optional[CaseSpec] = None, **overrides) -> RunResult:
    """Run `case` under `spec`, returning the trajectory and final state.

    Keyword `overrides` are applied on top of `spec` (or on top of the case's
    own defaults when no spec is given), so a test can say
    ``run(tgvCase, nx=32, nSteps=20)``.
    """
    if spec is None:
        spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params))
        spec = spec.merged(**case.defaults)
    if overrides:
        spec = spec.merged(**overrides)

    ctx = buildContext(case, spec)

    if case.configureScheme is not None:
        case.configureScheme(ctx)

    system = case.buildSystem(ctx)
    if case.initialConditions is not None:
        case.initialConditions(ctx, system)

    runningState = system.initializeNewState()

    if spec.verbose:
        _describe(ctx, runningState)

    # --- output setup -------------------------------------------------------
    outFile = None
    groups = None
    if spec.store or spec.plot:
        ctx.exportPath = prepExport(spec.caseName, ctx.config, ctx.schemeConfig,
                                    ctx.scheme, ctx.exportFunction,
                                    exportRoot=spec.exportRoot)
        spec.save(os.path.join(ctx.exportPath, 'caseSpec.json'))

    if spec.plot and case.setupPlot is not None:
        ctx.imagePath = os.path.join(ctx.exportPath, 'images')
        os.makedirs(ctx.imagePath, exist_ok=True)
        ctx.scratch['plot'] = case.setupPlot(ctx, runningState)

    extraData = case.extraData(ctx, runningState) if case.extraData is not None else {}

    if spec.store and spec.storeMode == 'trajectory':
        outFile = createOutFile(ctx.exportPath)
        groups = writeInitialData(ctx.exportPath, outFile, ctx.scheme, ctx.config,
                                  ctx.schemeConfig,
                                  SimpleNamespace(exportInterval=spec.exportInterval),
                                  runningState, extraData=extraData)

    # `config.dt` is only final once the case has configured it -- weakly
    # compressible cases derive it from the sound speed during setup.
    if ctx.config.dt is None:
        raise ValueError(
            'config.dt is unset after case setup; set spec.dt or have the case '
            'derive it (e.g. via setupWeaklyCompressibleTimestep).'
        )
    dt = _scalar(ctx.config.dt)
    nSteps = spec.nSteps if spec.nSteps is not None else int(spec.tLimit / dt)
    storeSteps = max(1, int(spec.exportInterval / dt)) if spec.storeMode == 'trajectory' \
        else max(1, spec.storeInterval)

    if spec.store and spec.storeMode == 'states':
        exportSimulationSystem(ctx.exportPath, 'initialState', ctx.scheme, runningState,
                               exportAdjacency=False, stages=None,
                               exportStagesAdjacency=False,
                               extraData=dict(extraData, frame_num=0))

    result = RunResult(ctx=ctx, state=runningState, exportPath=ctx.exportPath)
    if case.diagnostics is not None:
        result.trajectory.append(dict(case.diagnostics(ctx, runningState), step=-1, t=0.0,
                                      stepTime_ms=0.0))

    stepResult = None

    progress = _progressBar(range(nSteps), enabled=spec.progress)
    for i in progress:
        with _Timer(ctx.device) as timer:
            stepResult = ctx.integrator.function(
                state=runningState,
                f=ctx.stepFunction,
                dt=ctx.config.dt,
                config=ctx.config,
                verbose=False,
                schemeConfig=ctx.schemeConfig,
            )
        runningState = stepResult.state

        row = {'step': i, 't': _scalar(runningState.t), 'stepTime_ms': timer.elapsed_ms}
        if case.diagnostics is not None:
            row.update(case.diagnostics(ctx, runningState))
        result.trajectory.append(row)

        if hasattr(progress, 'set_description'):
            progress.set_description(_describeStep(i, nSteps, row))

        if spec.plot and case.updatePlot is not None and i > 0 and \
                (i % spec.plotInterval == 0 or i == nSteps - 1):
            case.updatePlot(ctx, runningState, ctx.scratch.get('plot'), i)

        if spec.store and i % storeSteps == 0:
            frameExtra = dict(extraData,
                              **(case.extraData(ctx, runningState) if case.extraData else {}),
                              frame_num=i)
            if spec.storeMode == 'trajectory':
                writeFrame(groups, i, stepResult.state, stepResult.stages,
                           config=ctx.config, schemeConfig=ctx.schemeConfig,
                           uniqueParticles=True, writeStages=False)
            else:
                exportSimulationSystem(ctx.exportPath, f'state_{i:04d}', ctx.scheme,
                                       runningState, exportAdjacency=False,
                                       stages=stepResult.stages,
                                       exportStagesAdjacency=True, extraData=frameExtra)

        if torch.any(torch.isnan(runningState.state.velocities)):
            print(f'NaN detected in velocities at step {i}; stopping.')
            result.diverged = True
            break

    result.state = runningState
    result.nSteps = len(result.trajectory) - (1 if case.diagnostics is not None else 0)

    if spec.store and spec.storeMode == 'states' and stepResult is not None:
        exportSimulationSystem(ctx.exportPath, 'finalState', ctx.scheme, runningState,
                               exportAdjacency=False, stages=stepResult.stages,
                               exportStagesAdjacency=True,
                               extraData=dict(extraData, frame_num=result.nSteps))
    if outFile is not None:
        outFile.close()

    if spec.video and ctx.imagePath is not None:
        result.videoPath = encodeFrames(ctx.imagePath, ctx.exportPath)

    return result


def _progressBar(iterable, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.autonotebook import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, leave=True)


def _describeStep(i: int, nSteps: int, row: Dict[str, float]) -> str:
    parts = [f'{i + 1}/{nSteps}', f't={row["t"]:.4g}']
    parts += [f'{k}={v:.4g}' for k, v in row.items()
              if k not in ('step', 't', 'stepTime_ms') and isinstance(v, (int, float))]
    parts.append(f'{row["stepTime_ms"]:.1f}ms')
    return ' | '.join(parts)


def _describe(ctx: RunContext, state) -> None:
    print('-' * 80)
    print(f'case: {ctx.case.name}  scheme: {ctx.scheme}')
    print(f'device: {ctx.device}, dtype: {ctx.dtype}')
    print(f'particles: {len(state.state.positions)}')
    print(f'domain: {ctx.config.domain.min.cpu().numpy()} to {ctx.config.domain.max.cpu().numpy()}')
    print(f'dt: {ctx.config.dt}, minDt: {ctx.config.minDt}, adaptiveDt: {ctx.config.adaptiveDt}, '
          f'cflFactor: {ctx.config.cflFactor}')
    print(f'kernel: {ctx.config.kernel}, targetNeighbors: {ctx.config.targetNeighbors}')
    print('-' * 80)
