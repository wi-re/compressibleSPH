"""The generic machinery: does one loop really drive every scheme?"""

import dataclasses

import pytest

from warpSPH.runner import buildContext, listCases
from warpSPH.runner.caseSpec import CaseSpec
from warpSPH.runner.runner import resolveEnum


def test_allCasesRegister():
    from warpSPH.cases import importAll
    importAll()
    assert set(listCases()) >= {'sod', 'tgv', 'dambreak'}


@pytest.mark.parametrize('caseName', ['sod', 'tgv', 'dambreak'])
def test_everySchemeNamesItsConfigTheSame(caseName):
    """One loop drives every scheme only because they agree on the keyword.

    ``compSPH_step``/``crkSPH_step``/``compressibleSPH_Monaghan`` used to call it
    ``compParams`` while ``deltaSPH_step``/``dfsph_step`` called it
    ``schemeConfig``; the integrator forwards ``**kwargs`` verbatim, so a caller
    that guessed wrong got a ``TypeError``. The runner passes ``schemeConfig=``
    unconditionally now, so this is the invariant holding that up.
    """
    from inspect import signature

    from warpSPH.cases import importAll
    from warpSPH.runner import getCase
    importAll()
    case = getCase(caseName)
    spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                    params=dict(case.params)).merged(**case.defaults)
    ctx = buildContext(case, spec)
    assert 'schemeConfig' in signature(ctx.stepFunction).parameters
    assert 'compParams' not in signature(ctx.stepFunction).parameters


def test_allStepFunctionsAgreeOnTheKeyword():
    """The same invariant for the schemes no registered case exercises."""
    from inspect import signature

    from warpSPH.schemes import buildScheme
    for name in ('compSPH', 'crkSPH', 'MonaghanCompressibleSPH', 'deltaSPH',
                 'divergenceFree'):
        parameters = signature(buildScheme(name).stepFunction).parameters
        assert 'schemeConfig' in parameters, name
        assert 'compParams' not in parameters, name


def test_resolveEnumIsCaseInsensitiveAndRejectsGarbage():
    from warpSPHCore import KernelFunctions
    assert resolveEnum(KernelFunctions, 'wendland2') is KernelFunctions.Wendland2
    assert resolveEnum(KernelFunctions, KernelFunctions.B7) is KernelFunctions.B7
    with pytest.raises(ValueError, match='Invalid KernelFunctions'):
        resolveEnum(KernelFunctions, 'NotAKernel')


def test_schemeBundleIsNamedAndStillUnpacks():
    """`SchemeBundle` replaced a bare 7-tuple. Named access is the point; the
    positional unpacking is kept so that adding an eighth member (a tangent
    propagator, once forward-mode AD lands) does not break old call sites."""
    from warpSPH.schemes import buildScheme
    from warpSPH.schemes.builder import SchemeBundle

    bundle = buildScheme('compSPH')
    assert isinstance(bundle, SchemeBundle)
    assert bundle.stepFunction.__name__ == 'compSPH_step'

    system, state, config, update, step, export, imp = bundle
    assert (system, state, config, update, step, export, imp) == (
        bundle.SimulationSystem, bundle.SimulationState, bundle.SimulationConfig,
        bundle.SimulationUpdate, bundle.stepFunction, bundle.exportFunction,
        bundle.importFunction)


def test_buildSchemeRejectsUnknownSchemes():
    import pytest as _pytest

    from warpSPH.schemes import buildScheme
    with _pytest.raises(ValueError, match='not recognized'):
        buildScheme('notAScheme')


def test_everyCaseModuleRegistersItsCase():
    """`CASE_MODULES` and the registry have to stay in step.

    A case module that stops registering (a renamed `registerCase` target, a
    module dropped from the tuple) is otherwise invisible until someone runs
    `warpsph-run` and gets "Unknown case".
    """
    from warpSPH.cases import CASE_MODULES, importAll
    importAll()
    # channelFlow declares two cases and dambreak's hooks back them, so the
    # count is not one-per-module; the floor is what matters.
    assert len(listCases()) >= len(CASE_MODULES)


def test_everyCaseNamesAResolvableScheme():
    """Each case's declared scheme has to exist in one of the three enums."""
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase
    from warpSPH.runner.runner import _resolveScheme
    importAll()
    for name in listCases():
        assert _resolveScheme(getCase(name).scheme) is not None, name


def test_everyCaseDeclaresItsParamsAsScalarsOrLists():
    """`params` becomes both CLI flags and HDF5 attributes.

    Only scalars get flags (`buildArgumentParser` skips lists and dicts) and
    only scalars are written per frame, so anything else has to be a deliberate
    list/dict rather than, say, an enum or a tensor that would fail at export.
    """
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase
    importAll()
    for name in listCases():
        for key, value in getCase(name).params.items():
            assert isinstance(value, (int, float, str, bool, list, dict)), \
                f'{name}.{key} is {type(value).__name__}'


def test_timeLimitedLoopStopsOnTimeNotStepCount():
    """A case with a `timestep` hook is bounded by `tLimit`, not by an estimate.

    `nSteps = tLimit / dt0` is wrong the moment dt changes, which is exactly
    what the hook exists to do -- so those runs loop on simulated time.
    """
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()
    case = getCase('woodwardColella')
    assert case.timestep is not None

    tLimit = 5e-4
    result = run(case, nx=100, tLimit=tLimit, progress=False, plot=False, store=False)
    assert not result.diverged
    assert result.trajectory[-1]['t'] >= tLimit
    # The step before the last must still be short of the limit, or the loop
    # ran past its stopping condition.
    assert result.trajectory[-2]['t'] < tLimit


def test_postStepHookRunsAfterEveryStep():
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()

    case = getCase('sod')
    calls = []
    case = dataclasses.replace(case, postStep=lambda ctx, state, i: calls.append(i))
    result = run(case, nx=100, nSteps=4, progress=False, plot=False, store=False)
    assert calls == list(range(result.nSteps))
