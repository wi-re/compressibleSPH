"""The generic machinery: does one loop really drive every scheme?"""

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
