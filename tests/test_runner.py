"""The generic machinery: does one loop really drive every scheme?"""

import pytest

from warpSPH.runner import buildContext, listCases
from warpSPH.runner.caseSpec import CaseSpec
from warpSPH.runner.runner import _schemeConfigKeyword, resolveEnum


def test_allCasesRegister():
    from warpSPH.cases import importAll
    importAll()
    assert set(listCases()) >= {'sod', 'tgv', 'dambreak'}


@pytest.mark.parametrize('caseName, expected', [
    ('sod', 'compParams'),        # compSPH_step
    ('tgv', 'schemeConfig'),      # dfsph_step
    ('dambreak', 'schemeConfig'), # deltaSPH_step
])
def test_schemeConfigKeywordIsDetectedPerScheme(caseName, expected):
    """The step functions disagree on what to call their config; the runner
    introspects rather than assuming, which is what lets one loop drive all
    three schemes."""
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase
    importAll()
    case = getCase(caseName)
    spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                    params=dict(case.params)).merged(**case.defaults)
    ctx = buildContext(case, spec)
    assert _schemeConfigKeyword(ctx.stepFunction) == expected


def test_resolveEnumIsCaseInsensitiveAndRejectsGarbage():
    from warpSPHCore import KernelFunctions
    assert resolveEnum(KernelFunctions, 'wendland2') is KernelFunctions.Wendland2
    assert resolveEnum(KernelFunctions, KernelFunctions.B7) is KernelFunctions.B7
    with pytest.raises(ValueError, match='Invalid KernelFunctions'):
        resolveEnum(KernelFunctions, 'NotAKernel')
