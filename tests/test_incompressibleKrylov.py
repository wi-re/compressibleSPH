"""End-to-end tests for the opt-in Krylov pressure solvers in the DFSPH
incompressible solve (BiCGStab / GMRES / CG / BiCG), plus the Phase-0 operator
probe and the relaxed-Jacobi regression guard. See ``INCOMPRESSIBLE_SOLVER_PLAN.md``.

One divergence-free TGV case is built once (session scope) and shared. The
pressure operator is matrix-free, so the tests that need the operator matrix
assemble it densely on the small case (``_assembleA``): this both verifies the
matvec closure and gives a reference for the symmetry / spectrum probe.

Findings the probe encodes (measured, fp32, this operator):
  * symmetric to ~1e-6 (so BiCG's ``A^T = A`` placeholder is exact);
  * negative-semi-definite with a near-zero gauge mode (so CG/BiCG sign-flip);
  * not diagonally dominant (so the damped relaxed-Jacobi can diverge here --
    the unchanged default is still exercised and fingerprinted, not required to
    converge).
"""

import math

import pytest
from types import SimpleNamespace

import torch

from warpSPH.cases.tgv import tgvCase
from warpSPH.runner import buildContext, CaseSpec
from warpSPH.configurations import (
    IncompressibleSPHConfig, PressureSolverType, RelaxedJacobiSolverConfig,
    incompressibleConfigToDict, dictToIncompressibleSPHConfig,
)
from warpSPH.modules.incompressible.divergenceFree import solveDivergenceFree
from warpSPH.modules.incompressible.incompressible import solveIncompressible
from warpSPH.modules.incompressible.krylov import (
    buildIISPHMatvec, buildIISPHPrecond, solvePressureKrylov,
)
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.momentum.incompressible import computeMomentumIncompressible
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH
from warpSPH.modules.incompressible.drift import computePressureShiftIISPH
from warpSPH.modules.density import computeDensities
from warpSPHCore import buildVerletList, SupportScheme

NX = 32
SEED = 0
VELOCITY_SCALE = 0.05


def _buildCase(nx=NX, velocityScale=VELOCITY_SCALE, seed=SEED):
    spec = (CaseSpec(caseName='tgv', scheme='divergenceFree',
                     params=dict(tgvCase.params)).merged(**tgvCase.defaults)
            .merged(nx=nx))
    ctx = buildContext(tgvCase, spec)
    tgvCase.configureScheme(ctx)
    system = tgvCase.buildSystem(ctx)
    state = system.state
    config = ctx.config
    schemeConfig = ctx.schemeConfig
    adjacency = buildVerletList(state, config.domain, verletScale=config.verletScale,
                                supportMode=SupportScheme.SuperSymmetric,
                                priorNeighborhood=None, verbose=False)
    system.adjacency = adjacency
    state.densities = computeDensities(state, config, schemeConfig, adjacency)
    state.masses = state.masses / state.densities.mean() * 1.0
    state.densities = computeDensities(state, config, schemeConfig, adjacency)
    torch.manual_seed(seed)
    state.velocities = velocityScale * torch.randn_like(state.velocities)
    state.pressures = torch.zeros_like(state.densities)
    return SimpleNamespace(state=state, config=config, schemeConfig=schemeConfig,
                           adjacency=adjacency, dt=config.dt)


@pytest.fixture(scope='session')
def incompCase():
    return _buildCase()


def _assembleA(case):
    """Dense form of the matrix-free IISPH operator (column j = A e_j)."""
    N = case.state.densities.shape[0]
    matvec = buildIISPHMatvec(case.state, case.config, case.schemeConfig,
                              case.adjacency, case.dt)
    A = torch.empty(N, N, device=case.state.densities.device,
                    dtype=case.state.densities.dtype)
    col = torch.zeros(N, device=case.state.densities.device,
                      dtype=case.state.densities.dtype)
    for j in range(N):
        col.zero_()
        col[j] = 1.0
        A[:, j] = matvec(col)
    return A


def _sourceTerm(case, dvdt=None):
    if dvdt is None:
        dvdt = torch.zeros_like(case.state.velocities)
    predicted = case.state.velocities + case.dt * dvdt
    return -computeMomentumIncompressible(
        case.state, case.config, case.schemeConfig, case.adjacency, predicted)


def _run(case, solverType, maxIter=200, rtol=1e-5, variant='divergenceFree'):
    scfg = (case.schemeConfig.solverConfig.divergenceFreeSolver
            if variant == 'divergenceFree'
            else case.schemeConfig.solverConfig.pressureSolver)
    scfg.solverType = solverType
    scfg.maxIterations = maxIter
    scfg.rtol = rtol
    scfg.atol = 0.0
    scfg.restart = 30
    scfg.tolerance = 1e-9
    scfg.relaxationFactor = 0.5
    scfg.minIterations = 0
    case.state.pressures.zero_()
    dvdt = torch.zeros_like(case.state.velocities)
    fn = solveDivergenceFree if variant == 'divergenceFree' else solveIncompressible
    a_p, pressure, errors, pressures = fn(
        case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
    return a_p, pressure, errors


def test_pressureSolverTypeAndDefault():
    names = [e.name for e in PressureSolverType]
    assert names == ['relaxedJacobi', 'cg', 'bicg', 'bicgStab', 'gmres']
    # The default must stay relaxedJacobi so the historical path is unchanged.
    assert RelaxedJacobiSolverConfig().solverType is PressureSolverType.relaxedJacobi
    assert IncompressibleSPHConfig().solverConfig.divergenceFreeSolver.solverType is \
        PressureSolverType.relaxedJacobi
    assert IncompressibleSPHConfig().solverConfig.pressureSolver.solverType is \
        PressureSolverType.relaxedJacobi


def test_solverConfigRoundTrip():
    cfg = IncompressibleSPHConfig()
    cfg.solverConfig.divergenceFreeSolver.solverType = PressureSolverType.gmres
    cfg.solverConfig.divergenceFreeSolver.rtol = 1e-6
    cfg.solverConfig.divergenceFreeSolver.atol = 1e-9
    cfg.solverConfig.divergenceFreeSolver.restart = 50
    cfg.solverConfig.pressureSolver.solverType = PressureSolverType.cg
    d = incompressibleConfigToDict(cfg)
    assert d['solverConfig']['divergenceFreeSolver']['solverType'] == 'gmres'
    assert d['solverConfig']['divergenceFreeSolver']['restart'] == 50
    rt = dictToIncompressibleSPHConfig(d)
    assert rt.solverConfig.divergenceFreeSolver.solverType is PressureSolverType.gmres
    assert rt.solverConfig.divergenceFreeSolver.rtol == pytest.approx(1e-6)
    assert rt.solverConfig.divergenceFreeSolver.atol == pytest.approx(1e-9)
    assert rt.solverConfig.divergenceFreeSolver.restart == 50
    assert rt.solverConfig.pressureSolver.solverType is PressureSolverType.cg
    # a dict missing the new keys falls back to the defaults
    d2 = incompressibleConfigToDict(IncompressibleSPHConfig())
    del d2['solverConfig']['divergenceFreeSolver']['solverType']
    rt2 = dictToIncompressibleSPHConfig(d2)
    assert rt2.solverConfig.divergenceFreeSolver.solverType is PressureSolverType.relaxedJacobi


def test_matvecMatchesJacobiInnerIteration(incompCase):
    """The matvec closure is exactly the relaxed-Jacobi inner iteration
    (``dt * shift(accel(p))``), for both the dt (div-free) and dt**2 (constant-
    density) scalings."""
    case = incompCase
    p = torch.randn_like(case.state.densities)
    a_p = computePressureAccelIISPH(case.state, p, case.config,
                                    supportScheme=SupportScheme.Scatter,
                                    adjacency=case.adjacency)
    for scale, dt in ((1, case.dt), (2, case.dt ** 2)):
        expected = dt * computePressureShiftIISPH(case.state, case.config, a_p,
                                                  supportScheme=SupportScheme.Scatter,
                                                  adjacency=case.adjacency)
        got = buildIISPHMatvec(case.state, case.config, case.schemeConfig,
                               case.adjacency, dt)(p)
        assert torch.allclose(got, expected, rtol=1e-5, atol=1e-8), f'scale={scale}'


def test_precondIsReciprocalDiagonal(incompCase):
    """The preconditioner is 1/(dt * computeAlpha) (clamped), and equals 1/Diag(A)
    against the densely-assembled operator."""
    case = incompCase
    precond = buildIISPHPrecond(case.state, case.config, case.schemeConfig,
                                case.adjacency, case.dt)
    D = case.dt * computeAlpha(case.state, case.config, case.schemeConfig,
                               case.adjacency, apparentVolumes=case.state.masses / case.state.densities)
    D = torch.clamp(D, max=-1e-6)
    assert torch.allclose(precond, 1.0 / D, rtol=1e-5, atol=1e-8)
    # the IISPH diagonal is the exact operator diagonal (relL2 ~ 1e-6)
    A = _assembleA(case)
    trueDiag = A.diag()
    rel = float((trueDiag.double() - D.double()).norm() / trueDiag.double().norm())
    assert rel < 1e-4, f'IISPH diagonal does not match the operator diagonal: {rel}'


def test_operatorIsSymmetricNegativeSemiDefinite(incompCase):
    """Phase-0 operator probe: the discrete IISPH operator is symmetric (to fp32)
    and negative-semi-definite with a near-zero gauge mode, and is NOT diagonally
    dominant (which is why the damped relaxed-Jacobi can diverge here)."""
    case = incompCase
    A = _assembleA(case).double()
    symm = float((A - A.T).norm() / A.norm())
    assert symm < 1e-4, f'operator is not symmetric: ||A-A^T||/||A||={symm}'
    eig = torch.linalg.eigvalsh((A + A.T) / 2.0)
    assert float(eig.max()) <= abs(float(eig.min())) * 1e-3 + 1e-12, \
        f'operator has a significant positive mode: {float(eig.max())}'
    assert float(eig.min()) < 0, 'expected a negative (definite) operator'
    # not diagonally dominant on every row
    diag = A.diag().abs()
    off = (A.norm(dim=1) ** 2 - diag ** 2).clamp(min=0).sqrt()
    assert bool((off > diag).any()), 'expected at least one non-diagonally-dominant row'


def _relResid(case, pressure, b=None):
    matvec = buildIISPHMatvec(case.state, case.config, case.schemeConfig,
                              case.adjacency, case.dt)
    if b is None:
        b = _sourceTerm(case)
    bn = float(torch.linalg.norm(b))
    return (float(torch.linalg.norm(b - matvec(pressure))) / bn) if bn > 0 else 0.0


def test_bicgStabReducesResidual(incompCase):
    _, pressure, errors = _run(incompCase, PressureSolverType.bicgStab)
    assert torch.isfinite(pressure).all()
    assert _relResid(incompCase, pressure) < 1e-2


def test_gmresReducesResidual(incompCase):
    _, pressure, errors = _run(incompCase, PressureSolverType.gmres)
    assert torch.isfinite(pressure).all()
    assert _relResid(incompCase, pressure) < 1e-2


def test_cgReducesResidual(incompCase):
    # CG is viable because the operator is symmetric; it converges more slowly on
    # the ill-conditioned/gauge-mode spectrum, so the bound is looser.
    _, pressure, errors = _run(incompCase, PressureSolverType.cg)
    assert torch.isfinite(pressure).all()
    assert _relResid(incompCase, pressure) < 1e-2


def test_bicgRuns(incompCase):
    # BiCG needs the adjoint (the placeholder A^T=A is exact here because the
    # operator is symmetric) but is the least robust of the four on the
    # indefinite/gauge-mode spectrum, so we only require that it runs to
    # completion and returns a finite iterate -- not that it converges.
    a_p, pressure, errors = _run(incompCase, PressureSolverType.bicg)
    assert torch.isfinite(pressure).all()
    assert torch.isfinite(a_p).all()


def test_krylovSolversAgree(incompCase):
    # BiCGStab and GMRES are the robust pair; their solutions should be close
    # (same operator, same preconditioner), a strong check that both are solving
    # the same A p = b.
    _, pStab, _ = _run(incompCase, PressureSolverType.bicgStab)
    _, pGmres, _ = _run(incompCase, PressureSolverType.gmres)
    # compare after removing the gauge (the pressure is defined up to a constant)
    d = (pStab - pStab.mean()) - (pGmres - pGmres.mean())
    scale = float((pStab - pStab.mean()).norm())
    assert scale > 0
    assert float(d.norm()) / scale < 0.5


def test_incompressibleVariantRuns(incompCase):
    # The constant-density variant (dt**2 operator, non-negative gauge) is wired
    # through the same dispatch; check it runs and yields a finite pressure.
    a_p, pressure, errors = _run(incompCase, PressureSolverType.bicgStab,
                                 variant='incompressible')
    assert torch.isfinite(pressure).all()
    assert float(pressure.min()) >= -1e-6, 'non-negative gauge should clamp at 0'


# --- relaxed-Jacobi regression guard -----------------------------------------
# A fixed 8-iteration relaxed-Jacobi run on the seeded state fingerprints the
# unchanged default path. The errors grow (the damped Jacobi diverges on this
# non-diagonally-dominant operator -- pre-existing behaviour); what the guard
# pins is that the *sequence* is stable, i.e. the code path was not altered by
# adding the Krylov branches.
_JACOBI_FP = {
    'errors': [0.046124004, 0.062533826, 0.10262572, 0.175545648,
               0.304131687, 0.532372236, 0.936860502, 1.656827211],
    'pmean': 5.722e-06,
}


def test_relaxedJacobiRegression(incompCase):
    case = incompCase
    scfg = case.schemeConfig.solverConfig.divergenceFreeSolver
    scfg.solverType = PressureSolverType.relaxedJacobi
    scfg.minIterations, scfg.maxIterations = 0, 8
    scfg.tolerance, scfg.relaxationFactor = 1e-9, 0.5
    case.state.pressures.zero_()
    dvdt = torch.zeros_like(case.state.velocities)
    _, pressure, errors, _ = solveDivergenceFree(
        case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
    assert len(errors) == 8
    for got, want in zip(errors, _JACOBI_FP['errors']):
        assert got == pytest.approx(want, rel=1e-4), f'error sequence changed: {errors}'
    assert float(pressure.mean()) == pytest.approx(_JACOBI_FP['pmean'], rel=1e-3, abs=1e-6)
    # determinism: a second run reproduces the same iterate
    case.state.pressures.zero_()
    _, pressure2, errors2, _ = solveDivergenceFree(
        case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
    assert torch.allclose(pressure, pressure2)
    assert errors == errors2


# --- opt-in fp64 Krylov bookkeeping ------------------------------------------
# The recurrence (x/r/p dot products, Givens/least-squares scalars) runs in
# fp64 while the SPH matvec stays at production fp32. On this state the fp32
# BiCGStab recurrence loses its shadow-system orthogonality (kappa(M^-1 A) is
# ~1e8, above the fp32 precision limit) and stagnates; the fp64 bookkeeping
# goes ~10x further at the same matvec cost. See the "BiCGStab deep-dive"
# section of INCOMPRESSIBLE_SOLVER_PLAN.md.


def test_krylovFp64ConfigRoundTrip():
    cfg = IncompressibleSPHConfig()
    cfg.solverConfig.divergenceFreeSolver.krylovFp64 = True
    cfg.solverConfig.pressureSolver.krylovFp64 = True
    d = incompressibleConfigToDict(cfg)
    assert d['solverConfig']['divergenceFreeSolver']['krylovFp64'] is True
    assert d['solverConfig']['pressureSolver']['krylovFp64'] is True
    rt = dictToIncompressibleSPHConfig(d)
    assert rt.solverConfig.divergenceFreeSolver.krylovFp64 is True
    assert rt.solverConfig.pressureSolver.krylovFp64 is True
    # default is off; a dict missing the key falls back to the default
    assert IncompressibleSPHConfig().solverConfig.divergenceFreeSolver.krylovFp64 is False
    d2 = incompressibleConfigToDict(IncompressibleSPHConfig())
    del d2['solverConfig']['divergenceFreeSolver']['krylovFp64']
    del d2['solverConfig']['pressureSolver']['krylovFp64']
    rt2 = dictToIncompressibleSPHConfig(d2)
    assert rt2.solverConfig.divergenceFreeSolver.krylovFp64 is False


def test_krylovFp64DoesNotWorsenResidual(incompCase):
    case = incompCase
    scfg = case.schemeConfig.solverConfig.divergenceFreeSolver
    rel = {}
    try:
        for fp64 in (False, True):
            scfg.krylovFp64 = fp64
            case.state.pressures.zero_()
            _, pressure, _ = _run(case, PressureSolverType.bicgStab, maxIter=200)
            # the fp64 iterate must be cast back to the production dtype
            assert pressure.dtype == case.state.densities.dtype
            assert torch.isfinite(pressure).all()
            rel[fp64] = _relResid(case, pressure)
    finally:
        scfg.krylovFp64 = False
    # fp64 bookkeeping must not do worse than the fp32 recurrence on the same
    # budget (on this state it does ~10x better; the 1.5x margin keeps the
    # assertion robust across seeds/devices)
    assert rel[True] <= 1.5 * rel[False], f'fp64 {rel[True]} vs fp32 {rel[False]}'