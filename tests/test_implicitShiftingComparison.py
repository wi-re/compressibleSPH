"""Phase 4 steps 4 and 6 of `warpSPHCore/warpier_forward_mode_plan.md`.

Step 4 ("swap the matvec, keep the solver") compares
`implicitShifting.computeImplicitShift` (hand-built per-pair
`sphKernelHessian` + `torch.einsum` matvec) against
`implicitShiftingAutomatic.computeImplicitShiftAutomatic` (same
`bicgstabSolve` call, same relaxation/boundary/initializer handling, but
`grad C`/`Hess C . v` sourced from `warpSPHCore.warpOperationJVP`/
`warpOperationHVP` -- Phase 4 steps 2/3's composed-JVP bridge) on the same
jittered-lattice case `test_implicitShifting.py`'s own convergence test uses.

Two checks: (a) a single Newton step from the same starting state produces
numerically the same shift (both solves are handed an equivalent linear
system -- same `B`, and matvecs that agree per steps 2/3's own standalone
tests -- so `bicgstabSolve` should land in the same place up to solver
tolerance); (b) several outer relaxation iterations (rebuilding the
adjacency each step, mirroring `wrapper.solveShifting`'s own loop) drive
both to comparable equilibrium density uniformity, not just agreeing on
step one and then drifting apart under compounding solver-tolerance error.

Step 6 (three-way comparison, "correctness and effort/robustness, not
speed") adds `delta.computeDeltaShift` -- the pre-existing explicit
delta-SPH baseline, no Newton solve at all -- as a third relaxation path
alongside the two implicit ones, on the same case. The quantitative bar for
`computeDeltaShift` is deliberately looser than the implicit-vs-implicit
checks above: it is a different (first-order, CFL-clamped) algorithm, not
another way to compute the same operator, so this is "does it also make
real progress on the same objective", not an equilibrium-agreement check.
The qualitative half of step 6's deliverable -- lines of hand-derived math
vs. composed calls to an existing bridge -- is written up in
`warpSPHCore/warpier_forward_mode_plan.md` Phase 4's status section, not
re-derived here as a test assertion.
"""

from __future__ import annotations

import torch

from warpSPH.utils import buildDomainDescription
from warpSPH.configurations.simulationConfig import buildConfig
from warpSPH.configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingImplicitOperator, ShiftingImplicitSolver
from warpSPH.sample.regular import sampleRegularParticles
from warpSPH.modules.density import computeDensities
from warpSPH.modules.shifting.delta import computeDeltaShift
from warpSPH.modules.shifting.implicitShifting import computeImplicitShift
from warpSPH.modules.shifting.implicitShiftingAutomatic import computeImplicitShiftAutomatic
from warpSPHCore import ParticleState, SupportScheme, buildVerletList


def _jitteredLatticeState(nx, dim, L, device, dtype, jitter, seed):
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                      dtype=dtype, dx=L / nx, cflFactor=0.3)
    template = sampleRegularParticles(nx=nx, domain=domain, targetNeighbors=config.targetNeighbors)
    n = template.positions.shape[0]
    particleDx = template.masses.pow(1 / dim).mean().item()

    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(template.positions.shape, generator=gen).to(device=device, dtype=dtype)
    positions = template.positions + jitter * particleDx * noise

    state = ParticleState(
        positions=positions, supports=template.supports, masses=template.masses,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        densities=torch.ones(n, device=device, dtype=dtype),
    )
    # computeDeltaShift (modules/shifting/delta.py) unconditionally reads
    # currentState.velocities for its Mach-number estimate, even with
    # computeMach disabled -- see test_implicitShifting.py's own comment.
    state.velocities = torch.zeros_like(state.positions)
    return state, config, domain


def _schemeConfig():
    schemeConfig = WeaklyCompressibleSPHConfig()
    schemeConfig.shiftProperties.active = True
    schemeConfig.surfaceDetectionConfig.active = False  # periodic, no real free surface -- see test_implicitShifting.py
    # computeImplicitShiftAutomatic's autodiff-sourced Hessian only ever
    # represents ShiftingImplicitOperator.exactHessian (there's no automatic-
    # differentiation counterpart to the default `legacyPairwise` operator,
    # since it isn't an actual Hessian of anything -- see implicitShifting.py's
    # module docstring). Pin it explicitly so this cross-check keeps comparing
    # the two Hessian code paths against each other, not against a different
    # operator entirely now that legacyPairwise is the default.
    schemeConfig.shiftProperties.implicitOperator = ShiftingImplicitOperator.exactHessian
    return schemeConfig


def _cloneState(state):
    cloned = ParticleState(
        positions=state.positions.clone(), supports=state.supports.clone(),
        masses=state.masses.clone(), kinds=state.kinds.clone(),
        densities=state.densities.clone() if state.densities is not None else None,
    )
    if getattr(state, 'velocities', None) is not None:
        cloned.velocities = state.velocities.clone()
    return cloned


def test_automaticImplicitShift_matchesHandBuilt_singleStep():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L = 2, 8, 1.0

    state, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                    jitter=0.1, seed=1234)
    schemeConfig = _schemeConfig()
    rho0 = schemeConfig.fluid.restDensity
    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
    state.densities = computeDensities(state, config, schemeConfig, adjacency)

    handBuilt, _ = computeImplicitShift(state, config, schemeConfig, domain, adjacency, iters=1)
    automatic, _ = computeImplicitShiftAutomatic(state, config, schemeConfig, domain, adjacency, iters=1)

    torch.testing.assert_close(automatic, handBuilt, rtol=1e-3, atol=1e-6)


def test_automaticImplicitShift_convergesLikeHandBuilt():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L = 2, 16, 1.0
    # 8 outer iterations: enough for both solves to visibly relax the lattice
    # (checked empirically to stay well clear of the marginal-stability
    # blow-up this plan's own "Status" section already documents for the
    # hand-built solver's undamped-per-step Newton iteration -- at
    # `implicitRelaxation=0.1`, the two histories start diverging by
    # iteration ~12 on this same seed/case, one further data point for step
    # 5's own question about robustness parity, not something to paper over
    # by cherry-picking a larger window here).
    outerIters = 8

    baseState, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                        jitter=0.1, seed=1234)
    schemeConfig = _schemeConfig()
    rho0 = schemeConfig.fluid.restDensity

    def relax(shiftFn):
        state = _cloneState(baseState)
        adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
        state.densities = computeDensities(state, config, schemeConfig, adjacency)
        history = [((state.densities - rho0) / rho0).std().item()]
        for _ in range(outerIters):
            adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
            update, adjacency = shiftFn(state, config, schemeConfig, domain, adjacency, iters=1)
            state.positions = state.positions + update
            adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
            state.densities = computeDensities(state, config, schemeConfig, adjacency)
            history.append(((state.densities - rho0) / rho0).std().item())
        return history

    handBuiltHistory = relax(computeImplicitShift)
    automaticHistory = relax(computeImplicitShiftAutomatic)

    # Both should relax the jittered lattice toward uniform density (the
    # quantity the implicit solve's RHS directly targets) ...
    assert handBuiltHistory[-1] < handBuiltHistory[0] * 0.5
    assert automaticHistory[-1] < automaticHistory[0] * 0.5
    # ... and land at comparable equilibria, not just agree on step one and
    # then drift apart under compounding solver-tolerance error.
    assert abs(automaticHistory[-1] - handBuiltHistory[-1]) < 0.1 * handBuiltHistory[0]


def test_threeWayShiftComparison_allRelaxTowardUniformDensity():
    """Phase 4 step 6: `computeDeltaShift` (explicit baseline) alongside the
    two implicit solves, all three run on the same jittered-lattice case and
    graded on the same relative-density-std metric `test_implicitShifting.py`
    already uses.

    Explicit and implicit shifting are not two ways to compute the same
    operator (unlike the two implicit variants above), so this does not
    assert equilibrium agreement between `computeDeltaShift` and the other
    two -- only that all three make substantial, non-trivial progress on the
    shared objective, and that the two implicit solves still agree with each
    other under the same weaker bound the explicit one has to clear (this
    reuses, rather than duplicates, the tighter equilibrium-agreement check
    in `test_automaticImplicitShift_convergesLikeHandBuilt` above).
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L = 2, 16, 1.0
    outerIters = 8  # see test_automaticImplicitShift_convergesLikeHandBuilt for why 8

    baseState, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                        jitter=0.1, seed=1234)
    schemeConfig = _schemeConfig()
    rho0 = schemeConfig.fluid.restDensity

    def relax(shiftFn):
        state = _cloneState(baseState)
        adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
        state.densities = computeDensities(state, config, schemeConfig, adjacency)
        history = [((state.densities - rho0) / rho0).std().item()]
        for _ in range(outerIters):
            adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
            update, adjacency = shiftFn(state, config, schemeConfig, domain, adjacency, iters=1)
            state.positions = state.positions + update
            adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
            state.densities = computeDensities(state, config, schemeConfig, adjacency)
            history.append(((state.densities - rho0) / rho0).std().item())
        return history

    explicitHistory = relax(computeDeltaShift)
    handBuiltHistory = relax(computeImplicitShift)
    automaticHistory = relax(computeImplicitShiftAutomatic)

    # All three reduce density-std substantially over the shared window --
    # correctness, not speed, is the bar. Bounds differ per scheme because
    # they genuinely converge at different rates on this case, confirmed
    # empirically rather than assumed a single shared bound would fit:
    # explicit delta-SPH's CFL-clamped per-step correction is markedly
    # slower here (only reaches ~0.79x over these same 8 steps -- consistent
    # with, not a regression from, test_implicitShifting.py's own comment
    # that its 25-iteration window needs a looser bound than the implicit
    # scheme's), while both Newton solves clear a much tighter bar in the
    # same window.
    bounds = {'deltaShift': 0.85, 'implicitShift': 0.7, 'implicitShiftAutomatic': 0.7}
    for name, history in (('deltaShift', explicitHistory),
                           ('implicitShift', handBuiltHistory),
                           ('implicitShiftAutomatic', automaticHistory)):
        assert history[-1] < bounds[name] * history[0], (
            f'{name}: relative density std only went from {history[0]:.4f} to {history[-1]:.4f} '
            f'over {outerIters} steps (history={history})')

    # The two implicit solves -- two ways to compute the same operator --
    # still land at comparable equilibria with the explicit baseline in the
    # loop too (same check as test_automaticImplicitShift_convergesLikeHandBuilt,
    # confirming that comparison isn't an artifact of running it in isolation).
    assert abs(automaticHistory[-1] - handBuiltHistory[-1]) < 0.1 * handBuiltHistory[0]


def test_nullSpaceLiftIsAppliedAndFinite():
    """The opt-in `implicitNullSpaceLift` (A -> A + lift*I) is off by default
    (see the config test) and aimed at the near-singular `exactHessian`
    operator -- this file's case. The pure-torch preconditioner/solver tests in
    test_implicitShiftingSolvers.py don't exercise this one-liner, so verify it
    end-to-end on the real pipeline:
      - a small lift perturbs the update by O(lift) (the Tikhonov bias) --
        visible above fp32 noise but small;
      - a large lift dominates the operator (lift*I >> A), so both Krylov
        solvers converge to the same well-conditioned answer;
      - the identical one-liner in `computeImplicitShiftAutomatic` stays
        finite and applied there too.
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L = 2, 8, 1.0
    state, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                   jitter=0.1, seed=1234)
    schemeConfig = _schemeConfig()  # exactHessian
    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
    state.densities = computeDensities(state, config, schemeConfig, adjacency)

    def run(shiftFn, lift):
        sc = _schemeConfig()
        sc.shiftProperties.implicitNullSpaceLift = lift
        return shiftFn(_cloneState(state), config, sc, domain, adjacency, iters=1)[0]

    # small lift: applied (change exceeds fp32 noise) but O(lift) (small)
    u0 = run(computeImplicitShift, 0.0)
    uSmall = run(computeImplicitShift, 1e-2)
    assert torch.isfinite(u0).all() and torch.isfinite(uSmall).all()
    diff = torch.linalg.norm(uSmall - u0)
    u0norm = torch.linalg.norm(u0)
    assert diff > 1e-4 * u0norm, f'lift=1e-2 had no visible effect (diff={diff})'
    assert diff < 0.1 * u0norm, f'lift=1e-2 perturbed the update more than O(lift) (diff/||u0||={diff / u0norm})'

    # large lift: the operator is dominated by lift*I (well-conditioned), so
    # both Krylov solvers converge to the same Tikhonov answer
    scB = _schemeConfig(); scB.shiftProperties.implicitNullSpaceLift = 10.0
    scB.shiftProperties.implicitSolver = ShiftingImplicitSolver.bicgstab
    scG = _schemeConfig(); scG.shiftProperties.implicitNullSpaceLift = 10.0
    scG.shiftProperties.implicitSolver = ShiftingImplicitSolver.gmres
    uB = computeImplicitShift(_cloneState(state), config, scB, domain, adjacency, iters=1)[0]
    uG = computeImplicitShift(_cloneState(state), config, scG, domain, adjacency, iters=1)[0]
    torch.testing.assert_close(uB, uG, rtol=1e-3, atol=1e-6)

    # the automatic path shares the identical lift one-liner: finite + applied
    a0 = run(computeImplicitShiftAutomatic, 0.0)
    aSmall = run(computeImplicitShiftAutomatic, 1e-2)
    assert torch.isfinite(a0).all() and torch.isfinite(aSmall).all()
    assert torch.linalg.norm(aSmall - a0) > 1e-4 * torch.linalg.norm(a0), 'automatic lift had no visible effect'
