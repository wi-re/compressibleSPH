"""Probe (2026-08-26, follow-on to `scratch_dfsph_kolmogorov_findings.md`
E1.9 finding #3): does enabling the commented-out velocity correction in
`IncompressibleSystem.finalize` (`warpSPH/src/warpSPH/systems/
incompressible.py`, ~line 275) fix or improve the `nx=128`/`alpha=0.01`
Kolmogorov divergence at step 720?

Background (found by direct code reading, not guessed): `finalize` runs a
second, constant-density pressure solve (`solveIncompressible`) whose output
`dvdt_incomp` is applied as a *position* correction (`dx = dt**2 *
dvdt_incomp; self.state.positions += dx`) -- the standard IISPH/particle-
shifting trick of getting a position correction from an acceleration via a
second pressure solve. This is DFSPH's own implicit particle-shifting
mechanism (distinct from, and unconditionally active regardless of, the
separate `schemeConfig.shiftProperties.active`-gated block earlier in the
same function, which is known dead debug scaffolding from a prior session
and is NOT touched here). Immediately before applying `dx`, `finalize`
already computes `gradVel` (velocity-field Jacobian) and `proj_vel =
einsum('nij,ni->nj', gradVel, dx)` -- the standard first-order Taylor
correction (`dV ~= grad(V).dx`) needed to keep velocities kinematically
consistent with particles that just got silently moved. But `self.state.
velocities -= proj_vel` is commented out right next to the `positions += dx`
line that *is* live -- so every step, particles are shifted without their
carried velocity ever being corrected to match the new position.

This script does NOT edit `systems/incompressible.py`. It monkeypatches
`IncompressibleSystem.finalize` at runtime by taking `inspect.getsource` of
the live method, textually enabling the one commented-out line, `exec`-ing
the patched source back into the module's own namespace (so every name the
method body already closes over -- `solveIncompressible`, `computeDensities`,
`warpOperation`, `detectFreeSurface`, etc. -- still resolves correctly), and
rebinding the class attribute for the duration of the process. This is a
throwaway probe process; the actual `.py` file on disk is never touched.

Usage: `python scripts/probe_kolmogorovIncompressibleVelCorrection.py
[--nx 128] [--nsteps 1000] [--seed 0] [--tag patched]`
Prints the same trace format as `probe_kolmogorovIncompressible.py`'s own
`run()` (this script imports and reuses that function directly, per the
project convention of not rebuilding an already-working probe).
"""

from __future__ import annotations

import argparse
import inspect
import sys
import types

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import probe_kolmogorovIncompressible as base  # noqa: E402  (script dir on sys.path)


def patchVelocityCorrection():
    """Enable `self.state.velocities -= proj_vel` in `IncompressibleSystem.
    finalize` for the remainder of this process. Returns the original
    (unpatched) function so a caller can restore it if needed."""
    import warpSPH.systems.incompressible as incompMod

    original = incompMod.IncompressibleSystem.finalize
    src = inspect.getsource(original)
    needle = "        # self.state.velocities -= proj_vel\n"
    replacement = "        self.state.velocities -= proj_vel\n"
    if needle not in src:
        raise RuntimeError(
            "Expected commented-out line not found verbatim in "
            "IncompressibleSystem.finalize's source -- the production file "
            "may have changed since this probe was written. Aborting rather "
            "than silently patching the wrong thing."
        )
    patchedSrc = src.replace(needle, replacement, 1)
    assert patchedSrc != src

    # `exec`-ing this function body outside its original class statement
    # loses the implicit `__class__` cell zero-arg `super()` needs (it is
    # normally synthesized by the compiler when `super`/`__class__` appear
    # lexically inside a `class` block). Bind it explicitly instead -- exactly
    # equivalent at runtime, `IncompressibleSystem` is resolvable from the
    # module's own globals since the class object already exists by the time
    # this patch runs.
    patchedSrc = patchedSrc.replace(
        "super().finalize(", "super(IncompressibleSystem, self).finalize(", 1)

    # `inspect.getsource` on a method returns it already dedented to column 0
    # relative to the class body is NOT guaranteed -- it keeps the original
    # file's indentation (4 spaces for a dataclass method body's `def` line).
    # `exec` needs top-level-compilable source, so dedent first.
    import textwrap
    patchedSrc = textwrap.dedent(patchedSrc)

    ns = dict(incompMod.__dict__)  # same globals the original method closes over
    exec(compile(patchedSrc, f"<patched:{incompMod.__file__}>", 'exec'), ns)
    patchedFn = ns['finalize']

    incompMod.IncompressibleSystem.finalize = patchedFn
    return original


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--xi', type=float, default=1.0)
    p.add_argument('--k', type=float, default=base.K_DEFAULT)
    p.add_argument('--alpha', type=float, default=base.ALPHA_REFERENCE)
    p.add_argument('--nu', type=float, default=None)
    p.add_argument('--jitter', type=float, default=0.01)
    p.add_argument('--nsteps', type=int, default=1000)
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--maxdt', type=float, default=1e-1)
    p.add_argument('--L', type=float, default=base.L_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--print-every', type=int, default=20)
    p.add_argument('--tag', type=str, default='patched')
    args = p.parse_args()

    original = patchVelocityCorrection()
    print(f'[{args.tag}] IncompressibleSystem.finalize PATCHED: '
         f'velocities -= proj_vel is now LIVE for this run.')

    if args.nu is None:
        _sys, _cfg, _, _, _h = base.buildKolmogorovIncompressibleSystem(
            args.nx, args.xi, args.k, 0.0, args.jitter, L=args.L, cflFactor=args.cfl,
            seed=args.seed)
        nu = base.alphaToNu(args.alpha, base.SOUND_SPEED_REFERENCE, _h, 2)
        del _sys, _cfg
    else:
        nu = args.nu

    base.run(args.nx, args.xi, args.k, nu, args.jitter, args.nsteps, args.cfl,
            args.print_every, 0, L=args.L, maxDt=args.maxdt, seed=args.seed,
            tag=args.tag)
