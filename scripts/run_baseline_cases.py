"""Run the three `divergenceFree` baseline cases (item 2 of
`DFSPH_IMPROVEMENT_PLAN.md`'s "what is left") to their full `tLimit` and report
a stability summary for each:

- `staticBlob` (IC): nothing should happen -- velocity stays ~0, no drift.
- `impact` (IC, `--integrationScheme semiImplicitEuler`): the collision should
  close the gap and merge the bodies, reproducing the WC outcome.
- `impact` (WC, `deltaSPH`): the reference the IC run is compared against.
- `hydrostaticColumn` (IC): at rest under gravity -- the scheme is expected to
  *fail* this (see the case docstring); the run reports where it diverges.

The trajectories are downsampled so the output stays readable, and a final
line per case gives the figures of merit (max velocity over the run, final
density band, final displacement / gap, and the divergence step if any).

Usage:
    python scripts/run_baseline_cases.py [--nx 64] [--tLimit 1.0] [--every 25]
"""
import argparse
import io
import contextlib

from warpSPH.runner import run

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=64)
args.add_argument('--tLimit', type=float, default=1.0)
args.add_argument('--every', type=int, default=25,
                  help='print one trajectory row per this many steps')
args = args.parse_args()


def report(case, title, extra=None):
    overrides = dict(nx=args.nx, tLimit=args.tLimit,
                     progress=False, quiet=True, plot=False, store=False)
    overrides.update(extra or {})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run(case, **overrides)
    traj = result.trajectory
    if not traj:
        print(f'=== {title}: no trajectory')
        return
    keys = [k for k in traj[0] if k not in ('step', 't')]
    print(f'=== {title}: diverged={result.diverged}  '
          f'steps={len(traj) - 1}  t_final={traj[-1]["t"]:.4g}')
    header = 'step'.rjust(7) + 't'.rjust(9) + ''.join(k[:14].rjust(15) for k in keys)
    print(header)
    for i, row in enumerate(traj):
        if i % args.every and i != len(traj) - 1:
            continue
        line = f'{row["step"]:>7}' + f'{row["t"]:>9.4f}'
        for k in keys:
            v = row.get(k)
            line += (f'{v:>15.4g}' if isinstance(v, (int, float))
                     else f'{str(v)[:14]:>15}')
        print(line)
    # Figures of merit over the whole run.
    def col(name):
        return [row.get(name) for row in traj if isinstance(row.get(name), (int, float))]
    vcol = col('maxVelocity')
    if vcol:
        print(f'  [FOM] maxVelocity_over_run={max(vcol):.4g}  '
              f'final_maxVelocity={vcol[-1]:.4g}')
    for name in ('maxDensity', 'minDensity', 'dispMax', 'gap',
                 'pressureSlopeRatio', 'pressureResidual'):
        c = col(name)
        if c:
            print(f'  [FOM] {name}: final={c[-1]:.4g}  '
                  f'extreme={max(abs(x) for x in c):.4g}')
    print()


if __name__ == '__main__':
    from warpSPH.cases import staticBlob, impact, hydrostaticColumn

    report(staticBlob.staticBlobCase, 'staticBlob (IC)')
    report(impact.impactCase, 'impact (IC, semiImplicitEuler)',
           extra=dict(scheme='divergenceFree',
                      integrationScheme='semiImplicitEuler'))
    report(impact.impactCase, 'impact (WC, deltaSPH)')
    report(hydrostaticColumn.hydrostaticColumnCase, 'hydrostaticColumn (IC)')
