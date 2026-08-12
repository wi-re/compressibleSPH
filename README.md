# warpSPH

*(formerly `compressibleSPH`; the SPH core previously hosted at this URL now lives at
[`wi-re/warpSPHCore`](https://github.com/wi-re/warpSPHCore))*

![](examples/compressible/outputs/15-Triple_Point_equal_mass.gif)

A GPU-oriented Python package for Smoothed Particle Hydrodynamics experiments,
built on [NVIDIA Warp](https://github.com/NVIDIA/warp) and PyTorch. It covers
three families of scheme against one shared infrastructure:

| family | schemes | typical cases |
|---|---|---|
| compressible | `Monaghan`, `CompSPH`, `CRKSPH` | shock tubes, blast waves, instabilities |
| weakly compressible | `deltaSPH` | free surfaces, dam breaks, channel flow |
| incompressible | `divergenceFree` (DFSPH) | Taylor-Green, decaying turbulence |

Everything a run needs — configuration, geometry, the step loop, export,
plotting, video encoding — is shared, so a *case* is only the physics that
distinguishes it: how the geometry is built, what the initial conditions are,
and what to measure.

**This repository is the frontend.** The kernels, neighbour search and data
types live in [`warpSPHCore`](https://github.com/wi-re/warpSPHCore), the time
integrators in [`warpSPHIntegrators`](https://github.com/wi-re/warpSPHIntegrators),
and the visualisation in [`warpSPHPlotting`](https://github.com/wi-re/warpSPHPlotting).

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Running cases](#running-cases) — the CLI, config files, output, plots, reports
- [The cases](#the-cases) — what ships, and which example each came from
- [Writing a case](#writing-a-case)
- [From Python and notebooks](#from-python-and-notebooks)
- [Configuration reference](#configuration-reference)
- [Package layout](#package-layout)
- [Tests](#tests)
- [Gallery](#gallery)
- [Precision, and other things that bite](#precision-and-other-things-that-bite)

---

## Install

The four repositories are developed together, so editable installs from a common
parent directory are the supported setup.

```bash
mkdir warp-sph-stack && cd warp-sph-stack
git clone https://github.com/wi-re/warpSPHCore
git clone https://github.com/wi-re/warpSPHIntegrators
git clone https://github.com/wi-re/warpSPHPlotting
git clone https://github.com/wi-re/warpSPH

conda create -n warp_env python=3.14 && conda activate warp_env
pip install tqdm seaborn pandas pyglet dill numba scipy scikit-image h5py ipykernel ipympl
pip install -e warpSPHCore/ warpSPHIntegrators/ warpSPHPlotting/
pip install -e warpSPH/
```

A CUDA device is required for anything beyond imports. `ffmpeg` is optional —
without it, `--video` is skipped rather than failing a run that already produced
its frames.

## Quick start

```bash
warpsph-run sod --plot                       # Sod shock tube, live plot
warpsph-run                                  # list every case, with a description
warpsph-run kelvinHelmholtz --nx 256 --store # write HDF5 as it goes
```

Or run the example script next to its notebook, which pins that example's
settings:

```bash
python examples/compressible/12-kelvin-helmholtz.py
```

Both print what they are about to do, then what they did:

```
------------------------------------------------------------------------------
  warpSPH | kelvinHelmholtz
  Kelvin-Helmholtz instability (2D), compressible SPH.
------------------------------------------------------------------------------
  scheme      CRKSPH (CompressibleSPHScheme) | solver crkSPH_step
  device      cuda:0 (NVIDIA GeForce RTX 3090) | float32
  particles   65,536 | dim 2 | nx 256
  domain      [0, 0] to [1, 1]  periodic
  kernel      B7 | n_h 4 | targetNeighbors 50.3
  integrator  rungeKutta2 | support KernelMeanSymmetric
  timestep    dt 0.000389 | adaptive, cfl 0.3
  duration    10,282 steps to t = 4
  output      export/12-kelvinHelmholtz
              frames every 1 steps (vispy)
------------------------------------------------------------------------------
```

---

## Running cases

A case is run through a shared runner rather than per-script boilerplate. Every
flag below is a field of `CaseSpec`, so a run is fully described by a file as
well as by a command line.

### The CLI

```bash
warpsph-run <case> [flags]                  # the console script
python -m warpSPH.cases.sod [flags]         # the same case, as a module
python examples/compressible/01-sod/sod_1d.py [flags]
```

`warpsph-run` with no case name lists everything with its description.

`warpsph-run --precision float64 ...` works because the CLI selects the precision
*before* `warpSPH` is imported. The other two entry points import `warpSPH`
first and are stuck with whatever is already active — see
[Precision](#precision-and-other-things-that-bite).

### Comparing solvers

`--scheme` picks the solver, so the same case can be run against every scheme in
its family and the results compared directly:

```bash
for solver in Monaghan CompSPH CRKSPH; do
  warpsph-run sod --scheme $solver --store --caseName sod-$solver
done
```

`warpsph-run <case> --help` lists the valid names and the case's own default.
The banner names the step function that was actually selected
(`solver compressibleSPH_Monaghan`), so a comparison cannot quietly run the same
solver twice.

The three compressible solvers differ in what they conserve — over a 20-step Sod
run, CompSPH holds total energy to **exactly** zero drift by construction, CRKSPH
to ~5e-6, and Monaghan to ~1e-3, the last because its artificial viscosity and
conductivity are dissipative by design. `tests/test_physics.py` pins all three.

### Config files and sweeps

```bash
warpsph-run tgv --config examples/sweeps/tgv_nu.yaml --nx 128
warpsph-run sod --saveConfig resolved.yaml           # write the resolved spec out
```

Precedence, lowest to highest: `CaseSpec` defaults → the case's own defaults →
the `--config` file → explicit CLI flags. That works because every generated flag
defaults to "not passed", so a `true` in a config file is still overridable from
the command line. Unknown keys raise rather than being silently ignored, so a
typo in a sweep file fails loudly. See [examples/sweeps/](examples/sweeps/).

Case-specific knobs live under a `params:` block; list- and dict-valued ones
(shock regions, a gravity vector) are settable from a config file only, since
they have no sensible flag form.

### Output

`--store` writes state, `--plot` writes frames, `--video` encodes them:

```
export/<caseName>_<YYYY-MM-DD_HH-MM-SS>/
  caseSpec.json         the fully resolved spec, so a run can be reproduced
  config.json           the simulation and scheme configuration
  trajectory/           one .h5 per stored step   (storeMode: states, the default)
  trajectory.h5         one growing file          (storeMode: trajectory)
  images/frame_*.png    plot frames
  output.mp4, out.gif   if --video and ffmpeg is present
```

Each run gets its own timestamped folder, so re-running a case accumulates
results instead of overwriting the previous one. To pick a run back up without
knowing its exact name:

```python
from warpSPH.io import latestExportPath, findExportRuns

path = latestExportPath('01-sodShockTube')   # newest run of that case
runs = findExportRuns('01-sodShockTube')     # all of them, oldest first
```

Set `WARPSPH_EXPORT_TIMESTAMP=0` (or pass `timestamped=False` to `prepExport`)
for the old flat `export/<caseName>/` layout. `latestExportPath` falls back to
that layout, so trees written before this change still resolve.

`--exportRoot` (or `$WARPSPH_EXPORT_ROOT`) moves the whole tree; it defaults to
`export/`.

### Plots

`--plot` opens a live, updating window *and* writes the frames. The rendering
backend is chosen by dimension — **vispy for 2D, matplotlib for 1D** — because a
matplotlib scatter of a large 2D particle set costs more per frame than the
physics step it is drawing. Measured on a 16k-particle Kelvin-Helmholtz run,
6 frames: matplotlib spent 19.97 s plotting against 1.65 s of physics; vispy
brings that to 3.97 s.

| flag | effect |
|---|---|
| `--plot` | draw, and write `frame_NNNNN.png` under the export directory |
| `--plotBackend` | `matplotlib`, `vispy` or `pyVista`; default is by dimension |
| `--plotInterval` | steps between redraws |
| `--no-show` | keep the frames, skip the window |
| `--no-holdPlot` | do not block on the final figure when the run ends |

A vispy canvas that cannot start — no GL context over ssh, or in a container —
falls back to matplotlib with a message rather than taking the run down. The
example scripts pass `--plot` for you, since an example is meant to be watched;
`--no-plot` turns it off.

### What a run prints

The **banner** goes out once setup is finished and `dt` is resolved, so it shows
what will actually run rather than what was asked for. The **report** closes the
run:

```
------------------------------------------------------------------------------
  kelvinHelmholtz finished in 3m 12s
------------------------------------------------------------------------------
  steps       2,564 | t = 4 | final dt 0.001557
  step time   mean 41.2 ms | min 38.1 | max 248.2 | 1m 45s in the loop (55% of wall)
  diagnostics                     initial        final          min          max
              kineticEnergy       0.15942      0.09318      0.09318      0.15945
              totalEnergy          3.9094       3.9094       3.9094       3.9094
  output      export/12-kelvinHelmholtz
              6 state files | 2564 frames (vispy)
------------------------------------------------------------------------------
```

This exists for the unattended run: coming back to a finished terminal should
answer "did it work, and did it stay sane" without re-reading the scrollback.
Diagnostics show min/max as well as initial/final, because an excursion that
recovered before the end is invisible in the final value alone; the file counts
are read off disk rather than inferred from the configured intervals.

A diverged run says so in the header, warns that the numbers are unusable, **and
exits non-zero**, so a shell script can tell.

`--quiet` / `-q` suppresses banner, report and progress bar (and warp's
per-module load logging). The progress bar is on only when a terminal is
watching — redirected to a file it would bury the report under carriage returns
— and `--progress` forces it back on.

---

## The cases

Every notebook under `examples/` that runs a simulation has a case, plus a `.py`
script next to the notebook that runs it with that example's settings. The
notebooks stay for exploration; the scripts are what you run unattended.

### Compressible

| case | script | notes |
|---|---|---|
| `sod` | [01-sod/sod_1d.py](examples/compressible/01-sod/sod_1d.py) | Sod shock tube, 1D; own directory ([01-sod/](examples/compressible/01-sod/)) with a resume script/notebook and a trajectory-export demo -- see its notebooks for the pattern |
| `linearWave` | [02-linear-wave.py](examples/compressible/02-linear-wave.py) | linear acoustic wave, 1D |
| `kidder` | [03-kidder-isentropic-compression.py](examples/compressible/03-kidder-isentropic-compression.py) | isentropic compression; analytically driven boundary bands |
| `noh` | [04-noh-implosion.py](examples/compressible/04-noh-implosion.py) | Noh implosion, 1D |
| `woodwardColella` | [05-woodward-colella.py](examples/compressible/05-woodward-colella.py) | interacting blast waves, 1D |
| `sedov` | [06](examples/compressible/06-sedov-taylor-blastwave-1d.py), [07](examples/compressible/07-sedov-taylor-blastwave-2d.py) | Sedov-Taylor blast; `--dim 1` or `--dim 2` |
| `hydrostatic` | [08-hydrostatic.py](examples/compressible/08-hydrostatic.py) | hydrostatic equilibrium; the exact answer is "nothing happens" |
| `gresho` | [09-gresho-chan-vortex.py](examples/compressible/09-gresho-chan-vortex.py) | Gresho-Chan vortex, a steady state |
| `yee` | [10-yee-vortex.py](examples/compressible/10-yee-vortex.py) | Yee isentropic vortex, sampled on shells |
| `shearingNoh` | [11-shearing-noh-implosion-2d.py](examples/compressible/11-shearing-noh-implosion-2d.py) | Noh implosion with transverse shear |
| `kelvinHelmholtz` | [12-kelvin-helmholtz.py](examples/compressible/12-kelvin-helmholtz.py) | Kelvin-Helmholtz instability |
| `rayleighTaylor` | [13-rayleigh-taylor.py](examples/compressible/13-rayleigh-taylor.py) | Rayleigh-Taylor instability |
| `triplePoint` | [14](examples/compressible/14-triple-point.py), [15](examples/compressible/15-triple-point-equal-mass.py) | `--equalMass` or `--no-equalMass` sampling |

### Weakly compressible (deltaSPH)

| case | script | notes |
|---|---|---|
| `impact` | [01](examples/weaklyCompressible/01-impact-spheres.py), [02](examples/weaklyCompressible/02-impact-squares.py) | two bodies colliding; `--shape circle` or `box` |
| `squarePatch` | [03-rotating-square-patch.py](examples/weaklyCompressible/03-rotating-square-patch.py) | rotating square patch of fluid |
| `droplet` | [04-oscillating-droplet.py](examples/weaklyCompressible/04-oscillating-droplet.py) | oscillating droplet in a central potential |
| `tgv-wc` | [05-taylor-green-vortex.py](examples/weaklyCompressible/05-taylor-green-vortex.py) | Taylor-Green vortex with explicit viscosity |
| `randomFlow` | [06](examples/weaklyCompressible/06-periodic-random-flow.py), [07](examples/weaklyCompressible/07-bounded-random-flow.py) | divergence-free noise; `--bounded` adds walls |
| `kolmogorov` | [08-kolmogorov-flow.py](examples/weaklyCompressible/08-kolmogorov-flow.py) | sinusoidally forced periodic box |
| `ldc` | [09-lid-driven-cavity.py](examples/weaklyCompressible/09-lid-driven-cavity.py) | lid-driven cavity |
| `movingObstacle` | [10-moving-obstacle.py](examples/weaklyCompressible/10-moving-obstacle.py) | flow past a spinning rigid body |
| `drivenSquare` | [11-driven-square.py](examples/weaklyCompressible/11-driven-square.py) | driven channel flow past a cylinder |
| `dambreak` | [12-dambreak.py](examples/weaklyCompressible/12-dambreak.py) | dam break with optional obstacle |
| `openFlow` | [13-open-flow.py](examples/weaklyCompressible/13-open-flow.py) | open channel flow past an obstacle |

### Incompressible (DFSPH)

| case | script | notes |
|---|---|---|
| `tgv` | [01-tgv-incomp.py](examples/incompressible/01-tgv-incomp.py) | Taylor-Green vortex, divergence-free |
| `squarePatch` | [03-rotating-square-patch.py](examples/incompressible/03-rotating-square-patch.py) | same case at `--scheme divergenceFree` |
| `randomFlow` | [periodic-random-flow.py](examples/incompressible/periodic-random-flow.py) | same case at `--scheme divergenceFree` |

Several cases cover more than one notebook, because those notebooks differed
only in a flag; the scripts pin the flag. Two notebooks deliberately have no
case: `weaklyCompressible/naca.ipynb` is a standalone airfoil-SDF visualisation
with no simulation in it, and `incompressible/1d-test.ipynb` is an exploratory
scratchpad rather than a published example.

---

## Writing a case

A `Case` is a name, a scheme, and a set of hooks over a `RunContext`. Only
`buildSystem` is required.

```python
from warpSPH.runner import Case, RunContext, caseMain, registerCase

def configureScheme(ctx: RunContext) -> None:
    """Mutate ctx.schemeConfig (and ctx.config) before anything is sampled."""
    ctx.schemeConfig.gamma = ctx.param('gamma')

def buildSystem(ctx: RunContext):
    """Geometry and particle sampling. Returns a SimulationSystem."""
    return sampleSomething(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                           ctx.SimulationState, ctx.SimulationSystem)

def diagnostics(ctx: RunContext, state) -> dict:
    """Scalars recorded every step; the progress bar, report and tests read these."""
    return {'kineticEnergy': ...}

myCase = registerCase(Case(
    name='myCase',
    scheme='CRKSPH',
    description='One line, shown by `warpsph-run` with no arguments.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    defaults=dict(dim=2, nx=128, L=1.0, tLimit=2.0),   # CaseSpec overrides
    params=dict(gamma=5 / 3),                          # become --flags
))

if __name__ == '__main__':
    caseMain(myCase)
```

Add the module name to `CASE_MODULES` in
[`warpSPH/cases/__init__.py`](src/warpSPH/cases/__init__.py) and it is reachable
as `warpsph-run myCase`.

The full hook set, in call order:

| hook | signature | for |
|---|---|---|
| `configureScheme` | `(ctx)` | scheme config, domain reshaping |
| `buildSystem` | `(ctx) -> system` | **required**; geometry and sampling |
| `initialConditions` | `(ctx, system)` | velocities/energies; often where `dt` is fixed |
| `setupPlot` / `updatePlot` | `(ctx, state)` / `(ctx, state, handle, step)` | figures |
| `diagnostics` | `(ctx, state) -> dict` | per-step scalars |
| `postStep` | `(ctx, state, step)` | re-impose something after each step |
| `timestep` | `(ctx, state) -> dt` | recompute `dt`; makes the run time-bounded |
| `extraData` | `(ctx, state) -> dict` | metadata on every exported frame |

`ctx.scratch` is a free dict for state between hooks. A case may also replace
`ctx.spec` during setup — Kidder and Sedov only learn their time limit from the
analytic solution — and the loop reads it back.

Most of a case is already written: `warpSPH.cases.compressible` and
`warpSPH.cases.weaklyCompressible` carry the setup each family shares, and
`warpSPH.cases.plotting` turns a list of fields into a `setupPlot`/`updatePlot`
pair.

---

## From Python and notebooks

```python
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')          # must precede any warpSPH import

from warpSPH.runner import run
from warpSPH.cases.tgv import tgvCase

result = run(tgvCase, nx=64, nSteps=200, quiet=True)
result.series('kineticEnergy')          # one diagnostic across the run
result.diverged, result.wallTime, result.exportPath
```

Keyword arguments to `run` override the case defaults, which is what makes a
parameter study a `for` loop rather than a shell script.

At a lower level, `buildScheme` returns a frozen `SchemeBundle` with named
fields — `SimulationSystem`, `SimulationState`, `SimulationConfig`,
`SimulationUpdate`, `stepFunction`, `exportFunction`, `importFunction`:

```python
from warpSPH import buildConfig, buildScheme, CompressibleSPHScheme

config, integrator = buildConfig(dim=1, dt=1e-3, adaptiveDt=True, cflFactor=0.3)

bundle = buildScheme(CompressibleSPHScheme.CRKSPH)
schemeConfig = bundle.SimulationConfig(gamma=5 / 3)

result = integrator.function(state=state, f=bundle.stepFunction, dt=config.dt,
                             config=config, schemeConfig=schemeConfig)
```

Access the bundle by name. It still unpacks positionally for old call sites, but
that ordering is pinned for compatibility, not something to rely on — and
binding `SimulationConfig` from a positional unpack shadows the *global*
`SimulationConfig` that `from warpSPH import *` provides, which is a trap the
notebooks used to fall into.

---

## Configuration reference

`buildConfig(...)` returns a `SimulationConfig` and the matching integrator.
Every option below is also a `CaseSpec` field, so it is reachable as a CLI flag
of the same name.

**Discretisation** — `dim`, `nx`, `L`, `n_h` (neighbours per smoothing length,
converted to `targetNeighbors`), `periodic`.

**Operators** — `kernel` (`Poly6`, `CubicSpline`, `QuarticSpline`,
`QuinticSpline`, `B7`, `Wendland2/4/6`, `Spiky`), `supportMode` (`Gather`,
`Scatter`, `MeanSymmetric`, `KernelMeanSymmetric`, `SuperSymmetric`,
`PartialSymmetric`), `gradientMode`, `laplacianMode`, `samplingScheme`.

**Time integration** — `integrationScheme` selects from 26 schemes in
`warpSPHIntegrators` (`forwardEuler`, `rungeKutta2/3/4`, `leapFrog`,
`velocityVerlet`, `symplecticEuler`, `sspRK3`, `dormandPrince`, …), with
`dt`, `adaptiveDt`, `cflFactor`, `minDt`, `maxDt`. Leaving `dt` unset means the
case derives it — compressible samplers from the acoustic CFL, weakly
compressible ones from a target timestep and the sound speed together.

**Compressible scheme options**, on the scheme config rather than the global one:
`ViscositySwitch` (`Balsara1995`, `Colagrossi2004`, `CullenDehnen2010`,
`CullenHopkins`, `MorrisMonaghan1997`, `Rosswog2000`, `NoneSwitch`),
`AdaptiveSupportScheme` (`NoScheme`, `Monaghan`, `Owen`), and `EnergyScheme`
(`equalWork`, `PdV`, `diminishing`, `monotonic`, `hybrid`, `CRK`).

## Package layout

```
src/
  warpSPHBootstrap.py      precision/warp setup; must be imported first
  warpSPHRun.py            the warpsph-run CLI
  warpSPH/
    runner/                everything generic to a run
      case.py              the Case hooks and the registry
      caseSpec.py          the run description, and the argparse it generates
      runner.py            the step loop
      display.py           live plot windows and their teardown
      report.py            the setup banner and the completion report
      media.py             ffmpeg encoding
      cli.py               the shared __main__ block
    cases/                 one module per case, plus what each family shares
      compressible.py      CRKSPH defaults, energy diagnostics
      weaklyCompressible.py  banded domain, SDF regions, sound-speed/dt setup
      plotting.py          field lists -> plot hooks
    schemes/               scheme implementations and buildScheme
    configurations/        simulation and scheme configuration dataclasses
    systems/               state and system containers
    modules/               timestep, viscosity, shifting, boundary conditions, …
    sample/                particle samplers (regular, shell, per-family)
    geometry/              what a sampler is defined in terms of: SDFs, NACA,
                           ParticleSet/PointCloud, SamplingScheme
    regions/               SDF regions and filtering
    math/                  periodic positions, noise, scatter
    utils/                 domain description, support radii, timers
    caseUtils/             per-case setup helpers shared with the notebooks
    io/                    HDF5/JSON export, import, parsing and datasets
examples/                  notebooks, runnable scripts, sweeps, rendered media
datagen/                   dataset generation on top of the same cases
scripts/                   check_imports.py, run_tests.sh, run_sweep.py
tests/                     pytest suite
```

## Tests

```bash
scripts/run_tests.sh          # or plain `pytest`
```

42 tests, a few seconds once the warp kernel cache is warm; a CUDA device is
required. The script just wraps pytest, silencing the third-party warnings that
otherwise bury the result; it forwards any extra arguments
(`scripts/run_tests.sh -k sod -v`). They assert *properties* rather than golden numbers — total-energy
conservation for Sod, Taylor-Green decay against the analytic rate, density
bounds and gravitational work for the dam break — plus the runner's own
invariants: that every case registers and names a resolvable scheme, that the
spec round-trips through JSON and YAML, and that the banner, report and
`--quiet` behave.

To exercise every case rather than the three the physics tests cover, sweep
them:

```bash
scripts/run_sweep.py                # every case, 5 steps each, ~3 min
scripts/run_sweep.py --full         # every case to its own tLimit (long)
scripts/run_sweep.py --cases sod noh
```

Each case runs in its own process, sequentially — a diverged run exits
non-zero, and one crashing case must not take the sweep down with it. Logs, a
per-case export tree and a `summary.json` land in a timestamped
`sweeps/sweep_<timestamp>/`, and failures are printed with the tail of their
log. Anything after `--` is forwarded to every case, so the configs in
`examples/sweeps/` compose with it.

After a refactor, the matching check is:

```bash
scripts/check_imports.py            # every module imports; every import resolves
```

It imports each module under `warpSPH` for real, then AST-scans every `.py` and
notebook cell in the repo for `warpSPH*` imports and verifies both the module
and the imported symbol exist — which catches function-level and notebook
imports that nothing else executes.

## Gallery

Previews and embedded videos for the compressible set:
[examples/compressible/EXAMPLES_SUMMARY.md](examples/compressible/EXAMPLES_SUMMARY.md)

| Case | Notebook | Preview | Video |
|---|---|---|---|
| 01. Sod Shock Tube (1D) | [ipynb](examples/compressible/01-sod/sod_1d.ipynb) | ![](examples/compressible/01-sod/outputs/01-Sod_Shock_Tube.png) | [MP4](examples/compressible/01-sod/outputs/01-Sod_Shock_Tube.mp4) |
| 02. Linear Wave | [ipynb](examples/compressible/02-Linear_Wave.ipynb) | ![](examples/compressible/outputs/02-Linear_wave.png) | [MP4](examples/compressible/outputs/02-Linear_wave.mp4) |
| 03. Kidder Isentropic Compression | [ipynb](examples/compressible/03-Kidder_Isentropic_Compression.ipynb) | ![](examples/compressible/outputs/03-Kidder_Isentropic_compression.png) | [MP4](examples/compressible/outputs/03-Kidder_Isentropic_compression.mp4) |
| 04. Noh Implosion | [ipynb](examples/compressible/04-Noh_Implosion.ipynb) | ![](examples/compressible/outputs/04-Noh_Implosion.png) | [MP4](examples/compressible/outputs/04-Noh_Implosion.mp4) |
| 05. Woodward-Colella Double Blastwave | [ipynb](examples/compressible/05-Woodward_Colella.ipynb) | ![](examples/compressible/outputs/05-Wodward_Colella_Double_Blastwave.png) | [MP4](examples/compressible/outputs/05-Wodward_Colella_Double_Blastwave.mp4) |
| 06. Sedov-Taylor Blastwave (1D) | [ipynb](examples/compressible/06-Sedov_Taylor_Blastwave_1D.ipynb) | ![](examples/compressible/outputs/06-Sedov_Taylor_Blastwave_1D.png) | [MP4](examples/compressible/outputs/06-Sedov_Taylor_Blastwave_1D.mp4) |
| 07. Sedov-Taylor Blastwave (2D) | [ipynb](examples/compressible/07-Sedov_Taylor_Blastwave_2D.ipynb) | ![](examples/compressible/outputs/07-Sedov_Taylor_Blastwave_2D.png) | [MP4](examples/compressible/outputs/07-Sedov_Taylor_Blastwave_2D.mp4) |
| 08. Hydrostatic | [ipynb](examples/compressible/08-Hydrostatic.ipynb) | ![](examples/compressible/outputs/08-Hydrostatic.png) | [MP4](examples/compressible/outputs/08-Hydrostatic.mp4) |
| 09. Gresho-Chan Vortex | [ipynb](examples/compressible/09-Gresho_Chan_Vortex.ipynb) | ![](examples/compressible/outputs/09-Gresho_Chan_Vortex.png) | [MP4](examples/compressible/outputs/09-Gresho_Chan_Vortex.mp4) |
| 10. Yee Vortex | [ipynb](examples/compressible/10-Yee_Vortex.ipynb) | ![](examples/compressible/outputs/10-Yee_Vortex.png) | [MP4](examples/compressible/outputs/10-Yee_Vortex.mp4) |
| 11. Shearing Noh Implosion (2D) | [ipynb](examples/compressible/11-Shearing_Noh_Implosion_2D.ipynb) | ![](examples/compressible/outputs/11-Shearing_Noh_2D.png) | [MP4](examples/compressible/outputs/11-Shearing_Noh_2D.mp4) |
| 12. Kelvin-Helmholtz | [ipynb](examples/compressible/12-Kelvin-Helmholtz.ipynb) | ![](examples/compressible/outputs/12-Kelvin_Helmholtz.png) | [MP4](examples/compressible/outputs/12-Kelvin_Helmholtz.mp4) |
| 13. Rayleigh-Taylor | [ipynb](examples/compressible/13-Rayleigh_Taylor.ipynb) | ![](examples/compressible/outputs/13-Rayleigh_Taylor.png) | [MP4](examples/compressible/outputs/13-Rayleigh_Taylor.mp4) |
| 14. Triple Point (Equal Resolution) | [ipynb](examples/compressible/14-Triple_point.ipynb) | ![](examples/compressible/outputs/14-Triple_Point_equal_resolution.png) | [MP4](examples/compressible/outputs/14-Triple_Point_equal_resolution.mp4) |
| 15. Triple Point (Equal Mass) | [ipynb](examples/compressible/15-Triple_point_equalMass.ipynb) | ![](examples/compressible/outputs/15-Triple_Point_equal_mass.png) | [MP4](examples/compressible/outputs/15-Triple_Point_equal_mass.mp4) |

## Precision, and other things that bite

**Precision is resolved when `warpSPHCore` is first imported** and cannot be
changed afterwards. Any `warpSPH.*` import pulls `warpSPHCore` in transitively,
so the choice has to be made first — which is why `bootstrap` is a top-level
module rather than part of the package:

```python
from warpSPHBootstrap import bootstrap
runtime = bootstrap(precision='float64')   # also runs wp.init(), pins TORCH_CUDA_ARCH_LIST
```

In a notebook, changing precision means restarting the kernel.
`warpsph-run --precision float64` handles the ordering for you;
`python -m warpSPH.cases.X` cannot, and warns instead of silently running at the
wrong precision.

**No GL context** (ssh without X forwarding, containers): 2D plots fall back
from vispy to matplotlib automatically. If matplotlib also has no interactive
backend, the run says so once and keeps writing frames.

**No `ffmpeg`**: `--video` is skipped with a message rather than failing a run
that already produced its frames.

**Old `.h5` files** written before the package renames hold dill-encoded
callables under the previous module names and will not import. They are
regenerable; there is no migration path and none is planned.
