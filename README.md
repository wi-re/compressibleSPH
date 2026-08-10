# warpSPH

*(formerly `compressibleSPH`; the SPH core previously hosted at this URL now lives at
[`wi-re/warpSPHCore`](https://github.com/wi-re/warpSPHCore))*

![](examples/compressible/outputs/15-Triple_Point_equal_mass.gif)

A GPU-oriented Python package for compressible Smoothed Particle Hydrodynamics (SPH) experiments and benchmarks.

The project is organized around reusable builders for simulation configuration, interchangeable SPH schemes, adaptive time integration, case setup helpers, and export/import utilities for trajectories and postprocessing.

## What This Package Implements

### Core SPH Schemes

The package currently exposes three compressible SPH schemes via `CompressibleSPHScheme`:

- `Monaghan`
- `CompSPH`
- `CRKSPH`

Scheme dispatch is handled by `buildScheme(...)` in `src/warpSPH/schemes/builder.py`.

### Kernels

Kernel selection is part of `SimulationConfig.kernel` (type `KernelFunctions`, imported from `warpSPHCore`).

The parser helper `parseKernelFunctions(...)` accepts any kernel enum name exposed by your installed `warpSPHCore` version.

### Integrators

Time integration is selected with `SimulationConfig.integrationScheme` (type `IntegrationSchemeType`, from `warpSPHIntegrators`).

The helper `buildConfig(...)` returns both:

- a configured `SimulationConfig`
- the integrator object from `getIntegrator(...)`

### Spatial Operators and Support Modes

Simulation configuration includes selectable operator and support modes:

- `supportMode` (`SupportScheme`)
- `gradientMode` (`GradientScheme`)
- `laplacianMode` (`LaplacianScheme`)
- `samplingScheme` (`SamplingScheme`)

### Viscosity Switches

`ViscositySwitch` includes:

- `Balsara1995`
- `Colagrossi2004`
- `CullenDehnen2010`
- `CullenHopkins`
- `MorrisMonaghan1997`
- `Rosswog2000`
- `NoneSwitch`

### Adaptive Support Schemes

`AdaptiveSupportScheme` includes:

- `NoScheme`
- `Monaghan`
- `Owen`

### Energy Schemes

`EnergyScheme` includes:

- `equalWork`
- `PdV`
- `diminishing`
- `monotonic`
- `hybrid`
- `CRK`

### I/O and Reproducibility Utilities

The package provides trajectory/state serialization utilities:

- `prepExport(...)`
- `exportSimulationSystem(...)`
- `importSimulationSystem(...)`
- `importConfigs(...)`

These write/read HDF5 state data and JSON config metadata for reproducible runs.

## Package Layout

- `src/warpSPH/configurations/`: simulation and scheme configuration dataclasses/builders
- `src/warpSPH/schemes/`: scheme implementations and scheme builder
- `src/warpSPH/systems/`: state/system containers
- `src/warpSPH/modules/`: runtime modules (for example timestep support)
- `src/warpSPH/sample/`: sampling utilities
- `src/warpSPH/caseUtils/`: case setup helpers used by examples
- `src/warpSPH/runner/`: `CaseSpec`, the `Case` hooks, and the shared step loop
- `src/warpSPH/cases/`: one runnable case per example, built on the runner
- `src/warpSPH/io.py`: export/import and parsing helpers
- `src/warpSPHBootstrap.py`: pre-import precision/warp setup (must be imported first)
- `src/warpSPHRun.py`: the `warpsph-run` CLI
- `examples/compressible/`: benchmark notebooks and generated media
- `examples/sweeps/`: example `CaseSpec` config files
- `tests/`: pytest suite (short runs asserting physical invariants)

## Environment Setup

Recommended setup uses a dedicated conda environment and editable installs for backend repositories.

```
mkdir warp-sph-stack
cd warp-sph-stack
git clone https://github.com/wi-re/warpSPHIntegrators
git clone https://github.com/wi-re/warpSPHPlotting
git clone https://github.com/wi-re/warpSPHCore
git clone https://github.com/wi-re/warpSPH
conda create -n warp_env python=3.14
conda activate warp_env
pip install tqdm seaborn pandas pyglet dill numba scipy scikit-image h5py ipykernel ipympl
pip install -e warpSPHIntegrators/ warpSPHPlotting/ warpSPHCore/ 
cd warpSPH
pip install -e .
```

Current status: cloning and editable installs are the accepted setup path for now.

## Quick Start

If you are running directly from this repository, ensure `src` is on your Python path:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Minimal usage pattern:

```python
import torch
from warpSPH import *

# Build global simulation config + integrator
config, integrator = buildConfig(
    dim=1,
    dt=1e-3,
    adaptiveDt=True,
    cflFactor=0.3,
)

# Pick an SPH scheme
scheme = CompressibleSPHScheme.CRKSPH
SimulationSystem, SimulationState, SimulationConfig, SimulationUpdate, fn, export_fn, import_fn = buildScheme(scheme)

# Build scheme-specific config
schemeConfig = SimulationConfig()
schemeConfig.gamma = 5.0 / 3.0

# Case setup is typically done with helpers in warpSPH.caseUtils
```

## Running Cases

Cases are run through a shared runner rather than per-script boilerplate. Each
case declares only its geometry, initial conditions and diagnostics; the config
construction, step loop, export, plotting and video encoding are shared.

```bash
warpsph-run sod --nx 800 --plot --store
warpsph-run tgv --nx 128 --tLimit 2.0
warpsph-run dambreak --config examples/sweeps/dambreak_obstacle.yaml
```

`warpsph-run` with no case name lists every case with its description.

`warpsph-run --precision float64 ...` works because the CLI selects the
precision before `warpSPH` is imported. `python -m warpSPH.cases.sod ...` runs
the same case but is stuck with whatever precision is already active (float32 by
default, or `warpSPHCore_PRECISION` from the environment).

### The cases

Every notebook under `examples/` that runs a simulation has a case, and a
`.py` script next to the notebook that runs it with that example's settings.
The notebooks stay for exploration; the scripts are what you run unattended.

| case | example | notes |
|---|---|---|
| `sod` | compressible/01 | Sod shock tube, 1D |
| `linearWave` | compressible/02 | linear acoustic wave, 1D |
| `kidder` | compressible/03 | isentropic compression; analytically driven boundary bands |
| `noh` | compressible/04 | Noh implosion, 1D |
| `woodwardColella` | compressible/05 | interacting blast waves, 1D |
| `sedov` | compressible/06, 07 | Sedov-Taylor blast; `--dim 1` or `--dim 2` |
| `hydrostatic` | compressible/08 | hydrostatic equilibrium |
| `gresho` | compressible/09 | Gresho-Chan vortex |
| `yee` | compressible/10 | Yee isentropic vortex |
| `shearingNoh` | compressible/11 | Noh implosion with transverse shear |
| `kelvinHelmholtz` | compressible/12 | Kelvin-Helmholtz instability |
| `rayleighTaylor` | compressible/13 | Rayleigh-Taylor instability |
| `triplePoint` | compressible/14, 15 | `--equalMass` or `--no-equalMass` sampling |
| `impact` | weaklyCompressible/01, 02 | two bodies colliding; `--shape circle\|box` |
| `squarePatch` | weaklyCompressible/03, incompressible/03 | rotating square patch |
| `droplet` | weaklyCompressible/04 | oscillating droplet in a central potential |
| `tgv-wc` | weaklyCompressible/05 | Taylor-Green vortex, deltaSPH |
| `randomFlow` | weaklyCompressible/06, 07, incompressible/periodic | `--bounded` adds walls |
| `kolmogorov` | weaklyCompressible/08 | Kolmogorov flow |
| `ldc` | weaklyCompressible/09 | lid-driven cavity |
| `movingObstacle` | weaklyCompressible/10 | flow past a spinning rigid body |
| `drivenSquare` | weaklyCompressible/11 | driven channel flow past a cylinder |
| `dambreak` | weaklyCompressible/12, datagen | dam break with optional obstacle |
| `openFlow` | weaklyCompressible/13 | open channel flow past an obstacle |
| `tgv` | incompressible/01 | Taylor-Green vortex, divergence-free |

Several cases cover more than one notebook, because those notebooks differed
only in a flag. Where that is the case the example scripts pin the flag, so
`examples/compressible/06-sedov-taylor-blastwave-1d.py` and `07-...-2d.py` are
the same case at `--dim 1` and `--dim 2`.

Every flag corresponds to a field of `CaseSpec`, so a run is fully described by
a JSON or YAML file. CLI flags override the file; the file overrides the case's
own defaults. `--saveConfig out.yaml` writes the fully resolved spec back out.
See [examples/sweeps/](examples/sweeps/).

From Python or a notebook:

```python
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')          # before importing warpSPH

from warpSPH.runner import run
from warpSPH.cases.tgv import tgvCase

result = run(tgvCase, nx=64, nSteps=200)
result.series('kineticEnergy')          # one diagnostic across the run
```

## Tests

```bash
pytest
```

The suite runs each converted case for 20 steps at a coarse resolution and
asserts a physical invariant: total energy conservation for Sod, Taylor-Green
kinetic-energy decay against the analytic rate, and density bounds plus
gravitational work for the dam break. It needs a CUDA device and takes a few
seconds once the warp kernel cache is warm.

## Compressible Examples

A full gallery page with previews and embedded videos is available at:

- [examples/compressible/EXAMPLES_SUMMARY.md](examples/compressible/EXAMPLES_SUMMARY.md)

### Table of Cases

| Case | Notebook | Preview (PNG) | Video (MP4) |
|---|---|---|---|
| 01. Sod Shock Tube (1D) | [01-Sod_Shock_Tube_1D.ipynb](examples/compressible/01-Sod_Shock_Tube_1D.ipynb) | ![](examples/compressible/outputs/01-Sod_Shock_Tube.png) | [MP4](examples/compressible/outputs/01-Sod_Shock_Tube.mp4) |
| 02. Linear Wave | [02-Linear_Wave.ipynb](examples/compressible/02-Linear_Wave.ipynb) | ![](examples/compressible/outputs/02-Linear_wave.png) | [MP4](examples/compressible/outputs/02-Linear_wave.mp4) |
| 03. Kidder Isentropic Compression | [03-Kidder_Isentropic_Compression.ipynb](examples/compressible/03-Kidder_Isentropic_Compression.ipynb) | ![](examples/compressible/outputs/03-Kidder_Isentropic_compression.png) | [MP4](examples/compressible/outputs/03-Kidder_Isentropic_compression.mp4) |
| 04. Noh Implosion | [04-Noh_Implosion.ipynb](examples/compressible/04-Noh_Implosion.ipynb) | ![](examples/compressible/outputs/04-Noh_Implosion.png) | [MP4](examples/compressible/outputs/04-Noh_Implosion.mp4) |
| 05. Woodward-Colella Double Blastwave | [05-Woodward_Colella.ipynb](examples/compressible/05-Woodward_Colella.ipynb) | ![](examples/compressible/outputs/05-Wodward_Colella_Double_Blastwave.png) | [MP4](examples/compressible/outputs/05-Wodward_Colella_Double_Blastwave.mp4) |
| 06. Sedov-Taylor Blastwave (1D) | [06-Sedov_Taylor_Blastwave_1D.ipynb](examples/compressible/06-Sedov_Taylor_Blastwave_1D.ipynb) | ![](examples/compressible/outputs/06-Sedov_Taylor_Blastwave_1D.png) | [MP4](examples/compressible/outputs/06-Sedov_Taylor_Blastwave_1D.mp4) |
| 07. Sedov-Taylor Blastwave (2D) | [07-Sedov_Taylor_Blastwave_2D.ipynb](examples/compressible/07-Sedov_Taylor_Blastwave_2D.ipynb) | ![](examples/compressible/outputs/07-Sedov_Taylor_Blastwave_2D.png) | [MP4](examples/compressible/outputs/07-Sedov_Taylor_Blastwave_2D.mp4) |
| 08. Hydrostatic | [08-Hydrostatic.ipynb](examples/compressible/08-Hydrostatic.ipynb) | ![](examples/compressible/outputs/08-Hydrostatic.png) | [MP4](examples/compressible/outputs/08-Hydrostatic.mp4) |
| 09. Gresho-Chan Vortex | [09-Gresho_Chan_Vortex.ipynb](examples/compressible/09-Gresho_Chan_Vortex.ipynb) | ![](examples/compressible/outputs/09-Gresho_Chan_Vortex.png) | [MP4](examples/compressible/outputs/09-Gresho_Chan_Vortex.mp4) |
| 10. Yee Vortex | [10-Yee_Vortex.ipynb](examples/compressible/10-Yee_Vortex.ipynb) | ![](examples/compressible/outputs/10-Yee_Vortex.png) | [MP4](examples/compressible/outputs/10-Yee_Vortex.mp4) |
| 11. Shearing Noh Implosion (2D) | [11-Shearing_Noh_Implosion_2D.ipynb](examples/compressible/11-Shearing_Noh_Implosion_2D.ipynb) | ![](examples/compressible/outputs/11-Shearing_Noh_2D.png) | [MP4](examples/compressible/outputs/11-Shearing_Noh_2D.mp4) |
| 12. Kelvin-Helmholtz | [12-Kelvin-Helmholtz.ipynb](examples/compressible/12-Kelvin-Helmholtz.ipynb) | ![](examples/compressible/outputs/12-Kelvin_Helmholtz.png) | [MP4](examples/compressible/outputs/12-Kelvin_Helmholtz.mp4) |
| 13. Rayleigh-Taylor | [13-Rayleigh_Taylor.ipynb](examples/compressible/13-Rayleigh_Taylor.ipynb) | ![](examples/compressible/outputs/13-Rayleigh_Taylor.png) | [MP4](examples/compressible/outputs/13-Rayleigh_Taylor.mp4) |
| 14. Triple Point (Equal Resolution) | [14-Triple_point.ipynb](examples/compressible/14-Triple_point.ipynb) | ![](examples/compressible/outputs/14-Triple_Point_equal_resolution.png) | [MP4](examples/compressible/outputs/14-Triple_Point_equal_resolution.mp4) |
| 15. Triple Point (Equal Mass) | [15-Triple_point_equalMass.ipynb](examples/compressible/15-Triple_point_equalMass.ipynb) | ![](examples/compressible/outputs/15-Triple_Point_equal_mass.png) | <video src="https://github.com/wi-re/warpSPH/blob/main/examples/compressible/outputs/15-Triple_Point_equal_mass.mp4" width="80%" controls></video>|

## Precision Note (Notebook Workflows)

Floating-point precision is resolved when `warpSPHCore` is first imported and
cannot be changed afterwards, so it has to be chosen before any other import:

```python
from warpSPHBootstrap import bootstrap
runtime = bootstrap(precision='float64')   # also runs wp.init() and pins TORCH_CUDA_ARCH_LIST
```

If precision is changed (for example float32 to float64), restart the Jupyter
kernel before rerunning.
