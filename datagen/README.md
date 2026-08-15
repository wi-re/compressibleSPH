# datagen

Dataset generation on top of the same cases the examples run. Where
`examples/` produces *one* run to look at, this produces *many* runs to train
on: batches of parameter-varied simulations, each archived as a single
trajectory file plus preview media.

Only the weakly compressible family has a generator so far
(`weaklyCompressible/`, built on `warpSPH.cases.dambreak`).

## The pipeline

```
cases/*.sh          batches of generator.py invocations, one file per family
   |
generator.py        runs the case, then archives the result
   |
compressed/         flat dataset directory, one set of files per run
   |
compressor.py       optional: downsample a trajectory to a coarser interval
```

### `generator.py`

The simulation itself is `warpSPH.cases.dambreak` — what lives here is only the
dataset-specific part: stamping the run directory with a timestamp and the
obstacle description, then collecting the results into `compressed/`.

It takes the **same flags as any other case** (it builds its parser with
`buildArgumentParser`, so `warpsph-run dambreak --help` documents them), plus
this directory's own geometry knobs (`--obstacleType`, `--fillRatio`,
`--fluidWidth`, `--maxExtent`, `--aoa`, `--offsetX`, `--W`, ...).

```bash
python generator.py --nx 128 --plot --store
python generator.py --config sweeps/obstacle.yaml
```

Each run archives into `compressed/` as a flat set, tagged
`<caseName>_<exportDirName>`:

| file | what |
|---|---|
| `trajectory_<tag>.hdf5` | the trajectory, **moved** out of the export tree |
| `video_<tag>.mp4` | the render, if `--video` produced one |
| `first_frame_<tag>.png`, `last_frame_<tag>.png` | first and last frames |

Note that the trajectory is *moved*, not copied — after archiving, the run's
own export directory no longer holds it.

### `cases/*.sh`

Flat lists of `generator.py` command lines, one file per case family — the
parameter sweep written out longhand, so a batch is reproducible by rerunning
the file. They are not scripts with logic in them; read them as data.

| file | runs |
|---|---|
| `examples.sh` | one representative run of each family — start here |
| `dambreak.sh` | 81 |
| `kolmogorov.sh` | 192 |
| `periodic_wObstacle.sh` | 96 |
| `periodic.sh` | 48 |
| `openChannel.sh` | 44 |
| `semiPeriodic.sh` | 26 |
| `fullyPeriodic.sh` | 17 |

Run them from **inside `datagen/weaklyCompressible/`** — the commands invoke
`python generator.py` by relative path. Each line is a full simulation, so a
whole file is many GPU-hours; take the lines you want rather than running the
file end to end.

### `compressor.py`

Re-writes an existing trajectory at a coarser export interval, for when a run
was stored more finely than the dataset needs.

```bash
python compressor.py --directory compressed --exportInterval 0.01
```

## Notebooks

Exploratory, and older than the `examples/` ones — these are still on the
pre-`warpSPHBootstrap` style, so they set precision by hand rather than through
`bootstrap()`.

| notebook | what |
|---|---|
| `generator.ipynb` | `generator.py`'s pipeline, step by step |
| `dataset.ipynb` | assembling and inspecting a dataset |
| `compressedLoader.ipynb` | reading `compressed/` back |
| `compressedResume.ipynb` | restarting a run from an archived trajectory |
| `obstacle_init.ipynb` | the obstacle SDF presets, visualised |
| `quartzTest.ipynb` | scratch |

## `utils.py`, `export_util.py`

Thin compatibility shims. Both just re-export from `warpSPH` — the real
implementations moved to `warpSPH.caseUtils` and `warpSPH.io`. They exist so
the notebooks' `from utils import *` keeps working; new code should import
from `warpSPH` directly.

## Output directories

`export/`, `compressed/` and the per-case preview PNGs under `cases/*/` are all
gitignored except for the PNGs, which are small and are kept as a visual index
of what each obstacle preset looks like.
