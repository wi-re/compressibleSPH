#!/usr/bin/env python
"""Dam-break data generator -- the runner, plus this directory's archival step.

The simulation itself is `warpSPH.cases.dambreak`; what stays here is the bit
that is specific to generating a dataset: stamping the run directory with a
timestamp and the obstacle description, and collecting the trajectory, video and
end frames into `compressed/`.

    python datagen/weaklyCompressible/generator.py --nx 128 --plot --store
    python datagen/weaklyCompressible/generator.py --config sweeps/obstacle.yaml
"""

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

import os                                                   # noqa: E402
import shutil                                               # noqa: E402

from warpSPH.cases.dambreak import dambreakCase             # noqa: E402
from warpSPH.runner import buildArgumentParser, run, specFromArgs  # noqa: E402
from warpSPH.utils import getCurrentTimestamp               # noqa: E402


def archive(result, caseName: str, destination: str = 'compressed') -> None:
    """Move the trajectory and copy the renders into a flat dataset directory."""
    exportPath = result.exportPath
    if exportPath is None:
        return
    os.makedirs(destination, exist_ok=True)
    tag = f'{caseName}_{os.path.basename(exportPath)}'

    trajectory = os.path.join(exportPath, 'trajectory.h5')
    if os.path.exists(trajectory):
        shutil.move(trajectory, os.path.join(destination, f'trajectory_{tag}.hdf5'))

    if result.videoPath and os.path.exists(result.videoPath):
        shutil.copy(result.videoPath, os.path.join(destination, f'video_{tag}.mp4'))

    imagePath = result.ctx.imagePath
    if imagePath and os.path.isdir(imagePath):
        frames = sorted((f for f in os.listdir(imagePath)
                         if f.startswith('frame_') and f.endswith('.png')),
                        key=lambda name: int(name.split('_')[1].split('.')[0]))
        if frames:
            shutil.copy(os.path.join(imagePath, frames[0]),
                        os.path.join(destination, f'first_frame_{tag}.png'))
            shutil.copy(os.path.join(imagePath, frames[-1]),
                        os.path.join(destination, f'last_frame_{tag}.png'))


def main():
    parser = buildArgumentParser(description=dambreakCase.description,
                                 caseParams=dambreakCase.params)
    args = parser.parse_args()

    defaults = dict(dambreakCase.defaults)
    defaults.update(caseName='3-dambreak', store=True, video=True)
    spec = specFromArgs(args, caseParams=dambreakCase.params, defaults=defaults)

    # The dataset layout wants one directory per run, self-describing.
    obstacleText = (f"obstacle_{spec.param('maxExtent'):.4g}_{spec.param('aoa'):.4g}"
                    f"_{spec.param('offsetX'):.4g}" if spec.param('obstacleActive')
                    else 'no_obstacle')
    baseName = spec.caseName
    spec = spec.merged(caseName=(
        f'{baseName}/{getCurrentTimestamp()}_{spec.nx}_{spec.n_h}_'
        f"{spec.L}_{spec.param('W')}_{obstacleText}"))

    result = run(dambreakCase, spec)
    archive(result, baseName)
    return result


if __name__ == '__main__':
    main()
