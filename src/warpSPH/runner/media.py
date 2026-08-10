"""The ffmpeg export block, extracted from the tail of every example script."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Optional

__all__ = ['encodeFrames']

_MP4 = ("ffmpeg -y -loglevel error -hide_banner -framerate {framerate} -f image2 "
        "-pattern_type glob -i '{pattern}' -c:v libx264 -pix_fmt yuv420p -b:v {bitrate} {out}")
_PALETTE = ('ffmpeg -y -loglevel error -hide_banner -i {mp4} '
            '-vf "fps={fps},scale={width}:-1:flags=lanczos,palettegen" palette.png')
_GIF = ('ffmpeg -y -loglevel error -hide_banner -i {mp4} -i palette.png -filter_complex '
        '"fps={gifFps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse" {gif}')


def encodeFrames(imagePath: str,
                 destination: Optional[str] = None,
                 *,
                 framerate: int = 50,
                 fps: int = 50,
                 gifFps: int = 25,
                 width: int = 540,
                 bitrate: str = '10M',
                 pattern: str = 'frame_*.png',
                 gif: bool = True) -> Optional[str]:
    """Encode ``imagePath/frame_*.png`` to ``output.mp4`` (and ``out.gif``).

    Returns the path of the mp4, or ``None`` if there was nothing to encode or
    ffmpeg is not installed -- a missing encoder should not fail a simulation
    that has already produced its frames.
    """
    if shutil.which('ffmpeg') is None:
        print('ffmpeg not found on PATH; skipping video export.')
        return None
    if not any(f.startswith('frame_') and f.endswith('.png') for f in os.listdir(imagePath)):
        return None

    mp4 = os.path.join(imagePath, 'output.mp4')
    subprocess.run(shlex.split(_MP4.format(framerate=framerate, pattern=pattern,
                                           bitrate=bitrate, out='output.mp4')),
                   check=True, cwd=imagePath)
    if gif:
        subprocess.run(shlex.split(_PALETTE.format(mp4='output.mp4', fps=fps, width=width)),
                       check=True, cwd=imagePath)
        subprocess.run(shlex.split(_GIF.format(mp4='output.mp4', gifFps=gifFps, width=width,
                                               gif='out.gif')),
                       check=True, cwd=imagePath)

    if destination and os.path.abspath(destination) != os.path.abspath(imagePath):
        os.makedirs(destination, exist_ok=True)
        shutil.copy(mp4, os.path.join(destination, 'output.mp4'))
        if gif:
            shutil.copy(os.path.join(imagePath, 'out.gif'), os.path.join(destination, 'out.gif'))
        return os.path.join(destination, 'output.mp4')

    return mp4
