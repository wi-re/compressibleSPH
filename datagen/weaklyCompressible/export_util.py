"""Compatibility layer for compressed trajectory export helpers.

The source-of-truth implementation now lives in warpSPH.io.
"""

from warpSPH.io.hdf5 import createOutFile
from warpSPH.io.export import writeInitialData, writeFrame

__all__ = ["createOutFile", "writeInitialData", "writeFrame"]
