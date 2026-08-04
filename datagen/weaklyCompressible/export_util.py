"""Compatibility layer for compressed trajectory export helpers.

The source-of-truth implementation now lives in warpSPH.io.
"""

from warpSPH.io import createOutFile, writeInitialData, writeFrame

__all__ = ["createOutFile", "writeInitialData", "writeFrame"]
