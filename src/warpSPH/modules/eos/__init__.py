"""Equations of state: ideal-gas (compressible) and weakly-compressible
(Tait/Murnaghan-family) pressure closures.
"""

from .idealGas import idealGasEOS
from .weaklyCompressible import weaklyCompressibleEOS
from ...enumTypes import EquationOfState

__all__ = ['idealGasEOS', 'weaklyCompressibleEOS', 'EquationOfState']