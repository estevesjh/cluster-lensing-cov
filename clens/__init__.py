"""Compatibility helpers for the legacy covariance implementation."""

import numpy as _np


# NumPy 2.5 removed the long-deprecated ``np.trapz`` spelling.  The
# calculation is unchanged; keep the upstream source compatible with both
# the modern astronomy stack and older NumPy installations.
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid
