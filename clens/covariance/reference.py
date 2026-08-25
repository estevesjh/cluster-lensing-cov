"""Legacy trapz-over-ln(ell) covariance — TEST-ONLY reference.

Straight port of the retired ``cov_DeltaSigma._calc_C_ell_integration``
loop (closed-form annulus-averaged J2 kernels + dense trapz), consuming
the same Limber spectra as the FFTLog path so equivalence tests isolate
the ell-integration method.  Never import this from production code.
"""

from __future__ import annotations

import numpy as np
from clenspy.utils.fftlog_cov import j2_bin_averaged

__all__ = ["reference_cov_trapz"]

# legacy bessel_for_cov_theta.py settings
SCALING_FOR_ELL_MIN = 1.0
SCALING_FOR_ELL_MAX = 100.0
DLNELL = 1e-3


def reference_cov_trapz(
    ell: np.ndarray,
    C_of_ell: np.ndarray,
    theta_edges: np.ndarray,
    f_sky: float,
    dlnell: float = DLNELL,
) -> np.ndarray:
    """Legacy integration of one covariance term.

    Cov_ij = 1/(4 pi f_sky) int dln(ell) ell^2/(2 pi)
             J2bar_i J2bar_j C(ell),
    with the legacy per-pair ell range
    [1/theta_max, 100/theta_min].
    """
    theta_edges = np.asarray(theta_edges, dtype=float)
    n = theta_edges.size - 1
    cov = np.zeros((n, n))
    lnC = np.log(np.clip(np.asarray(C_of_ell, dtype=float), 1e-300, None))
    for i in range(n):
        for j in range(i, n):
            thmin = min(theta_edges[i], theta_edges[j])
            thmax = max(theta_edges[i + 1], theta_edges[j + 1])
            lnell = np.arange(
                np.log(SCALING_FOR_ELL_MIN / thmax),
                np.log(SCALING_FOR_ELL_MAX / thmin),
                dlnell,
            )
            ell_loc = np.exp(lnell)
            C_loc = np.exp(
                np.interp(np.log(ell_loc), np.log(ell), lnC)
            )
            geom = (
                j2_bin_averaged(ell_loc, theta_edges[i], theta_edges[i + 1])
                * j2_bin_averaged(ell_loc, theta_edges[j], theta_edges[j + 1])
                * ell_loc**2
                / (2.0 * np.pi)
            )
            val = np.trapezoid(C_loc * geom, lnell) / (4.0 * np.pi * f_sky)
            cov[i, j] = cov[j, i] = val
    return cov
