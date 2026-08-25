"""Covariance of the binned cluster counts N_ij.

The matrix over flattened bins alpha = (z_i, lambda_a) (z-major,
richness-fast — matching the DeltaSigma block ordering) is

.. math::

    \\mathrm{Cov}[N_\\alpha, N_\\beta]
      = \\delta_{\\alpha\\beta}\\, N_\\alpha
      + \\langle bN\\rangle_\\alpha \\langle bN\\rangle_\\beta\\,
        \\sigma_W^2(z)\\, \\delta_{z_\\alpha z_\\beta}

- Poisson: strictly diagonal.
- Sample variance: same-z-shell blocks are fully correlated across
  richness (same volume), with the window variance
  :math:`\\sigma_W = \\sigma_R(R_{\\rm eff}) D(z_{\\rm mid})`,
  :math:`R_{\\rm eff} = (3V/4\\pi)^{1/3}` — exactly the orphaned ``sv``
  estimate of the legacy ``ClusterCounts.calc_counts``, now assembled
  into a matrix.  Cross-z blocks default to zero (disjoint shells).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .inputs import LensSample

__all__ = ["NCountsCov"]


class NCountsCov:
    """Poisson + sample-variance covariance of the binned counts."""

    def __init__(self, samples: Sequence[LensSample]) -> None:
        self.samples = tuple(samples)

    def matrix(self) -> np.ndarray:
        n = len(self.samples)
        cov = np.zeros((n, n))
        for a, sa in enumerate(self.samples):
            cov[a, a] += sa.counts  # Poisson
            for b, sb in enumerate(self.samples):
                same_shell = (
                    abs(sa.z_min - sb.z_min) < 1e-12
                    and abs(sa.z_max - sb.z_max) < 1e-12
                )
                if same_shell:
                    cov[a, b] += sa.bN * sb.bN * sa.sigma_w**2
        return cov

    def labels(self) -> list[str]:
        return [s.label for s in self.samples]
