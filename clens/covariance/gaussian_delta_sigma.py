"""Gaussian DeltaSigma covariance for one (z, lambda) cluster bin — FFTLog.

Replaces the legacy trapz-over-ln(ell) of ``cov_DeltaSigma.py`` with the
clenspy FFTLog engine (``clenspy.utils.fftlog_cov``; math in
``CLensPy/docs/covariance_fftlog_math.md``).  Term structure preserved
(Wu et al. 2019 eq. 22):

- ``cosmic_shear``: (C_ell^hh + shot) * C_ell^{Sigma Sigma}   [smooth]
- ``shape_noise`` : (C_ell^hh + shot) * N_shape
  = C_ell^hh * N_shape [smooth] + shot * N_shape [white -> exact
  analytic diagonal via bin-averaged-J2 orthogonality]
- ``cross``       : (C_ell^{h Sigma})^2                        [smooth]

All terms carry the 1/(4 pi f_sky) prefactor, the ell/(2 pi) measure
(combined: 1/(8 pi^2 f_sky) inside the engine) and the 1e-24 conversion
to (Msun/pc^2)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from clenspy.utils.fftlog_cov import GaussianCovFFTLog, white_noise_diagonal

from .inputs import CosmologyInputs, LensSample, SourceInputs, SurveyGeometry
from .limber import LimberProjector

__all__ = ["DeltaSigmaCovBlocks", "GaussianDeltaSigmaCov"]

MPC2_TO_PC2 = 1e-24


@dataclass
class DeltaSigmaCovBlocks:
    """Per-term covariance blocks [(Msun/pc^2)^2, comoving radii]."""

    rp_mid: np.ndarray  # comoving Mpc (no h)
    rp_min: np.ndarray
    rp_max: np.ndarray
    cosmic_shear: np.ndarray
    shape_noise: np.ndarray
    cross: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.cosmic_shear + self.shape_noise + self.cross


class GaussianDeltaSigmaCov:
    """FFTLog Gaussian DeltaSigma covariance for one cluster bin.

    Parameters
    ----------
    cosmo, source, geometry : contract dataclasses
    q : float
        FFTLog tilt (default 1.0; see the math doc).
    """

    def __init__(
        self,
        cosmo: CosmologyInputs,
        source: SourceInputs,
        geometry: SurveyGeometry,
        q: float = 1.0,
    ) -> None:
        self.cosmo = cosmo
        self.source = source
        self.geometry = geometry
        self.q = q
        self.limber = LimberProjector(cosmo, source)

    def compute(
        self,
        sample: LensSample,
        rp_min: float,
        rp_max: float,
        n_rp: int,
    ) -> DeltaSigmaCovBlocks:
        """Covariance blocks over ``n_rp`` log bins of comoving rp [Mpc]."""
        z_mid = sample.z_mid
        chi_h = float(self.cosmo.chi(z_mid))
        theta_edges = (
            np.exp(np.linspace(np.log(rp_min), np.log(rp_max), n_rp + 1))
            / chi_h
        )

        ell = self.limber.ell
        # radial-bin-independent spectra (once per cluster bin)
        c_sigma = self.limber.c_ell_sigma(
            0.1, min(2.0, self.source.zs_max - 0.1), z_mid
        )
        c_hh, shot = self.limber.c_ell_h(
            sample.z_min, sample.z_max, sample.bias,
            sample.counts, self.geometry.area_sr, pk_hh=sample.pk_hh,
        )
        c_cross = self.limber.c_ell_h_sigma(
            sample.z_min, sample.z_max, sample.bias, z_mid,
            pk_hm=sample.pk_hm,
        )
        n_shape = self.limber.shape_noise_sigma(z_mid)

        engine = GaussianCovFFTLog(
            ell, theta_edges, f_sky=self.geometry.f_sky, q=self.q
        )
        cov_cosmic = engine.covariance((c_hh + shot) * c_sigma)
        # smooth part of the shape-noise term ...
        cov_shape = engine.covariance(c_hh * n_shape)
        # ... plus the exactly-diagonal white part (shot x shape noise)
        cov_shape[np.diag_indices_from(cov_shape)] += white_noise_diagonal(
            theta_edges, shot * n_shape, self.geometry.f_sky
        )
        cov_cross = engine.covariance(c_cross**2)

        rp_edges = theta_edges * chi_h
        return DeltaSigmaCovBlocks(
            rp_mid=np.sqrt(rp_edges[:-1] * rp_edges[1:]),
            rp_min=rp_edges[:-1],
            rp_max=rp_edges[1:],
            cosmic_shear=cov_cosmic * MPC2_TO_PC2,
            shape_noise=cov_shape * MPC2_TO_PC2,
            cross=cov_cross * MPC2_TO_PC2,
        )
