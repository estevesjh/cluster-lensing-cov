"""Limber angular power spectra for the Gaussian DeltaSigma covariance.

Rewrite of the legacy ``angular_power_spectra.py`` on the input contract
(:mod:`clens.covariance.inputs`) — same discretization (direct summation
over Delta-chi slices, ell + 1/2 Limber wavenumber), consuming only
``CosmologyInputs`` / ``SourceInputs`` primitives:

- ``c_ell_sigma``   — Sigma_crit-weighted matter spectrum
  C_ell^{Sigma Sigma} = rho_mean^2 sum dchi q_Sigma^2 / chi^2 P(k, z)
- ``c_ell_h``       — halo slab spectrum b^2 P over volume^2, plus the
  shot noise 1 / n_h[sr]
- ``c_ell_h_sigma`` — linear-bias cross b P x Sigma kernel x rho_mean / vol
- ``shape_noise_sigma`` — sigma_gamma^2 <Sigma_crit>^2 / n_src^eff
"""

from __future__ import annotations

import numpy as np

from .inputs import ARCMIN_TO_RAD, CosmologyInputs, SourceInputs

__all__ = ["LimberProjector"]


class LimberProjector:
    """Angular power spectra on a shared log-ell grid (legacy grid:
    1000 points/decade over [1e-1, 2e7])."""

    def __init__(
        self,
        cosmo: CosmologyInputs,
        source: SourceInputs,
        n_ell: int = 8000,
        ell_range: tuple[float, float] = (1e-1, 2e7),
    ) -> None:
        self.cosmo = cosmo
        self.source = source
        self.ell = np.exp(
            np.linspace(np.log(ell_range[0]), np.log(ell_range[1]), n_ell)
        )

    # -- Sigma-Sigma -------------------------------------------------------
    def c_ell_sigma(self, zl_min: float, zl_max: float, z_h: float):
        """Sigma_crit-weighted matter C_ell (legacy calc_C_ell_Sigma:
        dz = 0.1 slices)."""
        nzl = max(int((zl_max - zl_min) / 0.1), 1)
        edges = np.linspace(zl_min, zl_max, nzl + 1)
        C = np.zeros_like(self.ell)
        for lo, hi in zip(edges[:-1], edges[1:]):
            z_mid = 0.5 * (lo + hi)
            chi_l = float(self.cosmo.chi(z_mid))
            dchi = float(self.cosmo.chi(hi) - self.cosmo.chi(lo))
            kern = float(self.source.q_sigma(z_mid, z_h))
            k = (self.ell + 0.5) / chi_l
            C += dchi * kern**2 / chi_l**2 * self.cosmo.pk_lin(k, z_mid)
        return C * self.cosmo.rho_mean0**2

    def shape_noise_sigma(self, z_h: float) -> float:
        """sigma_gamma^2 <Sigma_crit>^2 / n_src^eff [sr]."""
        n_src_sr = (
            self.source.n_src_arcmin2
            * self.source.f_src_behind(z_h)
            / ARCMIN_TO_RAD**2
        )
        return (
            self.source.sigma_gamma**2
            / n_src_sr
            * self.source.mean_sigma_crit(z_h) ** 2
        )

    # -- halo-halo -----------------------------------------------------------
    def c_ell_h(
        self, z_min: float, z_max: float, bias: float,
        counts: float, area_sr: float,
    ):
        """Halo slab C_ell (legacy calc_C_ell_h) and shot noise.

        Returns ``(C_ell, shot_noise)`` with shot = area_sr / counts.
        """
        nzh = max(int((z_max - z_min) / 0.1), 1)
        edges = np.linspace(z_min, z_max, nzh + 1)
        C = np.zeros_like(self.ell)
        vol = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            z_mid = 0.5 * (lo + hi)
            chi_h = float(self.cosmo.chi(z_mid))
            dchi = float(self.cosmo.chi(hi) - self.cosmo.chi(lo))
            vol += dchi * chi_h**2
            k = (self.ell + 0.5) / chi_h
            C += dchi * chi_h**2 * bias**2 * self.cosmo.pk_lin(k, z_mid)
        C /= vol**2
        shot = area_sr / counts
        return C, shot

    # -- halo-Sigma cross ------------------------------------------------------
    def c_ell_h_sigma(
        self, z_min: float, z_max: float, bias: float, z_h: float
    ):
        """Linear-bias halo-Sigma cross C_ell (legacy calc_C_ell_h_Sigma)."""
        nzh = max(int((z_max - z_min) / 0.1), 1)
        edges = np.linspace(z_min, z_max, nzh + 1)
        C = np.zeros_like(self.ell)
        vol = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            z_mid = 0.5 * (lo + hi)
            chi_h = float(self.cosmo.chi(z_mid))
            dchi = float(self.cosmo.chi(hi) - self.cosmo.chi(lo))
            vol += dchi * chi_h**2
            k = (self.ell + 0.5) / chi_h
            kern = float(self.source.q_sigma(z_mid, z_h))
            C += dchi * kern * bias * self.cosmo.pk_lin(k, z_mid)
        return C * self.cosmo.rho_mean0 / vol
