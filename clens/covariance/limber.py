"""Limber angular power spectra — thin adapter over clenspy.

The projection itself lives in :class:`clenspy.lensing.limber.
LimberProjector` (clenspy computes the angular power spectra as well as
the number counts); this module only binds the covariance input contract
(:mod:`clens.covariance.inputs`) to it.
"""

from __future__ import annotations

from clenspy.lensing.limber import ARCMIN_TO_RAD  # noqa: F401 (re-export)
from clenspy.lensing.limber import LimberProjector as _ClenspyLimber

from .inputs import CosmologyInputs, SourceInputs

__all__ = ["LimberProjector"]


class LimberProjector(_ClenspyLimber):
    """clenspy LimberProjector bound to the contract dataclasses."""

    def __init__(
        self,
        cosmo: CosmologyInputs,
        source: SourceInputs,
        n_ell: int = 8000,
        ell_range: tuple[float, float] = (1e-1, 2e7),
    ) -> None:
        super().__init__(
            chi=cosmo.chi,
            pk_lin=cosmo.pk_lin,
            rho_mean0=cosmo.rho_mean0,
            q_sigma=source.q_sigma,
            mean_sigma_crit=source.mean_sigma_crit,
            f_src_behind=source.f_src_behind,
            sigma_gamma=source.sigma_gamma,
            n_src_arcmin2=source.n_src_arcmin2,
            n_ell=n_ell,
            ell_range=ell_range,
        )
        self.cosmo = cosmo
        self.source = source
