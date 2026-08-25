"""The clenspy-facing input contract of the covariance package.

The covariance classes consume plain callables/arrays held in frozen
dataclasses — never model objects directly.  Two providers exist:

- :class:`FrozenTables` — loads the M0 snapshots written by
  ``scripts/freeze_inputs.py`` (Stage-A refactor-equivalence mode:
  identical physics to the legacy pipeline, EH98 P(k), legacy kernels).
- :func:`from_clenspy` — builds the same contract from live clenspy
  objects (Stage-B physics mode: CAMB PkGrid, clenspy observables).

Units: physical (Msun, comoving Mpc, steradians); P(k, z) in Mpc^3 with
k in 1/Mpc; surface densities in Msun/Mpc^2 until the final 1e-24
conversion to Msun/pc^2 inside the covariance assembly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "CosmologyInputs",
    "SourceInputs",
    "LensSample",
    "SurveyGeometry",
    "FrozenTables",
    "from_clenspy",
]


@dataclass(frozen=True)
class CosmologyInputs:
    """Distances, linear power and growth for the Limber projection."""

    chi: Callable  # comoving distance chi(z) [Mpc]
    pk_lin: Callable  # P_lin(k, z) [Mpc^3], vectorized in k
    rho_mean0: float  # comoving mean matter density [Msun/Mpc^3]
    growth: Callable  # D(z), D(0) = 1
    sigma_R0: Callable  # sigma(R, z=0) top-hat [R in Mpc]


@dataclass(frozen=True)
class SourceInputs:
    """Source population and lensing-kernel ingredients."""

    sigma_gamma: float
    n_src_arcmin2: float
    q_sigma: Callable  # Sigma_crit-weighted kernel q(z_l; z_h) [Mpc^-2... legacy units]
    mean_sigma_crit: Callable  # <Sigma_crit>(z_h) [Msun/Mpc^2]
    f_src_behind: Callable  # fraction of sources behind z_h
    zs_max: float = 2.0


@dataclass(frozen=True)
class LensSample:
    """One (z, lambda) cluster bin.

    ``pk_hh`` / ``pk_hm`` optionally carry the bin's halo-model spectra
    (S_ij-weighted; e.g. ``clenspy.clusters.BinHaloModelSpectra``): the
    Limber projector uses them when present and falls back to the
    linear-bias forms ``bias^2 P_lin`` / ``bias P_lin`` otherwise (the
    frozen Stage-A path always uses the fallback).
    """

    z_min: float
    z_max: float
    lam_min: float
    lam_max: float
    counts: float
    bias: float
    bN: float  # <bN> = b_eff * N (sample-variance weight)
    volume: float  # comoving shell volume over the footprint [Mpc^3]
    sigma_w: float  # sigma_R(R_eff) * D(z_mid) window r.m.s.
    pk_hh: Callable | None = None  # P_hh(k, z) [Mpc^3], 2-halo
    pk_hm: Callable | None = None  # P_hSigma(k, z) [Mpc^3], 2h + 1h
    intrinsic_cov: Callable | None = None  # C_intr(R) [(Msun/Mpc^2)^2],
    # halo-to-halo profile variance / N_cl (clenspy IntrinsicProfileVariance)

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.z_min + self.z_max)

    @property
    def label(self) -> str:
        return (
            f"z[{self.z_min:g},{self.z_max:g}]_"
            f"lam[{self.lam_min:g},{self.lam_max:g}]"
        )


@dataclass(frozen=True)
class SurveyGeometry:
    f_sky: float
    area_sr: float


ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)


@dataclass
class FrozenTables:
    """Contract provider backed by the M0 frozen snapshots."""

    cosmology: CosmologyInputs
    source: SourceInputs
    samples: tuple[LensSample, ...]
    geometry: SurveyGeometry
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenTables":
        path = Path(path)
        pk_data = np.load(path / "pk_eh_grid.npz")
        sig_data = np.load(path / "sigma_R0.npz")
        ker_data = np.load(path / "kernels.npz")
        cb = json.loads((path / "counts_bias.json").read_text())

        k_grid = pk_data["k"]
        z_grid = pk_data["z"]
        lnpk = np.log(pk_data["pk"])  # (nz, nk)
        pk_interp = RegularGridInterpolator(
            (z_grid, np.log(k_grid)), lnpk,
            bounds_error=False, fill_value=None,
        )

        def pk_lin(k, z):
            k = np.atleast_1d(np.asarray(k, dtype=float))
            pts = np.column_stack(
                [np.full(k.size, float(z)), np.log(k)]
            )
            return np.exp(pk_interp(pts))

        growth_tab = pk_data["growth"]

        def growth(z):
            return np.interp(z, z_grid, growth_tab)

        r_grid = sig_data["r"]
        s_tab = sig_data["sigma_R0"]

        def sigma_R0(r):
            return np.interp(r, r_grid, s_tab)

        cosmo_dict = json.loads(str(pk_data["cosmology"]))
        from astropy.cosmology import w0waCDM

        h = float(cosmo_dict["h"])
        om = float(cosmo_dict["OmegaM"])
        astropy_dist = w0waCDM(
            H0=100.0 * h, Om0=om,
            Ode0=float(cosmo_dict.get("OmegaDE", 1.0 - om)),
            w0=float(cosmo_dict.get("w0", -1.0)),
            wa=float(cosmo_dict.get("wa", 0.0)),
        )
        zs_ref = np.linspace(1e-4, 3.0, 3000)
        chi_ref = astropy_dist.comoving_distance(zs_ref).to_value("Mpc")

        def chi(z):
            return np.interp(z, zs_ref, chi_ref)

        rho_crit_h2 = 2.77533742639e11  # Msun h^2 / Mpc^3 (legacy constant)
        rho_mean0 = rho_crit_h2 * h**2 * om

        cosmology = CosmologyInputs(
            chi=chi, pk_lin=pk_lin, rho_mean0=rho_mean0,
            growth=growth, sigma_R0=sigma_R0,
        )

        zl = ker_data["zl"]
        z_h_rows = ker_data["z_h"]
        q_rows = ker_data["q_sigma"]  # (n_zh, n_zl)
        msc_rows = ker_data["mean_sigma_crit"]
        fsrc_rows = ker_data["f_src"]

        def _row(z_h):
            i = int(np.argmin(np.abs(z_h_rows - z_h)))
            if abs(z_h_rows[i] - z_h) > 1e-3:
                raise KeyError(
                    f"frozen kernels have no z_h row near {z_h}; "
                    f"available: {z_h_rows}"
                )
            return i

        def q_sigma(z_l, z_h):
            return np.interp(z_l, zl, q_rows[_row(z_h)])

        def mean_sigma_crit(z_h):
            return float(msc_rows[_row(z_h)])

        def f_src_behind(z_h):
            return float(fsrc_rows[_row(z_h)])

        source = SourceInputs(
            sigma_gamma=float(cb["sigma_gamma"]),
            n_src_arcmin2=float(cb["n_src_arcmin2"]),
            q_sigma=q_sigma,
            mean_sigma_crit=mean_sigma_crit,
            f_src_behind=f_src_behind,
            zs_max=float(ker_data["zs_max"]) if "zs_max" in ker_data else 3.0,
        )

        samples = tuple(
            LensSample(
                z_min=row["z_min"], z_max=row["z_max"],
                lam_min=row["lam_min"], lam_max=row["lam_max"],
                counts=row["counts"], bias=row["bias"], bN=row["bN"],
                volume=row["volume_mpc3"], sigma_w=row["sigma_w"],
            )
            for row in cb["bins"]
        )
        area_sr = float(cb["area_sr"])
        geometry = SurveyGeometry(
            f_sky=float(cb["survey_area_deg2"]) / 41253.0, area_sr=area_sr
        )
        return cls(
            cosmology=cosmology, source=source, samples=samples,
            geometry=geometry, meta=dict(config=cb.get("config", "")),
        )


def from_clenspy(
    *,
    pkgrid,
    cosmology,
    lensing_kernel,
    samples: Sequence[LensSample],
    sigma_gamma: float,
    n_src_arcmin2: float,
    survey_area_deg2: float,
) -> FrozenTables:
    """Build the contract from live clenspy objects (Stage-B).

    ``lensing_kernel`` must expose ``q_sigma(z_l, z_h)``,
    ``mean_sigma_crit(z_h)``, ``f_src_behind(z_h)`` — e.g. the legacy
    LensingKernel wrapped, or a clenspy kernel engine.
    """
    from astropy import units as u  # noqa: F401
    from clenspy.halo.mass_function import SigmaGrid

    sg = SigmaGrid(pkgrid, cosmo=cosmology)
    zs_ref = np.linspace(1e-4, 3.0, 3000)
    chi_ref = cosmology.comoving_distance(zs_ref).to_value("Mpc")

    def chi(z):
        return np.interp(z, zs_ref, chi_ref)

    z_grid = np.asarray(pkgrid.z)
    pk00 = pkgrid(pkgrid.k[0], z_grid)
    growth_tab = np.sqrt(pk00 / pk00[0])

    def growth(z):
        return np.interp(z, z_grid, growth_tab)

    def pk_lin(k, z):
        return pkgrid(k, float(z))

    def sigma_R0(r):
        # sigma of the mass enclosed in radius r at z=0
        M = 4.0 / 3.0 * np.pi * np.asarray(r, dtype=float) ** 3 * sg.rho_m0
        return sg(M, 0.0)

    cosmo_inputs = CosmologyInputs(
        chi=chi, pk_lin=pk_lin, rho_mean0=sg.rho_m0,
        growth=growth, sigma_R0=sigma_R0,
    )
    source = SourceInputs(
        sigma_gamma=sigma_gamma,
        n_src_arcmin2=n_src_arcmin2,
        q_sigma=lensing_kernel.q_sigma,
        mean_sigma_crit=lensing_kernel.mean_sigma_crit,
        f_src_behind=lensing_kernel.f_src_behind,
    )
    geometry = SurveyGeometry(
        f_sky=survey_area_deg2 / 41253.0,
        area_sr=4.0 * np.pi * survey_area_deg2 / 41253.0,
    )
    return FrozenTables(
        cosmology=cosmo_inputs, source=source,
        samples=tuple(samples), geometry=geometry,
    )
