#!/usr/bin/env python3
"""Reproduce the analytic DeltaSigma + N_ij covariances — FFTLog pipeline.

Successor of ``reproduce_covariance.py``: same JSON configs, same output
schema (z-major, richness-fast 180x180 block matrix, per-term files), but
the ell-integration runs through the clenspy FFTLog engine and all model
ingredients enter through the ``clens.covariance.inputs`` contract.

Providers:
  --provider frozen   Stage-A: the M0 frozen snapshots (legacy EH physics)
  --provider clenspy  Stage-B: live clenspy (CAMB PkGrid + legacy kernels)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clens.covariance import CovarianceAssembler, FrozenTables, LensSample

ROOT = Path(__file__).resolve().parents[1]


def _clenspy_tables(config: dict) -> FrozenTables:
    """Stage-B provider: CAMB P(k) via clenspy; kernels from the legacy
    LensingKernel (until the kernel engine moves to clenspy); counts/bias
    from the config (or clenspy BinnedClusterModel when a selection block
    is present)."""
    from astropy.cosmology import w0waCDM
    from clenspy.cosmology import PkGrid

    from clens.covariance.inputs import from_clenspy
    from clens.lensing.lensing_kernel import LensingKernel
    from clens.util.parameters import CosmoParameters
    from clens.util.survey import Survey

    co = CosmoParameters(**config["cosmology"])
    cosmology = w0waCDM(
        H0=100 * co.h, Om0=co.OmegaM, Ode0=co.OmegaDE, w0=co.w0, wa=co.wa
    )
    # attach amplitude/tilt so PkGrid does not fall back to its defaults
    cosmology.sigma8 = co.sigma8
    cosmology.n_s = co.ns
    pkgrid = PkGrid(
        backend="camb",
        cosmo=cosmology,
        nonlinear=False,
        k_range=(1e-5, 2e4),
        z_range=(0.0, 2.0),
        nk=700,
        nz=81,
    )

    pz = config["source_pz"]
    survey = Survey(
        z_star_src=pz["z_star"], m_src=pz["m"], beta_src=pz["beta"],
        n_src_arcmin=config["n_src_arcmin2"],
        sigma_gamma=config["sigma_gamma"],
    )
    lk = LensingKernel(co, survey)

    class _KernelShim:
        def q_sigma(self, z_l, z_h):
            lk.calc_kernel_Sigma(z_h)
            return lk.kernel_Sigma_z_interp(z_l)

        def mean_sigma_crit(self, z_h):
            return float(lk.mean_Sigma_crit(z_h))

        def f_src_behind(self, z_h):
            return float(lk.fsrc_behind_zh(z_h))

    area_sr = config["survey_area_deg2"] * (np.pi / 180.0) ** 2
    counts = np.asarray(config["counts"], dtype=float)
    bias = np.asarray(config["bias"], dtype=float)

    samples = []
    zs_ref = np.linspace(1e-4, 3.0, 3000)
    dv_ref = cosmology.differential_comoving_volume(zs_ref).to_value("Mpc3/sr")
    from clenspy.halo.mass_function import SigmaGrid

    sg = SigmaGrid(pkgrid, cosmo=cosmology)
    z_grid = np.asarray(pkgrid.z)
    pk00 = pkgrid(pkgrid.k[0], z_grid)
    growth_tab = np.sqrt(pk00 / pk00[0])
    for iz, (zmin, zmax) in enumerate(config["z_bins"]):
        zs = np.linspace(zmin, zmax, 200)
        volume = float(
            np.trapezoid(np.interp(zs, zs_ref, dv_ref), zs) * area_sr
        )
        r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        M_eff = 4.0 / 3.0 * np.pi * r_eff**3 * sg.rho_m0
        sigma_w = float(
            sg(M_eff, 0.0)
            * np.interp(0.5 * (zmin + zmax), z_grid, growth_tab)
        )
        for il, (lmin, lmax) in enumerate(config["lambda_bins"]):
            samples.append(
                LensSample(
                    z_min=zmin, z_max=zmax, lam_min=lmin, lam_max=lmax,
                    counts=float(counts[iz, il]), bias=float(bias[iz, il]),
                    bN=float(counts[iz, il] * bias[iz, il]),
                    volume=volume, sigma_w=sigma_w,
                )
            )

    return from_clenspy(
        pkgrid=pkgrid,
        cosmology=cosmology,
        lensing_kernel=_KernelShim(),
        samples=samples,
        sigma_gamma=config["sigma_gamma"],
        n_src_arcmin2=config["n_src_arcmin2"],
        survey_area_deg2=config["survey_area_deg2"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/des_y1.json")
    parser.add_argument("--output", default="output/covariance_v2")
    parser.add_argument(
        "--provider", choices=("frozen", "clenspy"), default="frozen"
    )
    parser.add_argument(
        "--frozen-dir", default=str(ROOT / "validation" / "frozen_inputs")
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if args.provider == "frozen":
        tables = FrozenTables.load(args.frozen_dir)
    else:
        tables = _clenspy_tables(config)

    rmin, rmax = config["radial_range_physical_mpc"]
    assembler = CovarianceAssembler(
        tables=tables,
        n_radial=int(config["n_radial"]),
        radial_range_physical_mpc=(float(rmin), float(rmax)),
    )
    result = assembler.run(args.output)
    print(
        f"wrote {Path(args.output) / 'covariance.npz'}: "
        f"{result['covariance'].shape}, provider={args.provider}"
    )


if __name__ == "__main__":
    main()
