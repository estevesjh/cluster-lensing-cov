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


def _counts_bias_from_selection(config: dict, pkgrid, cosmology):
    """Counts and bias from the clenspy cluster model (config has a
    'selection' block: HOD or lognormal MOR, EMG projection kernel when
    apply_projection, photo-z K_j, optional b_sel_ls large-scale bias).

    Legacy selection parameters are quoted in Msun/h (RichnessSelection
    convention) — converted to physical here.
    """
    import numpy as np
    from clenspy.clusters import (
        AnalyticLogNormalKernel,
        BinDefinition,
        BinnedClusterModel,
        EmgRichnessKernel,
        HodMOR,
        HodParams,
        LogNormalMOR,
        LogNormalParams,
        SelBiasEngine,
        XiNL,
        omega_z_const_factory,
    )
    from clenspy.cosmology import PkGrid
    from clenspy.halo.mass_function import (
        SigmaGrid,
        Tinker08MassFunction,
        Tinker10Bias,
    )

    sel = config["selection"]
    h = float(config["cosmology"]["h"])
    params = dict(sel.get("parameters", {}))
    model = str(sel["model"]).lower()
    if model in {"hod", "mor", "costanzi_hod"}:
        if "sigma_lambda" in params:
            params["sigma_intr"] = params.pop("sigma_lambda")
        mor = HodMOR(
            HodParams(
                M_min=10.0 ** params["log10_Mmin"] / h,
                M1=10.0 ** params["log10_M1"] / h,
                alpha=params["alpha"],
                sigma_intr=params["sigma_intr"],
                epsilon=params.get("epsilon", 0.0),
                z_pivot=params.get("z_pivot", 0.4544),
            )
        )
    elif model in {"lognormal", "log_normal", "logn"}:
        base = LogNormalParams.costanzi21(h)
        mor = LogNormalMOR(
            LogNormalParams(
                A_lambda=params.get("A_lambda", base.A_lambda),
                B_lambda=params.get("B_lambda", base.B_lambda),
                C_lambda=params.get("C_lambda", base.C_lambda),
                D_lambda=params.get("D_lambda", base.D_lambda),
                # legacy configs quote M_pivot in Msun/h
                M_pivot=params.get("M_pivot", 3.0e14) / h,
                z_pivot=params.get("z_pivot", base.z_pivot),
            )
        )
    else:
        raise ValueError(f"unknown selection model: {model}")

    kernel = (
        EmgRichnessKernel()
        if sel.get("apply_projection", False)
        else AnalyticLogNormalKernel()
    )
    photoz_sigma = sel["photoz_sigma"]
    if np.ndim(photoz_sigma) == 0:
        photoz_sigma = [float(photoz_sigma)] * len(config["lambda_bins"])

    bins = tuple(
        BinDefinition(lmin, lmax, zmin, zmax, sigma_z=photoz_sigma[il])
        for (zmin, zmax) in config["z_bins"]
        for il, (lmin, lmax) in enumerate(config["lambda_bins"])
    )
    area_sr = config["survey_area_deg2"] * (np.pi / 180.0) ** 2
    model_obj = BinnedClusterModel(
        pkgrid=pkgrid,
        mor=mor,
        kernel=kernel,
        bins=bins,
        cosmology=cosmology,
        omega_z=omega_z_const_factory(area_sr),
        n_q=int(sel.get("quadrature_order", 64)),
    )
    counts = model_obj.counts().reshape(
        len(config["z_bins"]), len(config["lambda_bins"])
    )
    bias = model_obj.mean_bias().reshape(counts.shape)

    bsel_cfg = sel.get("bsel_ls", {})
    if bsel_cfg.get("applied", False):
        pk_nl = PkGrid(
            backend="camb", cosmo=cosmology, nonlinear=True,
            k_range=(1e-5, 2e4), z_range=(0.0, 2.0), nk=700, nz=81,
        )
        sg = SigmaGrid(pkgrid, cosmo=cosmology)
        engine = SelBiasEngine(
            cosmology=cosmology,
            xi_nl=XiNL(pk_nl),
            hmf=Tinker08MassFunction(sg),
            bias=Tinker10Bias(sg),
            mor=mor,
        )
        open_lob = float(bsel_cfg.get("open_bin_lob", 80.0))
        for iz, (zmin, zmax) in enumerate(config["z_bins"]):
            zob = 0.5 * (zmin + zmax)
            for il, (lmin, lmax) in enumerate(config["lambda_bins"]):
                lob = open_lob if lmax >= 999.0 else 0.5 * (lmin + lmax)
                # large-scale plateau of the marginalised selection bias
                _, b_large = engine.plateaus(lob, zob)
                bias[iz, il] = b_large
    return counts, bias


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
    base_cosmology = w0waCDM(
        H0=100 * co.h, Om0=co.OmegaM, Ode0=co.OmegaDE, w0=co.w0, wa=co.wa
    )

    class _CosmoWithAmplitude:
        """Astropy cosmologies are frozen; delegate everything and add the
        sigma8/n_s attributes PkGrid reads for the CAMB spec."""

        sigma8 = co.sigma8
        n_s = co.ns

        def __getattr__(self, name):
            return getattr(base_cosmology, name)

    cosmology = _CosmoWithAmplitude()
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
    if "selection" in config:
        counts, bias = _counts_bias_from_selection(config, pkgrid, cosmology)
    else:
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


BUZZARD_R_COMOVING_MPC_H = np.array(
    [0.200, 0.286, 0.409, 0.585, 0.836, 1.196, 1.710, 2.445, 3.497, 5.000]
)


def reproduce(
    config: dict,
    output_dir: str | Path,
    provider: str = "clenspy",
    frozen_dir: str | Path | None = None,
    radial_grid: str = "paper",
) -> dict:
    """Run the full covariance pipeline for one config; returns the
    assembled matrices (also written to ``output_dir``).

    ``radial_grid="buzzard"`` uses the Buzzard jackknife comoving Mpc/h
    bin edges (converted to no-h comoving Mpc) instead of the config's
    physical range.
    """
    if provider == "frozen":
        tables = FrozenTables.load(
            frozen_dir or ROOT / "validation" / "frozen_inputs"
        )
    else:
        tables = _clenspy_tables(config)
    if radial_grid == "buzzard":
        h = float(config["cosmology"]["h"])
        ratio = BUZZARD_R_COMOVING_MPC_H[1] / BUZZARD_R_COMOVING_MPC_H[0]
        edges = np.geomspace(
            BUZZARD_R_COMOVING_MPC_H[0] / np.sqrt(ratio),
            BUZZARD_R_COMOVING_MPC_H[-1] * np.sqrt(ratio),
            len(BUZZARD_R_COMOVING_MPC_H) + 1,
        )
        assembler = CovarianceAssembler(
            tables=tables,
            n_radial=len(BUZZARD_R_COMOVING_MPC_H),
            radial_range_physical_mpc=(edges[0] / h, edges[-1] / h),
            radial_mode="comoving",
        )
    else:
        rmin, rmax = config["radial_range_physical_mpc"]
        assembler = CovarianceAssembler(
            tables=tables,
            n_radial=int(config["n_radial"]),
            radial_range_physical_mpc=(float(rmin), float(rmax)),
        )
    return assembler.run(output_dir)


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
    parser.add_argument(
        "--radial-grid", choices=("paper", "buzzard"), default="paper"
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    result = reproduce(
        config, args.output, provider=args.provider,
        frozen_dir=args.frozen_dir, radial_grid=args.radial_grid,
    )
    print(
        f"wrote {Path(args.output) / 'covariance.npz'}: "
        f"{result['covariance'].shape}, provider={args.provider}"
    )


if __name__ == "__main__":
    main()
