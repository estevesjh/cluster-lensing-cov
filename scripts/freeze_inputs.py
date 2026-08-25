#!/usr/bin/env python3
"""M0 freeze: snapshot the legacy pipeline's physics inputs.

Dumps, for the Stage-A refactor-equivalence gates (see the refactor plan):

- ``pk_eh_grid.npz``   — EH98 linear P(k, z) from ``clens.ying`` on dense
  grids (the exact spectrum the legacy Limber projection consumed).
- ``kernels.npz``      — lensing-kernel ingredients per z-bin:
  q_Sigma(z_l; z_h), <Sigma_crit>(z_h), f_src(z_h).
- ``counts_bias.json`` — per-(z, lambda)-bin counts, effective bias,
  <bN> = bias*counts, comoving shell volumes, and the legacy
  sample-variance window factor sigma_W = sigma_R(R_eff) * D(z_mid).

Run with the legacy virtualenv:
    .venv/bin/python scripts/freeze_inputs.py --config configs/des_y1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.cosmology import w0waCDM

from clens.lensing.lensing_kernel import LensingKernel
from clens.util.parameters import CosmoParameters
from clens.util.survey import Survey
from clens.ying.param_w0wa import CosmoParams as YingCosmoParams
from clens.ying.lineartheory import LinearTheory
from clens.ying.density_w0wa import Density

ROOT = Path(__file__).resolve().parents[1]


def _survey(config):
    pz = config["source_pz"]
    return Survey(
        z_star_src=pz["z_star"],
        m_src=pz["m"],
        beta_src=pz["beta"],
        n_src_arcmin=config["n_src_arcmin2"],
        sigma_gamma=config["sigma_gamma"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/des_y1.json")
    parser.add_argument("--output", default="validation/frozen_inputs")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    co = CosmoParameters(**config["cosmology"])
    su = _survey(config)
    ying = YingCosmoParams(
        omega_M_0=co.OmegaM, omega_b_0=co.OmegaB,
        omega_lambda_0=co.OmegaDE, h=co.h,
        sigma_8=co.sigma8, n=co.ns, tau=co.tau,
        w0=co.w0, wa=co.wa,
    )
    astropy_dist = w0waCDM(
        H0=co.h * 100.0, Om0=co.OmegaM, Ode0=co.OmegaDE, w0=co.w0, wa=co.wa
    )

    # ---- 1. EH98 linear P(k, z) grid (physical 1/Mpc, Mpc^3) ----------
    k = np.logspace(-5, np.log10(2.0e4), 700)
    z_grid = np.linspace(0.0, 2.0, 81)
    pk = np.empty((z_grid.size, k.size))
    growth = np.empty(z_grid.size)
    for iz, z in enumerate(z_grid):
        lt = LinearTheory(cosmo=ying, z=float(z))
        pk[iz] = lt.power_spectrum(k)
        growth[iz] = Density(ying).growth_factor(float(z))
    growth /= growth[0]
    np.savez(
        out / "pk_eh_grid.npz",
        k=k, z=z_grid, pk=pk, growth=growth,
        cosmology=json.dumps(config["cosmology"]),
    )

    # sigma_R(z=0) table for the counts sample-variance window
    lt0 = LinearTheory(cosmo=ying, z=0.0)
    r_grid = np.logspace(-2, np.log10(500.0), 200)
    sigma_R0 = np.array([lt0.sigma_r_0(float(r)) for r in r_grid])
    np.savez(out / "sigma_R0.npz", r=r_grid, sigma_R0=sigma_R0)

    # ---- 2. lensing kernels per z-bin ----------------------------------
    kernel = LensingKernel(co, su)
    # sample on the EXACT legacy interp1d nodes: the Sigma kernel crosses
    # the Sigma_crit singularity at z_s = z_h and is jagged below z_h, so
    # resampling on any other grid distorts it (np.interp on these nodes
    # reproduces the legacy linear interp1d exactly)
    zl_grid = np.linspace(0.1, su.zs_max - 0.01, 100)
    q_plain = kernel.kernel_z_interp(zl_grid)
    q_sigma = {}
    mean_sc = {}
    f_src = {}
    for zmin, zmax in config["z_bins"]:
        zh = 0.5 * (zmin + zmax)
        kernel.calc_kernel_Sigma(zh)
        q_sigma[f"{zh:.4f}"] = kernel.kernel_Sigma_z_interp(zl_grid)
        mean_sc[f"{zh:.4f}"] = float(kernel.mean_Sigma_crit(zh))
        f_src[f"{zh:.4f}"] = float(kernel.fsrc_behind_zh(zh))
    np.savez(
        out / "kernels.npz",
        zl=zl_grid,
        zs_max=np.array(su.zs_max, dtype=float),
        q_plain=q_plain,
        z_h=np.array([0.5 * (a + b) for a, b in config["z_bins"]]),
        q_sigma=np.array([q_sigma[f"{0.5*(a+b):.4f}"] for a, b in config["z_bins"]]),
        mean_sigma_crit=np.array(
            [mean_sc[f"{0.5*(a+b):.4f}"] for a, b in config["z_bins"]]
        ),
        f_src=np.array([f_src[f"{0.5*(a+b):.4f}"] for a, b in config["z_bins"]]),
    )

    # ---- 3. counts / bias / volumes / sigma_W ---------------------------
    counts = np.asarray(config["counts"], dtype=float)
    bias = np.asarray(config["bias"], dtype=float)
    area_sr = config["survey_area_deg2"] * (np.pi / 180.0) ** 2
    rows = []
    for iz, (zmin, zmax) in enumerate(config["z_bins"]):
        z_mid = 0.5 * (zmin + zmax)
        zs = np.linspace(zmin, zmax, 200)
        dv = astropy_dist.differential_comoving_volume(zs).to_value("Mpc3/sr")
        volume = float(np.trapz(dv, zs) * area_sr)  # comoving Mpc^3
        r_eff = float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))
        lt_z = LinearTheory(cosmo=ying, z=0.0)
        sigma_w = float(
            lt_z.sigma_r_0(r_eff) * np.interp(z_mid, z_grid, growth)
        )
        for il, (lam_min, lam_max) in enumerate(config["lambda_bins"]):
            rows.append(
                dict(
                    z_min=zmin, z_max=zmax,
                    lam_min=lam_min, lam_max=lam_max,
                    counts=float(counts[iz, il]),
                    bias=float(bias[iz, il]),
                    bN=float(bias[iz, il] * counts[iz, il]),
                    volume_mpc3=volume,
                    r_eff_mpc=r_eff,
                    sigma_w=sigma_w,
                )
            )
    (out / "counts_bias.json").write_text(
        json.dumps(
            dict(
                config=str(args.config),
                survey_area_deg2=config["survey_area_deg2"],
                area_sr=area_sr,
                n_src_arcmin2=config["n_src_arcmin2"],
                sigma_gamma=config["sigma_gamma"],
                bins=rows,
            ),
            indent=2,
        )
    )
    print(f"frozen inputs written to {out}")


if __name__ == "__main__":
    main()
