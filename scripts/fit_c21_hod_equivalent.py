#!/usr/bin/env python3
"""Fit the RichnessSelection HOD to the Costanzi et al. (2021) target.

Costanzi et al. (2021) do not publish HOD parameters.  Their SPT-OMR
calibration (the column using the SPT weak-lensing follow-up) publishes an
intrinsic log-normal ``lambda--M`` relation instead.  This script makes the
conversion explicit and reproducible: it fits the HOD's effective satellite
mean and variance to the first two linear-space moments implied by that
published relation over the DES Y1 mass/redshift range.

The fitted ``M_min`` is intentionally treated as an identifiability check.
The C21 power law contains no low-occupancy/central-galaxy information, so a
free fit drives it to the lower bound.  The default covariance conversion
therefore fixes it to the existing RichnessSelection HOD threshold and fits
the remaining four parameters.  The resulting values are an equivalent HOD
representation, not parameters quoted by C21.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]

# Costanzi et al. 2021, Table IV, DES-NC+SPT-OMR, BKG column.  The SPT-OMR
# sample includes the 32 weak-lensing shear profiles (19 Megacam + 13 HST).
C21_SPT_DES_SHEAR = {
    "A_lambda": 76.3,
    "B_lambda": 0.957,
    "C_lambda": 0.48,
    "D_lambda": 0.217,
    "M_pivot": 3.0e14,
    "z_pivot": 0.45,
}

# The current HOD threshold is below the cluster-mass range used by the
# covariance calculation.  C21's log-normal relation cannot determine it;
# retaining it isolates the requested change to the scaling parameters.
REFERENCE_LOG10_MMIN = 11.3852818


def target_moments(mass, redshift, params=C21_SPT_DES_SHEAR):
    """Return C21 linear-space mean and sigma on a broadcastable grid."""
    mass = np.asarray(mass, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    mu = (
        np.log(params["A_lambda"])
        + params["B_lambda"] * np.log(mass / params["M_pivot"])
        + params["C_lambda"]
        * np.log((1.0 + redshift) / (1.0 + params["z_pivot"]))
    )
    lam = np.exp(mu)
    sigma_ln2 = params["D_lambda"] ** 2 + (lam - 1.0) / lam ** 2
    mean = np.exp(mu + 0.5 * sigma_ln2)
    sigma = mean * np.sqrt(np.expm1(sigma_ln2))
    return mean, sigma


def hod_moments(mass, redshift, parameters):
    """Return the effective moments used by RichnessSelection.MOR."""
    mass = np.asarray(mass, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    mmin = 10.0 ** parameters["log10_Mmin"]
    m1 = 10.0 ** parameters["log10_M1"]
    satellite = np.clip((mass - mmin) / (m1 - mmin), 1.0e-30, None)
    satellite = satellite ** parameters["alpha"]
    satellite *= (
        (1.0 + redshift) / (1.0 + parameters["z_pivot"])
    ) ** parameters["epsilon"]
    sigma = np.sqrt(
        satellite + (parameters["sigma_intr"] * satellite) ** 2
    )
    return satellite, sigma


def fit_equivalent(
    *,
    log10_mmin=REFERENCE_LOG10_MMIN,
    mass_log10=(13.0, 15.5),
    redshifts=(0.2, 0.35, 0.5, 0.65),
    n_mass=80,
):
    """Fit HOD parameters to the C21 first two moments."""
    masses = np.logspace(*mass_log10, int(n_mass))
    redshifts = np.asarray(redshifts, dtype=float)
    mass_grid, redshift_grid = np.meshgrid(
        masses, redshifts, indexing="ij"
    )
    target_mean, target_sigma = target_moments(mass_grid, redshift_grid)

    def residual(vector):
        log10_m1, alpha, sigma_intr, epsilon = vector
        hod_mean, hod_sigma = hod_moments(
            mass_grid,
            redshift_grid,
            {
                "log10_Mmin": log10_mmin,
                "log10_M1": log10_m1,
                "alpha": alpha,
                "sigma_intr": sigma_intr,
                "epsilon": epsilon,
                "z_pivot": C21_SPT_DES_SHEAR["z_pivot"],
            },
        )
        # Relative errors give the mean and scatter comparable weight and
        # avoid the highest-richness points dominating the fit.
        return np.concatenate(
            [
                np.log(hod_mean / target_mean).ravel(),
                np.log(hod_sigma / target_sigma).ravel(),
            ]
        )

    fit = least_squares(
        residual,
        x0=np.array([12.5, 0.95, 0.22, 0.48]),
        bounds=(
            np.array([12.0, 0.2, 0.0, -2.0]),
            np.array([15.0, 1.8, 1.0, 2.0]),
        ),
        max_nfev=100_000,
    )
    params = {
        "log10_Mmin": float(log10_mmin),
        "log10_M1": float(fit.x[0]),
        "alpha": float(fit.x[1]),
        "sigma_lambda": float(fit.x[2]),
        "epsilon": float(fit.x[3]),
        "z_pivot": C21_SPT_DES_SHEAR["z_pivot"],
    }
    fit_residual = residual(fit.x)
    return params, {
        "success": bool(fit.success),
        "message": str(fit.message),
        "cost": float(fit.cost),
        "rmse_log_relative": float(np.sqrt(np.mean(fit_residual ** 2))),
        "max_abs_log_relative": float(np.max(np.abs(fit_residual))),
        "mass_log10_range": list(map(float, mass_log10)),
        "redshift_grid": list(map(float, redshifts)),
        "n_mass": int(n_mass),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs" / "c21_spt_des_shear_hod.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    hod_parameters, fit_summary = fit_equivalent()
    payload = {
        "source": {
            "paper": "Costanzi et al. 2021",
            "arxiv": "2010.13800",
            "bibcode": "2021PhRvD.103d3522C",
            "paper_pdf": str(
                ROOT / "papers" / "Costanzi2021.pdf"
            ),
            "data_combination": "DES-NC+SPT-OMR",
            "scatter_model": "BKG",
            "weak_lensing_followup": "32 SPT profiles: 19 Megacam + 13 HST",
            "table": "Table IV / tab:results",
        },
        "target_log_normal_parameters": C21_SPT_DES_SHEAR,
        "hod_parameters": hod_parameters,
        "fit": fit_summary,
        "interpretation": (
            "Empirical HOD-equivalent fit to the C21 intrinsic lambda-M "
            "moments. C21 publishes no native HOD parameters."
        ),
        "selection_method": (
            "Use the fitted HOD P(lambda_tr|M,z) in the existing S_ij "
            "Gauss-Legendre contraction, then apply the existing "
            "P(lambda_ob|lambda_tr,z) kernel."
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
