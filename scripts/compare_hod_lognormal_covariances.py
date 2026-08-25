#!/usr/bin/env python3
"""Generate and compare intrinsic-HOD and log-normal DES-Y1 covariances."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reproduce_covariance import ROOT, reproduce


def _load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


_C21_HOD_CONFIG = json.loads(
    (ROOT / "configs" / "c21_spt_des_shear_hod.json").read_text(
        encoding="utf-8"
    )
)
HOD_PARAMETERS = _C21_HOD_CONFIG["hod_parameters"]
LOGNORMAL_PARAMETERS = _C21_HOD_CONFIG["target_log_normal_parameters"]


def _model_config(base, model):
    config = copy.deepcopy(base)
    config.pop("counts", None)
    config.pop("bias", None)
    config["selection"] = {
        "model": model,
        "parameters": (HOD_PARAMETERS if model == "hod"
                        else LOGNORMAL_PARAMETERS),
        "quadrature_order": 64,
        # The RichnessSelection validation pipeline uses the DES-Y3
        # richness-bin value sigma_z=0.03 for this comparison.  Keep this
        # explicit because S_ij needs the redshift kernel K_j as well as K_i.
        "photoz_sigma": 0.03,
        "ltr_bracket_sigma": 6.0,
        "apply_projection": True,
        "bsel_ls": {
            "applied": True,
            "open_bin_lob": 80.0,
        },
        "projection": {
            "model": "Costanzi-2019",
            "applied": True,
            "parameters_source": (
                "github/RichnessSelection/data/"
                "prj_params_DESY3_lss_lin_dep_getdist_v1.txt"
            ),
        },
    }
    return config


def _load_diagonal(path, n_z, n_lambda):
    data = np.load(path)
    covariance = np.asarray(data["covariance"], dtype=float)
    radii_by_z = np.asarray(data["radii_phys_mpc"], dtype=float)
    radii = np.repeat(radii_by_z, n_lambda, axis=0).ravel()
    return radii, np.sqrt(np.diag(covariance))


def compare(base_config, output_root, radial_grid, reuse=False):
    output_root.mkdir(parents=True, exist_ok=True)
    products = {}
    for model in ("lognormal", "hod"):
        config = _model_config(base_config, model)
        model_dir = output_root / model
        covariance_path = model_dir / "covariance.npz"
        if reuse and covariance_path.is_file():
            products[model] = None
        else:
            products[model] = reproduce(config, model_dir, provider="clenspy")

    n_z = len(base_config["z_bins"])
    n_lambda = len(base_config["lambda_bins"])
    radii_logn, error_logn = _load_diagonal(
        output_root / "lognormal" / "covariance.npz", n_z, n_lambda)
    radii_hod, error_hod = _load_diagonal(
        output_root / "hod" / "covariance.npz", n_z, n_lambda)
    if radii_logn.shape != radii_hod.shape or not np.allclose(radii_logn, radii_hod):
        raise ValueError("the two covariance products do not share a radial grid")

    ratio = error_hod / error_logn
    per_bin = []
    for iz, (zmin, zmax) in enumerate(base_config["z_bins"]):
        for ilam, (lam_min, lam_max) in enumerate(base_config["lambda_bins"]):
            block = iz * n_lambda + ilam
            block_ratio = ratio[block * len(radii_logn) // (n_z * n_lambda):
                                (block + 1) * len(radii_logn) // (n_z * n_lambda)]
            per_bin.append({
                "z": [zmin, zmax],
                "lambda": [lam_min, lam_max],
                "median_ratio": float(np.median(block_ratio)),
                "min_ratio": float(np.min(block_ratio)),
                "max_ratio": float(np.max(block_ratio)),
            })
    summary = {
        "radial_grid": radial_grid,
        "ordering": "z-major, richness-fast, radial-fast",
        "comparison": "HOD diagonal error / log-normal diagonal error",
        "median_ratio": float(np.median(ratio)),
        "p16_ratio": float(np.percentile(ratio, 16)),
        "p84_ratio": float(np.percentile(ratio, 84)),
        "min_ratio": float(np.min(ratio)),
        "max_ratio": float(np.max(ratio)),
        "fraction_hod_errors_larger": float(np.mean(ratio > 1.0)),
        "per_bin": per_bin,
        "hod_parameters": HOD_PARAMETERS,
        "lognormal_parameters": LOGNORMAL_PARAMETERS,
        "projection_kernel": (
            "applied through RichnessSelection K_i and K_j; "
            "photoz_sigma=0.03"
        ),
        "projection_parameters_source": (
            "github/RichnessSelection/data/"
            "prj_params_DESY3_lss_lin_dep_getdist_v1.txt"
        ),
    }
    with (output_root / "comparison_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    n_blocks = n_z * n_lambda
    n_radial = len(radii_logn) // n_blocks
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax = axes[0]
    for block in range(n_blocks):
        sl = slice(block * n_radial, (block + 1) * n_radial)
        ax.loglog(radii_logn[sl], error_logn[sl], color="0.55", alpha=0.45)
        ax.loglog(radii_hod[sl], error_hod[sl], color="tab:blue", alpha=0.45)
    ax.plot([], [], color="0.55", label="log-normal")
    ax.plot([], [], color="tab:blue", label="HOD")
    ax.set_xlabel(r"$R$ [physical Mpc]")
    ax.set_ylabel(r"$\sigma(\Delta\Sigma)$ [M$_\odot$ pc$^{-2}$]")
    ax.set_title("DES Y1 diagonal errors")
    ax.legend(frameon=False)

    ax = axes[1]
    for block in range(n_blocks):
        sl = slice(block * n_radial, (block + 1) * n_radial)
        ax.semilogx(radii_logn[sl], ratio[sl], color="tab:purple", alpha=0.55)
    ax.axhline(1.0, color="0.2", lw=1)
    ax.axhline(1.1, color="0.6", lw=0.8, ls="--")
    ax.axhline(0.9, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel(r"$R$ [physical Mpc]")
    ax.set_ylabel("HOD / log-normal error")
    ax.set_title("Error ratio")
    fig.savefig(output_root / "hod_vs_lognormal_diagonal_errors.png", dpi=180)
    fig.savefig(output_root / "hod_vs_lognormal_diagonal_errors.pdf")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "des_y1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "des_y1_hod_lognormal")
    parser.add_argument("--radial-grid", choices=("paper", "buzzard"), default="paper")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse existing model covariance.npz files")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    summary = compare(_load_config(config_path), output_root, args.radial_grid,
                      reuse=args.reuse)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
