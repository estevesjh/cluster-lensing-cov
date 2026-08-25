#!/usr/bin/env python3
"""Compare the generated DES Y1 covariance with local validation products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
Z_MID = np.array([0.275, 0.425, 0.575])
N_LAMBDA = 4
N_RADIAL_ANALYTIC = 15

# The active Buzzard validation likelihood uses these 10 comoving Mpc/h
# centers.  The DES-Y1 analytic files use physical Mpc centers, so the script
# converts the Buzzard grid per lens-redshift bin before nearest-bin matching.
BUZZARD_R_COMOVING_MPC_H = np.array(
    [0.200, 0.286, 0.409, 0.585, 0.836, 1.196, 1.710, 2.445, 3.497, 5.000]
)


def _block(matrix: np.ndarray, iz: int, ilam: int, nrad: int) -> np.ndarray:
    start = (iz * N_LAMBDA + ilam) * nrad
    return matrix[start : start + nrad, start : start + nrad]


def compare(y1_path: Path, buzzard_path: Path, output_dir: Path, buzzard_grid_path: Path | None) -> dict:
    y1 = np.load(y1_path)
    buzzard = np.load(buzzard_path)
    buzzard_cov = np.linalg.inv(buzzard["invcov_Shear"])
    buzzard_cov = 0.5 * (buzzard_cov + buzzard_cov.T)
    y1_cov = y1["covariance"]
    y1_radii_phys = y1["radii_phys_mpc"]
    exact_grid = buzzard_grid_path is not None and buzzard_grid_path.exists()
    exact_grid_cov = np.load(buzzard_grid_path)["covariance"] if exact_grid else None

    variance_ratios = []
    error_ratios = []
    correlation_rms = []
    selected_indices = []
    for iz, zmid in enumerate(Z_MID):
        if exact_grid:
            indices = None
        else:
            r_target_phys = BUZZARD_R_COMOVING_MPC_H / (0.7 * (1.0 + zmid))
            indices = np.array(
                [np.argmin(np.abs(np.log(y1_radii_phys[iz] / r))) for r in r_target_phys]
            )
            selected_indices.append(indices.tolist())
        for ilam in range(N_LAMBDA):
            if exact_grid:
                analytic = _block(exact_grid_cov, iz, ilam, len(BUZZARD_R_COMOVING_MPC_H))
            else:
                analytic = _block(y1_cov, iz, ilam, N_RADIAL_ANALYTIC)[
                    np.ix_(indices, indices)
                ]
            buzz = _block(buzzard_cov, iz, ilam, len(BUZZARD_R_COMOVING_MPC_H))
            variance_ratios.extend(np.diag(analytic) / np.diag(buzz))
            error_ratios.extend(np.sqrt(np.diag(analytic) / np.diag(buzz)))
            analytic_corr = analytic / np.sqrt(np.outer(np.diag(analytic), np.diag(analytic)))
            buzz_corr = buzz / np.sqrt(np.outer(np.diag(buzz), np.diag(buzz)))
            correlation_rms.append(float(np.sqrt(np.mean((analytic_corr - buzz_corr) ** 2))))

    jk_variance_ratios = []
    sac_variance_ratios = []
    for iz in range(3):
        for ilam in range(N_LAMBDA):
            analytic = _block(y1_cov, iz, ilam, N_RADIAL_ANALYTIC)
            jk_path = ROOT / "data" / "desy1_jackknife" / (
                f"full-unblind-v2-mcal-zmix_y1subtr_l{ilam + 3}_z{iz}_dst_cov.dat"
            )
            sac_path = ROOT / "data" / "desy1_SAC" / f"SAC_z{iz}_l{ilam + 3}.txt"
            jk = np.loadtxt(jk_path)
            sac = np.loadtxt(sac_path)
            jk_variance_ratios.extend(np.diag(analytic) / np.diag(jk))
            sac_variance_ratios.extend(np.diag(analytic) / np.diag(sac))

    summary = {
        "buzzard_source": str(buzzard_path),
        "buzzard_inverse_was_inverted": True,
        "buzzard_comparison_note": (
            "exact re-integration on inferred logarithmic Buzzard windows"
            if exact_grid
            else "nearest analytic 15-bin center; not a re-integration on the Buzzard 10-bin window grid"
        ),
        "buzzard_selected_analytic_indices": selected_indices,
        "buzzard_variance_ratio_analytic_over_buzzard": {
            "median": float(np.median(variance_ratios)),
            "p16": float(np.percentile(variance_ratios, 16)),
            "p84": float(np.percentile(variance_ratios, 84)),
        },
        "buzzard_error_ratio_analytic_over_buzzard": {
            "median": float(np.median(error_ratios)),
            "p16": float(np.percentile(error_ratios, 16)),
            "p84": float(np.percentile(error_ratios, 84)),
        },
        "buzzard_correlation_rms": {
            "median": float(np.median(correlation_rms)),
            "p16": float(np.percentile(correlation_rms, 16)),
            "p84": float(np.percentile(correlation_rms, 84)),
        },
        "mcclintock_jackknife_variance_ratio_analytic_over_jk": {
            "median": float(np.median(jk_variance_ratios)),
            "p16": float(np.percentile(jk_variance_ratios, 16)),
            "p84": float(np.percentile(jk_variance_ratios, 84)),
        },
        "mcclintock_sac_variance_ratio_analytic_over_sac": {
            "median": float(np.median(sac_variance_ratios)),
            "p16": float(np.percentile(sac_variance_ratios, 16)),
            "p84": float(np.percentile(sac_variance_ratios, 84)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    np.savez_compressed(
        output_dir / "buzzard_comparison.npz",
        buzzard_covariance=buzzard_cov,
        buzzard_r_comoving_mpc_h=BUZZARD_R_COMOVING_MPC_H,
        variance_ratios=np.asarray(variance_ratios),
        error_ratios=np.asarray(error_ratios),
        correlation_rms=np.asarray(correlation_rms),
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y1", type=Path, default=ROOT / "results/des_y1/covariance.npz")
    parser.add_argument("--buzzard", type=Path, default=ROOT / "data/buzzard/dv_buzzard_jkcov.npz")
    parser.add_argument("--buzzard-grid", type=Path, default=ROOT / "results/des_y1_buzzard_grid/covariance.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/comparison")
    args = parser.parse_args()
    compare(args.y1, args.buzzard, args.output, args.buzzard_grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
