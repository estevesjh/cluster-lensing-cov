#!/usr/bin/env python3
"""Plot the plain physical-unit analytic/Buzzard JK error ratio."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
Z_LABELS = ["0.20–0.35", "0.35–0.50", "0.50–0.65"]
LAMBDA_LABELS = ["20–30", "30–45", "45–60", "60+"]
H_BUZZARD = 0.7


def _block(matrix: np.ndarray, iz: int, ilam: int, nrad: int = 10) -> np.ndarray:
    start = (iz * 4 + ilam) * nrad
    return matrix[start : start + nrad, start : start + nrad]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analytic",
        type=Path,
        default=ROOT / "results/des_y1_buzzard_grid/covariance.npz",
        help="analytic covariance on the Buzzard 10-radius grid",
    )
    parser.add_argument(
        "--output-stem",
        default="plain_physical_error_ratio",
        help="output filename stem under results/validation_plots",
    )
    parser.add_argument("--survey-label", default="DES Y1")
    args = parser.parse_args()

    analytic_path = args.analytic if args.analytic.is_absolute() else ROOT / args.analytic
    analytic = np.load(analytic_path)
    buzzard = np.load(ROOT / "data/buzzard/dv_buzzard_jkcov_hfix.npz")
    buzzard_cov_h = np.linalg.inv(buzzard["invcov_Shear"])
    buzzard_cov_h = 0.5 * (buzzard_cov_h + buzzard_cov_h.T)

    # hfix stores h * DeltaSigma. Convert both its covariance and errors back
    # to physical DeltaSigma before taking the ratio.
    buzzard_cov_phys = buzzard_cov_h / H_BUZZARD**2
    buzzard_sigma_phys = np.sqrt(np.diag(buzzard_cov_phys))

    output = ROOT / "results/validation_plots"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True, squeeze=False)
    ratios = []
    for iz in range(3):
        for ilam in range(4):
            analytic_cov = _block(analytic["covariance"], iz, ilam)
            analytic_sigma_phys = np.sqrt(np.diag(analytic_cov))
            buzzard_sigma = buzzard_sigma_phys[(iz * 4 + ilam) * 10 : (iz * 4 + ilam + 1) * 10]
            ratio = analytic_sigma_phys / buzzard_sigma
            ratios.extend(ratio)

            ax = axes[iz, ilam]
            ax.semilogx(analytic["radii_phys_mpc"][iz], ratio, "o-", lw=1.5, ms=4)
            ax.axhline(1.0, color="black", lw=0.9)
            ax.set_title(f"z {Z_LABELS[iz]}, λ {LAMBDA_LABELS[ilam]}", fontsize=10)
            ax.grid(True, which="both", alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$R$ [physical Mpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sigma_{\rm analytic}/\sigma_{\rm JK,phys}$")
    axes[0, 0].set_ylim(0, max(2.0, 1.08 * max(ratios)))
    fig.suptitle(
        f"Plain physical-unit error ratio: analytic {args.survey_label} / Buzzard JK",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output / f"{args.output_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{args.output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    ratios = np.asarray(ratios)
    print(
        "plain physical error ratio: "
        f"median={np.median(ratios):.3f}, "
        f"p16={np.percentile(ratios, 16):.3f}, "
        f"p84={np.percentile(ratios, 84):.3f}, "
        f"min={ratios.min():.3f}, max={ratios.max():.3f}"
    )
    print(f"wrote {output / (args.output_stem + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
