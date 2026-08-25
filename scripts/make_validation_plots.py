#!/usr/bin/env python3
"""Make reproducible DES Y1 covariance validation plots.

The plots compare the generated analytic covariance with the local McClintock
DES Y1 jackknife and SAC covariance products.  Radial ratios are evaluated at
the McClintock profile centres by log-linear interpolation of the analytic
variance, with log-slope extrapolation only at the two edge centres; this
avoids silently comparing different radial windows by index.
The Buzzard comparison is kept separate because its covariance file has a
different 10-point radial grid and unresolved unit/convention provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
Z_LABELS = ["0.20–0.35", "0.35–0.50", "0.50–0.65"]
LAMBDA_LABELS = ["20–30", "30–45", "45–60", "60+"]
N_Z = len(Z_LABELS)
N_LAMBDA = len(LAMBDA_LABELS)
N_RADIAL = 15
N_BUZZARD_RADIAL = 10
BUZZARD_R_COMOVING_MPC_H = np.array(
    [0.200, 0.286, 0.409, 0.585, 0.836, 1.196, 1.710, 2.445, 3.497, 5.000]
)


def _block(matrix: np.ndarray, iz: int, ilam: int, nrad: int) -> np.ndarray:
    start = (iz * N_LAMBDA + ilam) * nrad
    return matrix[start : start + nrad, start : start + nrad]


def _corr(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    return covariance / np.sqrt(np.outer(diagonal, diagonal))


def _interpolate_variance(r_source: np.ndarray, variance: np.ndarray, r_target: np.ndarray) -> np.ndarray:
    """Interpolate positive variance in log R, preserving covariance units."""

    if np.any(r_source <= 0) or np.any(r_target <= 0):
        raise ValueError("All radial centres must be positive")
    if np.any(variance <= 0):
        raise ValueError("All diagonal variances must be positive for interpolation")
    log_source = np.log(r_source)
    log_variance = np.log(variance)
    log_target = np.log(r_target)
    result = np.interp(log_target, log_source, log_variance)
    left_slope = (log_variance[1] - log_variance[0]) / (log_source[1] - log_source[0])
    right_slope = (log_variance[-1] - log_variance[-2]) / (log_source[-1] - log_source[-2])
    left = log_target < log_source[0]
    right = log_target > log_source[-1]
    result[left] = log_variance[0] + left_slope * (log_target[left] - log_source[0])
    result[right] = log_variance[-1] + right_slope * (log_target[right] - log_source[-1])
    return np.exp(result)


def _load_inputs(y1_path: Path) -> dict:
    analytic = np.load(y1_path)
    references = []
    for iz in range(N_Z):
        for ilam in range(N_LAMBDA):
            # The local McClintock products use l3..l6 for lambda 20..60+.
            source_lambda = ilam + 3
            profile_path = ROOT / "data/desy1_jackknife" / (
                f"full-unblind-v2-mcal-zmix_y1subtr_l{source_lambda}_z{iz}_profile.dat"
            )
            jk_path = ROOT / "data/desy1_jackknife" / (
                f"full-unblind-v2-mcal-zmix_y1subtr_l{source_lambda}_z{iz}_dst_cov.dat"
            )
            sac_path = ROOT / "data/desy1_SAC" / f"SAC_z{iz}_l{source_lambda}.txt"
            profile = np.loadtxt(profile_path, comments="#")
            references.append(
                {
                    "iz": iz,
                    "ilam": ilam,
                    "r_profile": profile[:, 0],
                    "delta_sigma": profile[:, 1],
                    "jk": np.loadtxt(jk_path),
                    "sac": np.loadtxt(sac_path),
                    "profile_path": str(profile_path),
                    "jk_path": str(jk_path),
                    "sac_path": str(sac_path),
                }
            )
    return {"analytic": analytic, "references": references}


def _records(inputs: dict) -> list[dict]:
    analytic = inputs["analytic"]
    records = []
    for reference in inputs["references"]:
        iz = reference["iz"]
        ilam = reference["ilam"]
        analytic_cov = _block(analytic["covariance"], iz, ilam, N_RADIAL)
        analytic_r = analytic["radii_phys_mpc"][iz]
        jk_r = reference["r_profile"]
        analytic_var_at_jk = _interpolate_variance(
            analytic_r, np.diag(analytic_cov), jk_r
        )
        jk_var = np.diag(reference["jk"])
        sac_var = np.diag(reference["sac"])
        jk_error_ratio = np.sqrt(analytic_var_at_jk / jk_var)
        sac_error_ratio = np.sqrt(analytic_var_at_jk / sac_var)
        records.append(
            {
                **reference,
                "analytic_cov": analytic_cov,
                "analytic_r": analytic_r,
                "analytic_sigma": np.sqrt(np.diag(analytic_cov)),
                "jk_sigma": np.sqrt(jk_var),
                "sac_sigma": np.sqrt(sac_var),
                "analytic_sigma_at_jk": np.sqrt(analytic_var_at_jk),
                "jk_error_ratio": jk_error_ratio,
                "sac_error_ratio": sac_error_ratio,
                "native_index_jk_error_ratio": np.sqrt(np.diag(analytic_cov) / jk_var),
                "native_index_sac_error_ratio": np.sqrt(np.diag(analytic_cov) / sac_var),
                "jk_variance_ratio": analytic_var_at_jk / jk_var,
                "sac_variance_ratio": analytic_var_at_jk / sac_var,
            }
        )
    return records


def _ratio_summary(ratio: np.ndarray) -> dict:
    fractional_difference = np.abs(ratio - 1.0)
    return {
        "median": float(np.median(ratio)),
        "p16": float(np.percentile(ratio, 16)),
        "p84": float(np.percentile(ratio, 84)),
        "fraction_points_outside_10_percent": float(
            np.mean(fractional_difference > 0.10)
        ),
        "max_abs_fractional_difference": float(np.max(fractional_difference)),
    }


def _save_figure(fig: plt.Figure, name: str, output: Path, pdf: PdfPages) -> None:
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_sigma(records: list[dict], output: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(
        N_Z, N_LAMBDA, figsize=(15, 10), sharex=True, sharey=True, squeeze=False
    )
    for record in records:
        iz, ilam = record["iz"], record["ilam"]
        ax = axes[iz, ilam]
        r = record["r_profile"]
        jk_y = r * record["jk_sigma"]
        sac_y = r * record["sac_sigma"]
        analytic_y = record["analytic_r"] * record["analytic_sigma"]
        ax.fill_between(r, 0.9 * jk_y, 1.1 * jk_y, color="0.85", label="±10% JK")
        ax.loglog(record["analytic_r"], analytic_y, "o-", ms=3, lw=1.5, label="analytic")
        ax.loglog(r, jk_y, "s-", ms=3, lw=1.2, label="McClintock JK")
        ax.loglog(r, sac_y, "^-", ms=3, lw=1.2, label="SAC")
        ax.set_title(f"z {Z_LABELS[iz]}, λ {LAMBDA_LABELS[ilam]}", fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("R [physical Mpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$R\,\sigma(\Delta\Sigma)$")
    fig.suptitle(
        "DES Y1 diagonal covariance validation\n"
        "Curves retain their native radial centres; the shaded band is ±10% around the JK result",
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, "mcclintock_sigma_comparison", output, pdf)


def _plot_ratios(records: list[dict], output: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(
        N_Z, N_LAMBDA, figsize=(15, 10), sharex=True, sharey=True, squeeze=False
    )
    for record in records:
        iz, ilam = record["iz"], record["ilam"]
        ax = axes[iz, ilam]
        r = record["r_profile"]
        ax.semilogx(r, record["jk_error_ratio"], "o-", ms=3, lw=1.3, label="analytic / JK")
        ax.semilogx(r, record["sac_error_ratio"], "^-", ms=3, lw=1.3, label="analytic / SAC")
        ax.axhspan(0.9, 1.1, color="0.88", zorder=0)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axhline(0.9, color="0.4", ls="--", lw=0.7)
        ax.axhline(1.1, color="0.4", ls="--", lw=0.7)
        # robust y-limits: median +/- 2 robust sigma over both ratio sets
        both = np.concatenate(
            [record["jk_error_ratio"], record["sac_error_ratio"]]
        )
        both = both[np.isfinite(both)]
        med = np.median(both)
        sig_rob = 0.7413 * (np.percentile(both, 84) - np.percentile(both, 16))
        ax.set_ylim(max(0.0, min(med - 2 * sig_rob, 0.85)),
                    max(med + 2 * sig_rob, 1.15))
        jk = _ratio_summary(record["jk_error_ratio"])
        sac = _ratio_summary(record["sac_error_ratio"])
        ax.text(
            0.03,
            0.04,
            f"JK med {jk['median']:.2f}\n>{10}%: {100*jk['fraction_points_outside_10_percent']:.0f}%\n"
            f"SAC med {sac['median']:.2f}\n>{10}%: {100*sac['fraction_points_outside_10_percent']:.0f}%",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.set_title(f"z {Z_LABELS[iz]}, λ {LAMBDA_LABELS[ilam]}", fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("McClintock profile R [physical Mpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"error ratio $\sigma_{\rm analytic}/\sigma_{\rm ref}$")
    fig.suptitle(
        "Point-by-point diagonal-error ratios\n"
        "Analytic variances are log-linearly interpolated to the McClintock radial centres",
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, "mcclintock_error_ratios", output, pdf)


def _plot_fractional_errors(records: list[dict], output: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(
        N_Z, N_LAMBDA, figsize=(15, 10), sharex=True, sharey=True, squeeze=False
    )
    for record in records:
        iz, ilam = record["iz"], record["ilam"]
        ax = axes[iz, ilam]
        signal = np.abs(record["delta_sigma"])
        r = record["r_profile"]
        ax.loglog(r, record["jk_sigma"] / signal, "s-", ms=3, lw=1.2, label="McClintock JK")
        ax.loglog(r, record["sac_sigma"] / signal, "^-", ms=3, lw=1.2, label="SAC")
        ax.loglog(
            r,
            record["analytic_sigma_at_jk"] / signal,
            "o-",
            ms=3,
            lw=1.2,
            label="analytic",
        )
        ax.set_title(f"z {Z_LABELS[iz]}, λ {LAMBDA_LABELS[ilam]}", fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("R [physical Mpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"fractional error $\sigma(\Delta\Sigma)/|\Delta\Sigma_t|$")
    fig.suptitle(
        "Fractional diagonal errors using the McClintock tangential signal as denominator",
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, "mcclintock_fractional_errors", output, pdf)


def _plot_correlation_differences(records: list[dict], output: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(
        2 * N_Z,
        N_LAMBDA,
        figsize=(14, 18),
        squeeze=False,
        constrained_layout=True,
    )
    difference_matrices = []
    for record in records:
        difference_matrices.append(
            _corr(record["analytic_cov"]) - _corr(record["jk"])
        )
        difference_matrices.append(
            _corr(record["analytic_cov"]) - _corr(record["sac"])
        )
    vmax = min(0.5, max(np.max(np.abs(matrix)) for matrix in difference_matrices))
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = None
    for record in records:
        iz, ilam = record["iz"], record["ilam"]
        for row_offset, reference_name, reference in [
            (0, "JK", record["jk"]),
            (N_Z, "SAC", record["sac"]),
        ]:
            difference = _corr(record["analytic_cov"]) - _corr(reference)
            ax = axes[iz + row_offset, ilam]
            image = ax.imshow(difference, cmap="coolwarm", norm=norm, origin="lower")
            rms = np.sqrt(np.mean(difference**2))
            ax.set_title(f"z {Z_LABELS[iz]}, λ {LAMBDA_LABELS[ilam]}\n{reference_name} RMS={rms:.3f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("analytic − JK", fontsize=10)
    axes[N_Z, 0].set_ylabel("analytic − SAC", fontsize=10)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.7, label="correlation coefficient difference")
    fig.suptitle(
        "Correlation-matrix validation (same radial ordering; radial centres are not forced identical)",
        fontsize=14,
    )
    _save_figure(fig, "correlation_matrix_differences", output, pdf)


def _plot_components(inputs: dict, records: list[dict], output: Path, pdf: PdfPages) -> None:
    analytic = inputs["analytic"]
    fig, axes = plt.subplots(
        N_Z, N_LAMBDA, figsize=(15, 10), sharex=True, sharey=True, squeeze=False
    )
    for record in records:
        iz, ilam = record["iz"], record["ilam"]
        covariance = record["analytic_cov"]
        total = np.diag(covariance)
        shape = np.diag(_block(analytic["covariance_shape_noise"], iz, ilam, N_RADIAL)) / total
        cosmic = np.diag(_block(analytic["covariance_cosmic_shear"], iz, ilam, N_RADIAL)) / total
        cross = np.diag(_block(analytic["covariance_cross"], iz, ilam, N_RADIAL)) / total
        ax = axes[iz, ilam]
        r = record["analytic_r"]
        ax.semilogx(r, shape, label="shape")
        ax.semilogx(r, cosmic, label="cosmic shear")
        ax.semilogx(r, cross, label="cross")
        ax.axhline(1, color="k", lw=0.7)
        ax.set_yscale("symlog", linthresh=1e-7)
        ax.set_title(f"z {Z_LABELS[iz]}, λ {LAMBDA_LABELS[ilam]}", fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("analytic R [physical Mpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel("component / total diagonal variance")
    fig.suptitle("Analytic covariance diagonal components", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, "analytic_covariance_components", output, pdf)


def _plot_radial_grids(inputs: dict, output: Path, pdf: PdfPages) -> None:
    analytic = inputs["analytic"]
    buzzard = np.load(ROOT / "results/des_y1_buzzard_grid/covariance.npz")
    fig, axes = plt.subplots(1, N_Z, figsize=(14, 4.5), sharey=True)
    for iz, ax in enumerate(axes):
        analytic_r = analytic["radii_phys_mpc"][iz]
        profile_r = inputs["references"][iz * N_LAMBDA]["r_profile"]
        buzzard_r = buzzard["radii_phys_mpc"][iz]
        ax.semilogx(analytic_r, np.ones_like(analytic_r), "o", label="analytic 15-bin")
        ax.semilogx(profile_r, np.ones_like(profile_r) * 0.8, "s", label="McClintock 15-bin")
        ax.semilogx(buzzard_r, np.ones_like(buzzard_r) * 0.6, "^", label="Buzzard 10-bin")
        ax.set_ylim(0.4, 1.2)
        ax.set_yticks([0.6, 0.8, 1.0])
        ax.set_yticklabels(["Buzzard", "McClintock", "analytic"])
        ax.set_title(f"z {Z_LABELS[iz]}")
        ax.set_xlabel("R [physical Mpc]")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Radial-window diagnostic: points are bin centres, not interchangeable by index", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, "radial_grid_diagnostic", output, pdf)


def _plot_buzzard(y1_path: Path, output: Path, pdf: PdfPages) -> dict:
    analytic = np.load(ROOT / "results/des_y1_buzzard_grid/covariance.npz")
    buzzard_file = ROOT / "data/buzzard/dv_buzzard_jkcov.npz"
    buzzard = np.load(buzzard_file)
    buzzard_cov = np.linalg.inv(buzzard["invcov_Shear"])
    buzzard_cov = 0.5 * (buzzard_cov + buzzard_cov.T)
    ratios = []
    fig, axes = plt.subplots(
        N_Z, N_LAMBDA, figsize=(15, 10), sharex=True, sharey=True, squeeze=False
    )
    for iz in range(N_Z):
        for ilam in range(N_LAMBDA):
            analytic_cov = _block(analytic["covariance"], iz, ilam, N_BUZZARD_RADIAL)
            buzz_cov = _block(buzzard_cov, iz, ilam, N_BUZZARD_RADIAL)
            ratio = np.sqrt(np.diag(analytic_cov) / np.diag(buzz_cov))
            ratios.extend(ratio)
            r = analytic["radii_comoving_mpc_noh"][iz]
            ax = axes[iz, ilam]
            ax.semilogx(BUZZARD_R_COMOVING_MPC_H, ratio, "o-", ms=3, lw=1.2)
            ax.axhspan(0.9, 1.1, color="0.88")
            ax.axhline(1.0, color="k", lw=0.8)
            ax.axhline(0.9, color="0.4", ls="--", lw=0.7)
            ax.axhline(1.1, color="0.4", ls="--", lw=0.7)
            # robust per-panel y-limits: median +/- 2 robust sigma
            med = np.median(ratio)
            sig_rob = 0.7413 * (np.percentile(ratio, 84) - np.percentile(ratio, 16))
            ax.set_ylim(max(0.0, min(med - 2 * sig_rob, 0.85)),
                        max(med + 2 * sig_rob, 1.15))
            ax.set_xlim(0.18, 5.6)
            ax.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0])
            ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
            ax.set_title(f"z {Z_LABELS[iz]}, λ {LAMBDA_LABELS[ilam]}\nmed={np.median(ratio):.2f}", fontsize=9)
            ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].text(
        0.03,
        0.97,
        "analytic / Buzzard",
        transform=axes[0, 0].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Buzzard grid $R$ [comoving Mpc/$h$]")
    for ax in axes[:, 0]:
        ax.set_ylabel("error ratio")
    fig.suptitle(
        "Buzzard validation on the exact 10-window re-integration grid\n"
        "Interpret amplitude cautiously: the local Buzzard covariance has unresolved unit/convention provenance",
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, "buzzard_error_ratios", output, pdf)
    return {
        "median_error_ratio": float(np.median(ratios)),
        "p16_error_ratio": float(np.percentile(ratios, 16)),
        "p84_error_ratio": float(np.percentile(ratios, 84)),
        "fraction_points_outside_10_percent": float(np.mean(np.abs(np.asarray(ratios) - 1) > 0.1)),
        "source": str(buzzard_file),
        "note": "The Buzzard file is plotted separately because local provenance flags unresolved DeltaSigma/Sigma_crit and h conventions.",
    }


def make_plots(y1_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(y1_path)
    records = _records(inputs)
    metrics = {
        "inputs": {
            "analytic": str(y1_path),
            "mcclintock_profile_and_jackknife": "data/desy1_jackknife",
            "mcclintock_sac": "data/desy1_SAC",
            "radial_ratio_method": "log-linear interpolation of analytic variance to McClintock profile centres, with log-slope edge extrapolation",
        },
        "mcclintock": {"jackknife": [], "sac": []},
    }
    for record in records:
        label = {"z": Z_LABELS[record["iz"]], "lambda": LAMBDA_LABELS[record["ilam"]]}
        metrics["mcclintock"]["jackknife"].append(
            {**label, **_ratio_summary(record["jk_error_ratio"])}
        )
        metrics["mcclintock"]["sac"].append(
            {**label, **_ratio_summary(record["sac_error_ratio"])}
        )
    for reference_name in ["jackknife", "sac"]:
        record_key = "jk_error_ratio" if reference_name == "jackknife" else "sac_error_ratio"
        all_ratios = np.concatenate(
            [record[record_key] for record in records]
        )
        metrics["mcclintock"][reference_name + "_overall"] = _ratio_summary(all_ratios)
        native_key = (
            "native_index_jk_error_ratio"
            if reference_name == "jackknife"
            else "native_index_sac_error_ratio"
        )
        native_ratios = np.concatenate([record[native_key] for record in records])
        metrics["mcclintock"][reference_name + "_native_index_overall"] = _ratio_summary(
            native_ratios
        )

    with PdfPages(output / "validation_plots.pdf") as pdf:
        _plot_sigma(records, output, pdf)
        _plot_ratios(records, output, pdf)
        _plot_fractional_errors(records, output, pdf)
        _plot_correlation_differences(records, output, pdf)
        _plot_components(inputs, records, output, pdf)
        _plot_radial_grids(inputs, output, pdf)
        metrics["buzzard"] = _plot_buzzard(y1_path, output, pdf)

    metrics["notes"] = [
        "The ±10% test is applied to error ratios, not variance ratios.",
        "McClintock and analytic curves are shown on their native radial centres in the amplitude plot.",
        "Ratio plots map the analytic variance to McClintock centres; native-index aggregates are retained to expose the earlier shortcut.",
        "The SAC and jackknife matrices are compared in their native 15-bin ordering for correlation plots.",
        "The analytic covariance has no cross-lens-bin terms in this reproduction, so block off-diagonal structure is not validated here.",
    ]
    with (output / "validation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y1", type=Path, default=ROOT / "results/des_y1/covariance.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/validation_plots")
    args = parser.parse_args()
    metrics = make_plots(args.y1, args.output)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
