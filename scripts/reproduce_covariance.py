#!/usr/bin/env python3
"""Reproduce the Wu et al. (2019) analytic DeltaSigma covariance.

The upstream implementation evaluates one lens redshift/richness bin at a
time.  This driver assembles those 15x15 blocks into the z-major, richness-
fast ordering used by the DES validation data.  Cross-bin terms are left zero
because the upstream API has no cross-bin covariance calculation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from clens.lensing.cov_DeltaSigma import CovDeltaSigma
from clens.util.cluster_counts import ClusterCounts
from clens.util.parameters import CosmoParameters
from clens.util.richness_selection_adapter import make_intrinsic_relation
from clens.util.scaling_relation import PrecalculatedCountsBias
from clens.util.survey import Survey


ROOT = Path(__file__).resolve().parents[1]
BUZZARD_R_COMOVING_MPC_H = np.array(
    [0.200, 0.286, 0.409, 0.585, 0.836, 1.196, 1.710, 2.445, 3.497, 5.000]
)


def _make_richness_selection_two_halo(config, selection):
    """Build the existing RichnessSelection TwoHalo b_sel_ls calculator."""
    root = Path(
        selection.get("richness_selection_root")
        or os.environ.get(
            "RICHNESS_SELECTION_ROOT",
            ROOT.parent / "github" / "RichnessSelection",
        )
    )
    source_root = root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from richness_selection import (  # pylint: disable=import-outside-toplevel
        Bias,
        Cosmology,
        HMF,
        MOR,
        PkGrid,
        SelBias,
        TwoHalo,
        XiNL,
    )
    from richness_selection.mor import (  # pylint: disable=import-outside-toplevel
        LogNormalMOR,
    )
    from richness_selection.sigma_m import (  # pylint: disable=import-outside-toplevel
        SigmaM,
    )

    cosmology = config["cosmology"]
    rs_cosmo = Cosmology(
        Om0=float(cosmology["OmegaM"]),
        H0=100.0 * float(cosmology["h"]),
        sigma8=float(cosmology["sigma8"]),
    )
    pk = PkGrid(rs_cosmo)
    sigma_m = SigmaM(pk)
    hmf = HMF(sigma_m)
    halo_bias = Bias(sigma_m)
    parameters = dict(selection.get("parameters", {}))
    model = str(selection["model"]).lower()
    if model in {"hod", "mor", "costanzi_hod"}:
        if "sigma_lambda" in parameters:
            parameters["sigma_intr"] = parameters.pop("sigma_lambda")
        mor = MOR(**parameters)
    elif model in {"lognormal", "log_normal", "logn"}:
        mor = LogNormalMOR(**parameters)
    else:
        raise ValueError(f"unknown intrinsic richness model: {model}")
    xi_nl = XiNL(rs_cosmo)
    sel_bias = SelBias(rs_cosmo, pk, hmf, halo_bias, mor, xi_nl=xi_nl)
    return TwoHalo(rs_cosmo, sel_bias)


def _bsel_cache_key(lam_min, lam_max, zmin, zmax, open_bin_lob):
    return "|".join(
        f"{float(value):.12g}"
        for value in (lam_min, lam_max, zmin, zmax, open_bin_lob)
    )


def _bin_bsel_ls(two_halo, lam_min, lam_max, zmin, zmax, open_bin_lob,
                 cache=None):
    """Evaluate and cache RichnessSelection's b_sel_ls for one bin."""
    lob = float(open_bin_lob) if float(lam_max) >= 999.0 else 0.5 * (
        float(lam_min) + float(lam_max)
    )
    zob = 0.5 * (float(zmin) + float(zmax))
    if cache is None:
        cache = getattr(two_halo, "_clens_bsel_ls_cache", None)
        if cache is None:
            cache = {}
            setattr(two_halo, "_clens_bsel_ls_cache", cache)
    key = _bsel_cache_key(lam_min, lam_max, zmin, zmax, open_bin_lob)
    if key not in cache:
        if two_halo is None:
            raise ValueError("missing TwoHalo calculator for uncached b_sel_ls")
        cache[key] = float(two_halo.b_sel_ls(lob, zob))
    return float(cache[key])


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "survey_area_deg2",
        "n_src_arcmin2",
        "sigma_gamma",
        "source_pz",
        "cosmology",
        "z_bins",
        "lambda_bins",
        "radial_range_physical_mpc",
        "n_radial",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    if ("counts" not in config) != ("bias" not in config):
        raise ValueError("counts and bias must be supplied together")
    if "selection" not in config and "counts" not in config:
        raise ValueError("config needs counts/bias or an intrinsic selection block")
    return config


def _counts_and_bias(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolve lens counts/bias from config or the intrinsic MOR model."""
    if "selection" not in config:
        return (np.asarray(config["counts"], dtype=float),
                np.asarray(config["bias"], dtype=float),
                {"model": "precalculated"})

    selection = config["selection"]
    relation = make_intrinsic_relation(
        model=selection["model"],
        root=selection.get("richness_selection_root"),
        quadrature_order=selection.get("quadrature_order", 64),
        apply_projection=selection.get("apply_projection", False),
        ltr_bracket_sigma=selection.get("ltr_bracket_sigma", 6.0),
        **selection.get("parameters", {}),
    )
    cosmo = CosmoParameters(**config["cosmology"])
    bsel_config = selection.get("bsel_ls", {})
    two_halo = (
        _make_richness_selection_two_halo(config, selection)
        if bsel_config.get("applied", False)
        else None
    )
    photoz_sigma = selection.get("photoz_sigma")
    if photoz_sigma is None:
        raise ValueError(
            "selection.photoz_sigma is required for the full S_ij path; "
            "provide one value per richness bin (or a scalar)"
        )
    if np.ndim(photoz_sigma) == 0:
        photoz_sigma = [float(photoz_sigma)] * len(config["lambda_bins"])
    else:
        photoz_sigma = [float(value) for value in photoz_sigma]
    if len(photoz_sigma) != len(config["lambda_bins"]):
        raise ValueError(
            "selection.photoz_sigma must be scalar or have one value per "
            "richness bin"
        )
    if any(value <= 0.0 for value in photoz_sigma):
        raise ValueError("selection.photoz_sigma values must be positive")
    counts = np.zeros((len(config["z_bins"]), len(config["lambda_bins"])))
    bias = np.zeros_like(counts)
    for iz, (zmin, zmax) in enumerate(config["z_bins"]):
        for ilam, (lam_min, lam_max) in enumerate(config["lambda_bins"]):
            cc = ClusterCounts(cosmo_parameters=cosmo, scaling_relation=relation)
            result = cc.calc_counts_full(
                zmin=zmin,
                zmax=zmax,
                lambda_min=lam_min,
                lambda_max=lam_max,
                survey_area_sq_deg=config["survey_area_deg2"],
                sigma_z=photoz_sigma[ilam],
                n_mass=selection.get("outer_mass_nodes", 32),
                n_redshift=selection.get("outer_redshift_nodes", 24),
                ltr_bracket_sigma=selection.get("ltr_bracket_sigma", 6.0),
            )
            counts[iz, ilam] = result[0]
            bias[iz, ilam] = result[2] if two_halo is None else _bin_bsel_ls(
                two_halo,
                lam_min,
                lam_max,
                zmin,
                zmax,
                bsel_config.get("open_bin_lob", 80.0),
            )
    provenance = {
        "model": selection["model"],
        "parameters": selection.get("parameters", {}),
        "quadrature_order": selection.get("quadrature_order", 64),
        "photoz_sigma": photoz_sigma,
        "outer_mass_nodes": selection.get("outer_mass_nodes", 32),
        "outer_redshift_nodes": selection.get("outer_redshift_nodes", 24),
        "projection_kernel": (
            "applied through RichnessSelection K_i; redshift through K_j"
            if selection.get("apply_projection", False)
            else "intrinsic P(lambda_true|M,z), redshift through K_j"
        ),
        "projection": selection.get("projection", {
            "applied": False,
            "reason": "intrinsic HOD/log-normal richness path",
        }),
        "large_scale_selection_bias": {
            "applied": bool(two_halo is not None),
            "definition": "existing RichnessSelection TwoHalo.b_sel_ls",
            "evaluation": "marginalized over lambda_tr at 30 cMpc/h",
            "representative_observed_richness": (
                "bin midpoint; open upper bin uses configured open_bin_lob"
            ),
            "open_bin_lob": bsel_config.get("open_bin_lob", 80.0),
        },
    }
    return counts, bias, provenance


def _survey(config: dict) -> Survey:
    pz = config["source_pz"]
    if pz.get("model") != "whale":
        raise ValueError("only the upstream whale-shaped source p(z) is supported")
    return Survey(
        z_star_src=pz["z_star"],
        m_src=pz["m"],
        beta_src=pz["beta"],
        n_src_arcmin=config["n_src_arcmin2"],
        sigma_gamma=config["sigma_gamma"],
    )


def _radial_grid(config: dict, mode: str) -> tuple[float, float, int, np.ndarray]:
    if mode == "buzzard":
        ratio = BUZZARD_R_COMOVING_MPC_H[1] / BUZZARD_R_COMOVING_MPC_H[0]
        edges = np.geomspace(
            BUZZARD_R_COMOVING_MPC_H[0] / np.sqrt(ratio),
            BUZZARD_R_COMOVING_MPC_H[-1] * np.sqrt(ratio),
            len(BUZZARD_R_COMOVING_MPC_H) + 1,
        )
        return edges[0], edges[-1], len(BUZZARD_R_COMOVING_MPC_H), BUZZARD_R_COMOVING_MPC_H
    if mode != "paper":
        raise ValueError(f"unknown radial grid: {mode}")
    rmin, rmax = config["radial_range_physical_mpc"]
    nrad = int(config["n_radial"])
    if not (0 < rmin < rmax and nrad > 0):
        raise ValueError("radial_range_physical_mpc and n_radial are invalid")
    edges = np.geomspace(rmin, rmax, nrad + 1)
    return rmin, rmax, nrad, np.sqrt(edges[:-1] * edges[1:])


def reproduce(config: dict, output_dir: Path, radial_grid: str = "paper") -> dict:
    cosmo = CosmoParameters(**config["cosmology"])
    survey = _survey(config)
    rmin_grid, rmax_grid, nrad, requested_grid = _radial_grid(config, radial_grid)
    z_bins = config["z_bins"]
    lambda_bins = config["lambda_bins"]
    counts, bias, selection_provenance = _counts_and_bias(config)
    expected_shape = (len(z_bins), len(lambda_bins))
    if counts.shape != expected_shape or bias.shape != expected_shape:
        raise ValueError(
            f"counts/bias shapes must be {expected_shape}; got "
            f"{counts.shape} and {bias.shape}"
        )
    if np.any(counts <= 0) or np.any(bias <= 0):
        raise ValueError("counts and bias must be positive")

    nblocks = len(z_bins) * len(lambda_bins)
    matrix_shape = (nblocks * nrad, nblocks * nrad)
    cov_comoving = np.zeros(matrix_shape)
    cov_cosmic = np.zeros(matrix_shape)
    cov_shape = np.zeros(matrix_shape)
    cov_cross = np.zeros(matrix_shape)
    radii_phys = np.zeros((len(z_bins), nrad))
    radii_comoving = np.zeros((len(z_bins), nrad))
    labels = []
    block_records = []

    for iz, (zmin, zmax) in enumerate(z_bins):
        zmid = 0.5 * (zmin + zmax)
        scale_factor = 1.0 / (1.0 + zmid)
        if radial_grid == "paper":
            # The upstream demo converts the requested physical Mpc range to
            # the no-h comoving range consumed by CovDeltaSigma.
            rp_min_noh = rmin_grid / scale_factor
            rp_max_noh = rmax_grid / scale_factor
        else:
            # Buzzard publishes comoving Mpc/h radii. CovDeltaSigma consumes
            # no-h comoving Mpc, so only divide by h here.
            rp_min_noh = rmin_grid / cosmo.h
            rp_max_noh = rmax_grid / cosmo.h

        for ilam, (lam_min, lam_max) in enumerate(lambda_bins):
            block = iz * len(lambda_bins) + ilam
            start = block * nrad
            stop = start + nrad
            label = f"z{iz}_lambda{ilam}"
            labels.append(label)

            sr = PrecalculatedCountsBias(counts[iz, ilam], bias[iz, ilam])
            cds = CovDeltaSigma(
                co=cosmo,
                su=survey,
                sr=sr,
                fsky=config["survey_area_deg2"] / 41253.0,
                survey_area_sq_deg=config["survey_area_deg2"],
            )
            rp_mid, _, _ = cds.calc_cov(
                rp_min=rp_min_noh,
                rp_max=rp_max_noh,
                n_rp=nrad,
                zh_min=zmin,
                zh_max=zmax,
                lambda_min=lam_min,
                lambda_max=lam_max,
                diag_only=False,
            )

            # Match the upstream validation convention: DeltaSigma is
            # converted from comoving to physical surface density by a^-2,
            # hence the covariance by a^-4.
            block_cosmic = cds.cov_cosmic_shear / scale_factor**4
            block_shape = cds.cov_shape_noise / scale_factor**4
            block_cross = cds.cov_cross / scale_factor**4
            block_total = cds.cov_sum / scale_factor**4
            for target, source in (
                (cov_cosmic, block_cosmic),
                (cov_shape, block_shape),
                (cov_cross, block_cross),
                (cov_comoving, cds.cov_sum),
            ):
                target[start:stop, start:stop] = source

            radii_comoving[iz] = rp_mid
            radii_phys[iz] = rp_mid * scale_factor
            block_records.append(
                {
                    "label": label,
                    "z_min": zmin,
                    "z_max": zmax,
                    "lambda_min": lam_min,
                    "lambda_max": lam_max,
                    "counts": counts[iz, ilam],
                    "bias": bias[iz, ilam],
                    "scale_factor": scale_factor,
                    "diagonal_shape_noise": np.diag(block_shape).tolist(),
                    "diagonal_cosmic_shear": np.diag(block_cosmic).tolist(),
                    "diagonal_cross": np.diag(block_cross).tolist(),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    covariance = cov_cosmic + cov_shape + cov_cross
    np.savez_compressed(
        output_dir / "covariance.npz",
        covariance=covariance,
        covariance_cosmic_shear=cov_cosmic,
        covariance_shape_noise=cov_shape,
        covariance_cross=cov_cross,
        covariance_comoving=cov_comoving,
        radii_phys_mpc=radii_phys,
        radii_comoving_mpc_noh=radii_comoving,
        requested_radii_grid=requested_grid,
        block_labels=np.asarray(labels),
    )
    np.savetxt(output_dir / "covariance.txt", covariance)
    np.savetxt(output_dir / "covariance_shape_noise.txt", cov_shape)
    np.savetxt(output_dir / "covariance_cosmic_shear.txt", cov_cosmic)
    np.savetxt(output_dir / "covariance_cross.txt", cov_cross)
    np.savetxt(output_dir / "radii_phys_mpc.txt", radii_phys[0])
    metadata = {
        "config": config,
        "matrix_shape": list(covariance.shape),
        "radial_grid": radial_grid,
        "ordering": "z-major, richness-fast, radial-fast",
        "units": "(M_sun / pc^2)^2; physical DeltaSigma",
        "cross_bin_covariance": "not implemented by upstream CovDeltaSigma; off-block terms are zero",
        "selection_provenance": selection_provenance,
        "block_records": block_records,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"wrote {output_dir / 'covariance.npz'}: {covariance.shape}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radial-grid", choices=("paper", "buzzard"), default="paper")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    reproduce(_load_config(config_path), output_dir, radial_grid=args.radial_grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
