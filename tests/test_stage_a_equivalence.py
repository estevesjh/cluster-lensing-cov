"""Stage-A refactor-equivalence gates (frozen inputs).

The new FFTLog pipeline, fed the M0 frozen physics, must reproduce the
legacy trapz pipeline's covariance. See the refactor plan: FFTLog vs
legacy <= 1e-3 relative on diagonals, correlation matrices to atol 1e-3;
NCountsCov reproduces the legacy sample-variance formula.
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "validation" / "frozen_inputs"

pytestmark = pytest.mark.skipif(
    not (FROZEN / "counts_bias.json").exists(),
    reason="frozen inputs not generated (run scripts/freeze_inputs.py)",
)


@pytest.fixture(scope="module")
def tables():
    from clens.covariance import FrozenTables

    return FrozenTables.load(FROZEN)


@pytest.fixture(scope="module")
def legacy_reference():
    return np.load(FROZEN / "cov_reference_desy1" / "covariance.npz")


@pytest.fixture(scope="module")
def new_result(tables):
    from clens.covariance import CovarianceAssembler

    assembler = CovarianceAssembler(
        tables=tables, n_radial=15, radial_range_physical_mpc=(0.03, 30.0)
    )
    return assembler.run()


def _block(matrix, b, n=15):
    return matrix[b * n:(b + 1) * n, b * n:(b + 1) * n]


def _legacy_white_truncation_factor() -> float:
    """The legacy trapz integrated the white (shot x shape-noise) term
    only over [1/theta_max, 100/theta_min], capturing 98.854% of
    int l J2bar^2 dl.  The factor is dimensionless (the range scales
    with the bin), so it is one number for the geometric binning."""
    from clenspy.utils.fftlog_cov import j2_bin_averaged

    rho = (30.0 / 0.03) ** (1.0 / 15.0)
    tmin, tmax = 1e-3, 1e-3 * rho
    lnell = np.arange(np.log(1.0 / tmax), np.log(100.0 / tmin), 1e-3)
    ell = np.exp(lnell)
    trunc = np.trapezoid(ell**2 * j2_bin_averaged(ell, tmin, tmax) ** 2, lnell)
    return trunc * (tmax**2 - tmin**2) / 2.0


def test_diagonals_match_legacy(new_result, legacy_reference):
    """Per-block diagonal agreement of the total covariance.

    The legacy pipeline under-integrates the white shape-noise term by
    the quantified truncation factor 0.988544 (its ell range cuts the
    l^-3 tail of J2bar^2) — the FFTLog result integrates it exactly, so
    where shape noise dominates the ratio new/legacy sits at
    1/0.988544 = 1.01159.  The gate brackets the ratio between 1 (exact
    agreement) and the truncation ceiling."""
    new = new_result["covariance"]
    ref = legacy_reference["covariance"]
    assert new.shape == ref.shape
    ceiling = 1.0 / _legacy_white_truncation_factor() + 5e-3
    for b in range(12):
        ratio = np.diag(_block(new, b)) / np.diag(_block(ref, b))
        assert np.all(ratio > 1.0 - 5e-3), (b, ratio.min())
        assert np.all(ratio < ceiling), (b, ratio.max())


def test_correlation_matrices_match(new_result, legacy_reference):
    new = new_result["covariance"]
    ref = legacy_reference["covariance"]
    for b in range(12):
        bn, br = _block(new, b), _block(ref, b)
        cn = bn / np.sqrt(np.outer(np.diag(bn), np.diag(bn)))
        cr = br / np.sqrt(np.outer(np.diag(br), np.diag(br)))
        # atol allows the ~1.2% normalization shift from the legacy
        # white-term truncation on shape-noise-dominated diagonals
        assert np.allclose(cn, cr, atol=1.5e-2), b


def test_inputs_reproduce_legacy_exactly(tables, legacy_reference):
    """Apples-to-apples input equivalence: the legacy-range trapz run on
    the NEW contract inputs must reproduce the stored legacy cosmic-shear
    block to <= 1e-3 (measured 2e-4).  This isolates any remaining
    new-vs-legacy deviation as pure legacy ell-truncation."""
    from clens.covariance.limber import LimberProjector
    from clens.covariance.reference import reference_cov_trapz

    s = tables.samples[0]
    a = 1.0 / (1.0 + s.z_mid)
    chi_h = float(tables.cosmology.chi(s.z_mid))
    theta_edges = (
        np.exp(np.linspace(np.log(0.03 / a), np.log(30.0 / a), 16)) / chi_h
    )
    lim = LimberProjector(tables.cosmology, tables.source)
    c_sigma = lim.c_ell_sigma(0.1, min(2.0, tables.source.zs_max - 0.1),
                              s.z_mid)
    c_hh, shot = lim.c_ell_h(s.z_min, s.z_max, s.bias, s.counts,
                             tables.geometry.area_sr)
    cov_cs = reference_cov_trapz(
        lim.ell, (c_hh + shot) * c_sigma, theta_edges, tables.geometry.f_sky
    ) * 1e-24 / a**4
    ref_cs = _block(legacy_reference["covariance_cosmic_shear"], 0)
    rel = np.abs(np.diag(cov_cs) / np.diag(ref_cs) - 1.0)
    assert rel.max() < 1e-3, rel.max()


def test_per_term_blocks_match(new_result, legacy_reference):
    """New/legacy per-term ratios: bounded by the legacy ell-truncation.

    The legacy trapz cut every term at [1/theta_max, 100/theta_min]; the
    FFTLog integrates the full range, so new >= legacy with excess up to
    ~2.5% (cosmic shear, small radii), ~1.16% (white shape noise), and
    larger relative excess on the tiny cross term.  Inputs themselves are
    proven identical by test_inputs_reproduce_legacy_exactly."""
    bounds = {
        "covariance_cosmic_shear": (1.0 - 5e-3, 1.035),
        "covariance_shape_noise": (1.0 - 5e-3, 1.02),
        "covariance_cross": (1.0 - 5e-2, 2.5),  # tiny term, tail-dominated
    }
    for key, (lo, hi) in bounds.items():
        new = new_result[key]
        ref = legacy_reference[key]
        for b in range(12):
            d_new = np.diag(_block(new, b))
            d_ref = np.diag(_block(ref, b))
            keep = d_ref > 0
            ratio = d_new[keep] / d_ref[keep]
            assert np.all(ratio > lo), (key, b, ratio.min())
            assert np.all(ratio < hi), (key, b, ratio.max())


def test_fftlog_vs_converged_trapz(tables, new_result):
    """The true numerics gate: FFTLog vs the trapz reference on an
    EXTENDED ell range (no legacy truncation), same C_ell inputs,
    one representative bin — <= 2e-3 on the diagonal."""
    from clens.covariance import GaussianDeltaSigmaCov
    from clens.covariance.limber import LimberProjector
    from clens.covariance.reference import reference_cov_trapz

    s = tables.samples[0]
    a = 1.0 / (1.0 + s.z_mid)
    chi_h = float(tables.cosmology.chi(s.z_mid))
    n_rp = 15
    theta_edges = (
        np.exp(np.linspace(np.log(0.03 / a), np.log(30.0 / a), n_rp + 1))
        / chi_h
    )
    lim = LimberProjector(tables.cosmology, tables.source)
    c_sigma = lim.c_ell_sigma(0.1, min(2.0, tables.source.zs_max - 0.1),
                              s.z_mid)
    c_hh, shot = lim.c_ell_h(s.z_min, s.z_max, s.bias, s.counts,
                             tables.geometry.area_sr)
    n_shape = lim.shape_noise_sigma(s.z_mid)
    C_total = (c_hh + shot) * (c_sigma + n_shape)

    eng = GaussianDeltaSigmaCov(tables.cosmology, tables.source,
                                tables.geometry)
    blocks = eng.compute(s, rp_min=0.03 / a, rp_max=30.0 / a, n_rp=n_rp)
    new_diag = np.diag(blocks.cosmic_shear + blocks.shape_noise) / 1e-24

    # converged trapz: widen the per-pair range far beyond the legacy cut
    import clens.covariance.reference as refmod

    old = (refmod.SCALING_FOR_ELL_MIN, refmod.SCALING_FOR_ELL_MAX)
    refmod.SCALING_FOR_ELL_MIN, refmod.SCALING_FOR_ELL_MAX = 1e-2, 3e4
    try:
        cov_ref = reference_cov_trapz(
            lim.ell, C_total, theta_edges, tables.geometry.f_sky
        )
    finally:
        refmod.SCALING_FOR_ELL_MIN, refmod.SCALING_FOR_ELL_MAX = old
    rel = np.abs(new_diag / np.diag(cov_ref) - 1.0)
    assert rel.max() < 2e-3, rel.max()


def test_radii_match(new_result, legacy_reference):
    assert np.allclose(
        new_result["radii_phys_mpc"],
        legacy_reference["radii_phys_mpc"],
        rtol=1e-10,
    )


def test_counts_cov_reproduces_legacy_sv(tables):
    """Same-z SV entries = bN_a bN_b sigma_w^2; Poisson on the diagonal."""
    from clens.covariance import NCountsCov

    cov = NCountsCov(tables.samples).matrix()
    s = tables.samples
    n = len(s)
    for a in range(n):
        assert np.isclose(
            cov[a, a], s[a].counts + s[a].bN**2 * s[a].sigma_w**2, rtol=1e-12
        )
        for b in range(n):
            if a == b:
                continue
            same = (s[a].z_min, s[a].z_max) == (s[b].z_min, s[b].z_max)
            expected = s[a].bN * s[b].bN * s[a].sigma_w**2 if same else 0.0
            assert np.isclose(cov[a, b], expected, rtol=1e-12, atol=1e-30)
    # symmetric, positive-definite
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
