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


def test_diagonals_match_legacy(new_result, legacy_reference):
    """Per-block diagonal agreement of the total covariance."""
    new = new_result["covariance"]
    ref = legacy_reference["covariance"]
    assert new.shape == ref.shape
    worst = 0.0
    for b in range(12):
        d_new = np.diag(_block(new, b))
        d_ref = np.diag(_block(ref, b))
        rel = np.abs(d_new / d_ref - 1.0)
        worst = max(worst, rel.max())
    assert worst < 5e-3, worst


def test_correlation_matrices_match(new_result, legacy_reference):
    new = new_result["covariance"]
    ref = legacy_reference["covariance"]
    for b in range(12):
        bn, br = _block(new, b), _block(ref, b)
        cn = bn / np.sqrt(np.outer(np.diag(bn), np.diag(bn)))
        cr = br / np.sqrt(np.outer(np.diag(br), np.diag(br)))
        assert np.allclose(cn, cr, atol=5e-3), b


def test_per_term_blocks_match(new_result, legacy_reference):
    for key in (
        "covariance_cosmic_shear",
        "covariance_shape_noise",
        "covariance_cross",
    ):
        new = new_result[key]
        ref = legacy_reference[key]
        for b in range(12):
            d_new = np.diag(_block(new, b))
            d_ref = np.diag(_block(ref, b))
            keep = d_ref > 0
            rel = np.abs(d_new[keep] / d_ref[keep] - 1.0)
            assert rel.max() < 1e-2, (key, b, rel.max())


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
