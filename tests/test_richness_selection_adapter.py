import numpy as np

from clens.util.richness_selection_adapter import make_intrinsic_relation


def test_hod_adapter_uses_richness_selection_mor():
    relation = make_intrinsic_relation(
        "hod",
        log10_Mmin=11.3852818,
        log10_M1=12.6964410,
        alpha=0.858693714,
        sigma_lambda=0.180949022,
        epsilon=0.283887020,
        z_pivot=0.4544,
    )
    assert type(relation.mor).__name__ == "MOR"
    assert np.isfinite(relation.pdf(30.0, 1.0e14, 0.4))
    assert relation.selection_probability(1.0e14, 0.4, 20.0, 30.0) > 0.0


def test_log_normal_adapter_is_intrinsic_only():
    relation = make_intrinsic_relation("lognormal")
    probability = relation.selection_probability(3.0e14, 0.45, 20.0, 100.0)
    assert 0.0 < probability < 1.0


def test_projected_bin_probability_uses_richness_selection_kernel():
    relation = make_intrinsic_relation(
        "hod",
        log10_Mmin=11.3852818,
        log10_M1=12.6964410,
        alpha=0.858693714,
        sigma_lambda=0.180949022,
        epsilon=0.283887020,
        z_pivot=0.4544,
        apply_projection=True,
    )
    probability = relation.selection_probability(1.0e14, 0.4, 20.0, 30.0)
    assert 0.0 < probability < 1.0


def test_joint_selection_includes_the_existing_photoz_kernel():
    relation = make_intrinsic_relation("lognormal", apply_projection=True)
    narrow = relation.joint_selection_probability(
        3.0e14, 0.4, 20.0, 100.0, 0.2, 0.35, 0.03
    )
    wide = relation.joint_selection_probability(
        3.0e14, 0.4, 20.0, 100.0, 0.2, 0.65, 0.03
    )
    assert 0.0 < narrow < wide < 1.0
