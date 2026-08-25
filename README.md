# Cluster Lensing Covariance

Analytical covariance matrices for stacked cluster weak lensing
($\Delta\Sigma_{ij}$) and binned cluster counts ($N_{ij}$).

This repository holds **only the covariance layer**. All cluster-cosmology
model ingredients — mass function, mass-observable relations (HOD /
lognormal), richness selection function $S_{ij}(\ln M, z)$, selection bias
$b_{\rm sel}(\theta)$, binned observables, the $\Delta\Sigma_{\rm 1h2hMax}$
profile, and the FFTLog covariance transform — live in
[CLensPy](../CLensPy) (`clenspy`) and enter through the narrow dataclass
contract in `clens/covariance/inputs.py`.

The $\ell$-integration of the Gaussian $\Delta\Sigma$ covariance runs
through an exact FFTLog evaluation (one transform per radial-diagonal
offset, summed Mellin kernels, analytic white-noise diagonal) — the
derivation is in `CLensPy/docs/covariance_fftlog_math.md`. The historical
trapz-over-$\ln\ell$ survives only as a test reference
(`clens/covariance/reference.py`).

## Installation

```bash
pip install -e ../CLensPy   # clenspy (model layer + FFTLog engine)
pip install -e .
```

## Contents

* `clens/covariance/` — the package:
  * `inputs.py` — the clenspy-facing contract (`CosmologyInputs`,
    `SourceInputs`, `LensSample`, `SurveyGeometry`) with two providers:
    `FrozenTables` (M0 snapshots, Stage-A) and `from_clenspy` (live
    CAMB/clenspy physics, Stage-B).
  * `limber.py` — Limber $C_\ell^{\Sigma\Sigma}, C_\ell^{hh},
    C_\ell^{h\Sigma}$ + noise levels.
  * `gaussian_delta_sigma.py` — the FFTLog Gaussian $\Delta\Sigma$
    covariance (cosmic shear, shape noise, halo–$\Sigma$ cross terms).
  * `counts_cov.py` — $N_{ij}$ covariance: Poisson diagonal + same-$z$
    sample-variance blocks $\langle bN\rangle_a \langle bN\rangle_b
    \sigma_W^2$.
  * `assemble.py` — z-major, richness-fast block assembly (legacy npz
    schema preserved) + counts covariance.
  * `reference.py` — legacy trapz (tests only).
* `clens/lensing/lensing_kernel.py` — lensing-kernel ingredients
  ($q_\Sigma$, $\langle\Sigma_{\rm crit}\rangle$, $f_{\rm src}$); slated
  to move into clenspy.
* `clens/util/` — cosmology/survey parameter containers.
* `scripts/reproduce_covariance.py` — the driver: JSON config →
  `covariance.npz` (180×180 for DES Y1) + counts covariance.
  `--provider frozen` reproduces the legacy physics exactly;
  `--provider clenspy` uses live CAMB + clenspy (config `selection`
  blocks resolved through `clenspy.clusters.BinnedClusterModel`, optional
  `bsel_ls` through `SelBiasEngine`).
* `validation/frozen_inputs/` — the M0 freeze: legacy EH98 $P(k,z)$,
  kernels, counts/bias, and the legacy-pipeline covariance reference used
  by the Stage-A refactor-equivalence gates.
* `tests/test_stage_a_equivalence.py` — the gates (see below).

## Make targets

```bash
make stage-a                  # refactor-equivalence gates (frozen inputs)
make covariance-v2            # frozen-provider covariance end-to-end
make hod-lognormal-comparison # HOD vs lognormal through live clenspy
make validation-plots         # plotting (npz-based, unchanged)
make plain-ratio              # analytic vs Buzzard jackknife
make test                     # full pytest suite
```

## Validation status

Stage-A (frozen inputs, identical physics): the FFTLog pipeline
reproduces the legacy covariance with all deviations **quantified as
legacy defects** — the legacy trapz truncated every term at
$\ell \in [1/\theta_{\max}, 100/\theta_{\min}]$, losing 1.146% of the
white shape-noise integral (exact factor 0.988544) and up to ~2.5% of
the cosmic-shear tail at small radii. The legacy-range trapz run on the
new inputs matches the stored legacy blocks to $2\times10^{-4}$; the
FFTLog matches a converged trapz to $\le 2\times10^{-3}$.

## Provenance

Forked from [hywu/cluster-lensing-cov](https://github.com/hywu/cluster-lensing-cov)
(Wu et al. 2019, MNRAS 490, 2606). The pre-refactor implementation
(including `clens/ying/`, `cov_DeltaSigma.py`, `cov_gammat*.py`) is
available in the git history — see the "M0 freeze" commit.
