# Cluster Lensing 

This package computes the cluster lensing signal and covariance matrices.  Both in terms of $\gamma_t$ and $\Delta\Sigma$.

<!-- Put this in your `~/.bashrc`

`export PYTHONPATH="/your_folder/cluster-lensing-cov/:$PYTHONPATH"` -->

## Installation
Run `pip install git+https://github.com/hywu/cluster-lensing-cov`

### Development installation
If you'd like to run modified version of the code or contribute to the repo. Please follow the following steps: 

Run `git clone https://github.com/hywu/cluster-lensing-cov`

Then run `pip install -e cluster-lensing-cov`

## Contents

* `clens/lensing/` contains the main code for computing lensing signals and covariance matrices.  

	* `lensing_profiles.py` calculates $\gamma_t$ and $\Delta\Sigma$ profiles assuming NFW

	* `cov_DeltaSigma.py` and `cov_gammat.py` calculate the covariance matrices  

* `clens/util/` contains classes for cosmology, nuisance parameters, survey conditions, etc.

* `clens/ying/` contains code copied from Ying Zu's package

* `tests/` contains unit tests.

* `examples/` 
	* `demo_analytic.py` shows the one example for 
	* `abacus_analytic_grafting.py` shows how to graft analytic and abacus together 


## Pre-computed covariance matrices

* `output/analytic_abacus_scatter*/zh*_zs*/*_*_R*_*_nrp*/` includes various components of covariance matrices.
	* `DeltaSigma_cov_combined.dat` is everything without shape noise
	* `DeltaSigma_cov_shape_noise.dat` is for nsrc = 10 arcmin^{-2}
	*  Shape noise is inversely proportional to nsrc

* For example, `output/analytic_abacus_scatter0/zh0.5_zs1.25/1e+14_1e+16_R0.1_100_nrp15/` contains zero scatter, zh=0.5, zs=1.25, M between 1e14 and 1e16 Msun/h, 15 log rp bins between 0.1 and 100 Mpc/h

## DES Y1/Y3 reproduction

The reproducible driver in `scripts/reproduce_covariance.py` evaluates the
Wu et al. (2019) `CovDeltaSigma` calculation for the 12 DES lens bins
(three redshift bins and four richness bins), then assembles the per-bin
15-by-15 blocks into a 180-by-180 matrix. The matrix ordering is z-major,
richness-fast, radial-fast. Off-block terms are zero because the upstream
implementation calculates one lens bin at a time.

The paper acquired through `mcp-ads-arxiv` is stored in `papers/` as
`Wu_2019MNRAS_490_2606.pdf` and `Wu_2019MNRAS_490_2606.tex`.

Using the project-local environment:

```bash
MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache \
  .venv/bin/python scripts/reproduce_covariance.py \
  --config configs/des_y1.json --output results/des_y1

MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache \
  .venv/bin/python scripts/reproduce_covariance.py \
  --config configs/des_y3.json --output results/des_y3

MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache \
  .venv/bin/python scripts/compare_covariances.py
```

The transferred Buzzard file is
`data/buzzard/dv_buzzard_jkcov.npz`. Its `invcov_Shear` array is an inverse
covariance and is inverted by the comparison script before use. The Buzzard
comparison also produces `results/des_y1_buzzard_grid/`, an analytic
re-integration on inferred logarithmic windows surrounding its 10-radius
comoving grid.

The DES Y1 configuration uses the counts, biases, source distribution, and
area in the upstream validation. The DES Y3 result is explicitly a forecast:
it uses 5000 deg², 5.59 sources arcmin⁻², sigma_gamma=0.261, and scales the
Y1 counts by 5000/1437 because no DES Y3 counts or source n(z) were supplied.
Those assumptions are recorded in `configs/des_y3.json` and
`results/des_y3/metadata.json` and must be replaced for a literal DES Y3
measurement.

### Intrinsic HOD versus log-normal selection

`github/RichnessSelection` is the source of the analytical mass--richness
functions. For the model comparison, the adapter uses its `MOR.pdf` or
`LogNormalMOR.pdf`, folds in the existing `K_i` projection kernel to obtain
`S_i(M,z)`, and multiplies by the existing `K_j` photo-z kernel to obtain
`S_ij(M,z)`. The outer mass/redshift integral then contracts this selection
into the counts and effective bias consumed by the covariance. The recovered
DES-Y1 HOD parameters, including `epsilon=0.283887020`, are recorded in the
comparison script. The comparison explicitly uses `sigma_z=0.03`, matching
the RichnessSelection validation configuration; change that input when a
bin-dependent DES calibration is available.

Generate the two DES-Y1 covariance matrices and their diagonal-error
comparison with:

```bash
make hod-lognormal-comparison
```

Outputs are written below `results/des_y1_hod_lognormal/` as `hod/`,
`lognormal/`, `comparison_summary.json`, and
`hod_vs_lognormal_diagonal_errors.png`/`.pdf`.

### Validation plots

After generating the DES Y1 covariance, make the full validation suite with:

```bash
make validation-plots
```

This writes `results/validation_plots/validation_plots.pdf`, individual PNGs,
and `validation_metrics.json`. The diagonal-error comparison reports the
fraction of McClintock radial points outside a ±10% error band. The analytic
and McClintock curves retain their native radial centres; ratio plots
log-linearly interpolate the analytic variance to the McClintock centres.
The radial-grid diagnostic and the separate Buzzard plot make this mapping
explicit rather than treating different annular windows as identical by
array index. Edge points outside the analytic range use a documented
log-slope extrapolation; exact annular-window re-integration is the next step
if the mapped comparison remains discrepant.
