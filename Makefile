.PHONY: validation-plots plain-ratio plain-ratio-y3 hod-lognormal-comparison test freeze covariance-v2 stage-a

freeze:
	.venv/bin/python scripts/freeze_inputs.py --config configs/des_y1.json

covariance-v2:
	.venv/bin/python scripts/reproduce_covariance_v2.py --config configs/des_y1.json \
		--output output/covariance_v2 --provider frozen

stage-a:
	MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache .venv/bin/python -m pytest -q tests/test_stage_a_equivalence.py

validation-plots:
	MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache .venv/bin/python scripts/make_validation_plots.py

plain-ratio:
	MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache .venv/bin/python scripts/make_plain_physical_ratio_plot.py

plain-ratio-y3:
	MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache .venv/bin/python scripts/make_plain_physical_ratio_plot.py \
		--analytic results/des_y3_buzzard_grid/covariance.npz \
		--output-stem plain_physical_error_ratio_y3 \
		--survey-label "DES Y3 forecast"

hod-lognormal-comparison:
	MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache \
	.venv/bin/python scripts/compare_hod_lognormal_covariances.py

test:
	MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache .venv/bin/python -m pytest -q
