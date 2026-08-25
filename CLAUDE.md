# cluster-lensing-cov

Covariance-only repository: analytical covariance of ΔΣ_ij and N_ij.
All model ingredients (mass function, MOR, S_ij selection tables, b_sel,
ΔΣ_1h2hMax, Limber C_ℓ's, FFTLog covariance engine) live in
[CLensPy](../CLensPy) (`clenspy`, branch `codex/clusters`) and enter
through the contract in `clens/covariance/inputs.py`.

- Run gates before touching covariance math: `make stage-a`
  (frozen-input equivalence vs the legacy pipeline; snapshots in
  `validation/frozen_inputs/`, legacy tree at the "M0 freeze" commit).
- Drivers: `scripts/reproduce_covariance.py` (`--provider frozen|clenspy`,
  `--radial-grid paper|buzzard`). Buzzard comparisons MUST use
  `configs/buzzard.json` (Tan Xing's exact mock HOD, ε=0.283887020, and
  the Buzzard fiducial cosmology) — never the DES Y1 tables.
- FFTLog math: `CLensPy/docs/covariance_fftlog_math.md`. Mellin kernels
  disk-cached (`CLENSPY_FFTLOG_CACHE`).
- Units: physical Msun/Mpc everywhere; published HOD/MOR pivots are
  Msun/h — convert via the `HodParams`/`LogNormalParams` classmethods.

## Next steps

1. **Implement the Gruen et al. (2015) correlated-halo covariance
   template `C^corr` and apply the simulation-calibrated rescale
   `c_corr = 5.0 ± 0.3`** (their eq. 6; ellipticity analog
   `c_ell = 3.7 ± 0.3`). Structure: secondary halos with
   `dP_c = b(M_cl) b(M) (dn/dM) W(θ, z_cl) 2πθ dθ dM`, each contributing
   a truncated (BMO-like) ΔΣ profile at offset θ; independent-Poisson
   occupancy gives `C^corr_ij ∝ ∫ dP_c ΔΣ_i ΔΣ_j / N_cl`, then ×c_corr
   for the non-Gaussian/filament boost. The θ-quadrature machinery of
   `clenspy.clusters.selection_bias.SelBiasEngine._P_operator` and the
   miscentered profiles in `clenspy.lensing.miscentering` are most of
   what it needs. Home: `clenspy.clusters` (new module), wired like
   `IntrinsicProfileVariance` via `LensSample`. This is the quantified
   remaining gap vs the McClintock et al. (2019) SAC: off-diagonal
   correlations in rich, low-z bins are ~half the SAC's (e.g. z0 λ60+
   d=1: 0.077 vs 0.164) — see `memory: gruen-sac-calibration` and the
   validation report artifact.
2. Optional: miscentering stochasticity in `IntrinsicProfileVariance`
   (small-scale covariance; extension hook noted in its docstring).
3. Optional: full ellipticity axis-ratio/orientation integral
   (q ~ N(0.6, 0.12) truncated, isotropic orientation) to replace the
   σ_amp=0.20 amplitude proxy.
