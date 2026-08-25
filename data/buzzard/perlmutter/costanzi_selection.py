"""Costanzi optical-selection forward model.

Ported from https://github.com/MCostanzi/SelectionBias. Implements:

- Costanzi+2026 (arXiv:2604.05833), Eq. 15: intrinsic mass-richness relation
  <lambda_sat | M, z>, with lambda_tr = 1 + lambda_sat when M >= M_min.
  Sampling uses a compound Poisson with lognormal broadening of the mean
  (equivalent to second order to the Poisson * Gaussian convolution stated
  in C26).
- Costanzi+2019 (arXiv:1807.07072), Eq. 6: P(lambda_obs | lambda_tr, z) as
  the convolution of a Gaussian background term and a delta + exponential
  projection term, equivalent to the (1-f_prj) Gaussian + f_prj EMG mixture
  used here.
- A small-scale (one-halo) approximation of the selection weight
  B_sel(lambda_obs, lambda_tr, z). NOTE: in the current pipeline the
  scale-dependent C26 Appendix C Eq. C1 fitting function is preferred;
  ``b_sel_one_halo`` is kept for completeness only.

Parameters are the DES Y1 NC best fit derived assuming the DES Y1 3x2pt
cosmology (C26, footnote 5), hard-coded in the upstream notebook.
"""
import os
import numpy as np
import scipy.special as spc
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# ---------------------------------------------------------------------------
# DES-Y1 NC+3x2pt best-fit intrinsic scaling relation parameters
# ---------------------------------------------------------------------------
M_MIN = 10. ** 1.13852818e+01
ALPHA = 8.58693714e-01
M_1 = 10. ** 1.26964410e+01
M_PIVOT = M_1 - M_MIN
SIGM_INTR = 1.80949022e-01
EPSI = 2.83887020e-01
PIVOT_Z0 = 0.4544

DEFAULT_PRJ_PARAM_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'data',
    'prj_params_DESY3_lss_lin_dep_getdist_v1.txt'
)
PRJ_Z_BINS = np.linspace(0.10, 0.80, 15)


# ---------------------------------------------------------------------------
# C26 Eq. 15:  <lambda_sat | M, z>  and the lambda_tr = lambda_cen + lambda_sat
# halo model. lambda_cen = 1 for M >= M_min, 0 otherwise (C26 Sec. III).
# ---------------------------------------------------------------------------
def l_sat(M, z, Mmin=M_MIN, Mpivot=M_PIVOT, a=ALPHA, epsilon=EPSI,
          pivot_z=PIVOT_Z0):
    """C26 Eq. 15 expectation value of the satellite richness.

    Sub-threshold halos (M <= M_min) host no satellites; clipping the base
    avoids (negative)**alpha producing NaNs.
    """
    M = np.asarray(M, dtype=float)
    z = np.asarray(z, dtype=float)
    base = np.clip((M - Mmin) / Mpivot, 0., None)
    return base ** a * ((1. + z) / (1. + pivot_z)) ** epsilon


def l_tr(M, z):
    """Mean intrinsic richness <lambda_tr | M, z> = lambda_cen + <lambda_sat>.

    Following C26 Sec. III, lambda_cen = 1 above M_min and 0 below it.
    """
    M = np.asarray(M, dtype=float)
    return np.where(M >= M_MIN, 1. + l_sat(M, z), 0.)


def sig_intr(M, z):
    return SIGM_INTR * l_sat(M, z)


def pltr_M(ltr, M, z):
    """P(lambda_tr | M, z) — compound Poisson + lognormal approximation
    to the C26 Poisson*Gaussian convolution (their Sec. III)."""
    m = l_sat(M, z)
    std = np.sqrt(m + (m * SIGM_INTR) ** 2.)
    x = ltr + (m * SIGM_INTR) ** 2.
    lam = std ** 2.
    ln_gamma_fun = spc.gammaln(x)
    return np.exp(-lam + (x - 1.) * np.log(lam) - ln_gamma_fun,
                  dtype='float128')


def sample_lambda_true(M, z, rng=None):
    """Monte-Carlo draw of lambda_tr per halo.

    Generative form: N_sat ~ Poisson(m) with mean m = l_sat(M, z) broadened
    by a multiplicative lognormal of width SIGM_INTR. To second order this
    matches the Poisson * Gaussian convolution prescription of C26 Sec. III
    for the parameter regime used here (sigma_lambda = 0.18). The central
    galaxy contributes the explicit "+1" (only for halos above M_min).

    Returns an integer-valued ``lambda_tr`` per halo.
    """
    rng = np.random.default_rng() if rng is None else rng
    M = np.asarray(M, dtype=float)
    m = l_sat(M, z)
    # lognormal broadening of the mean (mean=-sigma^2/2 keeps E[m_broad]=m).
    m_broad = m * rng.lognormal(mean=-0.5 * SIGM_INTR ** 2, sigma=SIGM_INTR,
                                size=m.shape)
    m_broad = np.clip(m_broad, 0., None)
    n_sat = rng.poisson(m_broad)
    has_central = (M >= M_MIN).astype(int)
    return has_central + n_sat


# ---------------------------------------------------------------------------
# Projection-parameter file (`prj_params_DESY3_lss_lin_dep_getdist_v1.txt`):
#   shape (N_samples, 10), columns
#   (a_tau, b_tau, a_mu, b_mu, a_sig, b_sig, a_fprj, b_fprj, a_fmsk, b_fmsk).
# Rows are getdist MCMC posterior samples (NOT redshift knots) — the per-z
# evolution is already encoded in the linear coefficients, see C19 Eq. A8.
# In practice we collapse the rows to the posterior mean via
# ``load_prj_posterior_mean``; ``load_prj_interpolators`` is kept for users
# who want to scan a single posterior sample.
# ---------------------------------------------------------------------------
def load_prj_interpolators(path=DEFAULT_PRJ_PARAM_FILE, row=None):
    """Return a dict of ``z -> coefficient`` callables for the 10 columns.

    If ``row`` is an integer we return a constant-in-z function for that
    posterior sample. The legacy branch with ``row=None`` is preserved for
    backward compatibility but is NOT the recommended entry point — use
    ``load_prj_posterior_mean`` instead, which collapses the chain to its
    posterior mean.
    """
    params = np.loadtxt(path).T  # shape (10, 15)
    if row is None:
        interp = {}
        labels = ['a_tau', 'b_tau', 'a_mu', 'b_mu', 'a_sig', 'b_sig',
                  'a_fprj', 'b_fprj', 'a_fmsk', 'b_fmsk']
        for i, key in enumerate(labels):
            interp[key] = ius(PRJ_Z_BINS, params[i, :], k=1)
        return interp

    const = params[:, row]
    labels = ['a_tau', 'b_tau', 'a_mu', 'b_mu', 'a_sig', 'b_sig',
              'a_fprj', 'b_fprj', 'a_fmsk', 'b_fmsk']
    return {k: (lambda v=const[i]: (lambda z: np.full_like(np.asarray(z,
                                                                      dtype=float), v)))()
            for i, k in enumerate(labels)}


# Upstream notebook's parameter-to-function forms (lin := lambda_true)
def _tau_model(lin, a, b):
    return b / lin ** a


def _fprj_model(lin, a, b):
    return b / (1. + np.exp(-lin / 1.)) ** a


def _mu_model(lin, a, b):
    return a + b * lin


def _sig_model(lin, a, b):
    return b * lin ** a


# ---------------------------------------------------------------------------
# P(lambda_obs | lambda_true, z) — delta + exponential projection mixture
# ---------------------------------------------------------------------------
def projection_params(ltr, z, interp):
    """
    Return (f_prj, tau, mu, sigma) at each (ltr, z) for the Costanzi+2019
    four-parameter convolution model (their Eq. 6, arXiv:1807.07072):

        P(lobs|ltr, z) = (1 - f_prj) * N(mu, sigma)
                      +  f_prj * EMG(mu, sigma, tau)

    where EMG is the exponentially-modified Gaussian = Gauss * Exp(tau).

    Per-parameter forms (C19 Appendix A, Eq. A8) — all functions of ltr,
    with the (a, b) coefficients themselves z-dependent through ``interp``:
        mu     = a_mu  + b_mu  * ltr               (FULL Gaussian mean)
        sigma  = b_sig * ltr ** a_sig              (Gaussian width)
        tau    = b_tau / ltr ** a_tau              (Exp rate)
        f_prj  = b_fprj / (1 + exp(-ltr)) ** a_fprj (sigmoid mixture weight)
    """
    ltr = np.asarray(ltr, dtype=float)
    z   = np.asarray(z,   dtype=float)

    f_prj = _fprj_model(ltr, interp['a_fprj'](z), interp['b_fprj'](z))
    tau   = _tau_model (ltr, interp['a_tau'](z),  interp['b_tau'](z))
    mu    = _mu_model  (ltr, interp['a_mu'](z),   interp['b_mu'](z))    # FULL mean
    sigma = _sig_model (ltr, interp['a_sig'](z),  interp['b_sig'](z))

    f_prj = np.clip(f_prj, 0.0, 1.0)
    tau   = np.clip(tau,   1e-6, None)
    sigma = np.clip(sigma, 1e-6, None)
    return f_prj, tau, mu, sigma


def sample_lambda_obs(ltr, z, interp, rng=None):
    """
    Draw lambda_obs from Costanzi+2019 Eq. 6 by mixture sampling.

    Mixture:
        u ~ U(0, 1)
        if u < 1 - f_prj:   lobs ~ N(mu, sigma)
        else:               lobs ~ N(mu, sigma) + Exp(1/tau)         (EMG)

    The EMG branch is sampled as Gaussian + Exponential because the
    sum of independent N(mu, sigma) and Exp(tau) random variables has
    exactly the EMG density (the awful closed form in Eq. 6); see
    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    Returns
    -------
    lobs   : (nh,)  real-valued observed richness
    f_prj  : (nh,)  projection fraction used
    tau    : (nh,)  Exp rate used
    mu     : (nh,)  Gaussian mean used (~ lambda_true)
    sigma  : (nh,)  Gaussian width used
    """
    rng = np.random.default_rng() if rng is None else rng
    ltr = np.asarray(ltr, dtype=float)
    z   = np.asarray(z,   dtype=float)

    f_prj, tau, mu, sigma = projection_params(ltr, z, interp)

    gauss = mu + sigma * rng.standard_normal(size=ltr.shape)
    u     = rng.uniform(size=ltr.shape)
    boost = rng.exponential(scale=1.0 / tau)
    lobs  = np.where(u < f_prj, gauss + boost, gauss)

    return lobs, f_prj, tau, mu, sigma


# ---------------------------------------------------------------------------
# B_sel weight (one-halo / small-scale limit)
# ---------------------------------------------------------------------------
def b_sel_one_halo(lob, ltr, z, interp):
    """Per-halo B_sel weight in the small-scale (single-scale) limit.

    Uses the C19 closed-form expectation <Delta_prj | ltr, z> = f_prj / tau
    of the delta + exponential mixture. The projection-model parameters
    (f_prj, tau) are evaluated at lambda_TRUE because the kernel
    P(lambda_obs | lambda_tr, z) is parameterized by lambda_tr (C19 Eq. 6).
    Each halo's weight is

        w = 1 + (Delta_prj_i - <Delta_prj>) / <Delta_prj>,

    so the population mean over the mixture recovers 1; w = 1 corresponds
    to no selection bias.

    NOTE: superseded in the current pipeline by the scale-dependent C26
    Appendix C fitting function ``Bsel_C1`` defined in the notebook.
    """
    lob = np.asarray(lob, dtype=float)
    ltr = np.asarray(ltr, dtype=float)
    z = np.asarray(z, dtype=float)
    f_prj, tau, _mu, _sigma = projection_params(ltr, z, interp)
    dprj_bar = f_prj / tau           # mean projection boost at given ltr, z
    dprj = np.maximum(lob - ltr, 0.)  # per-halo projection excess
    with np.errstate(divide='ignore', invalid='ignore'):
        w = 1. + (dprj - dprj_bar) / np.where(dprj_bar > 0, dprj_bar, 1.)
    return np.clip(w, 0., None)


# ---------------------------------------------------------------------------
# Convenience: posterior-mean projection interpolator
# ---------------------------------------------------------------------------
def load_prj_posterior_mean(path=DEFAULT_PRJ_PARAM_FILE):
    """Collapse the 15 posterior-sample rows to a single posterior-mean set
    of constants (useful when the 15 rows are interpreted as samples rather
    than redshift knots). Returns the same interpolator-dict signature so
    the rest of the pipeline does not change.
    """
    params = np.loadtxt(path)           # (15, 10)
    mean = params.mean(axis=0)          # (10,)
    labels = ['a_tau', 'b_tau', 'a_mu', 'b_mu', 'a_sig', 'b_sig',
              'a_fprj', 'b_fprj', 'a_fmsk', 'b_fmsk']
    return {k: (lambda v=mean[i]: (lambda z: np.full_like(
                np.asarray(z, dtype=float), v)))()
            for i, k in enumerate(labels)}
