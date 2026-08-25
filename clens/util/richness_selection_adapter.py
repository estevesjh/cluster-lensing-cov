"""Adapter for the analytical relations in the sibling RichnessSelection repo.

The covariance project should use the existing implementation in
``github/RichnessSelection`` rather than carry a second HOD implementation.
This module only performs the final richness-bin contraction needed by
``ClusterCounts``.  It delegates the observed-richness kernel to
``RichnessSelection.selection_function`` when projection is enabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "github" / "RichnessSelection"


def _load_richness_selection_api(root=None):
    """Load the existing RichnessSelection MOR and projection functions."""
    root = Path(root or os.environ.get("RICHNESS_SELECTION_ROOT", _DEFAULT_ROOT))
    source_root = root / "src"
    if not (source_root / "richness_selection" / "mor.py").is_file():
        raise FileNotFoundError(
            "could not find RichnessSelection source at "
            f"{source_root}; set RICHNESS_SELECTION_ROOT to its repository"
        )
    source_root = str(source_root)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        from richness_selection.gl import gl_nodes
        from richness_selection.mor import LogNormalMOR, MOR
        from richness_selection.selection_function.kernels import K_i, K_j
    except ImportError as error:
        raise ImportError(
            "RichnessSelection dependencies are unavailable; install its "
            "project dependencies in the covariance environment"
        ) from error
    return MOR, LogNormalMOR, K_i, K_j, gl_nodes


class IntrinsicRichnessRelation:
    """Use an existing RichnessSelection MOR and its selection kernels."""

    def __init__(self, mor, quadrature_order=64, apply_projection=False,
                 ltr_bracket_sigma=6.0, projection_api=None):
        self.mor = mor
        self.quadrature_order = int(quadrature_order)
        self.apply_projection = bool(apply_projection)
        self.ltr_bracket_sigma = float(ltr_bracket_sigma)
        self._projection_api = projection_api
        if self.quadrature_order < 8:
            raise ValueError("quadrature_order must be at least 8")
        self._nodes, self._weights = np.polynomial.legendre.leggauss(
            self.quadrature_order
        )

    def __getattr__(self, name):
        return getattr(self.mor, name)

    def pdf(self, lambda_true, M, z):
        return self.mor.pdf(lambda_true, M, z)

    def selection_probability(self, M, z, lambda_min, lambda_max):
        """Return the observed-richness bin probability.

        With ``apply_projection=False`` this is the direct intrinsic HOD/MOR
        integral.  With ``apply_projection=True`` it is the existing
        RichnessSelection contraction

        ``int dltr P(ltr|M,z) K_i(ltr,z)``,

        where ``K_i`` is the analytical bin integral of
        ``P(lob|ltr,z)``.  The lambda_true nodes and weights are fixed for the
        supplied mass vector, as in ``selection_function.number_counts``.
        """
        if not 0.0 <= lambda_min < lambda_max:
            raise ValueError("richness edges must satisfy 0 <= min < max")

        M_arr = np.asarray(M, dtype=float)
        if self.apply_projection:
            if self._projection_api is None:
                raise RuntimeError("projection API was not loaded")
            K_i, gl_nodes = self._projection_api[:2]
            mean = np.asarray(self.mor.ltr_mean(M_arr, z), dtype=float)
            sigma = np.asarray(self.mor.ltr_sigma(M_arr, z), dtype=float)
            upper = float(np.max(mean + self.ltr_bracket_sigma * sigma))
            if upper <= 0.0:
                return 0.0 if np.ndim(M) == 0 else np.zeros_like(M_arr)
            lam, weights = gl_nodes(0.0, upper, self.quadrature_order)
            kernel = K_i(lam, float(z), lambda_min, lambda_max)
            pdf = self.mor.pdf(
                lam[:, None], M_arr.reshape((1,) + M_arr.shape), z
            )
            out = np.einsum("q,q,qM->M", weights, kernel, pdf)
            return float(out[0]) if np.ndim(M) == 0 else out

        half = 0.5 * (float(lambda_max) - float(lambda_min))
        mid = 0.5 * (float(lambda_max) + float(lambda_min))
        lam = mid + half * self._nodes
        lam_shape = (lam.size,) + (1,) * M_arr.ndim
        mass_shape = (1,) + M_arr.shape
        pdf = self.mor.pdf(
            lam.reshape(lam_shape),
            M_arr.reshape(mass_shape),
            z,
        )
        weights = self._weights.reshape(lam_shape)
        result = half * np.sum(weights * pdf, axis=0)
        return float(result) if np.ndim(M) == 0 else result

    def joint_selection_probability(self, M, z, lambda_min, lambda_max,
                                    z_min, z_max, sigma_z):
        """Return the full observed richness/redshift selection ``S_ij``.

        The richness factor is delegated to this repository's ``K_i``
        projection when requested, and the redshift factor is the existing
        RichnessSelection ``K_j`` Gaussian-CDF kernel.  Thus this method is
        exactly the factorized selection used by its ``S_ij`` implementation:

        ``S_ij(M,z) = S_i(M,z) * K_j(z)``.
        """
        if sigma_z <= 0.0:
            raise ValueError("sigma_z must be positive for the K_j kernel")
        if self._projection_api is None:
            raise RuntimeError("RichnessSelection kernel API was not loaded")
        K_j = self._projection_api[2]
        richness_selection = self.selection_probability(
            M, z, lambda_min, lambda_max
        )
        redshift_selection = K_j(z, z_min, z_max, sigma_z)
        return richness_selection * redshift_selection


def make_intrinsic_relation(model="hod", root=None, quadrature_order=64,
                            apply_projection=False, ltr_bracket_sigma=6.0,
                            **parameters):
    """Build a HOD or log-normal relation from ``RichnessSelection``.

    Parameters are passed unchanged to ``MOR`` or ``LogNormalMOR``.  For the
    DES-Y1 HOD path, pass the recovered HOD values explicitly, including the
    redshift slope ``epsilon``.  Set ``apply_projection=True`` to use the
    existing ``K_i`` convolution over true richness.
    """
    MOR, LogNormalMOR, K_i, K_j, gl_nodes = _load_richness_selection_api(root)
    key = str(model).lower().replace("-", "_")
    if key in {"hod", "mor", "costanzi_hod"}:
        if "sigma_lambda" in parameters and "sigma_intr" not in parameters:
            parameters["sigma_intr"] = parameters.pop("sigma_lambda")
        if "log10_ratio" in parameters and "log10_M1" not in parameters:
            parameters["log10_M1"] = (
                float(parameters.pop("log10_Mmin"))
                + float(parameters.pop("log10_ratio"))
            )
        mor = MOR(**parameters)
    elif key in {"lognormal", "log_normal", "logn"}:
        mor = LogNormalMOR(**parameters)
    else:
        raise ValueError(f"unknown intrinsic richness model: {model}")
    # Keep K_j available even for the intrinsic-only richness path: S_ij
    # still requires the observed-redshift selection factor.
    projection_api = (K_i, gl_nodes, K_j)
    return IntrinsicRichnessRelation(
        mor,
        quadrature_order=quadrature_order,
        apply_projection=apply_projection,
        ltr_bracket_sigma=ltr_bracket_sigma,
        projection_api=projection_api,
    )
