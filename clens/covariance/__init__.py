"""Analytical covariance of DeltaSigma_ij and N_ij (the only physics
kept in this repository — all model ingredients come from clenspy
through the :mod:`clens.covariance.inputs` contract)."""

from .assemble import CovarianceAssembler
from .counts_cov import NCountsCov
from .gaussian_delta_sigma import DeltaSigmaCovBlocks, GaussianDeltaSigmaCov
from .inputs import (
    CosmologyInputs,
    FrozenTables,
    LensSample,
    SourceInputs,
    SurveyGeometry,
    from_clenspy,
)
from .limber import LimberProjector

__all__ = [
    "CosmologyInputs",
    "SourceInputs",
    "LensSample",
    "SurveyGeometry",
    "FrozenTables",
    "from_clenspy",
    "LimberProjector",
    "GaussianDeltaSigmaCov",
    "DeltaSigmaCovBlocks",
    "NCountsCov",
    "CovarianceAssembler",
]
