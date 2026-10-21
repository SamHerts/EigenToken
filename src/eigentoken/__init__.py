"""eigentoken -- token embedding geometry across LLM models."""

from .loader import LoadedModel, load
from .vocab import Resolution, case_variant_ids, resolve, resolve_all
from .geometry import (
    Axis,
    PairStats,
    analogy_vector,
    build_axis,
    cosine_matrix,
    dim_attribution,
    isotropy,
    neighbors,
    pair_direction_similarity,
    pair_stats,
    pca,
    project,
    reject,
)
from .crossmodel import neighborhood_overlap, procrustes, rsa
from .bias import PRESETS as BIAS_PRESETS
from .bias import WeatResult, weat

__version__ = "0.1.0"

__all__ = [
    "LoadedModel", "load",
    "Resolution", "case_variant_ids", "resolve", "resolve_all",
    "Axis", "PairStats", "analogy_vector", "build_axis", "cosine_matrix",
    "dim_attribution", "isotropy", "neighbors", "pair_direction_similarity",
    "pair_stats", "pca", "project", "reject",
    "neighborhood_overlap", "procrustes", "rsa",
    "BIAS_PRESETS", "WeatResult", "weat",
    "__version__",
]
