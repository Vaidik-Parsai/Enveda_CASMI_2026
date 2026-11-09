"""Retrieval evaluation, candidate search, and molecule-level aggregation modules."""

from enveda.evaluation.aggregation import (
    aggregate_by_max_similarity,
    aggregate_by_mean_similarity,
    aggregate_by_rrf,
    evaluate_molecule_retrieval,
)
from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.evaluation.retrieval import (
    build_candidate_bank,
    chunked_topk_search,
    evaluate_retrieval,
)

__all__ = [
    "compute_retrieval_metrics",
    "build_candidate_bank",
    "chunked_topk_search",
    "evaluate_retrieval",
    "aggregate_by_max_similarity",
    "aggregate_by_mean_similarity",
    "aggregate_by_rrf",
    "evaluate_molecule_retrieval",
]
