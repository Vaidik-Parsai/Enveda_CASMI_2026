"""Retrieval evaluation metrics (Recall@K, MRR@25, MRR, Mean Rank)."""

from typing import Dict, List, Optional, Union
import numpy as np


def compute_retrieval_metrics(
    ranks: Union[List[int], np.ndarray],
    top_k_list: List[int] = [1, 5, 10, 25],
    max_k_cutoff: Optional[int] = 25,
) -> Dict[str, float]:
    """Compute retrieval metrics from 1-based ranks of ground truth candidates.
    
    Args:
        ranks: 1D array or list of 1-based rank positions for true candidates.
        top_k_list: List of k values for Recall@K.
        max_k_cutoff: Top-K cutoff for truncated retrieval (default 25).
                      Unretrieved candidates (rank > max_k_cutoff) contribute 0.0 to MRR.
        
    Returns:
        Dictionary of metric names to values.
    """
    if len(ranks) == 0:
        return {}

    ranks = np.asarray(ranks, dtype=np.float64)

    metrics: Dict[str, float] = {}
    for k in top_k_list:
        metrics[f"recall@{k}"] = float(np.mean(ranks <= k))

    # MRR@25: Primary competition objective (1/rank if rank <= 25, else 0.0)
    mrr_25 = np.where(ranks <= 25, 1.0 / ranks, 0.0)
    metrics["mrr@25"] = float(np.mean(mrr_25))

    # General MRR: under top-K cutoff, unretrieved targets (rank > max_k_cutoff) contribute 0.0
    if max_k_cutoff is not None:
        reciprocal_ranks = np.where(ranks <= max_k_cutoff, 1.0 / ranks, 0.0)
    else:
        reciprocal_ranks = 1.0 / ranks
    metrics["mrr"] = float(np.mean(reciprocal_ranks))

    # Ranks among retrieved candidates (for meaningful rank distribution)
    retrieved_mask = ranks <= (max_k_cutoff if max_k_cutoff is not None else float("inf"))
    retrieved_ranks = ranks[retrieved_mask]
    if len(retrieved_ranks) > 0:
        metrics["mean_rank_retrieved"] = float(np.mean(retrieved_ranks))
        metrics["median_rank_retrieved"] = float(np.median(retrieved_ranks))
    else:
        metrics["mean_rank_retrieved"] = float("nan")
        metrics["median_rank_retrieved"] = float("nan")

    metrics["mean_rank"] = float(np.mean(ranks))
    metrics["median_rank"] = float(np.median(ranks))
    metrics["num_queries"] = float(len(ranks))

    return metrics
