"""Retrieval evaluation metrics (Recall@K, MRR, Mean Rank)."""

from typing import Dict, List, Union
import numpy as np


def compute_retrieval_metrics(
    ranks: Union[List[int], np.ndarray],
    top_k_list: List[int] = [1, 5, 25],
) -> Dict[str, float]:
    """Compute retrieval metrics from 1-based ranks of ground truth candidates.
    
    Args:
        ranks: 1D array or list of 1-based rank positions for true candidates.
        top_k_list: List of k values for Recall@K.
        
    Returns:
        Dictionary of metric names to values.
    """
    if len(ranks) == 0:
        return {}

    ranks = np.asarray(ranks, dtype=np.float64)

    metrics: Dict[str, float] = {}
    for k in top_k_list:
        metrics[f"recall@{k}"] = float(np.mean(ranks <= k))

    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = 1.0 / ranks
    metrics["mrr"] = float(np.mean(reciprocal_ranks))

    metrics["mean_rank"] = float(np.mean(ranks))
    metrics["median_rank"] = float(np.median(ranks))
    metrics["num_queries"] = float(len(ranks))

    return metrics
