"""Unit tests for retrieval metrics, candidate bank, and chunked top-k search."""

import numpy as np
import torch
from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.evaluation.retrieval import chunked_topk_search


def test_chunked_topk_matches_brute_force():
    num_queries = 20
    num_candidates = 150
    emb_dim = 128
    top_k = 10
    chunk_size = 25  # Force multiple small chunks

    torch.manual_seed(42)
    queries = torch.randn(num_queries, emb_dim)
    queries = queries / torch.norm(queries, p=2, dim=1, keepdim=True)

    candidates = torch.randn(num_candidates, emb_dim)
    candidates = candidates / torch.norm(candidates, p=2, dim=1, keepdim=True)

    # Brute-force calculation
    full_sims = queries @ candidates.T
    expected_scores, expected_indices = torch.topk(full_sims, k=top_k, dim=1, largest=True)

    # Chunked calculation
    chunked_indices, chunked_scores = chunked_topk_search(
        query_embeddings=queries,
        candidate_embeddings=candidates,
        top_k=top_k,
        chunk_size=chunk_size,
    )

    assert torch.allclose(expected_scores, chunked_scores, atol=1e-5)
    assert torch.equal(expected_indices, chunked_indices)


def test_compute_retrieval_metrics():
    # 4 queries: ranks are 1, 3, 5, 20
    ranks = [1, 3, 5, 20]
    metrics = compute_retrieval_metrics(ranks, top_k_list=[1, 5, 25])

    # Recall@1: 1/4 = 0.25
    assert np.isclose(metrics["recall@1"], 0.25)
    # Recall@5: 3/4 = 0.75
    assert np.isclose(metrics["recall@5"], 0.75)
    # Recall@25: 4/4 = 1.0
    assert np.isclose(metrics["recall@25"], 1.0)
    # MRR: (1/1 + 1/3 + 1/5 + 1/20) / 4 = (1 + 0.3333 + 0.2 + 0.05) / 4 = 1.5833 / 4 = 0.3958
    expected_mrr = np.mean([1.0 / 1, 1.0 / 3, 1.0 / 5, 1.0 / 20])
    assert np.isclose(metrics["mrr"], expected_mrr)
