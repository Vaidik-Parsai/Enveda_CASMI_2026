"""Unit tests for molecule-level aggregation algorithms (Max, Mean, RRF)."""

import numpy as np
import torch

from enveda.evaluation.aggregation import (
    aggregate_by_max_similarity,
    aggregate_by_mean_similarity,
    aggregate_by_rrf,
    evaluate_molecule_retrieval,
)


def test_aggregation_outputs_exactly_one_ranking_per_molecule():
    # 5 spectra from 2 molecules:
    # Molecule 0: spectra 0, 1, 2
    # Molecule 1: spectra 3, 4
    num_spectra = 5
    emb_dim = 16
    num_candidates = 50
    top_k = 25

    torch.manual_seed(42)
    query_embeddings = torch.randn(num_spectra, emb_dim)
    query_embeddings = query_embeddings / torch.norm(query_embeddings, p=2, dim=1, keepdim=True)

    candidates = torch.randn(num_candidates, emb_dim)
    candidates = candidates / torch.norm(candidates, p=2, dim=1, keepdim=True)

    molecule_groups = [[0, 1, 2], [3, 4]]

    # 1. Max Similarity
    max_idx, max_scores = aggregate_by_max_similarity(
        query_embeddings=query_embeddings,
        molecule_groups=molecule_groups,
        candidate_embeddings=candidates,
        top_k=top_k,
    )
    assert max_idx.shape == (2, top_k)
    assert max_scores.shape == (2, top_k)
    # Check strictly descending scores per molecule
    for m in range(2):
        scores = max_scores[m].numpy()
        assert np.all(np.diff(scores) <= 1e-6)

    # 2. Mean Similarity
    mean_idx, mean_scores = aggregate_by_mean_similarity(
        query_embeddings=query_embeddings,
        molecule_groups=molecule_groups,
        candidate_embeddings=candidates,
        top_k=top_k,
    )
    assert mean_idx.shape == (2, top_k)
    assert mean_scores.shape == (2, top_k)
    for m in range(2):
        scores = mean_scores[m].numpy()
        assert np.all(np.diff(scores) <= 1e-6)

    # 3. RRF
    rrf_idx, rrf_scores = aggregate_by_rrf(
        query_embeddings=query_embeddings,
        molecule_groups=molecule_groups,
        candidate_embeddings=candidates,
        top_k=top_k,
        rrf_k=60,
        spectrum_top_p=30,
    )
    assert rrf_idx.shape == (2, top_k)
    assert rrf_scores.shape == (2, top_k)
    for m in range(2):
        scores = rrf_scores[m].numpy()
        assert np.all(np.diff(scores) <= 1e-6)


def test_analytical_max_and_mean_similarity():
    # 2 spectra for 1 molecule
    # Spectrum 0: [1, 0]
    # Spectrum 1: [0, 1]
    # Candidate 0: [1, 0] -> sims: [1.0, 0.0]
    # Candidate 1: [0.7071, 0.7071] -> sims: [0.7071, 0.7071]
    # Candidate 2: [0, 1] -> sims: [0.0, 1.0]

    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    candidates = torch.tensor(
        [[1.0, 0.0], [0.70710678, 0.70710678], [0.0, 1.0]], dtype=torch.float32
    )

    molecule_groups = [[0, 1]]

    # For Candidate 0: Max = 1.0, Mean = 0.5
    # For Candidate 1: Max = 0.7071, Mean = 0.7071
    # For Candidate 2: Max = 1.0, Mean = 0.5
    # Therefore, under Mean similarity, Candidate 1 MUST rank FIRST (highest mean)!
    # Under Max similarity, Candidates 0 and 2 tie for first with 1.0.

    mean_idx, mean_scores = aggregate_by_mean_similarity(
        queries, molecule_groups, candidates, top_k=3
    )
    assert mean_idx[0, 0].item() == 1  # Candidate 1 is ranked #1 under mean!
    assert np.isclose(mean_scores[0, 0].item(), 0.70710678, atol=1e-4)

    max_idx, max_scores = aggregate_by_max_similarity(
        queries, molecule_groups, candidates, top_k=3
    )
    # Under max, top score must be 1.0
    assert np.isclose(max_scores[0, 0].item(), 1.0, atol=1e-4)
    assert max_idx[0, 0].item() in [0, 2]


def test_rrf_analytical_scores():
    # 2 spectra for 1 molecule. 3 candidates.
    # Spectrum 0 ranks: candidate 0 (rank 1), candidate 1 (rank 2)
    # Spectrum 1 ranks: candidate 1 (rank 1), candidate 2 (rank 2)
    # k = 60
    # Cand 0: 1/(60+1) = 1/61 ~ 0.016393
    # Cand 1: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.016129 + 0.016393 = 0.032522
    # Cand 2: 1/(60+2) = 1/62 ~ 0.016129
    # Cand 1 must be #1!

    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    # Candidates crafted so spectrum 0 prefers 0 > 1 > 2, spectrum 1 prefers 2 > 1 > 0
    candidates = torch.tensor(
        [
            [1.0, 0.0],     # Cand 0: sim with s0=1.0, s1=0.0
            [0.6, 0.6],     # Cand 1: sim with s0=0.6, s1=0.6
            [0.0, 1.0],     # Cand 2: sim with s0=0.0, s1=1.0
        ],
        dtype=torch.float32,
    )
    molecule_groups = [[0, 1]]

    rrf_idx, rrf_scores = aggregate_by_rrf(
        query_embeddings=queries,
        molecule_groups=molecule_groups,
        candidate_embeddings=candidates,
        top_k=3,
        rrf_k=60,
        spectrum_top_p=2,  # Top-2 per spectrum: Cand 0 in s0 only, Cand 2 in s1 only, Cand 1 in both!
    )

    # Candidate 1 appears in top-2 for both spectra, so it wins RRF:
    # Cand 1 score: 1/62 + 1/62 = 2/62 = ~0.03226
    # Cand 0 score: 1/61 = ~0.01639
    # Cand 2 score: 1/61 = ~0.01639
    assert rrf_idx[0, 0].item() == 1
    expected_cand1_rrf = 2.0 / 62
    assert np.isclose(rrf_scores[0, 0].item(), expected_cand1_rrf, atol=1e-5)


def test_evaluate_molecule_retrieval_end_to_end():
    queries = torch.randn(4, 16)
    queries = queries / torch.norm(queries, p=2, dim=1, keepdim=True)
    cands = torch.randn(20, 16)
    cands = cands / torch.norm(cands, p=2, dim=1, keepdim=True)

    mol_ids = ["MOL_A", "MOL_A", "MOL_B", "MOL_B"]
    true_smiles = ["SMI_A", "SMI_B"]
    cand_smiles = [f"SMI_{i}" for i in range(18)] + ["SMI_A", "SMI_B"]

    metrics = evaluate_molecule_retrieval(
        query_embeddings=queries,
        molecule_ids=mol_ids,
        true_smiles_per_molecule=true_smiles,
        candidate_smiles=cand_smiles,
        candidate_embeddings=cands,
        methods=["max", "mean", "rrf"],
        top_k_list=[1, 5, 25],
    )

    for m in ["max", "mean", "rrf"]:
        assert m in metrics
        assert "mrr@25" in metrics[m]
        assert "recall@1" in metrics[m]
        assert "recall@5" in metrics[m]
        assert "recall@25" in metrics[m]
        assert "candidate_coverage" in metrics[m]
        assert metrics[m]["candidate_coverage"] == 1.0
        assert metrics[m]["num_molecules"] == 2.0


def test_candidate_reranker():
    from enveda.evaluation.reranker import CandidateReranker

    reranker = CandidateReranker(
        weight_neural=1.0,
        weight_mass_ppm=2.0,
        sigma_ppm=25.0,
        weight_consistency=0.5,
    )

    candidate_masses = np.array([200.0, 200.002, 250.0], dtype=np.float32)
    candidate_indices = [0, 1, 2]
    expected_mass = 200.001

    # Spectrum sims: 2 spectra, 3 candidates
    spectrum_sims = np.array([
        [0.8, 0.79, 0.95],
        [0.82, 0.81, 0.50],
    ], dtype=np.float32)

    topk, scores = reranker.rerank(
        candidate_indices=candidate_indices,
        candidate_masses=candidate_masses,
        expected_neutral_mass=expected_mass,
        spectrum_sims=spectrum_sims,
        top_k=3,
    )

    # Candidate 2 has large mass difference (250 vs 200, 250,000 ppm), should be penalized heavily
    assert topk[2] == 2
    # Candidate 0 and 1 have mass within 5 ppm, should be top 2
    assert set(topk[:2]) == {0, 1}
    assert scores[0] >= scores[1] >= scores[2]

