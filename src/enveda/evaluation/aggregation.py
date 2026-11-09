"""Molecule-level aggregation algorithms (Max, Mean, RRF) for multi-spectra retrieval."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.evaluation.retrieval import chunked_topk_search


def aggregate_by_max_similarity(
    query_embeddings: torch.Tensor,
    molecule_groups: List[List[int]],
    candidate_embeddings: Union[np.ndarray, torch.Tensor],
    top_k: int = 25,
    chunk_size: int = 5000,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate spectra per molecule using Maximum Similarity:
    S(m, c) = max_{s in spectra(m)} sim(s, c).
    
    Args:
        query_embeddings: (num_spectra, emb_dim) normalized embeddings.
        molecule_groups: List of lists, where each inner list contains row indices of spectra for one molecule.
        candidate_embeddings: (num_candidates, emb_dim) normalized embeddings.
        top_k: Number of candidates to return per molecule.
        chunk_size: Chunk size for candidate search.
        device: Torch device.
        
    Returns:
        Tuple of (topk_indices, topk_scores) of shape (num_molecules, top_k).
    """
    num_molecules = len(molecule_groups)
    num_candidates = len(candidate_embeddings)
    top_k = min(top_k, num_candidates)

    if device is None:
        device = query_embeddings.device

    q_gpu = query_embeddings.to(device)

    running_scores: Optional[torch.Tensor] = None
    running_indices: Optional[torch.Tensor] = None

    for start_c in range(0, num_candidates, chunk_size):
        end_c = min(start_c + chunk_size, num_candidates)
        cand_chunk = candidate_embeddings[start_c:end_c]
        if isinstance(cand_chunk, np.ndarray):
            if not cand_chunk.flags.writeable:
                cand_chunk = cand_chunk.copy()
            cand_gpu = torch.from_numpy(cand_chunk).to(device)
        else:
            cand_gpu = cand_chunk.to(device)

        # Spectra-to-candidate similarities for this chunk: shape (num_spectra, chunk_len)
        chunk_sims = q_gpu @ cand_gpu.T

        # Aggregate for each molecule by taking max over its spectra
        mol_chunk_sims = torch.zeros(
            (num_molecules, chunk_sims.shape[1]),
            dtype=chunk_sims.dtype,
            device=device,
        )
        for m_idx, spec_indices in enumerate(molecule_groups):
            if len(spec_indices) == 1:
                mol_chunk_sims[m_idx] = chunk_sims[spec_indices[0]]
            else:
                mol_chunk_sims[m_idx] = torch.max(chunk_sims[spec_indices], dim=0).values

        # Top-k within this chunk
        k_chunk = min(top_k, mol_chunk_sims.shape[1])
        c_scores, c_idx = torch.topk(mol_chunk_sims, k=k_chunk, dim=1, largest=True)
        c_idx = c_idx + start_c

        if running_scores is None:
            running_scores = c_scores
            running_indices = c_idx
        else:
            combined_scores = torch.cat([running_scores, c_scores], dim=1)
            combined_indices = torch.cat([running_indices, c_idx], dim=1)
            best_scores, best_positions = torch.topk(combined_scores, k=top_k, dim=1, largest=True)
            running_scores = best_scores
            running_indices = torch.gather(combined_indices, 1, best_positions)

        del cand_chunk, cand_gpu, chunk_sims, mol_chunk_sims

    assert running_indices is not None and running_scores is not None
    return running_indices.cpu(), running_scores.cpu()


def aggregate_by_mean_similarity(
    query_embeddings: torch.Tensor,
    molecule_groups: List[List[int]],
    candidate_embeddings: Union[np.ndarray, torch.Tensor],
    top_k: int = 25,
    chunk_size: int = 5000,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate spectra per molecule using Mean Similarity:
    S(m, c) = mean_{s in spectra(m)} sim(s, c).
    
    Since sim is linear in the query vector:
    mean_s (q_s @ c) = (mean_s q_s) @ c.
    We normalize the mean query vector or compute cosine similarity with normalized mean.
    
    Args:
        query_embeddings: (num_spectra, emb_dim) normalized embeddings.
        molecule_groups: List of lists of spectrum indices for each molecule.
        candidate_embeddings: (num_candidates, emb_dim) normalized embeddings.
        top_k: Number of candidates to return per molecule.
        chunk_size: Chunk size for candidate search.
        device: Torch device.
        
    Returns:
        Tuple of (topk_indices, topk_scores) of shape (num_molecules, top_k).
    """
    num_molecules = len(molecule_groups)
    emb_dim = query_embeddings.shape[1]

    # Precompute mean spectrum embedding for each molecule
    mol_mean_embs = torch.zeros((num_molecules, emb_dim), dtype=torch.float32)
    for m_idx, spec_indices in enumerate(molecule_groups):
        mean_vec = query_embeddings[spec_indices].mean(dim=0)
        # Note: we do NOT L2-normalize if we want strict average cosine similarity:
        # (q_1 @ c + q_2 @ c)/2 = ((q_1 + q_2)/2) @ c
        mol_mean_embs[m_idx] = mean_vec

    # Search using chunked top-k
    return chunked_topk_search(
        query_embeddings=mol_mean_embs,
        candidate_embeddings=candidate_embeddings,
        top_k=top_k,
        chunk_size=chunk_size,
        device=device,
    )


def aggregate_by_rrf(
    query_embeddings: torch.Tensor,
    molecule_groups: List[List[int]],
    candidate_embeddings: Union[np.ndarray, torch.Tensor],
    top_k: int = 25,
    rrf_k: int = 60,
    spectrum_top_p: int = 200,
    chunk_size: int = 5000,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate spectra per molecule using Reciprocal Rank Fusion:
    RRF(m, c) = sum_{s in spectra(m)} 1 / (rrf_k + rank(c, s)).
    
    Args:
        query_embeddings: (num_spectra, emb_dim) spectrum embeddings.
        molecule_groups: List of lists of spectrum indices for each molecule.
        candidate_embeddings: Candidate embeddings array/tensor.
        top_k: Number of final aggregated candidates to return per molecule.
        rrf_k: Constant k parameter for RRF (default 60).
        spectrum_top_p: Number of top candidates retrieved per spectrum before fusion.
        chunk_size: Candidate chunk size.
        device: Compute device.
        
    Returns:
        Tuple of (topk_indices, topk_scores) of shape (num_molecules, top_k).
    """
    num_molecules = len(molecule_groups)
    num_candidates = len(candidate_embeddings)
    top_k = min(top_k, num_candidates)

    # 1. Retrieve top-P candidates for all spectra
    spec_top_indices, _ = chunked_topk_search(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        top_k=min(spectrum_top_p, num_candidates),
        chunk_size=chunk_size,
        device=device,
    )
    spec_top_indices_np = spec_top_indices.numpy()

    # 2. Fuse rankings per molecule
    mol_top_indices = torch.zeros((num_molecules, top_k), dtype=torch.int64)
    mol_top_scores = torch.zeros((num_molecules, top_k), dtype=torch.float32)

    for m_idx, spec_indices in enumerate(molecule_groups):
        cand_rrf_scores = defaultdict(float)
        for s_idx in spec_indices:
            ranked_cands = spec_top_indices_np[s_idx]
            for rank_0, c_idx in enumerate(ranked_cands):
                rank_1 = rank_0 + 1
                cand_rrf_scores[c_idx] += 1.0 / (rrf_k + rank_1)

        # Sort candidate items by descending RRF score
        sorted_cands = sorted(cand_rrf_scores.items(), key=lambda x: x[1], reverse=True)
        take_n = min(top_k, len(sorted_cands))
        for j in range(take_n):
            mol_top_indices[m_idx, j] = sorted_cands[j][0]
            mol_top_scores[m_idx, j] = sorted_cands[j][1]

    return mol_top_indices, mol_top_scores


def evaluate_molecule_retrieval(
    query_embeddings: torch.Tensor,
    molecule_ids: Union[List[Any], np.ndarray],
    true_smiles_per_molecule: List[str],
    candidate_smiles: List[str],
    candidate_embeddings: Union[np.ndarray, torch.Tensor],
    methods: List[str] = ["max", "mean", "rrf"],
    top_k_list: List[int] = [1, 5, 10, 25],
    rrf_k: int = 60,
    spectrum_top_p: int = 200,
    chunk_size: int = 5000,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Comprehensive molecule-level retrieval evaluation comparing aggregation strategies.
    
    Args:
        query_embeddings: (num_spectra, emb_dim) query spectrum embeddings.
        molecule_ids: Identifier per spectrum mapping it to a molecule.
        true_smiles_per_molecule: Ground truth SMILES for each unique molecule.
        candidate_smiles: Full list of candidate SMILES in the bank.
        candidate_embeddings: (num_candidates, emb_dim) candidate embeddings.
        methods: Aggregation methods to evaluate ("max", "mean", "rrf").
        top_k_list: List of K values for Recall@K and MRR@25.
        rrf_k: RRF k parameter.
        spectrum_top_p: Per-spectrum top candidates for RRF.
        chunk_size: Candidate search chunk size.
        device: Compute device.
        
    Returns:
        Dict mapping method name to its metrics dictionary.
    """
    # Group spectrum indices by molecule
    mol_to_indices = defaultdict(list)
    unique_mol_order = []
    for s_idx, m_id in enumerate(molecule_ids):
        if m_id not in mol_to_indices:
            unique_mol_order.append(m_id)
        mol_to_indices[m_id].append(s_idx)

    num_molecules = len(unique_mol_order)
    if num_molecules != len(true_smiles_per_molecule):
        raise ValueError(
            f"Mismatch between number of unique molecules ({num_molecules}) "
            f"and true SMILES list ({len(true_smiles_per_molecule)})"
        )

    molecule_groups = [mol_to_indices[m_id] for m_id in unique_mol_order]

    # Map SMILES to candidate index
    smiles_to_cand_idx = {smi: idx for idx, smi in enumerate(candidate_smiles)}
    true_cand_indices = [smiles_to_cand_idx.get(smi, -1) for smi in true_smiles_per_molecule]

    # Candidate coverage: fraction of unique molecules present in the candidate bank
    in_bank_count = sum(1 for idx in true_cand_indices if idx >= 0)
    candidate_coverage = float(in_bank_count / num_molecules) if num_molecules > 0 else 0.0

    top_k_max = max(top_k_list)
    results: Dict[str, Dict[str, float]] = {}

    for method in methods:
        method_lower = method.lower()
        if method_lower == "max":
            topk_idx, _ = aggregate_by_max_similarity(
                query_embeddings=query_embeddings,
                molecule_groups=molecule_groups,
                candidate_embeddings=candidate_embeddings,
                top_k=top_k_max,
                chunk_size=chunk_size,
                device=device,
            )
        elif method_lower == "mean":
            topk_idx, _ = aggregate_by_mean_similarity(
                query_embeddings=query_embeddings,
                molecule_groups=molecule_groups,
                candidate_embeddings=candidate_embeddings,
                top_k=top_k_max,
                chunk_size=chunk_size,
                device=device,
            )
        elif method_lower == "rrf":
            topk_idx, _ = aggregate_by_rrf(
                query_embeddings=query_embeddings,
                molecule_groups=molecule_groups,
                candidate_embeddings=candidate_embeddings,
                top_k=top_k_max,
                rrf_k=rrf_k,
                spectrum_top_p=spectrum_top_p,
                chunk_size=chunk_size,
                device=device,
            )
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        ranks = []
        for m_idx, true_c in enumerate(true_cand_indices):
            if true_c < 0:
                # Target not in candidate bank
                rank = top_k_max + 1
            else:
                preds = topk_idx[m_idx].tolist()
                if true_c in preds:
                    rank = preds.index(true_c) + 1
                else:
                    rank = top_k_max + 1
            ranks.append(rank)

        method_metrics = compute_retrieval_metrics(ranks, top_k_list=top_k_list)
        method_metrics["candidate_coverage"] = candidate_coverage
        method_metrics["num_molecules"] = float(num_molecules)
        results[method_lower] = method_metrics

    return results

