"""Memory-bounded candidate bank builder and chunked top-k retrieval evaluator."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from enveda.data.preprocessing import generate_morgan_fingerprint
from enveda.evaluation.metrics import compute_retrieval_metrics


def build_candidate_bank(
    smiles_list: List[str],
    model: nn.Module,
    output_dir: Union[str, Path] = "outputs/candidate_bank",
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    fp_radius: int = 2,
    fp_n_bits: int = 2048,
) -> Tuple[Path, Path]:
    """Encode candidate molecules in bounded batches and persist memory-mapped embeddings.
    
    Args:
        smiles_list: List of unique candidate SMILES strings.
        model: Trained SpectrumMoleculeModel or MoleculeEncoder.
        output_dir: Destination directory.
        batch_size: Batch size for fingerprinting and encoding.
        device: Compute device.
        fp_radius: Morgan radius.
        fp_n_bits: Fingerprint bit length.
        
    Returns:
        Tuple of (embeddings_npy_path, manifest_json_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    emb_path = out_path / "candidate_embeddings.npy"
    manifest_path = out_path / "candidate_manifest.json"

    n_candidates = len(smiles_list)
    emb_dim = 128

    # Create memory-mapped numpy array on disk
    mmap_arr = np.lib.format.open_memmap(
        str(emb_path),
        mode="w+",
        dtype=np.float32,
        shape=(n_candidates, emb_dim),
    )

    model.eval()
    with torch.inference_mode():
        for start_idx in range(0, n_candidates, batch_size):
            end_idx = min(start_idx + batch_size, n_candidates)
            batch_smiles = smiles_list[start_idx:end_idx]

            # Generate batch of Morgan fingerprints
            batch_fps = np.zeros((len(batch_smiles), fp_n_bits), dtype=np.float32)
            for j, smi in enumerate(batch_smiles):
                fp, _ = generate_morgan_fingerprint(smi, radius=fp_radius, n_bits=fp_n_bits)
                batch_fps[j] = fp

            fp_tensor = torch.from_numpy(batch_fps).to(device)
            if hasattr(model, "encode_molecule"):
                mol_embs = model.encode_molecule(fp_tensor)
            else:
                mol_embs = model(fp_tensor)

            mmap_arr[start_idx:end_idx] = mol_embs.cpu().numpy()

    # Flush changes to disk
    del mmap_arr

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_candidates": n_candidates,
                "embedding_dim": emb_dim,
                "smiles": smiles_list,
            },
            f,
        )

    return emb_path, manifest_path


def chunked_topk_search(
    query_embeddings: torch.Tensor,
    candidate_embeddings: Union[np.ndarray, torch.Tensor],
    top_k: int = 25,
    chunk_size: int = 5000,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Search nearest candidates for queries in bounded chunks to prevent VRAM spikes.
    
    Args:
        query_embeddings: (num_queries, emb_dim) tensor.
        candidate_embeddings: (num_candidates, emb_dim) array or tensor on CPU/disk.
        top_k: Number of nearest neighbors to return.
        chunk_size: Number of candidates to evaluate per matrix multiply.
        device: Compute device for matrix multiplication.
        
    Returns:
        Tuple of (top_k_indices, top_k_scores) of shape (num_queries, top_k).
    """
    num_queries, emb_dim = query_embeddings.shape
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

        # Dot product / cosine similarity for this chunk: shape (num_queries, chunk_len)
        chunk_sims = q_gpu @ cand_gpu.T

        # Top-k within this chunk
        k_chunk = min(top_k, chunk_sims.shape[1])
        c_scores, c_idx = torch.topk(chunk_sims, k=k_chunk, dim=1, largest=True)
        c_idx = c_idx + start_c  # offset to global candidate indices

        if running_scores is None:
            running_scores = c_scores
            running_indices = c_idx
        else:
            combined_scores = torch.cat([running_scores, c_scores], dim=1)
            combined_indices = torch.cat([running_indices, c_idx], dim=1)

            best_scores, best_positions = torch.topk(combined_scores, k=top_k, dim=1, largest=True)
            running_scores = best_scores
            running_indices = torch.gather(combined_indices, 1, best_positions)

        del cand_chunk, cand_gpu, chunk_sims

    assert running_indices is not None and running_scores is not None
    return running_indices.cpu(), running_scores.cpu()


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    true_candidate_indices: List[int],
    candidate_embeddings: Union[np.ndarray, torch.Tensor],
    top_k_list: List[int] = [1, 5, 25],
    chunk_size: int = 5000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate retrieval metrics for queries against candidate embeddings with bounded memory."""
    top_k_max = max(top_k_list)
    topk_indices, _ = chunked_topk_search(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        top_k=top_k_max,
        chunk_size=chunk_size,
        device=device,
    )

    ranks = []
    for q_idx, true_c in enumerate(true_candidate_indices):
        preds = topk_indices[q_idx].tolist()
        if true_c in preds:
            rank = preds.index(true_c) + 1  # 1-based
        else:
            rank = top_k_max + 1  # Missed in top_k
        ranks.append(rank)

    return compute_retrieval_metrics(ranks, top_k_list=top_k_list)
