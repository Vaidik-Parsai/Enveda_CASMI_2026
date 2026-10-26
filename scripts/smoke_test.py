"""End-to-end lightweight smoke test using synthetic fixtures."""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

# Add src to python path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enveda.data import (
    MetadataPreprocessor,
    bin_spectrum_vector,
    filter_spectrum,
    generate_morgan_fingerprint,
)
from enveda.evaluation import build_candidate_bank, chunked_topk_search, evaluate_retrieval
from enveda.models import SpectrumMoleculeModel
from enveda.training import (
    Trainer,
    load_checkpoint,
    multi_positive_contrastive_loss,
    save_checkpoint,
)
from enveda.utils import set_seed, setup_logger


def run_smoke_test():
    start_time = time.time()
    logger = setup_logger("smoke_test")
    logger.info("Starting lightweight end-to-end smoke test...")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device}")

    # 1. Synthetic data fixture
    n_samples = 16
    batch_size = 8
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "c1ccccc1",  # Benzene
    ] * 4

    synthetic_df = pd.DataFrame({
        "precursor_mz": np.random.uniform(100.0, 500.0, size=n_samples),
        "base_peak_intensity": np.random.uniform(1e4, 1e6, size=n_samples),
        "collision_energy_ev": [[10.0, 20.0, 30.0]] * n_samples,
        "adduct": ["[M+H]+"] * n_samples,
        "ionization_mode": ["positive"] * n_samples,
        "normalized_smiles": test_smiles,
        "inchikey14": [f"MOL_{i % 4}" for i in range(n_samples)],
    })

    # 2. Preprocessing & feature extraction
    logger.info("Testing MetadataPreprocessor fitting & transform...")
    preprocessor = MetadataPreprocessor(target_dim=130)
    meta_features = preprocessor.fit_transform(synthetic_df)
    assert meta_features.shape == (n_samples, 130), f"Expected shape ({n_samples}, 130), got {meta_features.shape}"

    # 3. Spectrum binning & Morgan fingerprints
    logger.info("Testing spectrum filtering, binning, and Morgan fingerprints...")
    spectra = []
    fingerprints = []
    for i in range(n_samples):
        dummy_mzs = [50.0, 100.0, 150.0, 200.0]
        dummy_ints = [0.1, 0.9, 0.4, 0.05]
        mzs, ints = filter_spectrum(dummy_mzs, dummy_ints, precursor_mz=synthetic_df.loc[i, "precursor_mz"])
        binned = bin_spectrum_vector(mzs, ints, num_bins=10000)
        assert binned.shape == (10000,)
        spectra.append(binned)

        fp, valid = generate_morgan_fingerprint(synthetic_df.loc[i, "normalized_smiles"])
        assert valid
        assert fp.shape == (2048,)
        fingerprints.append(fp.astype(np.float32))

    spectra_np = np.stack(spectra, axis=0)
    fps_np = np.stack(fingerprints, axis=0)
    mol_ids = np.array([i % 4 for i in range(n_samples)], dtype=np.int64)

    # 4. Model initialization & forward pass
    logger.info("Testing SpectrumMoleculeModel forward pass & finite outputs...")
    model = SpectrumMoleculeModel(
        spectrum_dim=10000,
        metadata_dim=130,
        fingerprint_dim=2048,
        embedding_dim=128,
    ).to(device)

    batch_s = torch.from_numpy(spectra_np[:batch_size]).to(device)
    batch_m = torch.from_numpy(meta_features[:batch_size]).to(device)
    batch_fp = torch.from_numpy(fps_np[:batch_size]).to(device)
    batch_ids = torch.from_numpy(mol_ids[:batch_size]).to(device)

    s_emb, m_emb = model(batch_s, batch_m, batch_fp)
    assert s_emb.shape == (batch_size, 128)
    assert m_emb.shape == (batch_size, 128)
    assert torch.isfinite(s_emb).all().item()
    assert torch.isfinite(m_emb).all().item()

    # 5. Loss calculation & backward pass
    logger.info("Testing contrastive loss and gradient flow...")
    loss = multi_positive_contrastive_loss(s_emb, m_emb, batch_ids)
    assert torch.isfinite(loss).item()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Gradient missing on parameter"
            assert torch.isfinite(p.grad).all().item(), "Non-finite gradient"

    # 6. Checkpoint save & reload
    logger.info("Testing checkpoint save and restore...")
    tmp_ckpt = Path("outputs/scratch/smoke_test_ckpt.pt")
    tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(tmp_ckpt, model=model, step=1, loss_history=[loss.item()])

    model.eval()
    model_reloaded = SpectrumMoleculeModel().to(device)
    load_checkpoint(tmp_ckpt, model=model_reloaded, device=device)
    model_reloaded.eval()

    with torch.no_grad():
        s_emb_eval, _ = model(batch_s, batch_m, batch_fp)
        s_emb_reload, _ = model_reloaded(batch_s, batch_m, batch_fp)
    assert torch.allclose(s_emb_eval, s_emb_reload, atol=1e-5), "Reloaded model outputs mismatch!"

    # 7. Candidate bank & chunked retrieval search
    logger.info("Testing candidate bank build and chunked similarity search...")
    unique_smiles = list(set(test_smiles))
    cand_emb_path, cand_manifest_path = build_candidate_bank(
        smiles_list=unique_smiles,
        model=model,
        output_dir="outputs/scratch/candidate_bank",
        batch_size=4,
        device=device,
    )
    cand_embs = np.load(str(cand_emb_path), mmap_mode="r")
    assert cand_embs.shape == (len(unique_smiles), 128)

    topk_idx, topk_scores = chunked_topk_search(
        query_embeddings=s_emb,
        candidate_embeddings=cand_embs,
        top_k=min(3, len(unique_smiles)),
        chunk_size=2,
        device=device,
    )
    assert topk_idx.shape == (batch_size, min(3, len(unique_smiles)))
    assert topk_scores.shape == (batch_size, min(3, len(unique_smiles)))

    metrics = evaluate_retrieval(
        query_embeddings=s_emb,
        true_candidate_indices=[i % len(unique_smiles) for i in range(batch_size)],
        candidate_embeddings=cand_embs,
        top_k_list=[1, 3],
        chunk_size=2,
        device=device,
    )
    assert "recall@1" in metrics
    assert "mrr" in metrics

    total_time = time.time() - start_time
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

    print("\n" + "=" * 55)
    print("SMOKE TEST RESULTS SUMMARY")
    print("=" * 55)
    print(f"Status:                 PASSED [All assertions OK]")
    print(f"Device used:            {device}")
    print(f"Peak GPU memory:        {mem_mb:.2f} MB")
    print(f"Batch size:             {batch_size}")
    print(f"Records processed:      {n_samples}")
    print(f"Total runtime:          {total_time:.2f} seconds")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_smoke_test()
