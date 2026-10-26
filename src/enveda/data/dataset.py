"""Memory-bounded PyTorch Dataset and iterator implementations."""

from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset

from enveda.data.features import MetadataPreprocessor
from enveda.data.preprocessing import (
    bin_spectrum_vector,
    filter_spectrum,
    generate_morgan_fingerprint,
)


class StreamingParquetDataset(IterableDataset):
    """Memory-bounded streaming Dataset that reads Parquet in chunks using PyArrow."""

    def __init__(
        self,
        parquet_path: str,
        meta_preprocessor: MetadataPreprocessor,
        batch_size: int = 128,
        chunk_size: int = 1024,
        row_indices_filter: Optional[set] = None,
        fingerprint_cache: Optional[Dict[str, np.ndarray]] = None,
        max_cached_fps: int = 50000,
        spectrum_threshold: float = 0.01,
        spectrum_top_n: int = 128,
        bin_size: float = 0.1,
        max_mz: float = 1000.0,
        num_bins: int = 10000,
        fp_radius: int = 2,
        fp_n_bits: int = 2048,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.meta_preprocessor = meta_preprocessor
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.row_indices_filter = row_indices_filter
        self.fingerprint_cache = fingerprint_cache if fingerprint_cache is not None else {}
        self.max_cached_fps = max_cached_fps
        self.spectrum_threshold = spectrum_threshold
        self.spectrum_top_n = spectrum_top_n
        self.bin_size = bin_size
        self.max_mz = max_mz
        self.num_bins = num_bins
        self.fp_radius = fp_radius
        self.fp_n_bits = fp_n_bits
        self.device = device

        self.columns = [
            "ms2_mzs",
            "ms2_normalized_intensities",
            "precursor_mz",
            "base_peak_intensity",
            "collision_energy_ev",
            "adduct",
            "ionization_mode",
            "normalized_smiles",
            "inchikey14",
        ]

    def _get_fingerprint(self, smiles: str) -> np.ndarray:
        if not smiles or not isinstance(smiles, str):
            return np.zeros(self.fp_n_bits, dtype=np.uint8)

        if smiles in self.fingerprint_cache:
            return self.fingerprint_cache[smiles]

        fp, _ = generate_morgan_fingerprint(smiles, radius=self.fp_radius, n_bits=self.fp_n_bits)
        if len(self.fingerprint_cache) < self.max_cached_fps:
            self.fingerprint_cache[smiles] = fp
        return fp

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        parquet_file = pq.ParquetFile(self.parquet_path)
        avail_cols = [c for c in self.columns if c in parquet_file.schema.names]

        # Accumulator for building mini-batches
        b_spectra: List[np.ndarray] = []
        b_meta_dfs: List[pd.DataFrame] = []
        b_fps: List[np.ndarray] = []
        b_mol_ids: List[int] = []

        global_row_offset = 0

        # Stream row-groups / record batches
        for record_batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=avail_cols):
            chunk_df = record_batch.to_pandas()
            chunk_len = len(chunk_df)

            # Filter rows if row_indices_filter is specified
            if self.row_indices_filter is not None:
                mask = [((global_row_offset + i) in self.row_indices_filter) for i in range(chunk_len)]
                if not any(mask):
                    global_row_offset += chunk_len
                    continue
                chunk_df = chunk_df[mask].reset_index(drop=True)

            global_row_offset += chunk_len

            if len(chunk_df) == 0:
                continue

            # Process metadata for this chunk
            meta_features = self.meta_preprocessor.transform(chunk_df)

            # Process each spectrum and molecule in the chunk
            for i, row in chunk_df.iterrows():
                mzs, ints = filter_spectrum(
                    row.get("ms2_mzs", []),
                    row.get("ms2_normalized_intensities", []),
                    row.get("precursor_mz", 0.0),
                    threshold=self.spectrum_threshold,
                    top_n=self.spectrum_top_n,
                )
                spec_vec = bin_spectrum_vector(
                    mzs,
                    ints,
                    bin_size=self.bin_size,
                    max_mz=self.max_mz,
                    num_bins=self.num_bins,
                )

                smi = str(row.get("normalized_smiles", ""))
                fp_vec = self._get_fingerprint(smi)

                # Molecule identity hash for contrastive loss
                mol_key = row.get("inchikey14", smi)
                if not mol_key or pd.isna(mol_key):
                    mol_key = smi
                mol_id = hash(str(mol_key)) & 0x7FFFFFFFFFFFFFFF  # positive 64-bit int

                b_spectra.append(spec_vec)
                b_fps.append(fp_vec.astype(np.float32))
                b_mol_ids.append(mol_id)

            # Accumulate metadata features
            if len(b_meta_dfs) == 0:
                curr_meta = meta_features
            else:
                curr_meta = np.vstack([b_meta_dfs.pop(0), meta_features])

            # Yield batches of size self.batch_size
            while len(b_spectra) >= self.batch_size:
                spec_batch = np.array(b_spectra[: self.batch_size], dtype=np.float32)
                fp_batch = np.array(b_fps[: self.batch_size], dtype=np.float32)
                meta_batch = curr_meta[: self.batch_size]
                id_batch = np.array(b_mol_ids[: self.batch_size], dtype=np.int64)

                b_spectra = b_spectra[self.batch_size :]
                b_fps = b_fps[self.batch_size :]
                curr_meta = curr_meta[self.batch_size :]
                b_mol_ids = b_mol_ids[self.batch_size :]

                batch_dict = {
                    "spectrum": torch.from_numpy(spec_batch),
                    "metadata": torch.from_numpy(meta_batch),
                    "fingerprint": torch.from_numpy(fp_batch),
                    "molecule_ids": torch.from_numpy(id_batch),
                }

                if self.device is not None:
                    batch_dict = {k: v.to(self.device, non_blocking=True) for k, v in batch_dict.items()}

                yield batch_dict

            if len(curr_meta) > 0:
                b_meta_dfs.append(curr_meta)


class MemoryMappedBatchDataset:
    """Pre-indexed batch provider for ultra-fast GPU training without high RAM usage."""

    def __init__(
        self,
        spectrum_data: Any,
        metadata_data: Any,
        fp_matrix: np.ndarray,
        fp_indices: np.ndarray,
        molecule_ids: np.ndarray,
        row_indices: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        self.spectrum_data = spectrum_data
        self.metadata_data = metadata_data
        self.fp_matrix = fp_matrix
        self.fp_indices = fp_indices
        self.molecule_ids = molecule_ids
        self.row_indices = row_indices
        self.device = device

    def __len__(self) -> int:
        return len(self.row_indices)

    def get_batch(self, batch_rows: np.ndarray) -> Dict[str, torch.Tensor]:
        """Fetch and densify only this specific batch slice."""
        actual_rows = self.row_indices[batch_rows]

        # Spectra
        if hasattr(self.spectrum_data, "toarray"):
            spec_np = self.spectrum_data[actual_rows].toarray().astype(np.float32)
        elif hasattr(self.spectrum_data, "iloc"):
            spec_np = self.spectrum_data.iloc[actual_rows].to_numpy().astype(np.float32)
        else:
            spec_np = np.asarray(self.spectrum_data[actual_rows], dtype=np.float32)

        # Metadata
        if hasattr(self.metadata_data, "toarray"):
            meta_np = self.metadata_data[actual_rows].toarray().astype(np.float32)
        elif hasattr(self.metadata_data, "iloc"):
            meta_np = self.metadata_data.iloc[actual_rows].to_numpy().astype(np.float32)
        else:
            meta_np = np.asarray(self.metadata_data[actual_rows], dtype=np.float32)

        # Fingerprint mapping
        batch_fp_indices = self.fp_indices[actual_rows]
        fp_np = self.fp_matrix[batch_fp_indices].astype(np.float32)

        # Molecule IDs
        mol_ids_np = self.molecule_ids[actual_rows].astype(np.int64)

        batch_tensors = {
            "spectrum": torch.from_numpy(spec_np),
            "metadata": torch.from_numpy(meta_np),
            "fingerprint": torch.from_numpy(fp_np),
            "molecule_ids": torch.from_numpy(mol_ids_np),
        }

        if self.device is not None:
            batch_tensors = {k: v.to(self.device, non_blocking=True) for k, v in batch_tensors.items()}

        return batch_tensors
