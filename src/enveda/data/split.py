"""Deterministic group-based train/validation splitting by molecular identity."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit


def create_or_load_group_split(
    parquet_path: Union[str, Path],
    split_cache_path: Union[str, Path] = "data/cache/group_split.json",
    group_col: str = "inchikey14",
    fallback_col: str = "normalized_smiles",
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create or load deterministic train/validation indices partitioned by molecular identity.
    
    Reads only the identity column via PyArrow to ensure low memory consumption.
    
    Args:
        parquet_path: Path to train.parquet.
        split_cache_path: Path to cached JSON manifest.
        group_col: Primary molecular identity column.
        fallback_col: Fallback column if primary is unavailable.
        val_fraction: Fraction of data allocated to validation (default 0.05).
        seed: Random seed for GroupShuffleSplit.
        
    Returns:
        Tuple of (train_row_indices, val_row_indices) as numpy arrays.
    """
    cache_file = Path(split_cache_path)
    if cache_file.is_file():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        train_rows = np.array(data["train_indices"], dtype=np.int64)
        val_rows = np.array(data["val_indices"], dtype=np.int64)
        return train_rows, val_rows

    parquet_file = Path(parquet_path)
    if not parquet_file.is_file():
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    # Inspect schema without loading data
    schema = pq.read_schema(str(parquet_file))
    chosen_col = group_col if group_col in schema.names else fallback_col
    if chosen_col not in schema.names:
        raise KeyError(f"Neither '{group_col}' nor '{fallback_col}' found in Parquet columns: {schema.names}")

    # Read ONLY the chosen identity column
    table = pq.read_table(str(parquet_file), columns=[chosen_col])
    groups_series = table.column(0).to_pandas().astype("string")

    # Factorize to compact integer IDs
    molecule_ids, unique_molecules = pd.factorize(groups_series, sort=False)
    n_rows = len(molecule_ids)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed,
    )

    train_rows, val_rows = next(
        splitter.split(
            np.zeros(n_rows, dtype=np.uint8),
            groups=molecule_ids,
        )
    )

    # Verify 0 overlap
    overlap = len(set(molecule_ids[train_rows]).intersection(set(molecule_ids[val_rows])))
    if overlap > 0:
        raise RuntimeError(f"Split error: detected {overlap} overlapping molecules between train and val!")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_rows": int(n_rows),
                "n_unique_molecules": int(len(unique_molecules)),
                "val_fraction": float(val_fraction),
                "seed": int(seed),
                "group_col": chosen_col,
                "train_indices": train_rows.tolist(),
                "val_indices": val_rows.tolist(),
            },
            f,
        )

    return train_rows, val_rows
