"""Unit tests for group splitting, metadata preprocessor, and batch data structures."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from enveda.data.features import MetadataPreprocessor
from enveda.data.split import create_or_load_group_split


def test_group_split_zero_overlap(tmp_path):
    # Create small synthetic parquet file
    parquet_path = tmp_path / "synthetic_train.parquet"
    split_cache = tmp_path / "split_cache.json"

    molecules = ["MOL_A", "MOL_B", "MOL_C", "MOL_D", "MOL_E"] * 20
    df = pd.DataFrame({
        "inchikey14": molecules,
        "dummy_val": range(len(molecules)),
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(parquet_path))

    train_rows, val_rows = create_or_load_group_split(
        parquet_path=parquet_path,
        split_cache_path=split_cache,
        group_col="inchikey14",
        val_fraction=0.2,
        seed=42,
    )

    train_mols = set(df.iloc[train_rows]["inchikey14"])
    val_mols = set(df.iloc[val_rows]["inchikey14"])

    # Must be completely disjoint
    assert len(train_mols.intersection(val_mols)) == 0
    assert len(train_rows) + len(val_rows) == len(df)


def test_metadata_preprocessor_consistency(tmp_path):
    train_df = pd.DataFrame({
        "precursor_mz": [100.0, 200.0, 300.0],
        "base_peak_intensity": [1000.0, 5000.0, 10000.0],
        "collision_energy_ev": [[10.0], [20.0, 30.0], [40.0]],
        "adduct": ["[M+H]+", "[M+Na]+", "[M+H]+"],
        "ionization_mode": ["positive", "positive", "negative"],
    })

    val_df = pd.DataFrame({
        "precursor_mz": [150.0],
        "base_peak_intensity": [2000.0],
        "collision_energy_ev": [[25.0]],
        "adduct": ["[M+H]+"],
        "ionization_mode": ["positive"],
    })

    preprocessor = MetadataPreprocessor(target_dim=130)
    train_feats = preprocessor.fit_transform(train_df)
    assert train_feats.shape == (3, 130)

    # Save and reload artifact
    art_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(art_path)
    loaded_preprocessor = MetadataPreprocessor.load(art_path)

    val_feats = loaded_preprocessor.transform(val_df)
    assert val_feats.shape == (1, 130)
    assert np.isfinite(val_feats).all()
