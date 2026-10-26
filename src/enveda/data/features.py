"""Tabular metadata feature extraction and preprocessing."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from enveda.data.preprocessing import extract_single_ce_stats


class MetadataPreprocessor:
    """Extracts, standardizes, and encodes tabular metadata features."""

    NUMERIC_COLS = [
        "precursor_mz",
        "base_peak_intensity",
        "ce_min",
        "ce_max",
        "ce_mean",
        "ce_median",
        "ce_count",
    ]
    CATEGORICAL_COLS = ["adduct", "ionization_mode"]

    def __init__(self, target_dim: int = 130):
        self.target_dim = target_dim
        self.transformer: Optional[ColumnTransformer] = None

    @classmethod
    def prepare_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical CE features and format tabular DataFrame."""
        ce_series = df.get("collision_energy_ev", [None] * len(df))
        
        # Fast extraction of collision energy stats
        mins, maxs, means, medians, counts = [], [], [], [], []
        for val in ce_series:
            c_min, c_max, c_mean, c_med, c_cnt = extract_single_ce_stats(val)
            mins.append(c_min)
            maxs.append(c_max)
            means.append(c_mean)
            medians.append(c_med)
            counts.append(c_cnt)

        meta_df = pd.DataFrame({
            "precursor_mz": df.get("precursor_mz", pd.Series([0.0] * len(df))).fillna(0.0).astype(float),
            "base_peak_intensity": df.get("base_peak_intensity", pd.Series([0.0] * len(df))).fillna(0.0).astype(float),
            "ce_min": mins,
            "ce_max": maxs,
            "ce_mean": means,
            "ce_median": medians,
            "ce_count": counts,
            "adduct": df.get("adduct", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str),
            "ionization_mode": df.get("ionization_mode", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str),
        })
        return meta_df

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        """Fit scaler on numerical columns and one-hot encoder on categories."""
        meta_df = self.prepare_dataframe(df)

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.CATEGORICAL_COLS),
            ]
        )
        self.transformer.fit(meta_df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to float32 numpy array matching target dimensions."""
        if self.transformer is None:
            raise RuntimeError("MetadataPreprocessor must be fitted or loaded before calling transform.")

        meta_df = self.prepare_dataframe(df)
        features = self.transformer.transform(meta_df).astype(np.float32)

        # Ensure exact match with model's expected dimension (e.g. 130)
        curr_dim = features.shape[1]
        if curr_dim < self.target_dim:
            pad = np.zeros((features.shape[0], self.target_dim - curr_dim), dtype=np.float32)
            features = np.hstack([features, pad])
        elif curr_dim > self.target_dim:
            features = features[:, : self.target_dim]

        return features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in a single call."""
        return self.fit(df).transform(df)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted preprocessor to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"transformer": self.transformer, "target_dim": self.target_dim}, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MetadataPreprocessor":
        """Load fitted preprocessor from disk."""
        data = joblib.load(filepath)
        obj = cls(target_dim=data.get("target_dim", 130))
        obj.transformer = data["transformer"]
        return obj
