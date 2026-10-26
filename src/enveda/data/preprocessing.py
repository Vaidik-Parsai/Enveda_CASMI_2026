"""Centralized spectrum, molecule, and collision energy preprocessing logic."""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# Suppress verbose RDKit warnings
RDLogger.DisableLog("rdApp.*")


def filter_spectrum(
    mzs: Union[List[float], np.ndarray],
    intensities: Union[List[float], np.ndarray],
    precursor_mz: float,
    threshold: float = 0.01,
    top_n: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter noise peaks and retain the top peaks within precursor mass bounds.
    
    Args:
        mzs: List or array of peak m/z values.
        intensities: List or array of normalized peak intensities.
        precursor_mz: Precursor m/z for the spectrum.
        threshold: Minimum relative intensity threshold.
        top_n: Maximum number of peaks to retain.
        
    Returns:
        Tuple of (filtered_mzs, filtered_intensities) as float32 NumPy arrays.
    """
    if mzs is None or intensities is None or len(mzs) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    mzs = np.asarray(mzs, dtype=np.float32)
    intensities = np.asarray(intensities, dtype=np.float32)

    if len(mzs) != len(intensities):
        min_len = min(len(mzs), len(intensities))
        mzs = mzs[:min_len]
        intensities = intensities[:min_len]

    # Handle non-finite values
    finite_mask = np.isfinite(mzs) & np.isfinite(intensities)
    mzs = mzs[finite_mask]
    intensities = intensities[finite_mask]

    if len(mzs) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # 1. Intensity threshold
    mask = intensities >= threshold

    # 2. Precursor mass limit with 2.0 Da allowance for isotopes
    valid_prec = precursor_mz if (precursor_mz is not None and np.isfinite(precursor_mz) and precursor_mz > 0) else 2000.0
    mask = mask & (mzs <= (valid_prec + 2.0)) & (mzs >= 0.0)

    mzs = mzs[mask]
    intensities = intensities[mask]

    # 3. Retain top-n most intense peaks
    if len(intensities) > top_n:
        top_indices = np.argsort(intensities)[-top_n:]
        mzs = mzs[top_indices]
        intensities = intensities[top_indices]

    return mzs, intensities


def bin_spectrum_vector(
    mzs: Union[List[float], np.ndarray],
    intensities: Union[List[float], np.ndarray],
    bin_size: float = 0.1,
    max_mz: float = 1000.0,
    num_bins: int = 10000,
) -> np.ndarray:
    """Convert variable peak lists into a fixed-length normalized vector.
    
    Transforms intensities using sqrt and max-normalization matching the baseline.
    
    Args:
        mzs: Filtered peak m/z values.
        intensities: Filtered peak intensities.
        bin_size: Width of each m/z bin.
        max_mz: Upper m/z limit.
        num_bins: Total number of bins (default 10,000).
        
    Returns:
        1D float32 array of shape (num_bins,).
    """
    binned = np.zeros(num_bins, dtype=np.float32)
    if len(mzs) == 0 or len(intensities) == 0:
        return binned

    mzs = np.asarray(mzs, dtype=np.float32)
    intensities = np.asarray(intensities, dtype=np.float32)

    col_indices = (mzs / bin_size).astype(np.int32)
    valid_mask = (col_indices >= 0) & (col_indices < num_bins)

    valid_cols = col_indices[valid_mask]
    valid_ints = intensities[valid_mask]

    if len(valid_cols) == 0:
        return binned

    # Take the maximum intensity in case multiple peaks fall in the same bin
    for col, val in zip(valid_cols, valid_ints):
        if val > binned[col]:
            binned[col] = val

    # Square root transformation
    binned = np.sqrt(binned)

    # Max normalization
    max_val = np.max(binned)
    if max_val > 0.0:
        binned /= max_val

    return binned


def extract_single_ce_stats(ce_val: Any) -> Tuple[float, float, float, float, int]:
    """Extract min, max, mean, median, count from collision energy value."""
    if isinstance(ce_val, (list, tuple, np.ndarray)) and len(ce_val) > 0:
        arr = np.asarray(ce_val, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            return float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)), float(np.median(arr)), len(arr)
    elif isinstance(ce_val, (int, float)) and np.isfinite(ce_val):
        v = float(ce_val)
        return v, v, v, v, 1

    # Fallback default matching existing notebook
    return 20.0, 20.0, 20.0, 20.0, 0


def generate_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[np.ndarray, bool]:
    """Generate Morgan fingerprint from SMILES string.
    
    Args:
        smiles: Normalized SMILES string.
        radius: Morgan radius (default 2 for ECFP4).
        n_bits: Number of bits in fingerprint (default 2048).
        
    Returns:
        Tuple of (fingerprint_uint8_array, is_valid_bool).
    """
    if not smiles or not isinstance(smiles, str):
        return np.zeros(n_bits, dtype=np.uint8), False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8), False

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return fp, True
