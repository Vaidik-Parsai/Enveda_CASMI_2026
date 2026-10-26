"""Unit tests for spectrum preprocessing and molecular fingerprint generation."""

import numpy as np
from enveda.data.preprocessing import (
    bin_spectrum_vector,
    extract_single_ce_stats,
    filter_spectrum,
    generate_morgan_fingerprint,
)


def test_filter_spectrum_normal():
    mzs = [10.0, 50.0, 100.0, 500.0]
    intensities = [0.005, 0.5, 0.8, 0.9]  # 0.005 is below 0.01 threshold
    filtered_mzs, filtered_ints = filter_spectrum(
        mzs, intensities, precursor_mz=150.0, threshold=0.01, top_n=128
    )
    # 10.0 filtered out by threshold, 500.0 filtered out by precursor_mz (150 + 2.0 = 152.0)
    assert len(filtered_mzs) == 2
    assert np.allclose(filtered_mzs, [50.0, 100.0])
    assert np.allclose(filtered_ints, [0.5, 0.8])


def test_filter_spectrum_empty_and_malformed():
    # Empty inputs
    m_empty, i_empty = filter_spectrum([], [], precursor_mz=100.0)
    assert len(m_empty) == 0 and len(i_empty) == 0

    # Non-finite values
    m_nan = [50.0, np.nan, 80.0]
    i_inf = [0.5, 0.2, np.inf]
    filtered_m, filtered_i = filter_spectrum(m_nan, i_inf, precursor_mz=100.0)
    assert len(filtered_m) == 1
    assert filtered_m[0] == 50.0

    # Mismatched lengths
    m_long = [50.0, 60.0, 70.0]
    i_short = [0.5]
    f_m, f_i = filter_spectrum(m_long, i_short, precursor_mz=100.0)
    assert len(f_m) == 1 and len(f_i) == 1


def test_bin_spectrum_vector_dimensions_and_normalization():
    mzs = [10.05, 20.35, 150.72]
    intensities = [0.2, 0.8, 1.0]
    binned = bin_spectrum_vector(mzs, intensities, bin_size=0.1, max_mz=1000.0, num_bins=10000)
    assert binned.shape == (10000,)
    assert binned.dtype == np.float32
    assert np.isclose(binned.max(), 1.0)
    assert binned.min() >= 0.0


def test_bin_spectrum_empty():
    binned = bin_spectrum_vector([], [], bin_size=0.1, max_mz=1000.0, num_bins=10000)
    assert binned.shape == (10000,)
    assert np.all(binned == 0.0)


def test_morgan_fingerprint_valid():
    smiles = "CCO"  # Ethanol
    fp, valid = generate_morgan_fingerprint(smiles, radius=2, n_bits=2048)
    assert valid is True
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8
    assert fp.sum() > 0


def test_morgan_fingerprint_invalid_smiles():
    invalid_smiles = "INVALID_SMILES_STRING_123"
    fp, valid = generate_morgan_fingerprint(invalid_smiles, radius=2, n_bits=2048)
    assert valid is False
    assert fp.shape == (2048,)
    assert fp.sum() == 0


def test_extract_single_ce_stats():
    ce_list = [10.0, 20.0, 30.0]
    c_min, c_max, c_mean, c_med, c_cnt = extract_single_ce_stats(ce_list)
    assert c_min == 10.0 and c_max == 30.0 and c_mean == 20.0 and c_cnt == 3

    # Fallback
    f_min, f_max, f_mean, f_med, f_cnt = extract_single_ce_stats(None)
    assert f_min == 20.0 and f_cnt == 0
