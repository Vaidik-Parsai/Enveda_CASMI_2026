"""Unit tests for precursor mass estimation, adduct rules, and candidate filtering."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from enveda.evaluation.mass_filter import (
    ADDUCT_RULES,
    calculate_exact_monoisotopic_mass,
    compute_soft_mass_scores,
    estimate_neutral_mass,
    filter_candidates_by_mass,
    get_molecule_consensus_neutral_mass,
)


def test_calculate_exact_monoisotopic_mass():
    # Ethanol: C2 H6 O -> 2*12 + 6*1.007825 + 15.994915 = 46.041865
    smi_ethanol = "CCO"
    mass = calculate_exact_monoisotopic_mass(smi_ethanol)
    assert np.isclose(mass, 46.041865, atol=1e-4)

    # Caffeine: C8 H10 N4 O2
    smi_caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    mass_caff = calculate_exact_monoisotopic_mass(smi_caffeine)
    assert np.isclose(mass_caff, 194.080376, atol=1e-4)

    # Malformed SMILES returns 0.0
    assert calculate_exact_monoisotopic_mass("INVALID_SMILES_999") == 0.0
    assert calculate_exact_monoisotopic_mass("") == 0.0


def test_estimate_neutral_mass_adducts():
    # True neutral mass = 300.000000
    m_true = 300.000000

    # 1. [M+H]+
    pmz_mh = m_true + 1.007276
    est_mh = estimate_neutral_mass(pmz_mh, "[M+H]+")
    assert np.isclose(est_mh, m_true, atol=1e-5)

    # 2. [M-H]-
    pmz_mmh = m_true - 1.007276
    est_mmh = estimate_neutral_mass(pmz_mmh, "[M-H]-")
    assert np.isclose(est_mmh, m_true, atol=1e-5)

    # 3. [M+Na]+
    pmz_mna = m_true + 22.989218
    est_mna = estimate_neutral_mass(pmz_mna, "[M+Na]+")
    assert np.isclose(est_mna, m_true, atol=1e-5)

    # 4. [M+K]+
    pmz_mk = m_true + 38.963158
    est_mk = estimate_neutral_mass(pmz_mk, "[M+K]+")
    assert np.isclose(est_mk, m_true, atol=1e-5)

    # 5. [M+NH4]+
    pmz_nh4 = m_true + 18.033823
    est_nh4 = estimate_neutral_mass(pmz_nh4, "[M+NH4]+")
    assert np.isclose(est_nh4, m_true, atol=1e-5)

    # 6. [M+CH2O2-H]- (Formate adduct)
    pmz_fa = m_true + 44.998201
    est_fa = estimate_neutral_mass(pmz_fa, "[M+CH2O2-H]-")
    assert np.isclose(est_fa, m_true, atol=1e-5)

    # 7. Multiply charged: [M+2H]2+ -> (M + 2*1.007276)/2
    pmz_2h = (m_true + 2.014552) / 2.0
    est_2h = estimate_neutral_mass(pmz_2h, "[M+2H]2+")
    assert np.isclose(est_2h, m_true, atol=1e-5)

    # 8. Dimer adduct: [2M+H]+ -> 2M + 1.007276
    pmz_2m = 2.0 * m_true + 1.007276
    est_2m = estimate_neutral_mass(pmz_2m, "[2M+H]+")
    assert np.isclose(est_2m, m_true, atol=1e-5)


def test_consensus_neutral_mass_multiple_spectra():
    # Molecule with 3 spectra in different modes:
    # Spectrum 1: [M+H]+ at 301.007276
    # Spectrum 2: [M-H]- at 298.992724
    # Spectrum 3: [M+Na]+ at 322.989218
    p_mzs = [301.007276, 298.992724, 322.989218]
    adducts = ["[M+H]+", "[M-H]-", "[M+Na]+"]

    consensus = get_molecule_consensus_neutral_mass(p_mzs, adducts)
    assert np.isclose(consensus, 300.0, atol=1e-4)


def test_filter_candidates_by_mass():
    expected_m = 300.0
    cand_masses = np.array([
        300.001,   # 3.3 ppm diff -> inside 20 ppm
        300.005,   # 16.7 ppm diff -> inside 20 ppm
        300.010,   # 33.3 ppm diff -> outside 20 ppm, inside 50 ppm
        250.0,     # far outside
        500.0,     # far outside
    ], dtype=np.float32)

    # 20 ppm filter
    mask_20 = filter_candidates_by_mass(expected_m, cand_masses, tolerance_ppm=20.0)
    assert np.array_equal(mask_20, [True, True, False, False, False])

    # 50 ppm filter
    mask_50 = filter_candidates_by_mass(expected_m, cand_masses, tolerance_ppm=50.0)
    assert np.array_equal(mask_50, [True, True, True, False, False])

    # Absolute Da filter (0.006 Da)
    mask_da = filter_candidates_by_mass(expected_m, cand_masses, tolerance_da=0.006)
    assert np.array_equal(mask_da, [True, True, False, False, False])


def test_compute_soft_mass_scores():
    expected_m = 300.0
    cand_masses = np.array([300.0, 300.009, 500.0], dtype=np.float32)

    scores = compute_soft_mass_scores(expected_m, cand_masses, sigma_ppm=30.0)
    # At 0 ppm diff, score = 1.0
    assert np.isclose(scores[0], 1.0, atol=1e-4)
    # At 30 ppm diff (delta = 0.009 at mass 300), score = exp(-0.5 * 1^2) = exp(-0.5) ~ 0.6065
    assert np.isclose(scores[1], np.exp(-0.5), atol=1e-3)
    # At large mass error (500 vs 300), score clips to min_score (1e-4)
    assert np.isclose(scores[2], 1e-4, atol=1e-6)

