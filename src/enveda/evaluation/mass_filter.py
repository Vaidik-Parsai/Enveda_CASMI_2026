"""Cheminformatics precursor mass calculation and candidate filtering module."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# Exact masses for adduct calculations (in Daltons)
# Reference: NIST Mass Spectrometry & GNPS standard adduct table
ADDUCT_RULES: Dict[str, Dict[str, float]] = {
    # Positive ionization mode
    "[M+H]+": {"charge": 1.0, "mass_shift": 1.007276, "multiplier": 1.0},
    "[M+Na]+": {"charge": 1.0, "mass_shift": 22.989218, "multiplier": 1.0},
    "[M+K]+": {"charge": 1.0, "mass_shift": 38.963158, "multiplier": 1.0},
    "[M+NH4]+": {"charge": 1.0, "mass_shift": 18.033823, "multiplier": 1.0},
    "[M+Li]+": {"charge": 1.0, "mass_shift": 7.015455, "multiplier": 1.0},
    "[M]+": {"charge": 1.0, "mass_shift": -0.0005486, "multiplier": 1.0},
    "[M+2H]2+": {"charge": 2.0, "mass_shift": 2.014552, "multiplier": 1.0},
    "[M+H+Na]2+": {"charge": 2.0, "mass_shift": 23.996494, "multiplier": 1.0},
    "[M-H2O+H]+": {"charge": 1.0, "mass_shift": -17.003289, "multiplier": 1.0},
    "[M-2H2O+H]+": {"charge": 1.0, "mass_shift": -35.013854, "multiplier": 1.0},
    "[2M+H]+": {"charge": 1.0, "mass_shift": 1.007276, "multiplier": 2.0},
    "[2M+Na]+": {"charge": 1.0, "mass_shift": 22.989218, "multiplier": 2.0},
    "[2M+NH4]+": {"charge": 1.0, "mass_shift": 18.033823, "multiplier": 2.0},
    "[2M+K]+": {"charge": 1.0, "mass_shift": 38.963158, "multiplier": 2.0},

    # Negative ionization mode
    "[M-H]-": {"charge": -1.0, "mass_shift": -1.007276, "multiplier": 1.0},
    "[M+Cl]-": {"charge": -1.0, "mass_shift": 34.969402, "multiplier": 1.0},
    "[M+CH2O2-H]-": {"charge": -1.0, "mass_shift": 44.998201, "multiplier": 1.0},  # Formate [M+HCOO]-
    "[M+HCOO]-": {"charge": -1.0, "mass_shift": 44.998201, "multiplier": 1.0},
    "[M+C2H4O2-H]-": {"charge": -1.0, "mass_shift": 59.013851, "multiplier": 1.0},  # Acetate [M+CH3COO]-
    "[M+CH3COO]-": {"charge": -1.0, "mass_shift": 59.013851, "multiplier": 1.0},
    "[M+Br]-": {"charge": -1.0, "mass_shift": 78.918836, "multiplier": 1.0},
    "[M-H2O-H]-": {"charge": -1.0, "mass_shift": -19.018390, "multiplier": 1.0},
    "[M]-": {"charge": -1.0, "mass_shift": 0.0005486, "multiplier": 1.0},
    "[2M-H]-": {"charge": -1.0, "mass_shift": -1.007276, "multiplier": 2.0},
    "[2M+FA-H]-": {"charge": -1.0, "mass_shift": 44.998201, "multiplier": 2.0},
    "[M-2H]2-": {"charge": -2.0, "mass_shift": -2.014552, "multiplier": 1.0},
}


def calculate_exact_monoisotopic_mass(smiles: str) -> float:
    """Calculate exact monoisotopic neutral molecular weight from SMILES using RDKit."""
    if not smiles or not isinstance(smiles, str):
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(Descriptors.ExactMolWt(mol))


def compute_candidate_bank_exact_masses(candidate_smiles: List[str]) -> np.ndarray:
    """Compute exact monoisotopic neutral molecular weights for all candidate SMILES."""
    masses = np.zeros(len(candidate_smiles), dtype=np.float32)
    for i, smi in enumerate(candidate_smiles):
        masses[i] = calculate_exact_monoisotopic_mass(smi)
    return masses


def estimate_neutral_mass(
    precursor_mz: float,
    adduct: Optional[str],
    ionization_mode: Optional[str] = None,
) -> Optional[float]:
    """Calculate the expected neutral molecular mass from precursor m/z and adduct metadata.
    
    Args:
        precursor_mz: Experimental precursor ion m/z.
        adduct: Adduct string (e.g. "[M+H]+", "[M-H]-").
        ionization_mode: "positive" or "negative" fallback if adduct is unknown.
        
    Returns:
        Estimated neutral mass in Daltons, or None if precursor_mz is invalid.
    """
    if precursor_mz is None or not np.isfinite(precursor_mz) or precursor_mz <= 0:
        return None

    rule = None
    if adduct and isinstance(adduct, str):
        adduct_clean = adduct.strip()
        rule = ADDUCT_RULES.get(adduct_clean)

    if rule is None:
        # Fallback based on ionization mode
        mode = str(ionization_mode).lower() if ionization_mode else "positive"
        if "neg" in mode:
            rule = ADDUCT_RULES["[M-H]-"]
        else:
            rule = ADDUCT_RULES["[M+H]+"]

    charge_mag = abs(rule["charge"])
    mass_shift = rule["mass_shift"]
    mult = rule["multiplier"]

    # (m/z) = (mult * M + mass_shift) / charge_mag
    # => M = ( (m/z) * charge_mag - mass_shift ) / mult
    neutral_m = (precursor_mz * charge_mag - mass_shift) / mult
    return float(neutral_m) if neutral_m > 0 else None


def get_molecule_consensus_neutral_mass(
    precursor_mzs: List[float],
    adducts: List[str],
    ionization_modes: Optional[List[str]] = None,
) -> Optional[float]:
    """Compute consensus neutral mass across multiple spectra of the same molecule."""
    estimates = []
    for i, pmz in enumerate(precursor_mzs):
        add = adducts[i] if i < len(adducts) else None
        mode = ionization_modes[i] if ionization_modes and i < len(ionization_modes) else None
        m_est = estimate_neutral_mass(pmz, add, mode)
        if m_est is not None:
            estimates.append(m_est)

    if len(estimates) == 0:
        return None
    return float(np.median(estimates))


def filter_candidates_by_mass(
    expected_neutral_mass: float,
    candidate_masses: np.ndarray,
    tolerance_ppm: Optional[float] = 50.0,
    tolerance_da: Optional[float] = None,
) -> np.ndarray:
    """Return boolean mask of candidate molecules within mass tolerance.
    
    Args:
        expected_neutral_mass: Expected neutral molecular mass in Da.
        candidate_masses: 1D array of exact neutral masses of candidates.
        tolerance_ppm: Tolerance window in parts-per-million (default 50.0 ppm).
        tolerance_da: Absolute mass tolerance in Daltons (overrides ppm if specified).
        
    Returns:
        1D boolean numpy array where True indicates candidate is within tolerance.
    """
    if tolerance_da is not None:
        delta = np.abs(candidate_masses - expected_neutral_mass)
        return delta <= tolerance_da
    elif tolerance_ppm is not None:
        delta = np.abs(candidate_masses - expected_neutral_mass)
        ppm = (delta / expected_neutral_mass) * 1e6
        return ppm <= tolerance_ppm
    else:
        return np.ones(len(candidate_masses), dtype=bool)


def compute_soft_mass_scores(
    expected_neutral_mass: float,
    candidate_masses: np.ndarray,
    sigma_ppm: float = 30.0,
    min_score: float = 1e-4,
) -> np.ndarray:
    """Compute soft Gaussian mass compatibility scores in [0, 1].
    
    S_mass(c) = exp( - 0.5 * (ppm_diff / sigma_ppm)^2 )
    
    Args:
        expected_neutral_mass: Expected neutral molecular mass in Da.
        candidate_masses: 1D array of exact neutral masses of candidates.
        sigma_ppm: Gaussian width in ppm.
        min_score: Minimum clipping score.
        
    Returns:
        1D float32 numpy array of soft compatibility scores.
    """
    delta = np.abs(candidate_masses - expected_neutral_mass)
    ppm = (delta / expected_neutral_mass) * 1e6
    scores = np.exp(-0.5 * (ppm / sigma_ppm) ** 2)
    return np.clip(scores, min_score, 1.0).astype(np.float32)

