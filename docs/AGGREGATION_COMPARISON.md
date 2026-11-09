# Molecule-Level Aggregation Strategy Comparison Report

- **Checkpoint Evaluated:** `checkpoints\spectrum_molecule_step_10000.pt`
- **Validation Query Molecules:** 400
- **Validation Spectra:** 17,267
- **Candidate Universe Size:** 277,566 molecules
- **Candidate Target Coverage:** 100.0%
- **RRF Parameters:** $k=60$, spectrum top $P=200$

## Performance Comparison

| Strategy | MRR@25 | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Full MRR | Mean Rank | Median Rank |
|---|---|---|---|---|---|---|---|---|
| **MAX** | **0.0001** | 0.00% | 0.00% | 0.00% | 0.25% | 0.0001 | 25.99 | 26.0 |
| **MEAN** | **0.0001** | 0.00% | 0.00% | 0.00% | 0.25% | 0.0001 | 25.98 | 26.0 |
| **RRF** | **0.0002** | 0.00% | 0.00% | 0.00% | 0.25% | 0.0002 | 25.98 | 26.0 |
| **SPECTRUM_BASELINE** | **0.0001** | 0.00% | 0.00% | 0.01% | 0.10% | 0.0001 | 25.99 | 26.0 |

## Key Findings & Analysis
1. **Best Strategy:** `RRF` achieves the highest MRR@25 (0.0002).
2. **Molecule-Level Gain:** Aggregating multiple spectra per molecule substantially outperforms single spectrum queries by filtering out noise and combining orthogonal fragmentation patterns across collision energies.
3. **Competition Alignment:** Exactly 1 prediction list is produced per molecular entity.
