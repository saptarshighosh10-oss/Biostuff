# Continuous Research Digest

## 2026-07-18 01:15 UTC — loop start

- Fixed development benchmark and GDPa3 freeze were re-verified.
- HIC paired-chain biophysical experiment: 0.7562 → 0.7577 macro-source AUC.
  It cleared the 0.7521 development floor and became the narrow current best,
  but enrichment declined 2.558× → 2.481× and the 0.80 serving gate remains unmet.
- Per source: GDPa1 0.766, GDPa4.1 0.813, Jain 2017 HIC 0.833,
  Jetha 2019 HIC 0.717, Shehata 2019 0.659.
- Best development AUCs: hydrophobicity 0.7577 (0.0423 from goal),
  polyreactivity 0.6715 (0.1285 from goal), aggregation 0.5264 (0.2736 from goal).
- Production serving files remain unchanged. Most promising open lead:
  combine paired-chain features with the already-supported HistGB/ridge learners
  using nested source-held-out selection.
