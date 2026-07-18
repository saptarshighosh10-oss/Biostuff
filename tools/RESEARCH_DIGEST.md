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

## 2026-07-18 03:00 UTC — 0.80 development goal reached

- Seven sealed HIC experiments moved the fixed held-out development score from
  0.7571 to 0.8093 macro-source ROC AUC (+0.0521). The final step added explicit
  whole-Fv composition and VH/VL imbalance descriptors to the continuous-rank
  ensemble; its immediate prior best was 0.7932.
- Final per-source AUC: GDPa1 0.8060, GDPa4.1 0.9237, Jain 2017 HIC 0.8552,
  Jetha 2019 HIC 0.7227, and Shehata 2019 0.7387. All five clear 0.65.
- Three declared seeds scored 0.8017, 0.8144, and 0.8117 (standard deviation
  0.0054). Mean enrichment at a 15% screening budget was 3.078x.
- A second calculation rebuilt the macro score from all 15 outer source/seed
  AUCs and matched 0.8092596742962376 exactly. GDPa3 remained unspent.
- This is honest development evidence, not final or lab-ready validation. The
  strict serving gate failed because no locked validation source is assigned
  and the rank learner does not yet provide train-only Platt calibration.
  Consequently `model/head_platform.py` and `model/saved/platform.pkl` remain
  unchanged.
- End condition reached for this loop: hydrophobicity exceeded the preregistered
  0.80 development goal. Polyreactivity remains 0.6715 and aggregation 0.5264.

## 2026-07-18 05:30 UTC — confirmation corrected the HIC claim

- The 0.8093 HIC development result did not survive the new selection-isolated
  Shehata confirmation. With train-only Platt calibration, the interaction
  family scored 0.7388 on Shehata versus 0.7486 without the interactions; the
  combined macro-source AUC was 0.7820. The HIC 0.85 stretch is stopped rather
  than tuned against this failure.
- Aggregation continuous-rank and molecule-consensus trials were quarantined.
  They scored 0.4855 and 0.5124, respectively, below the registered 0.5264
  development best.
- Reliable heavy-chain CDR1+CDR3 features parsed at 94.8% to 98.4% by source,
  but polyreactivity transfer reached only 0.6645, below the 0.6665 gate floor
  and the registered 0.6715 best. The experiment was reverted as a null result.
- A fixed domain/representation polyreactivity trial is running. It tests
  library-only biophysics and diverse/mixed CDR learners under the same nested
  whole-source holdouts; its candidate menu and gate were fixed before launch.
- GDPa3 remains unspent, and production serving files remain unchanged.
