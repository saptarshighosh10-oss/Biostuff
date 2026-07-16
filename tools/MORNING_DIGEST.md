# Morning digest — 2026-07-16

## Decisions

- **ACCEPTED:**
  - **Roche-34 relative HIC** — 17 paired, standard-AA VH/VL rows were retained from the 34-row workbook (relative-HIC range `-0.050` to `0.550`; higher = more hydrophobic). Against the never-trained-on 328-antibody Shehata holdout, the top-15%-HIC-risk AUC increased from `0.671` with GDPa HIC data alone to `0.681` after adding Roche (+`0.010`). It clears the `baseline - 0.005` gate.
  - **FLAb Jain 2017 clinical HIC** — 134 paired rows (3 cross-dataset VH duplicates skipped; HIC `8.5–25.0` min) improved the fixed Shehata top-15%-risk AUC from `0.681` to `0.697` (+`0.016`). The HIC platform head was refreshed.
  - **FLAb Jain 2024 clinical PSR** — 43 paired rows (PSR `0.0–0.92`) produced a strict, source-balanced AIMS mean of `0.571` versus `0.574` baseline (−`0.003`), within the predeclared −`0.005` tolerance. A new additive `PSR_high_risk_top15pct` serving tower was added; legacy assay-specific PR regressors remain unchanged.
- **QUARANTINED:**
  - **AbSci unlocking-de-novo** — the mechanical shell-quoting issue was corrected and the CSV was inspected. It has numeric AC-SINS, polyspecificity, and HIC endpoints but only HCDR3 strings, not full VH/VL sequences, so it is unusable.
  - **ANDD (Zenodo)** — 48,800 antibody/nanobody sequences, but the measured endpoint is antigen binding affinity, not AC-SINS, PSR/polyreactivity, or HIC.
  - **Harvey 2022 VHH PSR** — 140,901 valid VHH rows (120 duplicate VHs removed) were downloaded from Hugging Face, with binary PSR labels where `1` means high polyreactivity. The deterministic, source-balanced gate used 2,000 VHH rows and the same fixed 2,000-row cap for each large library source because the full Python-list transfer implementation exceeded local memory. Mean strict-holdout AUC fell from `0.574` to `0.566` (−`0.008`): AIMS-mouse improved `0.645`→`0.649`, but VH-deduplicated AIMS-flu fell `0.503`→`0.483` (only 38 unique VHs). It fails the −`0.005` threshold, so it remains in the CSV but is excluded from training.
  - **Boughter et al. 2020 ELISA polyreactivity** — the public 914-row export really has binary labels and full VH sequences, but 516 of its 835 unique strict-AA VHs overlap the frozen AIMS mouse/flu holdouts from that same study. It was not ingested: even after molecule filtering, treating the remainder as independent training data would undermine source-held-out evaluation.
  - **FLAb Jain 2024 clinical HIC** — all 43 valid paired rows were retained (HIC `8.4–16.0` min), but the Shehata-holdout AUC declined `0.681`→`0.675` (−`0.006`), just beyond tolerance.
  - **FLAb Jain 2017 clinical PSR** — 136 valid paired rows (one duplicate skipped; PSR `0.0–0.81`) reduced the source-balanced AIMS mean `0.574`→`0.561` (−`0.013`), so it is excluded from training.
  - **FLAb Jain 2024 clinical AC-SINS** — 43 exact paired rows were verified, but not ingested after the source-normalized AC-SINS AUC on a separate 137-antibody Jain-2017 clinical holdout fell `0.623`→`0.591` (−`0.032`).

## Current platform artifact

`model/saved/platform.pkl` now has a source-normalized `HIC_high_risk_top15pct` classifier trained on 866 rows: GDPa1 240, GDPa2 34, GDPa4.1 113, Shehata 328, Roche 17, and accepted Jain-2017 HIC 134. It also has the new additive `PSR_high_risk_top15pct` classifier trained on 893 per-assay/source-normalized observations (including accepted Jain-2024 PSR 43). Both new heads have feature width `1795` and were smoke-tested. The historical GDPa-only scorecard remains aggregation AUC `0.72`, hydrophobicity AUC `0.70`, polyreactivity AUC `0.70`; it is not a re-estimate of these external-data gates.

## Queues

- **Auto queue:** Roche-34, Jain-2017 HIC, and Jain-2024 PSR are `ingested`; Harvey VHH, Boughter, Jain-2024 HIC, and Jain-2017 PSR are `quarantined`; no remaining row is `new` or `downloaded`.
- **Human queue:**
  - Borowska et al. 2023 — Cell Reports supplementary for polyreactivity.
  - SEC developability models (bioRxiv 2025) — supplementary URL returned HTTP 403 to curl; exact URL is in `tools/human_queue.csv`.
  - Teixeira 2021 drug-like antibodies — free PMC article describes clone-level ELISA polyspecificity and AC-SINS, but its supplement is behind PMC's proof-of-work download page; the advertised OA-package URL also returned HTTP 404.
  - NB0268 integrin-alpha11 HIC variants — the article explicitly says its workbook has full sequences and HIC retention times for 97 variants. PMC returned a proof-of-work page instead of the XLSX and the Taylor & Francis mirror returned HTTP 403. The direct workbook URL is in `tools/human_queue.csv`.
  - Hierarchical-clustering mAb developability 2025 — open PMC article reports 65 affinity-matured or immunization mAbs with HIC, PSR, AC-SINS and SEC measurements across four XLSX supplements. Direct PMC links returned proof-of-work HTML, so sequence availability still needs confirmation after a browser download.
  - Anti-MSLN ADC CDR variants 2021 — open-access article says Supplementary Table S1 contains scFv CDR-variant sequences and numerical data, with HIC on selected IgG variants. The direct Springer workbook returned HTTP 403 to curl; exact link is in the human queue.
  - Armstrong et al. 2024 anti-IL8 mutants — open article reports AC-SINS on a ~25-variant panel, but its PMC page was reCAPTCHA-gated before the supplementary sequence/value pairing could be checked.

## Other checked lead

- The already-downloaded supplement for *Reduction of therapeutic antibody self-association using yeast-display selections and machine learning* was unusable: the machine-readable tables only contain donor characteristics and the AC-SINS results are figures.
- **CrossAbSense** was checked and quarantined: it is a code/model repository that downloads the already-known GDPa1 benchmark, not a new measured dataset.
- **Makowski et al. 2024 CS-SINS/PSP** was checked and quarantined: its public tables have 80 antibody names and measurements but no amino-acid sequences, so joining names to an outside sequence source would violate the ingestion rule.
- **Chen et al. 2024 / Zenodo 14735846** was verified as the exact origin of the already-ingested 373,457 Tessier rows, so it was correctly skipped as a duplicate. **Makowski et al. 2022** is likewise the existing 4,000-row `emi` source. Biogen's much larger PSR/HIC/AC-SINS panel is proprietary and cannot be ingested.
- **PSERM Project 1** has numeric OVA/antigen binding values but only 8-residue mutation strings, not full VH/VL sequences. **Xu et al. 2024** has PSR MFI values keyed to design names and only parent sequences/mutation tables, not exact per-row paired sequences. Both were rejected rather than reconstructed by assumption.
- A pre-specified sparse 3–5-mer sequence diagnostic (logistic regression; the same deterministic capped library training set and strict VH removal) scored `0.583` on AIMS-mouse and `0.321` on the 38-VH AIMS-flu holdout. It is below the current biophysical-plus-charge baseline and has not been promoted to a serving tower.

**Biggest real gain of the night:** the gate-passing 134-row Jain-2017 clinical HIC panel, improving Shehata-holdout AUC by `+0.016` (`0.681`→`0.697`). The holdout has 49 top-15%-HIC positives, so the gain is promising but must not be overstated.
