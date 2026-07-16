# Morning digest — 2026-07-16

## Decisions

- **ACCEPTED:**
  - **Roche-34 relative HIC** — 17 paired, standard-AA VH/VL rows were retained from the 34-row workbook (relative-HIC range `-0.050` to `0.550`; higher = more hydrophobic). Against the never-trained-on 328-antibody Shehata holdout, the top-15%-HIC-risk AUC increased from `0.671` with GDPa HIC data alone to `0.681` after adding Roche (+`0.010`). It clears the `baseline - 0.005` gate.
  - **FLAb Jain 2017 clinical HIC** — 134 paired rows (3 cross-dataset VH duplicates skipped; HIC `8.5–25.0` min) improved the fixed Shehata top-15%-risk AUC from `0.681` to `0.697` (+`0.016`). The HIC platform head was refreshed.
  - **FLAb Jetha 2019 homology HIC** — 90 paired rows were retained from 97 downloaded HIC measurements (7 existing VH duplicates skipped; `21.27–30.00` min). Against the separate 328-antibody Shehata holdout, the GDPa baseline was `0.671` and the Jetha trial was `0.677` (+`0.006`).
  - **FLAb Shanehsazzadeh 2023 HIC-RRT** — 12 paired rows were retained from 13 relative-HIC measurements (1 existing VH duplicate; `1.05–1.56`). The same fixed Shehata gate increased from `0.671` to `0.689` (+`0.018`).
  - **FLAb Jain 2024 clinical PSR** — 43 paired rows (PSR `0.0–0.92`) produced a strict, source-balanced AIMS mean of `0.571` versus `0.574` baseline (−`0.003`), within the predeclared −`0.005` tolerance. A new additive `PSR_high_risk_top15pct` serving tower was added; legacy assay-specific PR regressors remain unchanged.
- **QUARANTINED:**
  - **AbSci unlocking-de-novo** — the mechanical shell-quoting issue was corrected and the CSV was inspected. It has numeric AC-SINS, polyspecificity, and HIC endpoints but only HCDR3 strings, not full VH/VL sequences, so it is unusable.
  - **ANDD (Zenodo)** — 48,800 antibody/nanobody sequences, but the measured endpoint is antigen binding affinity, not AC-SINS, PSR/polyreactivity, or HIC.
  - **Harvey 2022 VHH PSR** — 140,901 valid VHH rows (120 duplicate VHs removed) were downloaded from Hugging Face, with binary PSR labels where `1` means high polyreactivity. The deterministic, source-balanced gate used 2,000 VHH rows and the same fixed 2,000-row cap for each large library source because the full Python-list transfer implementation exceeded local memory. Mean strict-holdout AUC fell from `0.574` to `0.566` (−`0.008`): AIMS-mouse improved `0.645`→`0.649`, but VH-deduplicated AIMS-flu fell `0.503`→`0.483` (only 38 unique VHs). It fails the −`0.005` threshold, so it remains in the CSV but is excluded from training.
  - **Boughter et al. 2020 ELISA polyreactivity** — the public 914-row export really has binary labels and full VH sequences, but 516 of its 835 unique strict-AA VHs overlap the frozen AIMS mouse/flu holdouts from that same study. It was not ingested: even after molecule filtering, treating the remainder as independent training data would undermine source-held-out evaluation.
  - **FLAb Jain 2024 clinical HIC** — all 43 valid paired rows were retained (HIC `8.4–16.0` min), but the Shehata-holdout AUC declined `0.681`→`0.675` (−`0.006`), just beyond tolerance.
  - **FLAb Jain 2017 clinical PSR** — 136 valid paired rows (one duplicate skipped; PSR `0.0–0.81`) reduced the source-balanced AIMS mean `0.574`→`0.561` (−`0.013`), so it is excluded from training.
  - **FLAb Jain 2024 clinical AC-SINS** — 43 exact paired rows were verified, but not ingested after the source-normalized AC-SINS AUC on a separate 137-antibody Jain-2017 clinical holdout fell `0.623`→`0.591` (−`0.032`).
  - **FLAb Shanehsazzadeh 2023 AC-SINS** — 13 exact paired rows were verified, but likewise not ingested: the same external AUC fell `0.623`→`0.605` (−`0.018`).

## Current platform artifact

`model/saved/platform.pkl` now has a source-normalized `HIC_high_risk_top15pct` classifier trained on 968 rows: GDPa1 240, GDPa2 34, GDPa4.1 113, Shehata 328, Roche 17, accepted Jain-2017 HIC 134, accepted Jetha HIC 90, and accepted Shanehsazzadeh HIC-RRT 12. It also has the new additive `PSR_high_risk_top15pct` classifier trained on 893 per-assay/source-normalized observations (including accepted Jain-2024 PSR 43). Both new heads have feature width `1795` and were smoke-tested. The historical GDPa-only scorecard remains aggregation AUC `0.72`, hydrophobicity AUC `0.70`, polyreactivity AUC `0.70`; it is not a re-estimate of these external-data gates.

## Queues

- **Auto queue:** Roche-34, Jain-2017 HIC, and Jain-2024 PSR are `ingested`; Harvey VHH, Boughter, Jain-2024 HIC, and Jain-2017 PSR are `quarantined`; no remaining row is `new` or `downloaded`.
- **Human queue:**
  - SEC developability models (bioRxiv 2025) — supplementary URL returned HTTP 403 to curl; exact URL is in `tools/human_queue.csv`.

## Other checked lead

- The already-downloaded supplement for *Reduction of therapeutic antibody self-association using yeast-display selections and machine learning* was unusable: the machine-readable tables only contain donor characteristics and the AC-SINS results are figures.
- **CrossAbSense** was checked and quarantined: it is a code/model repository that downloads the already-known GDPa1 benchmark, not a new measured dataset.
- **Borowska et al. 2023 natural polyreactivity** was resolved from its direct Cell Reports assets: the available files are PDFs with plotted ELISA and alanine-scan results, not a machine-readable per-antibody VH/VL/value table. It was removed from the manual-download queue; no figure values were transcribed.
- **Assessment of in-vitro PK correlates 2024** was recovered from the official Europe PMC supplement endpoint. Its 43 paired Fv sequence/value rows are exactly the previously ingested Jain-2024 molecule panel; minor label differences are formatting/rounding only. It was not duplicated or re-gated.
- **Bailly et al. 2020 early-stage developability** was recovered from the official Europe PMC supplement endpoint. Its nested workbooks include 152 numerical HIC/AC-SINS measurements but only anonymized mAb numbers, with no VH/VL amino-acid sequences. It was rejected without an external identity join.
- **NB0268 integrin-alpha11 HIC variants** was likewise recovered from Europe PMC. The workbook has 97 paired variants, 66 with numeric HIC values; every numeric VH is already in the accepted Jetha 2019 HIC source, so it was not duplicated.
- **Hierarchical-clustering mAb developability 2025** was recovered from Europe PMC. Its XLSX supplements provide numerical HIC/PSR/AC-SINS results for anonymous IDs but no VH/VL sequence table, so it was rejected without an identity join.
- **Anti-MSLN ADC CDR variants 2021** was recovered from Europe PMC. Its supplementary PDF has HIC values but only CDR strings, not full paired VH/VL sequences, so no chains or labels were reconstructed.
- **Teixeira et al. 2021 drug-like antibodies** was recovered from Europe PMC. Its nested supplement provides donor metadata and figure files, but no complete antibody sequences or machine-readable property table, so no records were inferred.
- **Armstrong anti-IL8** was removed after auditing its queued PII: it points to an unrelated cell-morphology review, not an antibody data source.
- **ABGen’s 137-row clinical HIC table** was also checked directly: it has paired sequences and values, but every strict VH already occurs in the local Jain-2017/Roche HIC data. It is a rounded duplicate, not a new panel.
- **Hie et al. 2023 language-model antibody evolution** supplied raw PSR flow-cytometry measurements for 32 named designs through Zenodo, but no explicit paired sequence for each design. Parent-related files and aliases are not enough to reconstruct those rows, so it was rejected.
- **Jain et al. 2017 delayed-HIC prediction** was recovered from its Oxford supplement. The public portion has 112 paired clinical sequences with delta-HIC values (`0.00–5.34` min), but all 112 VHs are already present in Jain-2017/Roche HIC rows; the paper’s 5,089-sequence panel was not released.
- **Hoffmann et al. 2024 rabbit polyreactivity NGS** provides raw VH reads at BioProject `PRJNA1052988`, but no per-clone numeric value or released sequence-to-label table (only a PDF supplement). I did not infer labels from bulk-sorted samples.
- **Lecerf et al. 2021 heme-induced polyreactivity** is a real, locally available 113-clinical-antibody sequence/value panel (heme binding and post-heme binding ratios to three unrelated proteins). It is deliberately *not* merged into PSR: that would create a mislabeled endpoint. A separately validated heme-polyreactivity tower would need an independent holdout before it can be trained.
- **Lim et al. 2022 scFv polyreactivity** exposes 19,426 public FACS labels, but only as engineered full-scFv constructs with added overhangs, not explicit VH/VL fields. I did not guess where to split chains.
- **Marks-lab nanobody-polyreactivity repository** was traced to the already-ingested Harvey 2022 VHH experiment (71,772 high and 69,702 low ANARCI-positioned rows). It is not a second independent panel.
- **Makowski et al. 2024 CS-SINS/PSP** was checked and quarantined: its public tables have 80 antibody names and measurements but no amino-acid sequences, so joining names to an outside sequence source would violate the ingestion rule.
- **Chen et al. 2024 / Zenodo 14735846** was verified as the exact origin of the already-ingested 373,457 Tessier rows, so it was correctly skipped as a duplicate. **Makowski et al. 2022** is likewise the existing 4,000-row `emi` source. Biogen's much larger PSR/HIC/AC-SINS panel is proprietary and cannot be ingested.
- The locally downloaded `mmc5.xlsx` is not a new HIC panel: its 18 numeric clone names all occur in the already-ingested Shehata `mmc2.xlsx` data. Seventeen HIC values match exactly; ADI-47166 differs (`9.833` vs `10.09`). It also contains only VH fragments through CDR3 and no VL, so it was not used to duplicate or amend the existing data.
- **PSERM Project 1** has numeric OVA/antigen binding values but only 8-residue mutation strings, not full VH/VL sequences. **Xu et al. 2024** has PSR MFI values keyed to design names and only parent sequences/mutation tables, not exact per-row paired sequences. Both were rejected rather than reconstructed by assumption.
- A pre-specified sparse 3–5-mer sequence diagnostic (logistic regression; the same deterministic capped library training set and strict VH removal) scored `0.583` on AIMS-mouse and `0.321` on the 38-VH AIMS-flu holdout. It is below the current biophysical-plus-charge baseline and has not been promoted to a serving tower.

**Biggest real gain of the night:** the gate-passing 134-row Jain-2017 clinical HIC panel, improving Shehata-holdout AUC by `+0.016` (`0.681`→`0.697`). The latest 12-row Shanehsazzadeh HIC trial also scored `+0.018` on the GDPa-only reference, but its very small size means it should not be overstated. The holdout has 49 top-15%-HIC positives.
