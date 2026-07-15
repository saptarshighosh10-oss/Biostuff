# Ginkgo GDPa Antibody Developability Datasets — Loader Research Notes

Research date: 2026-07-14. Sources are web-verified (firecrawl scrape) unless a
line is explicitly tagged `[ASSUMED]`. These notes back `data/gdpa.py`.

## Sources
- Benchmark repo README + data format: https://github.com/ginkgobioworks/abdev-benchmark
- Ground-truth schema (authoritative column list): https://github.com/ginkgobioworks/abdev-benchmark/blob/main/data/schema/README.md
  (raw: https://raw.githubusercontent.com/ginkgobioworks/abdev-benchmark/main/data/schema/README.md)
- Data directory README: https://github.com/ginkgobioworks/abdev-benchmark/tree/main/data
- Dataset catalog + assay lists: https://datapoints.ginkgo.bio/dataset-access

## GDPa1 (primary benchmark) — VERIFIED
File in the benchmark repo: `data/GDPa1_v1.2_20250814.csv` (~247 IgGs).

Ground-truth CSV columns (from `data/schema/README.md`, "Ground Truth Format"):
- **Antibody id:** `antibody_name` (string, unique), `antibody_id` (numeric)
- **Heavy/VH sequence:** `vh_protein_sequence` (also full `hc_protein_sequence`)
- **Light/VL sequence:** `vl_protein_sequence` (also full `lc_protein_sequence`)
- **Alignments:** `heavy_aligned_aho`, `light_aligned_aho`
- **CV folds:** `hierarchical_cluster_IgG_isotype_stratified_fold` (0–4), `random_fold`

Developability assay columns (VERIFIED names + direction from README "Predicted Properties"):
| Column               | Assay family         | Direction   | Notes |
|----------------------|----------------------|-------------|-------|
| `AC-SINS_pH7.4`      | self_association     | higher_bad  | self-interaction at pH 7.4, lower better |
| `HIC`                | hydrophobicity       | higher_bad  | HIC retention time, lower better |
| `Tm2`                | thermostability      | lower_bad   | 2nd melting temp °C, higher better |
| `Titer`              | expression           | lower_bad   | expression titer mg/L, higher better |
| `PR_CHO`             | polyreactivity       | higher_bad  | polyreactivity vs CHO, lower better |
| `SEC %Monomer`       | aggregation          | lower_bad   | % monomer, higher better `[ASSUMED direction; column named in schema README "Additional measurements"]` |
| `SMAC`               | colloidal_stability  | higher_bad  | `[ASSUMED direction]` retention time, lower better |
| `Purity`             | purity               | lower_bad   | rCE-SDS purity, higher better `[ASSUMED direction]` |

The 5 headline benchmark endpoints are `AC-SINS_pH7.4, HIC, Tm2, Titer, PR_CHO`.
GDPa1 also reports (dataset-access page) HAC (heparin), DLS-kD, and multiple
nanoDSF/DSF temps — handled generically by the loader's assay spec where present.

## GDPa2 / GDPa2.1 — VERIFIED (catalog), column names `[ASSUMED]`
GDPa2.1 = 18 VHH (nanobody) constructs (VHH-His and VHH-Fc). Assays (dataset-access):
hydrophobicity (HIC), aggregation (SEC % monomer), colloidal stability (SMAC),
heparin binding (HAC), self-association (AC-SINS at **pH 7.4 and pH 6.0**),
thermostability (nanoDSF Tonset/Tm1/Tm2/Tm3, DSF Tonset/Tm1).

`[ASSUMED]` GDPa2 reuses GDPa1 column conventions (`antibody_name`,
`vh_protein_sequence`, `vl_protein_sequence`, same assay headers). **VHH is
single-domain**: expect `vl_protein_sequence` empty/absent — loader tolerates a
missing VL. Adds `AC-SINS_pH6.0` (same family/direction as pH7.4). Needs
verification against the real GDPa2 file header when access is granted.

## GDPa3 — HARD-GATED, NOT INSPECTED
GDPa3 = 80 IgGs from Observed Antibody Space (published 2026-01-08 on
dataset-access; assays HIC/SEC/nanoDSF/titer/CHO-PR). Per design doc §11
(`docs/superpowers/specs/2026-07-13-antibody-developability-rebuild-design.md`)
it is a **frozen single-shot holdout**: never downloaded, inspected, or logged in
development. `data/gdpa.py:load_gdpa3()` is a tripwire that always raises. Its
columns were deliberately NOT researched.

## Splits / metrics (VERIFIED)
- 5-fold CV via `hierarchical_cluster_IgG_isotype_stratified_fold` (stratified by
  IgG isotype + hierarchical sequence cluster). `heldout-set-sequences.csv` is a
  labels-withheld competition test set (`antibody_name, vh_protein_sequence,
  vl_protein_sequence`).
- Metrics: **Spearman ρ** and **Top-10% recall**, per property, averaged over folds.

## Loader consequences (`data/gdpa.py`)
- Emits one row per (antibody, assay) — an antibody on N assays → N rows — matching
  `data/flab.py::load_all_developability_data` row schema.
- `group_id` = antibody id (all assay rows for one antibody share a group so
  grouped/leave-cluster-out CV never splits the same VH/VL across folds).
  `study_id` = dataset name (`gdpa1`/`gdpa2`).
- Alias map absorbs minor header variants (`VH`/`heavy`/`vh_sequence` → VH, etc.).
- Loader is stdlib-only and never downloads; the file must be placed locally.
