# Thera-SAbDab (Therapeutic Structural Antibody Database) — Loader Research Notes

Research date: 2026-07-14. Sources web-verified (firecrawl scrape) unless a line
is tagged `[ASSUMED]`. These notes back `data/thera_sabdab.py`.

## What it is
Thera-SAbDab tracks every WHO-recognised antibody- / nanobody-derived
**therapeutic** (INN-assigned) and links it to near/exact structural matches in
SAbDab. Every entry with released sequence data is a molecule a company paid the
WHO ~$12k to name and intends to (or did) carry into the clinic — i.e. a
**known-developable** antibody. We use these as **background / OOD reference**
rows of "working" antibodies, NOT assay-supervised training rows.

## Sources
- Search / download UI (current host): https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/
  (redirects to https://sabdab.opig.stats.ox.ac.uk/therasabdab/search/ ; browse-all: `?all=true`)
- About page: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/about
- Paper (authoritative): Raybould, Marks, Lewis, Shi, Bujotzek, Taddese, Deane.
  "Thera-SAbDab: the Therapeutic Structural Antibody Database." Nucleic Acids
  Res. 2020;48(D1):D383-D388. https://doi.org/10.1093/nar/gkz827
  (open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC6943036/)
- ACTUAL download column header (verified): Merck/Sapiens notebook reading the
  Thera-SAbDab bulk CSV —
  https://github.com/Merck/Sapiens/blob/main/notebooks/01_sapiens_antibody_infilling.ipynb

## Download file format — VERIFIED
- Bulk download lives under the search page's **Downloads** tab. Filename as
  distributed: `TheraSAbDab_SeqStruc_OnlineDownload.csv`.
- It is **comma-separated** (the Sapiens notebook reads it with `pd.read_csv`
  default `sep=,`). Despite the task calling it a "TSV", the OPIG online download
  is a CSV; the loader sniffs comma-vs-tab so either works.

## ACTUAL columns (verified from the Sapiens notebook `pd.read_csv` header)
Header row, left to right:
- `Therapeutic` — INN / therapeutic name (unique id, e.g. `adalimumab`).
- `Format` — molecular format: `Whole mAb`, `Fab`, `scFv`, `VHH`/single-domain,
  `Bispecific`, etc.
- `CH1 Isotype` — e.g. IgG1, IgG4, Kappa/Lambda flags.
- `VD LC` — variable-domain light-chain category (Kappa / Lambda / NA for VHH).
- `Highest_Clin_Trial (Oct '21)` — highest clinical trial phase / status reached.
  **The parenthetical date changes with each release** (`(Feb '25)`, `(Oct '21)`,
  ...), so the loader matches on the `highest_clin_trial` prefix and ignores the
  `(...)` suffix.
- `Est. Status` — estimated developmental status (Active / Discontinued / ...).
- `Heavy Sequence` — **VH** amino-acid sequence.
- `Light Sequence` — **VL** amino-acid sequence (empty for single-domain / VHH).
- `Heavy Sequence (if bispec)` — second VH for bispecifics `[ignored by loader]`.
- `Light Sequence (if bispec)` — second VL for bispecifics `[ignored by loader]`.
- `Target` — intended antigen target (e.g. TNF, ERBB2, IL17A).
- `Year Proposed` — INN proposal year `[not emitted]`.

Field-set corroborated by the paper's "Accessibility of the data" section:
"Sequences are supplied alongside the therapeutic INN, format, isotype, light
chain category, highest clinical trial stage reached, and estimated
developmental status."

## VHH / single-domain rows
Single-domain therapeutics (e.g. caplacizumab, ozoralizumab domains) have a VH
only — `Light Sequence` is blank. The loader emits `vl_sequence=None` for these
and still keeps the row (VH is the usable sequence).

## Loader consequences (`data/thera_sabdab.py`)
- Emits one row per therapeutic: `variant_sequence` (==VH), `vh_sequence`,
  `vl_sequence` (None for VHH), `source="thera_sabdab"`,
  `supervision_status="background_ood"`, `study_id="thera_sabdab"`,
  `group_id` (== therapeutic name), plus `clinical_phase`, `format`, `target`
  when the columns are present.
- Rows with no usable VH (`Heavy Sequence` blank) are skipped — some INNs have no
  released sequence yet (paper: 87.1% of INNs mapped to sequence).
- `background_ood` matches the repo's registry role for non-assay reference
  sources (see `data/registry.py` / `tests/test_normalize.py`).
- Alias map absorbs minor header variants and the moving date suffix on
  `Highest_Clin_Trial`. Stdlib-only, never downloads: file must be placed locally.
