# OAS (Observed Antibody Space) — loader research notes

Research backing `data/oas.py` (loader) and `tests/test_oas.py`.

## What OAS is (and why it's background/OOD, not supervised)

The Observed Antibody Space (OAS) is a database of natural antibody repertoire
sequences collated from NGS studies — **over a billion unpaired** and ~120k+
**paired** sequences from 80+ studies. It is cleaned, annotated (AIRR/IMGT), and
translated. Crucially, **OAS has no developability assay labels** (no AC-SINS,
HIC, Tm, titer, etc.). It is natural-repertoire *background*: useful as a
negative reference and for out-of-distribution (OOD) detection, not as a
supervised training signal.

Accordingly the loader tags every row `supervision_status="background_ood"`,
consistent with the other background sources in `data/contract.py::SOURCE_ROLES`
(`abdev`, `antiref`, `sabdab`, `anchors` — all `background_ood`).

## File format — the metadata-header-line quirk (verified)

Data is organized as **studies → data-units**. Each data-unit is a single
`.csv.gz` file. Verified structure:

- **Line 1 = data-unit metadata**, a JSON object (Species, Chain, Isotype,
  Author, Run, Disease, Subject, Longitudinal, Total/Unique sequences, ...).
- **Line 2 = the AIRR-style CSV column header.**
- **Lines 3+ = one antibody sequence + annotations per row.**

The official OAS read idiom (from the OAS docs and OPIG/blopig blog) is:

```python
import json, pandas as pd
data_unit = "..._Heavy_IGHG.csv.gz"
metadata  = json.loads(','.join(pd.read_csv(data_unit, nrows=0).columns))  # line 1
sequences = pd.read_csv(data_unit, header=1)                                # rows, header on line 2
```

The `','.join(... .columns)` trick reconstructs the JSON metadata string because
pandas splits that first line on the JSON's internal commas. Our stdlib loader
does the equivalent without pandas: open with `gzip`, `readline()` to consume and
parse line 1 as metadata, then hand the remaining stream to `csv.DictReader`,
which then reads line 2 as the header. Metadata parse is best-effort (also
tolerates a CSV-quoted `"{""k"": ...}"` variant); on failure `study_id` falls
back to the filename stem.

Note: Safari may auto-unzip `.csv.gz` on download and corrupt it — OAS docs
recommend Chrome/Firefox or disabling auto-unzip. The loader handles both
`.csv.gz` and already-decompressed `.csv`.

## Verified column names

### Unpaired unit (single chain per file)
Key AIRR columns: `sequence` (nt), **`sequence_alignment_aa`** (the aa sequence),
`germline_alignment_aa`, **`v_call`**, `d_call`, **`j_call`**, **`locus`**
(`IGH` = heavy; `IGK`/`IGL` = light), `cdr1_aa`/`cdr2_aa`/`cdr3_aa`,
`junction_aa`, `v_identity`/`d_identity`/`j_identity`, `Redundancy`,
`ANARCI_numbering`, `ANARCI_status`. Chain is fixed per file (filename tokens
`_Heavy_`/`_Light_` and the `locus` column).

### Paired unit (10x, VH+VL per row)
Every per-chain column is suffixed `_heavy` / `_light`, e.g.:
**`sequence_alignment_aa_heavy`**, **`sequence_alignment_aa_light`**,
`v_call_heavy` / `v_call_light`, `j_call_heavy` / `j_call_light`,
`cdr3_aa_heavy` / `cdr3_aa_light`, `ANARCI_status_heavy` / `_light`.
Missing light annotations appear as `NaN`.

The loader's alias map keys on these (normalized lowercase): heavy aa =
{`sequence_alignment_aa_heavy`, ...}, light aa = {`sequence_alignment_aa_light`,
...}, unpaired single-chain aa = {`sequence_alignment_aa`}. Presence of a
`_heavy`/`_light` aa column ⇒ paired schema; otherwise unpaired (chain from
`locus`). `sequence_alignment_aa` can carry IMGT gap chars (`.`/`-`), so the
loader strips them to recover the real amino-acid sequence.

## Sources (cited)

- OAS home: https://opig.stats.ox.ac.uk/webapps/oas/
- Unpaired OAS documentation (format + metadata-line + read idiom):
  https://opig.stats.ox.ac.uk/webapps/oas/documentation/
- Paired OAS documentation (`_heavy`/`_light` schema):
  https://opig.stats.ox.ac.uk/webapps/oas/documentation_paired/
- OPIG/blopig, "Exploring the Observed Antibody Space (OAS)" (2023) — read
  idioms, `header=1`, metadata via `json.loads(','.join(...columns))`,
  `ANARCI_status`/`Redundancy` filtering:
  https://www.blopig.com/blog/2023/06/exploring-the-observed-antibody-space-oas/
- Olsen, Boyles, Deane (2021/2022), *Protein Science* 31:141–146 — updated OAS:
  https://doi.org/10.1002/pro.4205
- Kovaltsuk, Leem et al. (2018), *J. Immunol.* 201(8):2502–2509 — original OAS:
  https://doi.org/10.4049/jimmunol.1800708
