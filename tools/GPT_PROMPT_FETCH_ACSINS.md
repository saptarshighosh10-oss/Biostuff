You are helping me gather more public antibody self-association (AC-SINS) data with
sequences, to add to a machine-learning model. Assume I'm a beginner. Be rigorously honest:
many antibody papers do NOT publish sequences (IP reasons) or only give plots, not tables —
if a source doesn't have usable sequence + value pairs, SAY SO and skip it. Do not fabricate
sequences or numbers, and do not guess values off a figure.

## Goal
Produce ONE file: ~/Downloads/antibody_sources/extra_acsins_dataset.csv
Columns exactly: source, antibody_id, vh, vl, assay_metric, value, ph, direction, notes
  - vh, vl = amino-acid sequences (vl empty if only VH/VHH given). Standard 20 AAs only.
  - assay_metric = the self-association metric name as published (e.g. "AC-SINS", "CSI-BLI").
  - value = the numeric measurement (keep the published units; put units in notes).
  - ph = assay pH if stated, else empty.
  - direction = "higher_bad" if a higher value means MORE self-association (usual for AC-SINS),
    else "lower_bad". State your reasoning in notes if unsure.
Only include rows where you have BOTH a sequence and a numeric value for the same antibody.

## Sources to check (in priority order)
1. Phan et al. 2022, "High-throughput profiling of antibody self-association in multiple
   formulations" — https://pmc.ncbi.nlm.nih.gov/articles/PMC9291693/
   Check its Data Availability + supplementary tables for sequences + AC-SINS values.
2. Arsiwala et al. 2025, "A high-throughput platform for biophysical antibody
   [developability]" — https://pmc.ncbi.nlm.nih.gov/articles/PMC12674332/
   Look for HIC + AC-SINS values with sequences (HEK/CHO-expressed antibodies).
3. Mieczkowski et al. 2021, "Characterization and Modeling of Reversible Antibody
   Self-association" — https://www.mdpi.com/2073-4468/10/1/8
   Check supplementary for AC-SINS + sequences.

## For each source, do this
- Open the paper, find "Data Availability" / "Supplementary Materials" / any linked GitHub or
  Zenodo. Download the supplementary tables (Excel/CSV/PDF) if present.
- Extract antibody id, VH, VL, the self-association metric, value, pH.
- If sequences are NOT provided (common!), record that source as "no usable sequences" and move on.

## Output & honest report
- Write the CSV. Then print: for each source — how many usable rows you got, or why you got
  zero (no sequences / only figures / gated). Total usable rows.
- If ALL three yield nothing usable, say so plainly — that is a valid, useful result. Do not
  pad the file to look productive.

## Rules
- No fabricated sequences or values, ever. No reading numbers off charts.
- Keep per-source labels/units as published; I normalize per-source later.
- Dependency-light; a short Python script + manual supplementary download is fine.
