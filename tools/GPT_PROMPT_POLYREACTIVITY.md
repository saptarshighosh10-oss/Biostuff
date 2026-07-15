You are helping me consolidate several antibody datasets into ONE clean labeled file, on
my Mac. Assume I'm a beginner. Write a single Python script I run locally (python3
consolidate.py) — do not ask me to upload anything. Be honest: if a file's format differs
from what I describe, inspect its first lines and adapt; don't guess silently.

## Goal
Build a "polyreactivity / non-specificity" training table from the sources I already
downloaded. Output ONE file: ~/Downloads/antibody_sources/polyreactivity_dataset.csv
with columns exactly: source, vh, vl, label, label_type
  - vh = heavy-chain amino-acid sequence (required)
  - vl = light-chain amino-acid sequence (empty string if not available)
  - label = a number where HIGHER = MORE polyreactive/non-specific
  - label_type = which raw scale it came from (see below)
Higher-polyreactive-is-larger must hold within each source. Do NOT try to make labels
comparable ACROSS sources — I evaluate per-source. De-duplicate identical vh (+vl) within
each source. Drop rows with empty/invalid sequences (keep only standard 20 amino acids).

## Sources and formats (all under ~/Downloads/antibody_sources/)

1) Emi_Pareto_Opt_ML/emi_binding.csv, igg_binding.csv, iso_binding.csv
   CSV with columns: "VH Sequence", "ANT Binding", "OVA Binding", "pI_seq".
   -> vh = VH Sequence; vl = "" (VH only)
   -> label = OVA Binding  (ovalbumin binding = non-specificity; 1 = polyreactive)
   -> label_type = "ova_binary"
   -> source = "emi" / "igg" / "iso" respectively
   (Ignore "ANT Binding" — that's antigen binding, a different property.)

2) AIMS_manuscripts/app_data/full_sequences/ — paired FASTA + reactivity, matched by order:
   - flu_fastaH.txt + flu_fastaL.txt + flu_NumReact.txt        (source="aims_flu")
   - gut_hiv_fastaH.txt + gut_hiv_fastaL.txt + gut_hiv_NumReact.txt (source="aims_gut_hiv")
   - mouse_fastaH.dat + mouse_fastaL.dat + mouse_YN.txt         (source="aims_mouse")
   For each set: the Nth FASTA heavy record pairs with the Nth light record and the Nth
   line of the reactivity file (they are in the same order).
   -> vh = heavy seq, vl = light seq
   -> flu/gut_hiv: label = the NumReact number (count of reactivities), label_type="num_react"
   -> mouse: label = 1 if the YN line is Y/1/yes else 0, label_type="yn_binary"
   FIRST inspect the first ~3 lines of each file (FASTA headers, number formats) and adapt
   the parser to what you actually see. If lengths mismatch, print a warning and align by
   the shortest.

## Output & report
- Write the combined CSV, then print a summary: rows per source, label range per source,
  and total rows. Tell me any files you couldn't parse and why.

## Rules
- Keep it dependency-light (csv + standard library; no pandas required, but pandas is fine
  if already installed).
- Do not invent labels or merge sources. One tidy long table, per-source labels preserved.
- If something is ambiguous, print a clear NOTE rather than guessing.
