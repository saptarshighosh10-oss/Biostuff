You are extracting a large antibody polyreactivity dataset into an existing project's format,
on my Mac, then re-running the model. Assume I'm a beginner. Write a Python script I run
locally with ~/Biostuff/.venv/bin/python. Be honest: report the real row counts; if a file's
layout differs from what I describe, inspect it and adapt — don't guess or fabricate.

## Context
I have a polyreactivity model. It reads ~/Downloads/antibody_sources/polyreactivity_dataset.csv
(columns: source, vh, vl, label, label_type ; higher label = more polyreactive). I just
downloaded a big Tessier-lab dataset; add it in the same format, then re-run the model.

## Source files (already unzipped)
Folder: ~/Downloads/antibody_sources/tessier_polyreactivity/Tessier-Lab-UMich-Human_Ab_Polyreactivity-c42966e/Supplemental Datasets/
- "Human Ab Poly Dataset S1_v2.xlsx"  (library #1, ~246,293 scFvs)
- "Human Ab Poly Dataset S2_v2.xlsx"  (library #2, ~127,000 scFvs)
Layout of each (read with openpyxl, read_only):
  row 1 = a long description (ignore)
  row 2 = a group label "Sequence" (ignore)
  row 3 = the real headers: Name , VL , VH
  row 4+ = data: Name, VL sequence, VH sequence
The LABEL is in the Name text: if it contains "high" -> label 1 (polyreactive);
if it contains "low" -> label 0 (not polyreactive). Skip rows where Name has neither.

## What to build (consolidate_tessier.py)
1. Read S1 and S2. For each row extract vh (VH col), vl (VL col), label (from Name).
   Keep only standard-20-amino-acid sequences of length >= 20. De-duplicate identical (vh,vl).
2. Use ALL rows — do NOT subsample. Keep every unique (vh,vl) with a valid high/low label
   from both libraries. (Print the count kept per library.) This will be a large training
   run (~300k+ rows); that is intended — just let it run, it may take a while.
3. Append the rows to ~/Downloads/antibody_sources/polyreactivity_dataset.csv with:
     source = "tessier_lib1" (from S1) / "tessier_lib2" (from S2)
     vh, vl, label (0/1), label_type = "polyreactivity_highlow"
   Do NOT delete or alter the existing rows in that file.
4. Print final per-source counts in the combined file.

## Then re-run the model and report
Run:  cd ~/Biostuff && ~/Biostuff/.venv/bin/python -m model.head_c_polyreactivity --out results/head_c_polyreactivity.json
Report the printed per-source AUC table, especially the two new tessier_lib sources.

## Honesty rules
- Real counts only; no fabricated sequences/labels.
- If openpyxl isn't installed: ~/Biostuff/.venv/bin/pip install openpyxl
- Do NOT modify any file under ~/Biostuff/model/ — only add consolidate_tessier.py under
  ~/Downloads/antibody_sources/ and append to the dataset CSV.
- If head_c errors, paste the exact error and stop; don't hack around it.
