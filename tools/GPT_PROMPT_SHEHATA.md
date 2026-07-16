You are fetching and merging a DIVERSE antibody polyreactivity dataset into a project at
~/Biostuff, run locally with ~/Biostuff/.venv/bin/python. Assume I'm a beginner. Write clean
code. Be rigorously honest.

## READ FIRST — rules you must follow (we already made these mistakes; do not repeat them)
1. USABLE only if it has BOTH sequences (VH/VL) AND numeric values. If Shehata's supplementary
   does NOT actually contain sequences, STOP and tell me — do not fabricate or proceed.
2. Keep properties separate. We only want the PSR (polyreactivity) values here — ignore Tm and
   any other columns for this task.
3. This is a DIVERSE dataset (400 different antibodies) — that is exactly why it's valuable.
   Mark its source accordingly.
4. Higher polyreactivity value = more polyreactive (label direction). State units in notes.
5. Do NOT modify any file under ~/Biostuff/model/. Only append to the dataset CSV and add your
   own script under ~/Downloads/antibody_sources/.
6. Report REAL counts. If sequences are missing or the format differs, inspect, adapt, and if
   still unusable, log why and stop. Never read numbers off a figure.

## The dataset
Shehata et al. 2019, Cell Reports, "Affinity Maturation Enhances Antibody Specificity but
Compromises Conformational Stability." DOI 10.1016/j.celrep.2019.05.066.
~400 antibodies with VH/VL sequences + PSR (polyspecificity reagent) binding + HIC + Tm.
Find its supplementary data (Cell Reports supplemental tables / any linked repo) and download
the table that has, per antibody: VH sequence, VL sequence, and a PSR value.

## What to build (~/Downloads/antibody_sources/consolidate_shehata.py)
1. Download the supplementary table with sequences + PSR.
2. Extract per antibody: vh, vl, PSR value. Keep only standard-20-AA sequences length >= 20.
3. Append to ~/Downloads/antibody_sources/polyreactivity_dataset.csv with:
     source = "shehata2019", vh, vl, label = PSR value, label_type = "psr_continuous"
   Preserve all existing rows.
4. Print how many antibodies were added and the PSR value range.

## Then gate-check it (the important part)
Run the quality gate that GPT built earlier:
  cd ~/Biostuff && ~/Biostuff/.venv/bin/python -m tools.auto_pipeline
Report whether "shehata2019" was ACCEPTED or QUARANTINED, and the baseline-vs-trial DIVERSE
generalization numbers. (Unlike the Tessier library data, this diverse set SHOULD have a
chance of helping the generalization number — but report honestly whatever it says.)

## Report back
- Did Shehata's supplementary actually contain sequences? (yes/no)
- How many antibodies added + PSR range.
- The gate decision (accepted/quarantined) with the baseline and trial diverse AUCs.
