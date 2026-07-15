# Full handoff — antibody aggregation project (read this whole thing first)

You are taking over a task on an antibody machine-learning project. Assume you know
nothing about it. This document re-educates you on **what it is, how to think about it,
where it stands, and exactly what to do.** Read all of it before acting. The person you
are helping is a motivated student — explain things plainly, never bluff.

---

## PART 1 — What the project is (plain English)

Antibodies are drugs. A common way they fail in development is **aggregation** — the
molecules clump together. One key measurable form of this is **self-association**,
measured by an assay called **AC-SINS**. Higher AC-SINS = stickier = worse.

The project builds a model that **predicts, from an antibody's amino-acid sequence, how
sticky it will be** — so a lab can decide which candidates to test first instead of
testing all of them. It is a **triage / ranking tool**, not a replacement for the lab.

---

## PART 2 — How to THINK about this (the part that matters most)

These mental models were learned the hard way. Do not regress on them:

1. **Honesty over hype. Always report the real number, even when it's worse.**
   We repeatedly found the flattering number was wrong. The credibility of this whole
   project rests on numbers that survive scrutiny. If something looks too good, hunt for
   the reason before believing it.

2. **Cross-validation flatters; generalization is the real test.**
   Testing a model on antibodies similar to its training data overstates how good it is.
   The number that matters is performance on **antibodies from a lab/source it has never
   seen** (we simulate this by holding out an entire study at a time).

3. **Data leakage is the enemy.** If the same antibody appears in both training and test
   (even under a different dataset's name), the score is inflated and fake. We check
   overlaps at the **molecule/sequence level** and remove shared antibodies from training.
   The classic clinical-antibody panels (Jain, etc.) appear in *multiple* datasets — this
   is a real trap.

4. **Score the useful thing, not the hard thing.** Ranking every antibody perfectly
   (Spearman) is near-impossible (~0.5 ceiling in the whole field; even the lab assay
   disagrees with itself). Instead we score **"does it catch the worst ~15% sticky ones?"**
   — measured by **AUC** and **catch-rate**. That's both achievable AND what a triage tool
   is actually for. Current honest AUC ≈ **0.72**.

5. **Anchor on one thing done well.** Self-association (AC-SINS) is the strongest, most
   data-rich, best-generalizing signal. We focus there. A strong shared model on the rich
   signal lifts the weaker related assays (HIC, SEC) — not the other way around.

6. **Tiny datasets are noise.** With <~30–50 antibodies, a correlation/AUC is basically a
   coin flip and swings wildly run-to-run. Don't trust or chase them.

7. **Additive only — do not overwrite existing files.** Another AI (Codex) has in-progress
   work in `model/head_b_gbm.py`, `data/*.py`, etc. All improvements are **new files**.
   Keep it that way.

8. **GDPa3 is a locked final exam — NEVER touch it.** `GDPa3_20260106_full.xlsx` is a
   frozen single-shot holdout. If the model ever trains on or even inspects it, the one
   honest external test is destroyed forever. The loader deliberately refuses to read it.
   Do not move it into `data/external/gdpa/`. Do not fold it. Do not use it.

---

## PART 3 — Where the project stands right now

- **Journey:** a reported "0.15" that looked broken → found it was mostly noise, mixed-up
  assay scales, and reporting artifacts → rebuilt as a pooled/blended model anchored on
  AC-SINS.
- **Honest, leakage-free result (leave-one-study-out, shared antibodies removed):**
  **AUC ≈ 0.72** on solid datasets (up to **0.78** on the best), for catching sticky
  antibodies among molecules it has never seen. That is a real 7/10 triage tool.
- **What's been built (all additive, in `model/` and `tools/`):** serving_policy,
  head_b_pooled, head_b_hybrid, head_b_blend, head_b_generalization, head_b_catch_rate,
  head_b_acsins_unified, head_b_acsins_ensemble, plus tools/overnight_tune (a config
  search running locally) and the folding tools below.
- **The next real lever is DATA, not model tricks** (those are largely tapped out):
  specifically, **3D structure features** — because stickiness happens on the antibody's
  *surface*, which you can only see in a folded 3D structure.

---

## PART 4 — YOUR TASK: fold the antibodies on Colab

The local machine can't fold (only 16 GB RAM, no GPU, Python 3.14 breaks the folding
libraries). So folding must happen on **Google Colab (free GPU)**. That's your job.

**Inputs the user will give you (upload both to Colab):**
- `acsins_sequences_to_fold.csv` — 606 rows, columns `seq_id, sequence`.
- `tools/fold_structures_colab.py` — the ready-to-run script. Its docstring has the exact
  Colab cells.

**Do exactly this:**
1. Colab → Runtime → Change runtime type → **T4 GPU**.
2. Install: `!pip -q install transformers accelerate biotite numpy`
3. Upload `acsins_sequences_to_fold.csv`.
4. Run the body of `tools/fold_structures_colab.py`. It uses **ESMFold via HuggingFace**
   (`transformers.EsmForProteinFolding`, `facebook/esmfold_v1`).
5. Let it run (~15–30 s/sequence, ~2–4 h total). It is **resumable** — if Colab
   disconnects, rerun the last cell; it skips finished sequences.

**Hard constraints:**
- Do **NOT** install `openfold` or `igfold` — they break; use the HuggingFace ESMFold path.
- Do **NOT** change the feature formulas or output column names in the script.
- Do **NOT** split sequences into chains — fold each sequence as given.

**Output:** `structure_features.csv` with columns:
`seq_id, plddt_mean, plddt_variance, low_confidence_fraction, sap_score, sap_per_residue,
max_hydrophobic_patch, surface_hydrophobic_fraction, n_res`.

`sap_score` / `max_hydrophobic_patch` = surface hydrophobic patches = the actual
aggregation signal we're after. `plddt_*` = structure confidence.

**Report back to the user:** how many of the 606 folded successfully, any failures, and
hand them the `structure_features.csv` to place at
`Biostuff/data/external/structures/structure_features.csv`.

---

## PART 5 — What happens after you finish

The user brings `structure_features.csv` back to the local machine. Claude (the other
assistant) merges these structure features into the model's feature set and re-runs the
**same honest leakage-free AC-SINS test** to measure whether structure pushed AUC above
0.72 toward the goal of 0.75. You do not need to do that part — just deliver clean folded
features.

---

## PART 6 — If something goes wrong

- ESMFold out-of-memory on Colab: reduce to shorter sequences first, or add
  `model.trunk.set_chunk_size(64)` after loading (lower memory).
- A sequence errors: the script already catches it, logs FAILED, and continues. That's fine
  — a few failures out of 606 is acceptable; report the count.
- If you're unsure about anything, ask the user rather than guessing or "improving" the
  approach. The schema and method are deliberate.
