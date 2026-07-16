You are running the antibody-data flywheel OVERNIGHT, unattended, in ~/Biostuff with
~/Biostuff/.venv/bin/python. Work steadily until you run out of leads or hit morning. Assume
I'm asleep — be safe, be honest, leave me a clear digest to read when I wake up.

## Read these first (obey them exactly)
- ~/Biostuff/tools/ORCHESTRATION.md   (the loop + queues)
- ~/Biostuff/tools/INGEST_RULES.md    (hard rules — never break)
- ~/Biostuff/tools/GPT_PROMPT_DECIDER.md (the per-source decider steps)

## Your overnight job (repeat this cycle until done or morning)
1. **Process the auto-queue.** For each row in tools/auto_queue.csv with status new/downloaded:
   download → extract to the right property CSV → run the QUALITY GATE (train WITH vs WITHOUT,
   score on the DIVERSE holdout, NOT the mean) → ACCEPT (retrain, refresh platform.pkl) or
   QUARANTINE. Update the row's status. Log to tools/pipeline_log.jsonl.
2. **Hunt for more** (if you have web access). Find more antibody datasets with BOTH sequences
   AND values for aggregation(AC-SINS) / polyreactivity(PSR) / hydrophobicity(HIC). Classify
   each: OPEN_AUTO → append to auto_queue.csv; FREE_MANUAL (open-access but bot-blocked) →
   append to human_queue.csv for me; PAYWALLED/irrelevant → log in data_hunt_log.csv and skip.
   Then loop back to step 1 to process the new OPEN_AUTO ones.
3. Keep going until no unprocessed OPEN_AUTO rows remain or it's morning.

## Self-fix fixable problems — do NOT stop on superficial issues
If a failure is MECHANICAL/superficial, FIX IT YOURSELF and keep going — do not quarantine or
stop for something that was fixable in the first place. This includes: shell-quoting/glob bugs
(quote the URL), a wrong API/URL format, a missing dependency (pip install it into the venv), a
parser that needs adapting to the real file layout, a rate-limit (wait and retry), or a missing
baseline/holdout you can construct honestly. **All changes that genuinely improve the pipeline
or the model are welcome** — make them, commit them, and note them in the digest.

BUT this NEVER overrides data integrity. "Fixable" means the CODE/TOOLING — never the SCIENCE.
Absolute lines you may never cross to "keep going": never fabricate sequences or values, never
read numbers off a figure, never weaken/bypass the honest gate, never train on a holdout, never
touch GDPa3, never delete data. If the only way to "fix" something is to lower the honesty bar,
DON'T — quarantine it and log the real reason. Fix the plumbing freely; never fudge the result.

## Hard safety rules (unattended = extra careful)
- ADDITIVE ONLY. Do NOT modify model/head_*.py, model/head_platform.py, or any file that is
  Claude's. Add your own scripts + append to data CSVs only.
- NEVER delete data. Quarantine (exclude from training) — never remove.
- NEVER fabricate sequences or values. If a download 403s, route it to human_queue and move on.
- NEVER touch GDPa3 (frozen holdout).
- Judge every dataset by the DIVERSE-holdout generalization AUC, never the padded mean.
- Commit your progress to git periodically (git add -A on your own added files + data logs;
  do NOT force, do NOT rewrite history).
- If something errors, log the exact error, skip that source, keep going. Don't hack around it.

## Leave me a morning digest: tools/MORNING_DIGEST.md
Write a short summary I can read in 30 seconds:
- Sources processed, with ACCEPTED vs QUARANTINED and the baseline→trial diverse AUC for each.
- Whether any property head improved, and platform.pkl's current honest numbers.
- New leads added to auto_queue and to human_queue (what I need to manually grab).
- Any errors or things you skipped and why.
- One line: "biggest real gain of the night" (or "no real gains — here's why", honestly).

Do NOT overstate. A quarantine or a null result is a valid, useful outcome. I trust an honest
digest far more than an impressive-looking one.
