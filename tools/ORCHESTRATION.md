# Autonomous data flywheel — orchestration spec

Three roles, three queues. Each role only does its part. Honest quality gate protects the model.

```
   HUNTER (finds everything)
        |  classifies each find: OPEN_AUTO | FREE_MANUAL | PAYWALLED
        v
   +----+-------------------+------------------+
   |                        |                  |
OPEN_AUTO              FREE_MANUAL          PAYWALLED
   |                        |                  |
   v                        v                  v
auto_queue.csv        human_queue.csv        (skip, logged)
   |                        |
   |                        +--> HUMAN grabs it in a browser, drops the file in
   |                             ~/Downloads/antibody_sources/, moves the row to auto_queue
   v
   GPT DECIDER (auto): for each new source -> download -> extract to the right
   property dataset -> QUALITY GATE -> accept/quarantine -> retrain platform on accept
```

## The three queues (CSV files under tools/)
- **auto_queue.csv** — OPEN_AUTO finds (GitHub/Zenodo/PMC direct). GPT processes these hands-off.
- **human_queue.csv** — FREE_MANUAL finds (open-access but bot-blocked, e.g. Elsevier). The
  HUMAN downloads these in a browser; nothing auto-processes them until the file exists locally.
- Paywalled/irrelevant finds are logged in data_hunt_log.csv and ignored.

Columns for both queues: date, source, url, property, download_link, status
(status: new | downloaded | ingested | quarantined)

## Roles
1. **HUNTER** (the search agent, or Claude on request): find datasets, classify fetchability
   (see DATA_HUNT_BRIEF.md), append each to auto_queue.csv OR human_queue.csv OR data_hunt_log.
   Never downloads paywalled; never touches GDPa3.
2. **HUMAN** (you): periodically open human_queue.csv, grab the FREE_MANUAL files in a browser,
   save to ~/Downloads/antibody_sources/, and move that row into auto_queue.csv (status=downloaded).
3. **GPT DECIDER** (auto): watches auto_queue.csv; for each row not yet ingested, runs the
   decider loop below. This is the only role that trains the model.

## The QUALITY GATE (the rule that protects everything)
Before any dataset is kept:
- Retrain the relevant property head WITH and WITHOUT the new data.
- Score on the fixed DIVERSE holdout (never train on it). Use the honest generalization AUC —
  **NOT the padded overall mean.** (We got fooled once: 373k library rows pushed the mean to
  0.815 while generalization stayed 0.617.)
- If generalization holds or improves (>= baseline - 0.005): **ACCEPT** -> add to training pool
  -> retrain and update the serving platform.
- Else: **QUARANTINE** (keep the file, exclude from training, log the reason).

## When GPT trains
Only when a dataset PASSES the gate. Accepting new data == retraining that property head and
refreshing model/saved/platform.pkl. Quarantined data never triggers a retrain.

## Everyone obeys INGEST_RULES.md
Sequences AND values required · keep properties separate · higher = worse · molecule-dedup ·
skip Tm/expression · never touch GDPa3 · never fabricate · additive files only.

## What stays human-in-the-loop
- Grabbing FREE_MANUAL (bot-blocked) files.
- Reviewing the periodic report of what got accepted/quarantined (discuss next steps).
Everything else runs itself.
