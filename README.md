# Antibody Aggregation Failure Pipeline

A pipeline for predicting *why antibodies fail* — not just scoring how good a candidate looks, but building a model on confirmed failures, quantifying how confident that model is, and closing the loop with real wet-lab results.

## The core idea

Most protein ML models are trained almost entirely on **working** proteins — the ones that got published, crystallized, or approved. Failures are underrepresented because negative results rarely get reported. That means the "failure space" for antibody aggregation is largely unmapped.

This pipeline does the opposite: it deliberately generates and collects failure candidates, trains directly on public failure data (deep mutational scans, developability assays, nucleation datasets), and is built around an active-learning loop so that real wet-lab results — however few — get fed straight back into the model.

Runs entirely free on Google Colab: no GPU required, no paid APIs.

## Pipeline phases

**Phase 1 — Anchor → Perturb → Predict → Flag** (`pipeline.py`)
- Pull real antibody/nanobody/VHH/scFv sequences from the PDB
- Generate mutated variants (ESM-2 guided, or BLOSUM62 fallback)
- Score each variant: CamSol solubility approximation, hydrophobicity, and (for top candidates) ESMFold structural confidence
- Flag high-risk / high-disagreement candidates

**Phase 2 — Multi-predictor consensus + explanation** (`pipeline_phase2.py`)
- Run TANGO, AGGRESCAN, and Zyggregator approximations from published scales
- Find consensus aggregation hotspots (≥2 of 3 predictors agree)
- Optional short OpenMM implicit-solvent MD for per-residue flexibility
- Generate a plain-English explanation per candidate

**Phase 3 — Database cross-referencing** (`pipeline_phase3.py`)
- Map each candidate's parent PDB entry to UniProt
- Check for aggregation/disease annotations, known pathogenic variants at the mutation sites, and PDBe structural context (buried vs. surface)
- Re-rank candidates by database-backed confidence

**Model — trained failure predictor** (`model/`)
- Logistic regression + random forest ensemble, trained on public failure/working data (see Data sources below)
- Grouped cross-validation, probability calibration, and a permutation significance test (see Model details)
- Nearest-neighbor lookup and a rescue-mutation suggester
- An active-learning feedback loop that merges wet-lab results back into training

## Quickstart (Colab)

Open `colab_phase1.ipynb` in Google Colab and run cells top to bottom, or from a shell:

```bash
pip install -r requirements.txt

# Phase 1 — generate and score candidates
python pipeline.py --entries 2000 --variants 20 --esm2 --esmfold --top 200

# Phase 2 — multi-predictor consensus + explanations
python pipeline_phase2.py --top 20

# Phase 3 — database cross-referencing
python pipeline_phase3.py --top 50

# Train the model on public failure/working data
python -m model.train --flab --abdev --anchors --proteingym --sabdab

# Score candidates with the trained model
python -m model.predict --file results/phase3_candidates.json --top 20

# See a unified explanation for the top candidates
python -m model.report --file results/phase3_candidates.json --top 5

# Visualize results (risk distribution, mutation hotspots, substitution types)
python visualize.py
```

## Data sources

The model trains on public data so it works before any wet-lab result exists. Sources are combined with `--flag`s on `model/train.py`:

| Flag | Source | What it contributes |
|---|---|---|
| `--flab` | [FLAb](https://github.com/Graylab/FLAb) | Experimental HIC/SEC/AC-SINS developability assays → failures + working |
| `--abdev` | AbDev | Clinical-stage antibody sequences → clean negatives |
| `--anchors` | Your own Phase 1 output | Crystallized PDB chains → free negatives (already on disk) |
| `--proteingym` | [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) | Deep mutational scanning failures — low-fitness mutants that no longer fold/express, including a 14,811-variant amyloid-beta aggregation scan. The only source of true *mutant* failures, which activates the mutation-delta features. |
| `--sabdab` | SAbDab | Structural antibody database negatives (network-dependent; fails gracefully) |
| `--canya`, `--figshare-agg`, `--antiref` | CANYA / Figshare Aggrescan3D / AntiRef | Additional nucleation/structural/germline sources (availability varies by network — GitHub's API is rate-limited on some hosts, so loaders fall back to direct raw-file URLs where possible) |
| `--input` | Your own `results/labeled_results.json` | Real wet-lab results, always combined with public data |

Recommended baseline command:
```bash
python -m model.train --flab --abdev --anchors --proteingym --sabdab --flab-percentile 0.4
```

## Model details

**23+ features** per sequence: physical properties (hydrophobicity, charge, CamSol), aggregation-mechanism features (aromatic content, beta-sheet propensity, longest hydrophobic run, gatekeeper-residue density, aromatic cross-beta "zipper" score), mutation deltas, and the TANGO/AGGRESCAN/Zyggregator consensus scores.

**Grouped cross-validation** — mutants of the same wild-type/assay are kept together in the same fold (never split across train/test), closing a leakage path that would otherwise inflate the AUC.

**Probability calibration** — LR and RF are each wrapped in `CalibratedClassifierCV` so a predicted "0.7" reflects an actual ~70% empirical failure rate on held-out folds, not a raw (often overconfident) classifier score.

**Permutation significance test** (`model/significance.py`) — shuffles labels and retrains many times to build a null distribution, then reports a p-value for the real AUC. On the FLAb+AbDev baseline: real AUC 0.875 vs. a null distribution centered at 0.51, p ≈ 0.02.

```bash
python -m model.significance --flab --abdev --anchors --permutations 50
```

**Held-out benchmark** (`model/validate.py`) — scores 10 textbook amyloid aggregators (amyloid-beta, alpha-synuclein, IAPP, prion PrP fragment, etc.) against 10 textbook soluble proteins (ubiquitin, GFP, lysozyme, etc.), none seen in training. This is a rough sanity check on generalization beyond antibodies, not a precision benchmark — with n=20 the rank-AUC swings noticeably per point.

```bash
python -m model.validate --plot
```

## Interpreting a candidate

`model/predict.py` reports a calibrated failure probability, a confidence gap (disagreement between the LR and RF sub-models — high gap + high risk is the most valuable case to test next), and the top contributing features. `model/report.py` stitches this together with the closest known working/failure reference protein and a rescue-mutation suggestion into one narrative per candidate:

```bash
python -m model.report --file results/phase3_candidates.json --index 0
```

`model/rescue.py` searches for the smallest sequence change that most reduces predicted risk — reverting a mutation to wild-type, or an exhaustive single-position substitution search that can find a better-than-wild-type fix. These are computational hypotheses from the same model that scored the candidate, not verified fixes — treat them as a prioritized shortlist for the wet-lab queue.

## Active learning loop

```
pipeline → phase2 → phase3 → train → predict
                                          |
                             model/feedback.py --priority   (what to test next,
                                          |                   ranked by risk × uncertainty)
                                    [wet-lab testing]
                                          |
                             model/feedback.py --results     (merge results, retrain)
                                          |
                                       predict              (re-score, repeat)
```

Fill in `results/wetlab_results_template.json` → `results/wetlab_results.json` with real results, then:
```bash
python -m model.feedback --results results/wetlab_results.json
```
This merges your data with the public sources and retrains automatically.

## Repo structure

```
data/              PDB fetching, sequence/mutation loaders, all public data-source loaders
features/          sequence-level physical property + aggregation-mechanism features
predictors/        CamSol, ESMFold, multi-predictor (TANGO/AGGRESCAN/Zyggregator), database lookups
model/             feature extraction, trained failure model, train/predict/rescue/report/
                   significance/feedback/validate CLIs
pipeline.py        Phase 1 orchestration
pipeline_phase2.py Phase 2 orchestration
pipeline_phase3.py Phase 3 orchestration
visualize.py       results plotting (risk distribution, mutation hotspots, substitution types)
colab_phase1.ipynb one-notebook walkthrough of the whole pipeline
```

## Known limitations

- All public training failures except ProteinGym are *original* sequences, not mutants — so mutation-delta features only carry real signal from the ProteinGym source (and eventually your own wet-lab results).
- The held-out textbook benchmark is small (n=20) and antibody-specific features transfer imperfectly to non-antibody amyloid mechanisms (e.g. polyglutamine aggregation, which involves no charge/hydrophobicity change at all).
- Some data sources (SAbDab, CANYA, Figshare Aggrescan3D, AntiRef) depend on external hosts that are inconsistently reachable depending on network/proxy — loaders are written to fail gracefully and fall back where possible, but availability varies by environment.
- This is a computational triage tool, not a synthesis or wet-lab replacement. Every prediction, rescue suggestion, and nearest-neighbor match is a hypothesis to prioritize testing, not a verified result.
