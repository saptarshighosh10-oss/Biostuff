# 2026-07-17 — Evidence-Tier Data Recovery Design

## Goal

Use trustworthy information from incomplete antibody datasets without treating
partial or reconstructed records as equivalent to directly published paired
VH/VL measurements.

## Decision

Keep two independent axes in the registry:

- `supervision_tier` describes what sequence is present: paired Fv,
  single-chain, CDR-only, or unlabeled sequence;
- `evidence_tier` describes how the sequence-to-measurement pairing was
  established.

This avoids the current failure mode where molecular completeness and evidence
quality are collapsed into one label.

## Evidence tiers

1. `direct`: the source directly pairs the reported sequence with the measured
   endpoint. These records may train and, when benchmark policy permits, serve
   as held-out evaluation rows.
2. `reconstructed_verified`: the source gives a verified parent sequence and an
   explicit substitution list, and a second artifact supplies the expected
   reconstructed sequence or its SHA-256 digest. These records may train
   auxiliary experiments but are never eligible for holdout evaluation.
3. `unlabeled`: a sequence is available without an eligible measurement. It may
   support representation adaptation, similarity search, or applicability-domain
   modeling, but not supervised assay training.
4. Rejected or ambiguous evidence remains in the rejection ledger. Figure-only
   values, guessed chain boundaries, and unresolved cross-paper identity joins
   are never promoted into supervised rows.

## Reconstruction contract

The first implementation supports substitutions such as `A23G;S45T`, using
one-based positions independently for VH and VL. Insertions, deletions,
ambiguous residues, and mutation strings without an explicit wild-type residue
are rejected.

A reconstruction passes only when all checks succeed:

1. every declared wild-type residue matches the parent at the stated position;
2. a fresh parent-versus-child diff recreates exactly the declared mutation set;
3. the child sequence matches the independently supplied full sequence or
   SHA-256 digest;
4. two distinct provenance references are present: one for the parent/mutation
   evidence and one for the independent sequence verification.

These are validation checks, not proof that a publication is correct. The
registry preserves both references and the check results.

## Training and evaluation rules

- Direct supervised records: `training_eligible=1`, `holdout_eligible=1`.
- Verified reconstructed records: `training_eligible=1`,
  `holdout_eligible=0`.
- Unlabeled records: `training_eligible=0`, `holdout_eligible=0` for supervised
  assay heads.
- The matrix loader must filter training and test rows by these flags even if a
  future caller selects the same supervision tier.
- Reconstructed rows are source-balanced and overlap-filtered exactly like
  direct rows.
- Reconstructed data earn promotion only through improvement on direct,
  source-held-out paired-Fv development sources across the declared seeds.

## Output and audit

Registry reports include counts by evidence tier and counts eligible for
training and holdout use. Every reconstructed record stores a compact JSON
verification report. Registry builds remain deterministic and replace the
database atomically.

## Non-goals

- No OCR or numeric transcription from figures.
- No guessed scFv linker split or chain boundary.
- No reconstruction without an independent expected sequence or digest.
- No insertion/deletion notation in the first parser.
- No automatic serving-model promotion.
