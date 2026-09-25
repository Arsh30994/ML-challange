# Amazon ML Challenge 2026: Business Entity Resolution (key rules, condensed from the official problem statement)

## Task
Source 1 is the deduplicated reference. For every S1 entity, find all matching S2/S3 records (0..many).

## Scoring (leaderboard: matching_results.tsv only)
- F_0.5 = 1.25*P*R / (0.25*P + R), computed PER Source 1 entity, then MACRO-averaged over all S1 entities.
- Singletons (no true matches) are included: an empty prediction scores 1.0, and any prediction scores 0.0.
- Public LB = subset of test, private LB = the rest, final ranking = private.

## Candidate pairs (part of final ranking review)
- candidate_pairs.tsv = the exact final candidate set the model runs inference on (not an earlier blocking pass).
- Smaller candidate set per S1 entity is ranked higher in the final evaluation beyond the leaderboard.
- Every matched ID should appear among the candidates (the validator warns if not).
- Same format rules as matching_results; column name is candidate_entity_ids.

## Format
- TSV, header `source1_entity_id\tmatched_entity_ids`, exactly one row per test S1 ID, an empty list for no match.
- Only S2-/S3- IDs that exist in test; no duplicates in a list; no duplicate rows. Violations lead to rejection.

## Country
- Train: US + India. Test adds France (unseen). Country is an open set: do not hard-code, filter or one-hot to {US, India}.

## Constraints
- No external data lookup, APIs, registries, geocoding or internet augmentation. Leads to disqualification.
- Final model must be MIT/Apache-2.0 licensed and up to 8B parameters.

## Final package
<team>_submission.zip: output/{matching_results.tsv,candidate_pairs.tsv}, code/business_entity_resolution/{src/,README.md,requirements.txt}, Documentation_template.md (filled in).
Validate: python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
