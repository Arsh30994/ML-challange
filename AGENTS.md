# AGENTS.md

Amazon ML Challenge 2026: Business Entity Resolution.

## Rules for every agent
- Use only the supplied challenge data. No external business lookup, APIs, geocoding, registries, or data augmentation.
- Read TSV files with `sep="\t"`. Never modify the original data.
- Never train on test data or leak ground-truth labels. Keep entities grouped when splitting train/validation.
- Do not hard-code countries. Use deterministic seeds.
- Do not assume the test singleton rate matches train. Threshold tuning must hold up across different singleton rates.
- Never claim an improvement without measured validation results committed under `reports/`.
- Work on a branch and open a PR; do not push directly to `main`.

## Layout
- `code/business_entity_resolution/src/` pipeline code
- `tests/` pytest tests
- `reports/<experiment>/metrics.json` + config, committed for every baseline and experiment
- `docs/` design notes
- `utils/validate_submission.py` official validator
- `dataset/` challenge TSVs (git-ignored, see `dataset/README.md`)
- `output/` `matching_results.tsv`, `candidate_pairs.tsv` (git-ignored)

## Pipeline (build and measure in this order)
1. TSV loading and schema validation
2. Cleaning and normalization
3. Blocking: country-aware tokens, character n-grams, address/numeric, then candidate union (report candidate recall and P95/P99 candidates before any model)
4. Pairwise features
5. LightGBM classifier
6. Entity-level macro F0.5 threshold tuning
7. `matching_results.tsv` + `candidate_pairs.tsv`, validated with `utils/validate_submission.py`

## Roles
- Coordinator (New Bot): plan, assign tasks, compare experiments
- Codex (Codex CLI, run by Arshdeep): implementation, tests, validation
- chatgpt: architecture and metric review
- claudu: independent code and methodology review
- majdoor: Kaggle full-scale runs and submission
- Gitu: GitHub repo, branches and PRs
