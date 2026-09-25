# Reference offline scorer (AMLC 2026 business entity resolution)

    # full scoring
    /workspace/kvenv/bin/python scorer.py --gt GT.tsv --pred matching_results.tsv \
        --cand candidate_pairs.tsv [--s1-ids val_ids.txt] [--source1 train_source1.tsv] [--out metrics.json]
    # blocker ceiling only (no predictions needed; cheap enough to run per k)
    /workspace/kvenv/bin/python scorer.py --gt GT.tsv --cand candidate_pairs.tsv --oracle-only [--s1-ids ...] [--source1 ...]

    from scorer import score, oracle, read_tsv
    res = score(gt_df, pred_df, cand_df, s1_ids=None, source1_df=None)   # -> dict
    ceil = oracle(gt_df, cand_df, s1_ids=None, source1_df=None)          # == score(..., oracle_only=True)

## Output JSON (top-level, in order)
* `nonsingleton` (headline): n, macro_f05, mean_precision, mean_recall over evaluated S1 with >=1 true match.
* `macro_f05`: the official leaderboard metric (macro over ALL evaluated S1, singletons included).
* `mean_precision_all_incl_singletons` / `mean_recall_all_incl_singletons`: means over all S1; a singleton
  counts P=R=1 if predicted empty, else P=R=0 (so these are inflated by the ~5.6% singletons).
* `candidates` (when --cand): `entity_coverage` (fraction of evaluated NON-singleton S1 whose true matches are
  ALL in their candidate list), `candidate_recall` (pair level: fraction of true pairs in candidates),
  `oracle_macro_f05` (per S1 predict exactly true ∩ candidates; singleton -> empty -> 1.0; non-singleton with no
  true match in candidates -> empty -> 0.0), `oracle_nonsingleton_macro_f05`, candidates per S1
  (mean/P50/P95/P99/max; S1 absent from the candidate file count as 0).
* `breakdown`: singletons / non_singletons / by_country (with --source1; includes coverage and oracle per country).
* `pair_level`, `checks` (all format/validation counters), `warnings`, `errors`, `valid`.
* In `--oracle-only` mode only `candidates`, `n_s1_evaluated`, `singleton_rate`, `by_country`, `checks` are emitted.

## Rules
* Reads with READ_KWARGS mirrored from gt_checks/.../io_config.py (sep=\t, dtype=str, keep_default_na=False, quoting=QUOTE_NONE).
* Exit code 1 (and `"valid": false`) if any predicted (s1, id) pair is missing from candidate_pairs.
* `--s1-ids`: one ID per line (no header), or a TSV with a `source1_entity_id` / `entity_id` header column.
* Candidates: `source1_entity_id, candidate_entity_ids` (comma list, same format as matching_results);
  long format `source1_entity_id, candidate_entity_id` (one pair per row) is also accepted.
* Lists are scored as sets: duplicate IDs are reported and deduplicated, duplicate S1 rows are reported and unioned,
  surrounding whitespace / empty tokens (trailing comma) are reported and dropped.

Tests: `/workspace/kvenv/bin/python -m pytest -q tests`

sanity/: full-size sanity inputs + outputs (self-score, all-empty, oracle with cand=GT and with an empty
candidate file, every-5th-ID subset) and run_timed.py (wall time + peak RSS).
