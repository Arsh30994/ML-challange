# Pipeline spec v1 (architecture review, chatgpt)

Goal: a valid end-to-end submission first, then improve only against measured
macro F0.5 on the grouped validation split. Every stage writes
`reports/<run>/metrics.json` + config before the next stage starts.

## 0. Loading and split
- All reads go through `io_config.READ_KWARGS` (sep="\t", dtype=str,
  keep_default_na=False, quoting=QUOTE_NONE). Use polars on Kaggle; keep
  integer row ids internally, map back to string IDs only at write time.
- Grouped split (seeded): each train S1 goes to train or val together with
  its matched S2/S3 records; unreferenced S2/S3 are split in the same
  proportion (stratify by country). Val approx 20%.
- Extra check: leave-one-country-out (train US, validate India, and the reverse)
  to estimate transfer to France.

## 1. Normalization (applied to all sources, no country list)
- Unicode NFKC, lowercase, strip punctuation, collapse whitespace.
- Transliterate every non-Latin script to Latin (indic-transliteration for
  Devanagari/Bengali/Gurmukhi/Gujarati/Oriya/Tamil/Telugu/Kannada/Malayalam,
  unidecode fallback for anything else, including French accents).
- Keep both raw and normalized strings. Extract number tokens from the address
  (house number, postcode-like runs) into a separate field.
- Legal-suffix / stopword list learned from token frequency in the data
  (top-N most frequent name tokens), not a hand-written country list.

## 2. Blocking (S1-centred, only S1-S2 and S1-S3, same-country value only)
Candidate sources, unioned:
- A. rare-token blocking on normalized names (IDF-weighted; skip tokens with
  doc-frequency above a cap)
- B. char 3-gram TF-IDF on names, approximate nearest neighbours per S1
  (sparse dot product in country/first-char chunks)
- C. address number and postcode tokens
Cap: keep top-k S2 and top-k S3 per S1 by the combined blocking score.
Report on val (with unreferenced records kept in): recall at k = 5/10/15/20 per
source, mean/P95/P99 candidates per S1, recall by country and by name script,
runtime, and peak memory.
Pick the smallest k where recall is within 0.5 pt of the plateau.

## 3. Pair features (all language-agnostic, no country feature)
- name: Jaro-Winkler, token-set ratio, char 3-gram cosine, token Jaccard,
  rare-token overlap (IDF-weighted), length diff, raw-vs-translit flag
- address: token-set ratio, char 3-gram cosine, number-token exact match /
  overlap, postcode match, address-missing flag
- rank/margin: rank of this S1 among the record's candidates, gap between the
  record's best and second-best S1 blocking score, number of S1 candidates
  within 10% of its best; S1-side top score and gap to its next candidate
- source flag (S2 vs S3)

## 4. Model
- LightGBM binary, seeded, early stopping on grouped val.
- Negatives = all non-matching pairs the blocker actually produced (including
  pairs involving unreferenced records). No random negatives.

## 5. Decoding (the part that moves the metric)
1. Keep pairs with p >= t.
2. One-S1-per-record: each S2/S3 is assigned only to its highest-p S1.
3. S1-empty rule: if the S1's best p < t_empty, predict empty (singleton bet).
Tune (t, t_empty) jointly by grid search for entity-level macro F0.5 on val.
Report macro F0.5, F0.5 on non-singletons, singleton accuracy, pair P/R.

## 6. Outputs
- candidate_pairs.tsv = exactly the capped set scored in step 4.
- matching_results.tsv = step 5 output; one row per test S1, empty allowed.
- Run utils/validate_submission.py; a run with any validator error is invalid.

## Experiment ledger (compare every run against the baseline)
candidate recall, mean/P95/P99 candidates per S1, pair P/R, macro F0.5,
singleton accuracy, runtime, peak memory. Accept a change only if macro F0.5
goes up on val (or it fixes a verified failure) without unacceptable
candidate growth.

Planned order: baseline (A+C blocking, string features) -> add B -> add
margin features -> multilingual-e5-small name embeddings as a blocking source
and feature (MIT, 118M), kept only if India/France-proxy recall or F0.5 improves.
