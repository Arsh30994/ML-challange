# Blocking baseline — validation split (grouped, seed 42)

All numbers measured on the val side only: 441,363 S1, 1,006,884 S2, 1,056,679 S3 (unreferenced S2/S3 included). k = cap per source per S1 (so up to 2k candidates per S1). Oracle columns come from /workspace/amlc/scorer/scorer.py --oracle-only (recall/coverage/mean/P95/P99 from the scorer match metrics.json exactly).

Config: M=30 per retrieval method (A word tokens, B char 3-gram, B2 consonant-skeleton 3-gram, C address tokens) + record-centred reverse top-5; blocking score = logistic-regression weights on 4 cosines + address-missing flag and interactions, fitted on a train-side subset (weights.json).

## Recall ceiling at fixed k

| k | pair recall | S2 | S3 | entity all kept | entity >=1 kept | cand/S1 mean | P95 | P99 | oracle macro F0.5 | oracle F0.5 non-singleton | oracle US | oracle India |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.9737 | 0.9762 | 0.9713 | 0.9222 | 0.9973 | 10.0 | 10 | 10 | 0.9910 | 0.9905 | 0.9962 | 0.9832 |
| 10 | 0.9789 | 0.9804 | 0.9776 | 0.9390 | 0.9975 | 20.0 | 20 | 20 | 0.9923 | 0.9918 | 0.9971 | 0.9851 |
| 15 | 0.9798 | 0.9812 | 0.9786 | 0.9420 | 0.9976 | 30.0 | 30 | 30 | 0.9926 | 0.9921 | 0.9972 | 0.9855 |
| 20 | 0.9803 | 0.9816 | 0.9792 | 0.9436 | 0.9976 | 40.0 | 40 | 40 | 0.9927 | 0.9923 | 0.9973 | 0.9858 |
| 30 | 0.9809 | 0.9820 | 0.9798 | 0.9455 | 0.9976 | 60.0 | 60 | 60 | 0.9929 | 0.9925 | 0.9974 | 0.9861 |

## Top-k plus score threshold (smaller candidate sets)

Thresholds correspond to tune-subset LR probability 0.002 / 0.005 / 0.01 (score 7.966 / 8.885 / 9.584).

| variant | pair recall | entity all kept | cand/S1 mean | P95 | P99 | oracle macro F0.5 | oracle US | oracle India |
|---|---|---|---|---|---|---|---|---|
| k10_t7.966 | 0.9774 | 0.9336 | 12.942 | 20 | 20 | - | - | - |
| k15_t7.966 | 0.9778 | 0.9349 | 15.888 | 30 | 30 | 0.9919 | 0.9967 | 0.9848 |
| k20_t7.966 | 0.9780 | 0.9355 | 18.011 | 40 | 40 | - | - | - |
| k30_t7.966 | 0.9781 | 0.9359 | 20.944 | 60 | 60 | - | - | - |
| k10_t8.885 | 0.9745 | 0.9242 | 8.017 | 19 | 20 | - | - | - |
| k15_t8.885 | 0.9747 | 0.9248 | 8.466 | 21 | 30 | 0.9909 | 0.9958 | 0.9836 |
| k20_t8.885 | 0.9748 | 0.9250 | 8.701 | 22 | 40 | - | - | - |
| k30_t8.885 | 0.9748 | 0.9252 | 8.938 | 22 | 51 | - | - | - |
| k15_t9.584 | 0.9709 | 0.9130 | 6.538 | 15 | 29 | 0.9897 | 0.9947 | 0.9820 |
| k30_t9.584 | 0.9710 | 0.9132 | 6.724 | 15 | 33 | - | - | - |

## By country (pair recall / entity all kept / entity >=1 kept)

| k | US | India |
|---|---|---|
| 5 | 0.9869 / 0.9559 / 0.9992 | 0.9540 / 0.8718 / 0.9944 |
| 10 | 0.9908 / 0.9693 / 0.9993 | 0.9611 / 0.8937 / 0.9947 |
| 15 | 0.9914 / 0.9714 / 0.9994 | 0.9625 / 0.8980 / 0.9948 |
| 20 | 0.9917 / 0.9724 / 0.9994 | 0.9633 / 0.9006 / 0.9949 |
| 30 | 0.9920 / 0.9735 / 0.9994 | 0.9643 / 0.9035 / 0.9950 |

## Pair recall by S2/S3 name script

| k | Latin (n=1,417,165) | Bengali (n=7,011) | Devanagari (n=62,430) | Gujarati (n=7,147) | Gurmukhi (n=1,629) | Kannada (n=8,759) | Malayalam (n=4,399) | Oriya (n=1,716) | Tamil (n=7,963) | Telugu (n=8,972) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.9811 | 0.8783 | 0.8762 | 0.8881 | 0.9398 | 0.8443 | 0.9373 | 0.9306 | 0.8357 | 0.9032 |
| 10 | 0.9859 | 0.8913 | 0.8842 | 0.8987 | 0.9534 | 0.8536 | 0.9588 | 0.9452 | 0.8611 | 0.9110 |
| 15 | 0.9868 | 0.8937 | 0.8855 | 0.8998 | 0.9552 | 0.8552 | 0.9632 | 0.9464 | 0.8671 | 0.9128 |
| 20 | 0.9872 | 0.8940 | 0.8860 | 0.9000 | 0.9570 | 0.8556 | 0.9657 | 0.9464 | 0.8710 | 0.9134 |
| 30 | 0.9878 | 0.8952 | 0.8864 | 0.9001 | 0.9570 | 0.8557 | 0.9666 | 0.9464 | 0.8742 | 0.9136 |

## Runtime and peak memory (val)

| step | seconds | peak RSS GB |
|---|---|---|
| normalize S1 (load 5.6 s + normalize 2.5 s, 6 procs) | 8.1 | 1.07 (cumulative process peak) |
| normalize S2 (load 14.7 s + normalize 5.5 s) | 20.2 | 2.24 (cumulative) |
| normalize S3 (load 15.2 s + normalize 6.0 s) | 21.2 | 2.59 (cumulative) |
| load_normalized_and_truth | 4.1 | 1.531 |
| blocking_retrieve_score_topk | 191.2 | 3.903 |
| write_full_candidates | 1.4 | 3.416 |
| evaluate | 4.8 | 4.92 |
| write_outputs | 15.9 | 4.047 |
| (inside blocking: vectorize / retrieve+score) | 62.7 / 128.5 | |
| scorer --oracle-only per k (separate process) | 16-32 | 2.88-5.06 (k=30 highest) |

## Recommendation

Spec rule (smallest k within 0.5 pt of the k=30 recall, 0.9809): **k = 10** (0.9789; k=5 is 0.9737). candidates_val_k10.parquet is saved for stage 3; the full k=30 table with features is in candidates_val_k30_full.parquet.

See metrics.json for the complete numbers, and run1_*/run2_* json for the earlier configs (run1: M=30 without reverse, 4-cosine LR; run2: + reverse top-5, 4-cosine LR).

Candidate-size option (the rules judge candidate set size; candidate_pairs.tsv must be the exact set the model scores):
k=10 plus score >= 7.966 keeps 0.9774 pair recall (-0.12 pt against fixed k=10) with 12.9 mean candidates per S1 instead of 20;
score >= 8.885 gives 0.9745 with 8.0. There is a cliff between score 8.25 and 8.5 in the 4-cosine score (exact-name pairs with an
empty address scored about 8.37); the address-missing features fixed most of it. Pick the threshold after stage-3 macro F0.5 is measured on both sets.
Largest remaining gap: Indic-script S2/S3 names (0.85-0.96 pair recall at k=10 against 0.986 for Latin), mostly generic transliterated
names with short or empty addresses.

## Reviewer checks
- Reverse top-5 retrieval cannot push an S1 past k per source: reverse pairs join the union before the per-(S1, source)
  top-k cut. Measured at k=10: max 10 S2 and 10 S3 per S1 in candidates_val_k10.parquet (max rank 10, 0 duplicate pairs) and
  in candidate_pairs_val_k10.tsv (max 10 S2- and 10 S3- IDs per row). 4,418,191 of the kept pairs carry the reverse bit; 196,019 were found only by reverse retrieval.
- Score-floor provenance: the floors 7.966 / 8.885 / 9.584 belong to the blocking LR in weights.json (fit on the train-side tuning
  union, M=30 + reverse top-5, features = 4 cosines + addr_missing + 3 interactions, intercept -14.1787). They are
  log-odds cuts at p = 0.002 / 0.005 / 0.01 under that fit; they were NOT tuned on validation. If the blocking LR is refit for any
  reason, the floor must be re-tuned on validation rather than reusing these numbers.
- Stopwords are learned per country value from the rows of the current run only (build_country asserts a single country
  value and at least one S1 row). For test, France gets its own list from test S1+S2+S3 (see reports/loco_check/test_stopwords.json).
