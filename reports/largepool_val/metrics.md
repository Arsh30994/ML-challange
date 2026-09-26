# Test-scale ("large-pool") validation

Setup: blocking (k=10 per source, m=30, reverse top-5, weights.json), features and matcher were run on the whole
train universe (all train S1/S2/S3, both split sides, per country). Stopwords were learned on that universe
(India: 36, from 5,016,534 docs; US: 28, from 7,510,506 docs). Only val S1 are scored. The one-S1-per-record
decoding runs over the whole country universe, so train-side S1 compete for records the same way test S1 do.
Val halves: hash(entity_id, 7) % 2 (A tunes, B reports). Code: src/tuning/largepool.py, largepool_models.py, train_v2.py.

| | India | US | all |
|---|---|---|---|
| universe S1 / S2 / S3 | 883,188 / 2,017,799 / 2,115,547 | 1,323,633 / 3,016,817 / 3,170,056 | |
| k10 pairs | 17,663,756 | 26,472,660 | |
| blocking time, peak RSS | 1415 s, 5.10 GB | 4835 s, 7.08 GB | |
| cand per val S1, floor 7.966 | 17.76 | 16.41 | 16.95 (test: 17.28) |
| cand per val S1, no floor | 20.0 | 20.0 | 20.0 |

## v1 matcher (floor 7.966), half B
| | small pool (earlier) | large pool, t=0.70/te=0.70 | large pool, tuned on A: t=0.90/te=0.92 |
|---|---|---|---|
| macro F0.5 (all) | 0.97144 | 0.93255 | 0.94907 |
| India | | 0.91933 | 0.93429 |
| US | | 0.94133 | 0.95887 |
| singleton accuracy (all) | 0.928 | 0.720 | 0.921 |
| pair precision / recall | 0.990 / 0.945 | 0.959 / 0.925 | 0.986 / 0.896 |
| empty rate | | 0.0468 | 0.0629 |

By true-match script (half B, locked → tuned): Latin-only matches 0.9512 → 0.9571; some Indic-script match 0.9048 → 0.9086;
singletons 0.720 → 0.921. Without the floor (20 cand per S1) half B is 0.94914 against 0.94907 with it, which is within 0.001, so the floor stays.

## Tuning-subset exclusion (India, v1, floor 7.966)
Universe without the 177,123 tuning S1; their S2/S3 records are kept.
Result: locked t half B 0.91439 (versus 0.91933 with them in); tuned on A t=0.92/te=0.92 gives half B 0.93234 (versus 0.93429).
Removing them does not help and makes the score slightly worse, because precision drops from 0.955 to 0.946. Their records become orphans that attach to look-alike val S1.
So the drop comes from pool size, not from overconfidence on the training S1.

## Submission-2 matcher (large-pool trained, floor 7.966, FEATURES_V2)
Half B, tuned on pooled half A:
| model | t / te | all | India | US | singleton acc |
|---|---|---|---|---|---|
| v1 (tune subset, small pool) | 0.90 / 0.92 | 0.94907 | 0.93429 | 0.95887 | 0.921 |
| v2 ALL (large pool) | 0.66 / 0.80 | 0.95062 | 0.93542 | 0.96072 | 0.951 |
| v2 ALL at 0.90/0.92 | | 0.94232 | 0.92576 | 0.95331 | 0.978 |

LOCO (US-only large-pool model):
- Tuned on US half A: t=0.68/te=0.82. US half B is 0.96390 and held-out India half B is 0.85235.
- Held-out-country tuned t (India half A): 0.84/0.84, which gives India half B 0.85742.
- Robust (t, te), which maximizes min(US A, India A): 0.84/0.84. US half B is 0.96228 and India half B is 0.85742. The cost on US half A versus the US-best is 0.00169.
- Held-out gap on India: 0.857, against 0.935 for the all-country v2 model in-country.
