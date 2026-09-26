# Pool-size check (small-pool val vs large-pool val)

Pools (S1 per country): small = val-only universe (India 176,637, US 264,726). Large = full train universe (India 883,188, US 1,323,633).
Both use floor 7.966. Decoding runs over each pool's whole population. Val halves come from hash(entity_id, 7): half A tunes, half B reports.
Code: src/tuning/poolsize_check.py. Data: poolsize_check.json.

## Fixed thresholds, half B (macro F0.5 / singleton acc / empty rate), pooled
| model | t/te | small pool | large pool |
|---|---|---|---|
| v1 | 0.70/0.70 | 0.97154 / 0.929 / 0.057 | 0.93255 / 0.720 / 0.047 |
| v1 | 0.84/0.84 | 0.97050 / 0.964 / 0.060 | 0.94504 / 0.837 / 0.055 |
| v1 | 0.90/0.92 | 0.96701 / 0.984 / 0.064 | 0.94907 / 0.921 / 0.063 |
| v2 ALL | 0.66/0.80 | 0.96584 / 0.985 / 0.065 | 0.95062 / 0.951 / 0.066 |
| v2 ALL | 0.84/0.84 | 0.95955 / 0.988 / 0.066 | 0.94794 / 0.961 / 0.068 |
| v2 ALL | 0.90/0.92 | 0.95199 / 0.994 / 0.071 | 0.94232 / 0.978 / 0.073 |
Per-country numbers are in the json.

## Half-A tuned, half B
| model | tuned on | t/te | B small | B large |
|---|---|---|---|---|
| v1 | small A | 0.68/0.82 | 0.97204 | 0.93589 |
| v1 | large A | 0.90/0.92 | 0.96701 | 0.94907 |
| v2 ALL | small A | 0.42/0.54 | 0.96979 | 0.94347 |
| v2 ALL | large A | 0.66/0.80 | 0.96584 | 0.95062 |

No single global (t, te) comes within 0.003 of both pool bests. The min-max choice lands on the large-pool optimum for both models.
It costs 0.00512 (v1) and 0.00404 (v2) on small-pool half A.

## log(pool S1) linear fit (4 per-country half-A-tuned points)
| model | France 259,452 | US test 663,106 | India test 809,986 |
|---|---|---|---|
| v1 | t 0.720 / te 0.822 | 0.833 / 0.887 | 0.858 / 0.900 |
| v2 ALL | t 0.465 / te 0.553 | 0.593 / 0.684 | 0.621 / 0.712 |

Checks behind the v2 fit values:
- Large-pool half B at the rounded fit values is flat: India 0.62/0.71 gives 0.93554 (0.66/0.80 gives 0.93542). US 0.59/0.68 gives 0.96010 (0.66/0.80 gives 0.96072). Source: logN_thresholds_largepool_halfB.json.
- Small-pool held-out country (US-only v2 model on India val, poolsize_loco_small.json):
  - tuned t is 0.40/0.40 (grid edge), half B 0.93012.
  - at 0.465/0.553: 0.92759. At 0.66/0.80: 0.91552. At 0.84/0.84: 0.90406.
  - So at small pool size a held-out country also prefers LOW t. The high t that large-pool LOCO preferred is a large-pool effect.
- Small-pool in-country, v2 ALL: India 0.465/0.553 gives 0.96139 against 0.95673 at 0.66/0.80. US gives 0.97521 against 0.97188.

## Decision
The model comparison depends on pool size. v1 wins small pools (0.97204 vs 0.96979) and v2 wins large pools (0.95062 vs 0.94907).
Test pools are 259k (France), 663k (US) and 810k (India), and India and US carry 85% of test S1, so I kept v2 ALL.
The best measured option is v2 ALL with per-country thresholds from the log-N fit: France 0.47/0.55, India 0.62/0.71, US 0.59/0.68.
It was built as a pure re-decode of the v3 test scores into submissions/v4 (validator PASS).

Caveat (feature_dists.json): on the record-competition features, France does not look like a small pool.
- s1_n is 10 for nearly every France pair (p5 = 10), the floor-anomaly signature.
- France has the lowest rec_rank=1 share (0.227 vs India 0.296 and US 0.360 on test) and the highest median rec_gap (7.0 vs 4.0 to 5.8).
So the France threshold from pool size alone is an extrapolation.
