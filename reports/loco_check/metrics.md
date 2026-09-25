# Leave-one-country-out check (France stand-in), first LightGBM matcher

All numbers measured. Candidates: blocking k=10 per source (reports/blocking_baseline). Features: src/pair_features.py
(21 features, no country feature). Decoding per spec §5 (p >= t, one S1 per record, S1 empty if best p < t_empty); (t, t_empty) grid
0.10..0.95 step 0.05, tuned on the named val population only.

Training rows = k=10 candidates of the train-side tuning subset (25% of train-side S1 + matches + 25% of unreferenced), minus a 10%
S1 hold-out used only for early stopping:
| model | train rows | train S1 | positives | early-stop rows | best iter |
|---|---|---|---|---|---|
| ALL | 7,954,900 | 397,745 | 1,347,289 | 880,460 | 133 |
| US-only | 4,765,160 | 238,258 | 816,788 | 527,740 | 209 |
| India-only | 3,189,740 | 159,487 | 530,501 | 352,720 | 388 |
Eval: val k=10 candidates, India val 176,637 S1, US val 264,726 S1.

| run | model | thresholds tuned on | evaluated on | t / t_empty | macro F0.5 | non-singleton F0.5 | singleton acc | pair P | pair R |
|---|---|---|---|---|---|---|---|---|---|
| held-out India (LOCO) | US-only | US val | India val | 0.70 / 0.85 | **0.93545** | 0.93992 | 0.85998 | 0.96867 | 0.90568 |
| normal, same procedure | ALL | US val | India val | 0.70 / 0.85 | **0.96312** | 0.96313 | 0.96302 | 0.98927 | 0.92656 |
| normal, tuned on all val | ALL | all val | India val | 0.70 / 0.80 | 0.96333 | 0.96405 | 0.95117 | 0.98904 | 0.92690 |
| held-out US (LOCO, reverse) | India-only | India val | US val | 0.70 / 0.80 | **0.96621** | 0.96920 | 0.91562 | 0.98361 | 0.94246 |
| normal, same procedure | ALL | India val | US val | 0.70 / 0.80 | **0.97816** | 0.97960 | 0.95392 | 0.99116 | 0.95811 |
| normal, all val (optimistic) | ALL | all val | all val | 0.70 / 0.80 | 0.97223 | 0.97338 | 0.95282 | 0.99033 | 0.94561 |

The last row is cross-checked with scorer.py on loco_pred_val_ALL.tsv: macro_f05 0.9722279 (mine 0.97223), valid, no errors or warnings.

Reading: dropping the target country costs 2.77 pt on India (0.9355 vs 0.9631) and 1.20 pt on US (0.9662 vs 0.9782). Most of the India
loss is singleton accuracy (0.860 vs 0.963) and pair recall. The US-only model's top gain features are len_diff and score, and len_diff
shifts under transliteration. Expect a France gap somewhere in this range. Candidate mitigations for stage 3: drop or normalize len_diff,
use relative/rank features, and add country-agnostic threshold rules. Thresholds were identical (t=0.70) across all tunings, so
t is stable across countries while t_empty moved between 0.80 and 0.85.

Runtime: features 25 s (train) + 28 s (val); training 29 / 22 / 33 s; total 234 s; peak RSS 3.96 GB.

## Update: honest half-split, relative length (v2), robust t_empty
Val S1 split 50/50 by hash(entity_id, seed=7): half A 220,326 S1 (tune t/t_empty), half B 221,037 S1 (report). Half-B numbers
are the ones to compare. v1 = features above (len_diff); v2 = len_diff replaced by len_rel = |la-lb|/max(la,lb) (models retrained).

| run (tune half A -> report half B) | v1 (len_diff) | v2 (len_rel) |
|---|---|---|
| ALL model, all val | **0.97205** | 0.97096 |
| ALL model -> India half B | 0.96297 | 0.96238 |
| held-out India (US-only model, tuned US-A) | 0.93547 | 0.93200 |
| normal India (ALL model, tuned US-A) | 0.96297 | 0.96238 |
| held-out US (India-only model, tuned India-A) | 0.96607 | 0.96595 |
| normal US (ALL model, tuned India-A) | 0.97815 | 0.97665 |
Tuned on same data (full val -> full val): v1 0.97223, v2 0.97103. These are labelled 'tuned on same data' and not used for comparisons.

The relative-length fix did not help. v2 is worse in-country (-0.0011) and on held-out India (-0.0035), and the LOCO gap stays at
about 2.7-3.0 pt for India. v1 is kept.

Robust t_empty (maximize min(held-out India, held-out US) on half A, LOCO models): the maximum is at t_empty <= t, which means
t_empty = t = 0.70, i.e. no separate S1-empty rule. v1 in-country cost on half B: 0.97205 (t_empty 0.85) -> 0.97164 (t_empty 0.70)
= 0.00041 <= 0.003, so the test submission uses t = 0.70 and t_empty = 0.70. At that setting on half B: held-out India 0.93628 and
held-out US 0.96552 (v1). The v2 result is the same (cost 0.00083).

Relative features that already exist, with gain share (v2 models; v1 is similar): rec_rank (rank of this S1 among the record's
candidate S1s) 0.267 ALL / 0.183 US / 0.428 India; rec_gap (margin to the record's best S1) 0.134 / 0.053 / 0.101; rec_n_close 0.009 / 0.011 / 0.014;
s1_gap (margin to the S1's best candidate) 0.008 / 0.027 / 0.007; rank 0.001-0.002; s1_n about 0; len_rel 0.008-0.012. Total relative share
0.43 ALL / 0.28 US / 0.56 India. Not yet present: margin to the S1's next candidate (second-best gap) and score / top score ratio.
The absolute blocking 'score' dominates the US-only model (0.63 gain), which is the likely transfer weakness. Deferred (no big rework now).
Files: metrics_v2.json, metrics_v1_halfB.json. v2 run peak RSS 5.29 GB.
