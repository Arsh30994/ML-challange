import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorer import score, f05_arrays, main  # noqa: E402

S1, M, C = "source1_entity_id", "matched_entity_ids", "candidate_entity_ids"


def gt(rows):
    return pd.DataFrame(rows, columns=[S1, M], dtype=str)


def pred(rows):
    return pd.DataFrame(rows, columns=[S1, M], dtype=str)


def cand(rows):
    return pd.DataFrame(rows, columns=[S1, C], dtype=str)


def f05(p, r):
    return 1.25 * p * r / (0.25 * p + r)


# ---------------------------------------------------------------- edge cases
def test_true_empty_pred_empty_is_one():
    r = score(gt([["S1-1", ""]]), pred([["S1-1", ""]]), cand([["S1-1", ""]]))
    assert r["macro_f05"] == 1.0 and r["mean_precision_all_incl_singletons"] == 1.0 and r["mean_recall_all_incl_singletons"] == 1.0


def test_true_empty_pred_nonempty_is_zero():
    r = score(gt([["S1-1", ""]]), pred([["S1-1", "S2-1"]]), cand([["S1-1", "S2-1"]]))
    assert r["macro_f05"] == 0.0 and r["mean_precision_all_incl_singletons"] == 0.0 and r["mean_recall_all_incl_singletons"] == 0.0


def test_true_nonempty_pred_empty_is_zero():
    r = score(gt([["S1-1", "S2-1"]]), pred([["S1-1", ""]]), cand([["S1-1", "S2-1"]]))
    assert r["macro_f05"] == 0.0


def test_tp_zero_with_predictions_is_zero():
    r = score(gt([["S1-1", "S2-1,S3-1"]]), pred([["S1-1", "S2-9"]]), cand([["S1-1", "S2-9,S2-1"]]))
    assert r["macro_f05"] == 0.0 and r["mean_precision_all_incl_singletons"] == 0.0


def test_never_nan_arrays():
    import numpy as np
    P, R, F = f05_arrays(np.array([0, 0, 0, 0, 1]), np.array([0, 1, 0, 2, 1]), np.array([0, 0, 1, 2, 1]))
    assert not np.isnan(F).any()
    assert F.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]


def test_missing_s1_in_pred_counts_as_empty():
    g = gt([["S1-1", ""], ["S1-2", "S2-1"]])
    r = score(g, pred([]), cand([]))
    assert r["macro_f05"] == 0.5
    assert r["checks"]["evaluated_s1_missing_from_pred"] == 2


def test_pred_for_unknown_s1_reported_not_scored():
    g = gt([["S1-1", "S2-1"]])
    p = pred([["S1-1", "S2-1"], ["S1-999", "S2-5"]])
    c = cand([["S1-1", "S2-1"], ["S1-999", "S2-5"]])
    r = score(g, p, c)
    assert r["macro_f05"] == 1.0
    assert r["checks"]["pred_s1_rows_not_in_gt"] == 1
    assert r["valid"]


# ---------------------------------------------------------------- validations
def test_pred_pair_not_in_candidates_is_error_and_cli_exit_nonzero(tmp_path):
    g = gt([["S1-1", "S2-1"]])
    p = pred([["S1-1", "S2-1,S3-2"]])
    c = cand([["S1-1", "S2-1"]])
    r = score(g, p, c)
    assert r["checks"]["pred_pairs_not_in_candidates"] == 1 and not r["valid"]
    for name, df in [("g", g), ("p", p), ("c", c)]:
        df.to_csv(tmp_path / f"{name}.tsv", sep="\t", index=False)
    rc = main(["--gt", str(tmp_path / "g.tsv"), "--pred", str(tmp_path / "p.tsv"),
               "--cand", str(tmp_path / "c.tsv"), "--out", str(tmp_path / "m.json")])
    assert rc == 1
    assert json.loads((tmp_path / "m.json").read_text())["checks"]["pred_pairs_not_in_candidates"] == 1


def test_candidate_pair_under_other_s1_does_not_count():
    # S2-1 is a candidate of S1-2, not S1-1 -> violation for (S1-1, S2-1)
    g = gt([["S1-1", "S2-1"], ["S1-2", ""]])
    r = score(g, pred([["S1-1", "S2-1"], ["S1-2", ""]]), cand([["S1-1", ""], ["S1-2", "S2-1"]]))
    assert r["checks"]["pred_pairs_not_in_candidates"] == 1
    assert r["checks"]["candidate_recall"] == 0.0


def test_duplicate_ids_reported_and_deduped():
    r = score(gt([["S1-1", "S2-1"]]), pred([["S1-1", "S2-1,S2-1"]]), cand([["S1-1", "S2-1"]]))
    assert r["checks"]["pred_duplicate_ids_within_list"] == 1
    assert r["macro_f05"] == 1.0


def test_id_predicted_for_multiple_s1_reported():
    g = gt([["S1-1", "S2-1"], ["S1-2", "S2-2"]])
    p = pred([["S1-1", "S2-1,S3-7"], ["S1-2", "S2-2,S3-7"]])
    c = cand([["S1-1", "S2-1,S3-7"], ["S1-2", "S2-2,S3-7"]])
    r = score(g, p, c)
    assert r["checks"]["pred_ids_assigned_to_multiple_s1"] == 1


# ---------------------------------------------------------------- subset
def test_s1_ids_subset():
    g = gt([["S1-1", "S2-1"], ["S1-2", "S2-2"], ["S1-3", ""]])
    p = pred([["S1-1", "S2-1"], ["S1-2", "S2-9"], ["S1-3", "S3-1"]])
    c = cand([["S1-1", "S2-1"], ["S1-2", "S2-9,S2-2"], ["S1-3", "S3-1"]])
    full = score(g, p, c)
    assert full["macro_f05"] == pytest.approx(1 / 3)
    sub = score(g, p, c, s1_ids=["S1-1", "S1-3"])
    assert sub["checks"]["n_s1_evaluated"] == 2
    assert sub["macro_f05"] == 0.5
    assert sub["checks"]["pred_s1_rows_in_gt_not_evaluated"] == 1
    assert sub["checks"]["candidates_per_s1"]["mean"] == 1.0
    assert score(g, p, c, s1_ids=["S1-1"])["macro_f05"] == 1.0
    # unknown ids in the subset list are ignored and reported
    assert score(g, p, c, s1_ids=["S1-1", "S1-XX"])["checks"]["s1_ids_not_in_gt"] == 1


# ---------------------------------------------------------------- hand-computed mixed example
def test_mixed_example_hand_computed():
    g = gt([
        ["S1-a", "S2-1,S2-2,S3-1,S3-2"],  # pred 3 of which 2 correct: P=2/3, R=1/2
        ["S1-b", "S2-3"],                 # exact: 1
        ["S1-c", ""],                     # singleton, pred empty: 1
        ["S1-d", ""],                     # singleton, pred nonempty: 0
        ["S1-e", "S3-5,S3-6"],            # missing from pred: 0
        ["S1-f", "S2-7,S2-8"],            # pred 1 correct: P=1, R=1/2
    ])
    p = pred([
        ["S1-a", "S2-1,S3-2,S2-99"],
        ["S1-b", "S2-3"],
        ["S1-c", ""],
        ["S1-d", "S2-50"],
        ["S1-f", "S2-7"],
    ])
    c = cand([
        ["S1-a", "S2-1,S3-2,S2-99,S2-2"],        # 4 cands, has 2 of 4 true
        ["S1-b", "S2-3,S2-4"],                   # 2
        ["S1-c", "S2-40"],                       # 1
        ["S1-d", "S2-50"],                       # 1
        ["S1-e", "S3-5"],                        # 1 (1 of 2 true)
        ["S1-f", "S2-7,S2-8,S3-9"],              # 3
    ])
    r = score(g, p, c)
    fa = f05(2 / 3, 1 / 2)  # = (1.25*1/3)/(1/6+1/2) = 0.625
    ff = f05(1.0, 0.5)      # = 0.625/0.75 = 0.8333...
    assert fa == pytest.approx(0.625)
    assert ff == pytest.approx(5 / 6)
    assert r["macro_f05"] == pytest.approx((fa + 1 + 1 + 0 + 0 + ff) / 6)
    assert r["mean_precision_all_incl_singletons"] == pytest.approx((2 / 3 + 1 + 1 + 0 + 0 + 1) / 6)
    assert r["mean_recall_all_incl_singletons"] == pytest.approx((1 / 2 + 1 + 1 + 0 + 0 + 1 / 2) / 6)
    assert r["breakdown"]["singletons"]["macro_f05"] == 0.5
    assert r["breakdown"]["non_singletons"]["macro_f05"] == pytest.approx((fa + 1 + 0 + ff) / 4)
    assert r["checks"]["pred_pairs_not_in_candidates"] == 0 and r["valid"]
    # true pairs = 4+1+2+2 = 9; in candidates: a:3 (S2-1,S3-2,S2-2), b:1, e:1, f:2 -> 7
    assert r["checks"]["candidate_recall"] == pytest.approx(7 / 9)
    cps = r["checks"]["candidates_per_s1"]
    assert cps["mean"] == pytest.approx(12 / 6) and cps["max"] == 4 and cps["p50"] == 1.5
    assert r["pair_level"]["tp"] == 4 and r["pair_level"]["pred_pairs"] == 6 and r["pair_level"]["true_pairs"] == 9
    # country breakdown
    src = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c", "S1-d", "S1-e", "S1-f"],
                        "country": ["US", "US", "US", "India", "India", "India"]})
    rc = score(g, p, c, source1_df=src)["breakdown"]["by_country"]
    assert rc["US"]["macro_f05"] == pytest.approx((fa + 2) / 3)
    assert rc["India"]["macro_f05"] == pytest.approx(ff / 3)


def test_whitespace_and_trailing_comma_tolerated_and_reported():
    r = score(gt([["S1-1", "S2-1,S3-1"]]), pred([["S1-1", "S2-1, S3-1,"]]), cand([["S1-1", "S2-1,S3-1"]]))
    assert r["macro_f05"] == 1.0
    assert r["checks"]["pred_tokens_with_whitespace"] == 1 and r["checks"]["pred_empty_tokens"] == 1


def test_long_format_candidates_accepted():
    c = pd.DataFrame([["S1-1", "S2-1"], ["S1-1", "S3-1"]], columns=[S1, "candidate_entity_id"])
    r = score(gt([["S1-1", "S2-1"]]), pred([["S1-1", "S2-1"]]), c)
    assert r["valid"] and r["checks"]["candidates_per_s1"]["mean"] == 2.0


def test_self_score_and_empty_small():
    g = gt([["S1-1", "S2-1,S3-1"], ["S1-2", ""], ["S1-3", "S3-3"], ["S1-4", ""]])
    r = score(g, g, g.rename(columns={M: C}))
    assert r["macro_f05"] == 1.0 and r["checks"]["candidate_recall"] == 1.0
    e = score(g, g.assign(**{M: ""}), g.rename(columns={M: C}))
    assert e["macro_f05"] == 0.5 == e["singleton_rate"]


# ---------------------------------------------------------------- follow-up: coverage / oracle / labels
from scorer import oracle  # noqa: E402


def _mixed():
    g = gt([["S1-a", "S2-1,S2-2,S3-1,S3-2"], ["S1-b", "S2-3"], ["S1-c", ""], ["S1-d", ""],
            ["S1-e", "S3-5,S3-6"], ["S1-f", "S2-7,S2-8"]])
    p = pred([["S1-a", "S2-1,S3-2,S2-99"], ["S1-b", "S2-3"], ["S1-c", ""], ["S1-d", "S2-50"], ["S1-f", "S2-7"]])
    c = cand([["S1-a", "S2-1,S3-2,S2-99,S2-2"], ["S1-b", "S2-3,S2-4"], ["S1-c", "S2-40"], ["S1-d", "S2-50"],
              ["S1-e", "S3-5"], ["S1-f", "S2-7,S2-8,S3-9"]])
    return g, p, c


def test_entity_coverage_and_oracle_mixed_hand_computed():
    g, p, c = _mixed()
    r = score(g, p, c)
    cb = r["candidates"]
    # non-singletons a (3/4), b (1/1), e (1/2), f (2/2) -> fully covered: b, f
    assert cb["entity_coverage"] == 0.5
    assert cb["candidate_recall"] == pytest.approx(7 / 9) == r["checks"]["candidate_recall"]
    fa, fe = f05(1.0, 0.75), f05(1.0, 0.5)
    assert fa == pytest.approx(0.9375) and fe == pytest.approx(5 / 6)
    assert cb["oracle_macro_f05"] == pytest.approx((fa + 1 + 1 + 1 + fe + 1) / 6)
    assert cb["oracle_nonsingleton_macro_f05"] == pytest.approx((fa + 1 + fe + 1) / 4)
    # oracle-only gives the same candidate numbers without predictions
    o = oracle(g, c)
    for k in ("entity_coverage", "candidate_recall", "oracle_macro_f05", "oracle_nonsingleton_macro_f05"):
        assert o["candidates"][k] == pytest.approx(cb[k])
    assert "macro_f05" not in o and o["mode"] == "oracle_only" and o["valid"]


def test_oracle_nonsingleton_with_zero_true_in_candidates_is_zero():
    g = gt([["S1-1", "S2-1,S2-2"], ["S1-2", ""], ["S1-3", "S3-1"]])
    c = cand([["S1-1", "S2-9"], ["S1-2", "S2-5"]])  # S1-3 absent from candidates entirely
    o = oracle(g, c)["candidates"]
    assert o["oracle_macro_f05"] == pytest.approx(1 / 3)  # only the singleton scores 1.0
    assert o["oracle_nonsingleton_macro_f05"] == 0.0
    assert o["entity_coverage"] == 0.0
    assert o["nonsingleton_s1_zero_true_in_candidates"] == 2


def test_oracle_self_is_one_and_empty_candidates_is_singleton_rate():
    g = gt([["S1-1", "S2-1,S3-1"], ["S1-2", ""], ["S1-3", "S3-3"], ["S1-4", ""], ["S1-5", "S2-5"]])
    o = oracle(g, g.rename(columns={M: C}))
    assert o["candidates"]["oracle_macro_f05"] == 1.0 and o["candidates"]["entity_coverage"] == 1.0
    e = oracle(g, cand([]))
    assert e["candidates"]["oracle_macro_f05"] == 0.4 == e["singleton_rate"]
    assert e["candidates"]["entity_coverage"] == 0.0 and e["candidates"]["candidate_recall"] == 0.0


def test_oracle_respects_s1_subset_and_country():
    g, p, c = _mixed()
    o = oracle(g, c, s1_ids=["S1-a", "S1-c"])
    assert o["n_s1_evaluated"] == 2
    assert o["candidates"]["oracle_macro_f05"] == pytest.approx((0.9375 + 1) / 2)
    assert o["candidates"]["entity_coverage"] == 0.0
    src = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c", "S1-d", "S1-e", "S1-f"],
                        "country": ["US", "US", "US", "India", "India", "India"]})
    bc = score(g, p, c, source1_df=src)["breakdown"]["by_country"]
    assert bc["US"]["entity_coverage"] == 0.5 and bc["India"]["entity_coverage"] == 0.5
    assert oracle(g, c, source1_df=src)["by_country"]["India"]["oracle_macro_f05"] == pytest.approx((1 + 5 / 6 + 1) / 3)


def test_cli_oracle_only_without_pred(tmp_path):
    g, _, c = _mixed()
    g.to_csv(tmp_path / "g.tsv", sep="\t", index=False)
    c.to_csv(tmp_path / "c.tsv", sep="\t", index=False)
    rc = main(["--gt", str(tmp_path / "g.tsv"), "--cand", str(tmp_path / "c.tsv"), "--oracle-only",
               "--out", str(tmp_path / "o.json")])
    assert rc == 0
    o = json.loads((tmp_path / "o.json").read_text())
    assert o["candidates"]["entity_coverage"] == 0.5
    with pytest.raises(SystemExit):
        main(["--gt", str(tmp_path / "g.tsv"), "--oracle-only"])  # needs --cand
    with pytest.raises(SystemExit):
        main(["--gt", str(tmp_path / "g.tsv"), "--cand", str(tmp_path / "c.tsv")])  # needs --pred


def test_headline_nonsingleton_block_first_and_labels():
    g, p, c = _mixed()
    r = score(g, p, c)
    assert list(r)[0] == "nonsingleton"
    assert "mean_precision" not in r and "mean_recall" not in r
    ns = r["nonsingleton"]
    fa, ff = f05(2 / 3, 1 / 2), f05(1.0, 0.5)
    assert ns["n"] == 4
    assert ns["macro_f05"] == pytest.approx((fa + 1 + 0 + ff) / 4) == r["breakdown"]["non_singletons"]["macro_f05"]
    assert ns["mean_precision"] == pytest.approx((2 / 3 + 1 + 0 + 1) / 4)
    assert ns["mean_recall"] == pytest.approx((1 / 2 + 1 + 0 + 1 / 2) / 4)
    # singletons: S1-c empty -> P=R=1, S1-d predicted -> 0
    assert r["mean_precision_all_incl_singletons"] == pytest.approx((2 / 3 + 1 + 1 + 0 + 0 + 1) / 6)
