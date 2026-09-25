#!/usr/bin/env python
"""Reference offline scorer for the Amazon ML Challenge 2026 business entity resolution task.

Metric (per Source-1 entity, macro-averaged over the evaluated S1 set):
    TP = |pred ∩ true|, P = TP/|pred|, R = TP/|true|, F0.5 = 1.25*P*R / (0.25*P + R)
Edge cases (explicit, never NaN):
    true empty & pred empty      -> F=1.0 (P=R=1.0)
    true empty & pred nonempty   -> F=0.0 (P=R=0.0)
    true nonempty & pred empty   -> F=0.0 (P=R=0.0)
    TP == 0 with predictions     -> F=0.0
S1 entities missing from the predictions file count as an empty prediction. Predictions for S1 IDs
outside the evaluated GT set are counted but not scored. ID lists are treated as sets (duplicates are
reported, then removed); duplicate rows for the same S1 are reported and unioned.

Usage:
    python scorer.py --gt GT.tsv --pred matching_results.tsv --cand candidate_pairs.tsv \
        [--s1-ids val_ids.txt] [--source1 train_source1.tsv] [--out metrics.json]
    python scorer.py --gt GT.tsv --cand candidate_pairs.tsv --oracle-only [...]   # blocker ceiling only

Importable: score(gt_df, pred_df, cand_df, s1_ids=None, source1_df=None, oracle_only=False) -> dict
            oracle(gt_df, cand_df, s1_ids=None, source1_df=None) -> dict
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Mirror io_config.READ_KWARGS from gt_checks (not imported, so the scorer stays standalone).
READ_KWARGS = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
_IO_CONFIG = Path("/workspace/amlc/gt_checks/code/business_entity_resolution/io_config.py")
if _IO_CONFIG.exists():  # sanity: keep in sync with the shared config if it is present
    try:
        import importlib.util

        _spec = importlib.util.spec_from_file_location("_amlc_io_config", _IO_CONFIG)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        READ_KWARGS = dict(_mod.READ_KWARGS)
    except Exception:  # pragma: no cover - fall back to the identical local copy
        pass

S1_COL = "source1_entity_id"
MATCH_COL = "matched_entity_ids"
CAND_COL = "candidate_entity_ids"
CAND_LONG_COL = "candidate_entity_id"  # tolerated alternative: one pair per row


# pyarrow-backed strings make the vectorised str ops / factorize much faster; fall back to object.
try:
    import pyarrow  # noqa: F401
    STR_DTYPE = pd.StringDtype("pyarrow")
except ImportError:  # pragma: no cover
    STR_DTYPE = object


def _uniq(a: np.ndarray) -> np.ndarray:
    """Sorted unique for int64 keys (sort + diff; faster than np.unique's hash path at this size)."""
    if len(a) == 0:
        return a
    a = np.sort(a)
    m = np.empty(len(a), dtype=bool)
    m[0] = True
    np.not_equal(a[1:], a[:-1], out=m[1:])
    return a[m]


def read_tsv(path, usecols=None):
    return pd.read_csv(path, usecols=usecols, **READ_KWARGS)


# ----------------------------------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------------------------------
def _explode(df: pd.DataFrame, list_col: str, name: str, report: dict):
    """Turn a (source1_entity_id, comma-list) frame into (row_s1, pair_row_index, pair_match).

    row_s1 / pair_match are string Series; pair_row_index maps each pair to its row in row_s1.

    Vectorised: one big ",".join / split instead of per-row splitting. Tokens are stripped of
    surrounding whitespace; empty tokens (e.g. trailing commas) are dropped and counted.
    Also accepts long format where list_col holds a single ID per row (same code path).
    """
    if S1_COL not in df.columns or list_col not in df.columns:
        raise ValueError(f"{name}: expected columns [{S1_COL}, {list_col}], got {list(df.columns)}")
    s1 = df[S1_COL]
    lst = df[list_col]
    # rows with too few fields are padded by pandas with NaN even with keep_default_na=False
    report[f"{name}_malformed_null_fields"] = int(s1.isna().sum() + lst.isna().sum())
    s1 = s1.fillna("").astype(STR_DTYPE)
    lst = lst.fillna("").astype(STR_DTYPE)
    s1_st = s1.str.strip()
    report[f"{name}_s1_ids_with_whitespace"] = int((s1_st != s1).sum())
    report[f"{name}_empty_s1_id_rows"] = int((s1_st == "").sum())
    report[f"{name}_rows"] = int(len(df))
    report[f"{name}_duplicate_s1_rows"] = int(s1_st.duplicated().sum())
    ne = (lst.str.len() > 0).to_numpy()
    ne_vals = lst[ne]
    if len(ne_vals):
        tokens = pd.Series(",".join(ne_vals.tolist()).split(","), dtype=STR_DTYPE)
        counts = (ne_vals.str.count(",") + 1).to_numpy()
    else:
        tokens = pd.Series([], dtype=STR_DTYPE)
        counts = np.zeros(0, dtype=np.int64)
    row_idx = np.repeat(np.flatnonzero(ne).astype(np.int64), counts)
    tok_st = tokens.str.strip() if len(tokens) else tokens
    report[f"{name}_tokens_with_whitespace"] = int((tok_st != tokens).sum())
    keep = (tok_st != "").to_numpy()
    report[f"{name}_empty_tokens"] = int((~keep).sum())
    tok_st = tok_st[keep]
    report[f"{name}_ids_bad_prefix"] = int((~tok_st.str.startswith(("S2-", "S3-"))).sum()) if len(tok_st) else 0
    return s1_st.reset_index(drop=True), row_idx[keep], tok_st.reset_index(drop=True)


def _pct(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    p50, p95, p99 = np.percentile(x, [50, 95, 99])
    return {"mean": float(x.mean()), "p50": float(p50), "p95": float(p95), "p99": float(p99), "max": int(x.max())}


def f05_arrays(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray):
    """Per-entity P, R, F0.5 with the explicit edge cases. Inputs are integer arrays."""
    tp = tp.astype(np.float64)
    n_pred_f = n_pred.astype(np.float64)
    n_true_f = n_true.astype(np.float64)
    P = np.zeros_like(tp)
    R = np.zeros_like(tp)
    F = np.zeros_like(tp)
    both_empty = (n_true == 0) & (n_pred == 0)
    regular = (n_true > 0) & (n_pred > 0) & (tp > 0)  # everything else is 0.0 by construction
    P[regular] = tp[regular] / n_pred_f[regular]
    R[regular] = tp[regular] / n_true_f[regular]
    F[regular] = 1.25 * P[regular] * R[regular] / (0.25 * P[regular] + R[regular])
    P[both_empty] = 1.0
    R[both_empty] = 1.0
    F[both_empty] = 1.0
    return P, R, F


def _country_masks(source1_df, s1_index, eval_mask):
    if "entity_id" not in source1_df.columns or "country" not in source1_df.columns:
        raise ValueError("source1 file needs entity_id and country columns")
    country = pd.Series(source1_df["country"].to_numpy(), index=source1_df["entity_id"].to_numpy())
    country = country[~country.index.duplicated()]
    c_eval = np.asarray(country.reindex(s1_index[eval_mask]).fillna("<missing>").to_numpy(), dtype=object)
    for c in pd.unique(c_eval):
        yield str(c), c_eval == c


def oracle(gt_df: pd.DataFrame, cand_df: pd.DataFrame, s1_ids=None, source1_df=None) -> dict:
    """Candidate-set ceiling only (no predictions): coverage, candidate recall, oracle macro F0.5."""
    return score(gt_df, None, cand_df, s1_ids=s1_ids, source1_df=source1_df, oracle_only=True)


# ----------------------------------------------------------------------------------------------
# main scoring function
# ----------------------------------------------------------------------------------------------
def score(gt_df: pd.DataFrame, pred_df: pd.DataFrame | None, cand_df: pd.DataFrame | None,
          s1_ids=None, source1_df: pd.DataFrame | None = None, oracle_only: bool = False) -> dict:
    """Score predictions. Returns a JSON-serialisable dict.

    gt_df:   columns source1_entity_id, matched_entity_ids
    pred_df: columns source1_entity_id, matched_entity_ids
    cand_df: columns source1_entity_id, candidate_entity_ids (comma list) or candidate_entity_id
             (one per row); None skips candidate checks.
    s1_ids:  optional iterable of S1 IDs to restrict evaluation to (must be in GT).
    source1_df: optional frame with entity_id, country for a per-country breakdown.
    oracle_only: skip prediction scoring (pred_df may be None) and return only candidate metrics:
             candidate recall, entity coverage, oracle macro F0.5, candidates per S1.
    """
    if oracle_only:
        if cand_df is None:
            raise ValueError("oracle_only requires cand_df")
        pred_df = pd.DataFrame({S1_COL: pd.Series([], dtype=STR_DTYPE), MATCH_COL: pd.Series([], dtype=STR_DTYPE)})
    elif pred_df is None:
        raise ValueError("pred_df is required unless oracle_only=True")
    t0 = time.time()
    rep: dict = {}
    warnings: list[str] = []
    errors: list[str] = []

    # ---- ground truth ------------------------------------------------------------------------
    gt_s1, gt_rows, gt_tok = _explode(gt_df, MATCH_COL, "gt", rep)
    s1_index = pd.Index(pd.unique(gt_s1))
    n_s1_all = len(s1_index)

    eval_mask = np.ones(n_s1_all, dtype=bool)
    if s1_ids is not None:
        req = pd.Index(pd.unique(pd.Series(list(s1_ids), dtype=STR_DTYPE).str.strip()))
        req = req[req != ""]
        pos = s1_index.get_indexer(req)
        rep["s1_ids_requested"] = int(len(req))
        rep["s1_ids_not_in_gt"] = int((pos < 0).sum())
        if (pos < 0).any():
            warnings.append(f"{int((pos < 0).sum())} --s1-ids entries are not in the ground truth (ignored)")
        eval_mask = np.zeros(n_s1_all, dtype=bool)
        eval_mask[pos[pos >= 0]] = True
    n_eval = int(eval_mask.sum())
    rep["n_s1_evaluated"] = n_eval

    pr_s1, pr_rows, pr_tok = _explode(pred_df, MATCH_COL, "pred", rep)
    if cand_df is not None:
        ccol = CAND_COL if CAND_COL in cand_df.columns else (CAND_LONG_COL if CAND_LONG_COL in cand_df.columns else CAND_COL)
        rep["cand_format"] = "list" if ccol == CAND_COL else "long"
        ca_s1, ca_rows, ca_tok = _explode(cand_df, ccol, "cand", rep)
    else:
        ca_s1, ca_rows, ca_tok = pd.Series([], dtype=STR_DTYPE), np.zeros(0, np.int64), pd.Series([], dtype=STR_DTYPE)

    # one shared integer vocabulary for all S2/S3 IDs
    m_codes_all, m_uniques = pd.factorize(pd.concat([gt_tok, pr_tok, ca_tok], ignore_index=True))
    n_m = max(len(m_uniques), 1)
    ng, npr = len(gt_tok), len(pr_tok)
    m_codes_all = m_codes_all.astype(np.int64)
    gt_m, pr_m, ca_m = m_codes_all[:ng], m_codes_all[ng:ng + npr], m_codes_all[ng + npr:]

    # S1 codes per row; pred/cand S1 not in GT get distinct codes >= n_s1_all
    gt_rc = s1_index.get_indexer(gt_s1).astype(np.int64)
    pr_rc = s1_index.get_indexer(pr_s1).astype(np.int64)
    ca_rc = s1_index.get_indexer(ca_s1).astype(np.int64)
    if (pr_rc < 0).any() or (ca_rc < 0).any():
        extra = pd.Index(pd.unique(pd.concat([pr_s1[pr_rc < 0], ca_s1[ca_rc < 0]], ignore_index=True)))
        pr_rc[pr_rc < 0] = n_s1_all + extra.get_indexer(pr_s1[pr_rc < 0])
        ca_rc[ca_rc < 0] = n_s1_all + extra.get_indexer(ca_s1[ca_rc < 0])
    gt_ps, pr_ps_k, ca_ps_k = gt_rc[gt_rows], pr_rc[pr_rows], ca_rc[ca_rows]

    # ---- GT sanity ---------------------------------------------------------------------------
    gt_keys_raw = gt_ps * n_m + gt_m
    gt_keys = _uniq(gt_keys_raw)
    rep["gt_duplicate_ids_within_list"] = int(len(gt_keys_raw) - len(gt_keys))
    gk_m = gt_keys % n_m
    rep["gt_ids_matched_to_multiple_s1"] = int((np.bincount(gk_m, minlength=n_m) > 1).sum()) if len(gk_m) else 0
    if rep["gt_duplicate_s1_rows"]:
        warnings.append(f"ground truth has {rep['gt_duplicate_s1_rows']} duplicate S1 rows (unioned)")

    # ---- predictions -------------------------------------------------------------------------
    pr_row_known = pr_rc[(pr_s1 != "").to_numpy()]
    pr_s1_codes = np.unique(pr_row_known)
    rep["pred_s1_rows_not_in_gt"] = int((pr_s1_codes >= n_s1_all).sum())
    in_gt_pos = pr_s1_codes[pr_s1_codes < n_s1_all]
    rep["pred_s1_rows_in_gt_not_evaluated"] = int((~eval_mask[in_gt_pos]).sum())
    present = np.zeros(n_s1_all, dtype=bool)
    present[in_gt_pos] = True
    rep["evaluated_s1_missing_from_pred"] = int((eval_mask & ~present).sum())
    if rep["evaluated_s1_missing_from_pred"]:
        warnings.append(f"{rep['evaluated_s1_missing_from_pred']} evaluated S1 IDs missing from predictions (scored as empty)")
    if rep["pred_duplicate_s1_rows"]:
        warnings.append(f"predictions have {rep['pred_duplicate_s1_rows']} duplicate S1 rows (unioned; rejected by the official validator)")

    pr_keys_raw = pr_ps_k * n_m + pr_m
    pr_keys = _uniq(pr_keys_raw)
    rep["pred_duplicate_ids_within_list"] = int(len(pr_keys_raw) - len(pr_keys))
    if rep["pred_duplicate_ids_within_list"]:
        warnings.append(f"{rep['pred_duplicate_ids_within_list']} duplicate predicted IDs within an S1 list (deduplicated for scoring)")
    pk_m = pr_keys % n_m
    per_m = np.bincount(pk_m, minlength=n_m) if len(pk_m) else np.zeros(n_m, dtype=np.int64)
    rep["pred_ids_assigned_to_multiple_s1"] = int((per_m > 1).sum())
    rep["pred_pairs_total"] = int(len(pr_keys))
    if rep["pred_ids_bad_prefix"]:
        warnings.append(f"{rep['pred_ids_bad_prefix']} predicted IDs do not start with S2-/S3-")

    # ---- candidates --------------------------------------------------------------------------
    if cand_df is not None:
        ca_keys = _uniq(ca_ps_k * n_m + ca_m)
        rep["cand_duplicate_ids_within_list"] = int(len(ca_m) - len(ca_keys))
        rep["cand_pairs_total"] = int(len(ca_keys))
        not_in_cand = ~np.isin(pr_keys, ca_keys, assume_unique=True)
        rep["pred_pairs_not_in_candidates"] = int(not_in_cand.sum())
        if rep["pred_pairs_not_in_candidates"]:
            errors.append(f"{rep['pred_pairs_not_in_candidates']} predicted (s1, match) pairs are not in candidate_pairs")
        # candidates per evaluated S1 (S1 absent from candidates -> 0)
        ca_s1c = ca_keys // n_m
        ca_known = ca_s1c < n_s1_all
        n_cand = np.bincount(ca_s1c[ca_known], minlength=n_s1_all)
        rep["candidates_per_s1"] = _pct(n_cand[eval_mask])
        rep["evaluated_s1_without_candidates"] = int((n_cand[eval_mask] == 0).sum())
        gt_eval_keys = gt_keys[eval_mask[gt_keys // n_m]]
        rep["true_pairs_evaluated"] = int(len(gt_eval_keys))
        hit = np.isin(gt_eval_keys, ca_keys, assume_unique=True)
        rep["candidate_recall"] = float(hit.mean()) if len(gt_eval_keys) else 1.0
        rep["true_pairs_missing_from_candidates"] = int((~hit).sum())
        # oracle: per S1 predict exactly true ∩ candidates
        oracle_tp = np.bincount(gt_eval_keys[hit] // n_m, minlength=n_s1_all)[:n_s1_all]
    else:
        rep["pred_pairs_not_in_candidates"] = None
        warnings.append("no candidate file given: candidate checks skipped")

    # ---- per-entity metric -------------------------------------------------------------------
    n_true = np.bincount(gt_keys // n_m, minlength=n_s1_all)[:n_s1_all]
    pk_s1 = pr_keys // n_m
    known = pk_s1 < n_s1_all
    n_pred = np.bincount(pk_s1[known], minlength=n_s1_all)[:n_s1_all]
    tp_keys = pr_keys[known][np.isin(pr_keys[known], gt_keys, assume_unique=True)]
    tp = np.bincount(tp_keys // n_m, minlength=n_s1_all)[:n_s1_all]
    rep["pred_pairs_for_unscored_s1"] = int((~known).sum() + (~eval_mask[pk_s1[known]]).sum())

    P, R, F = f05_arrays(tp[eval_mask], n_pred[eval_mask], n_true[eval_mask])
    nt, npd, tpe = n_true[eval_mask], n_pred[eval_mask], tp[eval_mask]
    single = nt == 0

    cand_block = None
    if cand_df is not None:
        otp = oracle_tp[eval_mask]
        _, _, OF = f05_arrays(otp, otp, nt)
        covered = otp == nt  # all true matches present (vacuously true for singletons)

        def cand_summary(mask):
            ns = mask & ~single
            return {
                "n": int(mask.sum()),
                "entity_coverage": float(covered[ns].mean()) if ns.any() else None,
                "oracle_macro_f05": float(OF[mask].mean()) if mask.any() else None,
                "oracle_nonsingleton_macro_f05": float(OF[ns].mean()) if ns.any() else None,
            }

        allm = np.ones(n_eval, dtype=bool)
        cs = cand_summary(allm)
        cand_block = {
            "entity_coverage": cs["entity_coverage"],
            "entity_coverage_definition": "fraction of evaluated NON-singleton S1 whose true matches are all in their candidate list",
            "candidate_recall": rep["candidate_recall"],
            "candidate_recall_definition": "pair level: fraction of evaluated true (s1, id) pairs present in candidates",
            "oracle_macro_f05": cs["oracle_macro_f05"],
            "oracle_nonsingleton_macro_f05": cs["oracle_nonsingleton_macro_f05"],
            "oracle_definition": "per S1 predict exactly true ∩ candidates; singletons -> empty -> 1.0",
            "nonsingleton_s1_fully_covered": int(covered[~single].sum()),
            "nonsingleton_s1_zero_true_in_candidates": int(((otp == 0) & ~single).sum()),
            "candidates_per_s1": rep["candidates_per_s1"],
            "evaluated_s1_without_candidates": rep["evaluated_s1_without_candidates"],
            "cand_pairs_total": rep["cand_pairs_total"],
        }

    def summary(mask):
        k = int(mask.sum())
        if k == 0:
            return {"n": 0, "macro_f05": None, "mean_precision": None, "mean_recall": None}
        return {"n": k, "macro_f05": float(F[mask].mean()), "mean_precision": float(P[mask].mean()),
                "mean_recall": float(R[mask].mean())}

    ns = ~single
    out = {}
    if not oracle_only:
        out["nonsingleton"] = {
            "n": int(ns.sum()),
            "macro_f05": float(F[ns].mean()) if ns.any() else None,
            "mean_precision": float(P[ns].mean()) if ns.any() else None,
            "mean_recall": float(R[ns].mean()) if ns.any() else None,
        }
        out["macro_f05"] = float(F.mean()) if n_eval else None  # official leaderboard metric
        out["mean_precision_all_incl_singletons"] = float(P.mean()) if n_eval else None
        out["mean_recall_all_incl_singletons"] = float(R.mean()) if n_eval else None
        out["note"] = ("macro_f05 is the leaderboard metric (all S1). *_all_incl_singletons count singletons as "
                       "P=R=1 if predicted empty, else 0; 'nonsingleton' excludes singletons.")
    if cand_block is not None:
        out["candidates"] = cand_block
    out["n_s1_evaluated"] = n_eval
    out["singleton_rate"] = float(single.mean()) if n_eval else None
    if oracle_only:
        out["mode"] = "oracle_only"
        if source1_df is not None:
            out["by_country"] = {c: cand_summary(m) for c, m in _country_masks(source1_df, s1_index, eval_mask)}
        out["checks"] = {k: v for k, v in rep.items() if not k.startswith("pred")}
        out["warnings"] = [w for w in warnings if "predict" not in w]
        out["errors"] = []
        out["valid"] = True
        out["score_seconds"] = round(time.time() - t0, 3)
        return out
    out["breakdown"] = {"singletons": summary(single), "non_singletons": summary(~single)}
    out["breakdown"]["singletons"]["accuracy_pred_empty"] = (float((npd[single] == 0).mean()) if single.any() else None)
    ptp, pp, pt = int(tpe.sum()), int(npd.sum()), int(nt.sum())
    out["pair_level"] = {"tp": ptp, "pred_pairs": pp, "true_pairs": pt,
                         "precision": ptp / pp if pp else None, "recall": ptp / pt if pt else None}
    out["pred_empty_rate"] = float((npd == 0).mean()) if n_eval else None

    if source1_df is not None:
        by_c = {}
        for c, m in _country_masks(source1_df, s1_index, eval_mask):
            s = summary(m)
            s["singleton_rate"] = float(single[m].mean())
            s["non_singleton_macro_f05"] = summary(m & ~single)["macro_f05"]
            if cand_block is not None:
                s.update({k: v for k, v in cand_summary(m).items() if k != "n"})
            by_c[c] = s
        out["breakdown"]["by_country"] = by_c

    out["checks"] = rep
    out["warnings"] = warnings
    out["errors"] = errors
    out["valid"] = not errors
    out["score_seconds"] = round(time.time() - t0, 3)
    return out


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def _read_ids(path) -> list[str]:
    """--s1-ids: a plain list (one ID per line) or a TSV with a source1_entity_id / entity_id column."""
    df = pd.read_csv(path, header=None, **READ_KWARGS)
    first = df.iloc[0].tolist() if len(df) else []
    for col in (S1_COL, "entity_id"):
        if col in first:
            j = first.index(col)
            return df.iloc[1:, j].tolist()
    return df.iloc[:, 0].tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="ground truth TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--pred", help="matching_results.tsv (required unless --oracle-only)")
    ap.add_argument("--cand", help="candidate_pairs.tsv (source1_entity_id, candidate_entity_ids)")
    ap.add_argument("--s1-ids", help="optional file of S1 IDs to evaluate (e.g. validation split)")
    ap.add_argument("--source1", help="optional source1 TSV (entity_id, country) for per-country breakdown")
    ap.add_argument("--out", help="write metrics JSON here as well")
    ap.add_argument("--oracle-only", action="store_true",
                    help="only candidate metrics (coverage, candidate recall, oracle F0.5); needs --gt and --cand")
    a = ap.parse_args(argv)
    if a.oracle_only and not a.cand:
        ap.error("--oracle-only requires --cand")
    if not a.oracle_only and not a.pred:
        ap.error("--pred is required unless --oracle-only")

    t0 = time.time()
    gt = read_tsv(a.gt, usecols=[S1_COL, MATCH_COL])
    pred = read_tsv(a.pred) if (a.pred and not a.oracle_only) else None
    cand = read_tsv(a.cand) if a.cand else None
    src1 = read_tsv(a.source1, usecols=["entity_id", "country"]) if a.source1 else None
    ids = _read_ids(a.s1_ids) if a.s1_ids else None
    t_read = time.time() - t0
    res = score(gt, pred, cand, s1_ids=ids, source1_df=src1, oracle_only=a.oracle_only)
    res["read_seconds"] = round(t_read, 3)
    res["inputs"] = {"gt": a.gt, "pred": None if a.oracle_only else a.pred, "cand": a.cand, "s1_ids": a.s1_ids, "source1": a.source1}
    s = json.dumps(res, indent=2)
    print(s)
    if a.out:
        Path(a.out).write_text(s + "\n")
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
