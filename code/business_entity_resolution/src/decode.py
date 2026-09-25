"""Decoding + entity-level macro F0.5 (spec §5) on a scored candidate table.

decode: keep p >= t; one-S1-per-record (each S2/S3 goes only to its highest-p S1 within the given
population); S1 predicted empty when its best p < t_empty.
macro_f05: per S1 over ALL evaluated S1 (singletons: empty -> 1, else 0), true matches missing from
candidates count as false negatives.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def prepare(scored: pl.DataFrame, true_count: pl.DataFrame, s1_all: pl.DataFrame):
    """scored: (s1_idx, src, cand_idx, p, y). true_count: (s1_idx, n_true). s1_all: (s1_idx)."""
    s = scored.select("s1_idx", "src", "cand_idx", "p", "y").with_columns(
        (pl.col("p") == pl.col("p").max().over("src", "cand_idx")).alias("rec_best"),
        pl.col("p").max().over("s1_idx").alias("s1_best"))
    # break exact ties on record-best deterministically (lowest s1_idx)
    s = s.with_columns((pl.col("rec_best") & (pl.col("s1_idx") == pl.col("s1_idx").filter(pl.col("rec_best")).min()
                                                 .over("src", "cand_idx"))).alias("rec_best"))
    base = s1_all.select("s1_idx").join(true_count, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0))
    return s.filter(pl.col("rec_best")).select("s1_idx", "p", "y", "s1_best"), base


def macro_f05(kept: pl.DataFrame, base: pl.DataFrame, t: float, t_empty: float) -> dict:
    k = kept.filter((pl.col("p") >= t) & (pl.col("s1_best") >= t_empty))
    agg = k.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("y").sum().alias("tp"))
    d = base.join(agg, on="s1_idx", how="left").with_columns(pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0))
    npred, tp, ntrue = (d[c].to_numpy().astype(np.float64) for c in ("n_pred", "tp", "n_true"))
    P = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    R = np.divide(tp, ntrue, out=np.zeros_like(tp), where=ntrue > 0)
    den = 0.25 * P + R
    F = np.divide(1.25 * P * R, den, out=np.zeros_like(tp), where=den > 0)
    F = np.where((ntrue == 0) & (npred == 0), 1.0, F)
    sing = ntrue == 0
    return {"macro_f05": float(F.mean()), "nonsingleton_f05": float(F[~sing].mean()),
            "singleton_acc": float((npred[sing] == 0).mean()) if sing.any() else None,
            "pair_precision": float(tp.sum() / max(npred.sum(), 1)), "pair_recall": float(tp.sum() / max(ntrue.sum(), 1)),
            "n_s1": int(len(F)), "t": t, "t_empty": t_empty}


def tune(kept, base, ts=np.round(np.arange(0.1, 0.96, 0.05), 2), tes=np.round(np.arange(0.1, 0.96, 0.05), 2)) -> dict:
    best = None
    for t in ts:
        for te in tes:
            if te < t:  # t_empty below t has no effect (all kept pairs already have p >= t)
                continue
            r = macro_f05(kept, base, float(t), float(te))
            if best is None or r["macro_f05"] > best["macro_f05"]:
                best = r
    return best
