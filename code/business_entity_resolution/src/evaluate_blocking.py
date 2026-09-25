"""Recall ceiling of a candidate table against ground-truth pairs (memory-lean: one truth join,
candidate counts from per-(S1, source) group sizes)."""
from __future__ import annotations

import numpy as np
import polars as pl


def truth_pairs(gt: pl.DataFrame, n1: pl.DataFrame, n2: pl.DataFrame, n3: pl.DataFrame) -> pl.DataFrame:
    """GT restricted to the S1 in n1 -> (s1_idx, src, cand_idx, script). Every matched ID must exist."""
    from business_entity_resolution.src.loader import explode_ground_truth
    g = gt.join(n1.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id", how="semi")
    p = explode_ground_truth(g)
    s1map = n1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("idx").alias("s1_idx"))
    out = []
    for src, nn in ((2, n2), (3, n3)):
        q = (p.filter(pl.col("source") == f"S{src}")
             .join(s1map, on="source1_entity_id", how="left")
             .join(nn.select(pl.col("entity_id").alias("matched_id"), pl.col("idx").alias("cand_idx"), "script"),
                   on="matched_id", how="left"))
        if q["cand_idx"].null_count() or q["s1_idx"].null_count():
            raise ValueError(f"S{src}: GT IDs missing from the evaluated frames")
        out.append(q.select("s1_idx", pl.lit(src, dtype=pl.Int8).alias("src"), "cand_idx", "script"))
    return pl.concat(out)


def _key(s1, src, cidx) -> np.ndarray:
    return (s1.to_numpy().astype(np.int64) << 33) | (src.to_numpy().astype(np.int64) << 31) | cidx.to_numpy().astype(np.int64)


def _dist(x: np.ndarray) -> dict:
    return {"mean": round(float(x.mean()), 3), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": int(x.max())}


def _with_rank(cand: pl.DataFrame, score_col: str) -> pl.DataFrame:
    if "rank" in cand.columns:
        return cand.select("s1_idx", "src", "cand_idx", "rank", score_col)
    return cand.select("s1_idx", "src", "cand_idx", score_col).with_columns(
        pl.col(score_col).rank("ordinal", descending=True).over("s1_idx", "src").cast(pl.Int16).alias("rank"))


def _metrics(t: pl.DataFrame, counts: pl.DataFrame, s1: pl.DataFrame) -> dict:
    """t: truth with boolean 'hit' and country/script; counts: (s1_idx, n2, n3) kept per S1."""
    r = {"pair_recall": round(float(t["hit"].mean()), 5), "true_pairs": t.height, "pairs_kept": int(t["hit"].sum())}
    r["pair_recall_by_source"] = {f"S{s}": round(float(g["hit"].mean()), 5)
                                  for (s,), g in t.sort("src").group_by("src", maintain_order=True)}
    ent = t.group_by("s1_idx").agg(pl.col("hit").all().alias("all"), pl.col("hit").any().alias("any"),
                                   pl.col("country").first())
    r["entity_all_matches_kept"] = round(float(ent["all"].mean()), 5)
    r["entity_at_least_one_kept"] = round(float(ent["any"].mean()), 5)
    r["non_singleton_s1"] = ent.height
    r["by_country"] = {}
    for (c,), g in t.group_by("country"):
        e = ent.filter(pl.col("country") == c)
        r["by_country"][c] = {"pair_recall": round(float(g["hit"].mean()), 5), "true_pairs": g.height,
                              "entity_all_kept": round(float(e["all"].mean()), 5),
                              "entity_any_kept": round(float(e["any"].mean()), 5)}
    r["by_script"] = {sc: {"pair_recall": round(float(g["hit"].mean()), 5), "true_pairs": g.height}
                      for (sc,), g in t.group_by("script")}
    full = s1.join(counts, on="s1_idx", how="left").fill_null(0).with_columns((pl.col("n2") + pl.col("n3")).alias("n"))
    r["candidates_per_s1"] = _dist(full["n"].to_numpy())
    r["candidates_per_s1_S2"] = _dist(full["n2"].to_numpy())
    r["candidates_per_s1_S3"] = _dist(full["n3"].to_numpy())
    r["candidates_total"] = int(full["n"].sum())
    r["candidates_per_s1_by_country"] = {c: round(float(g["n"].mean()), 3) for (c,), g in full.group_by("country")}
    return r


def recall_at_k(cand: pl.DataFrame, truth: pl.DataFrame, n1: pl.DataFrame, ks=(5, 10, 15, 20, 30),
                score_col: str = "score", thresholds=None) -> dict:
    """Fixed top-k per (S1, source). If thresholds is given, also top-k AND score >= t variants
    (keyed "k{k}_t{t}")."""
    cand = _with_rank(cand, score_col)
    s1 = n1.select(pl.col("idx").alias("s1_idx"), "country")
    # truth lookup by sorted int64 keys (numpy searchsorted; avoids a 26M-row hash join)
    ck = _key(cand["s1_idx"], cand["src"], cand["cand_idx"])
    order = np.argsort(ck, kind="stable")
    ck = ck[order]
    tk = _key(truth["s1_idx"], truth["src"], truth["cand_idx"])
    pos = np.clip(np.searchsorted(ck, tk), 0, max(len(ck) - 1, 0))
    found = (ck[pos] == tk) if len(ck) else np.zeros(len(tk), dtype=bool)
    rank_arr = cand["rank"].to_numpy()[order][pos]
    score_arr = cand[score_col].to_numpy()[order][pos]
    del ck, order
    t = truth.join(s1, on="s1_idx", how="left").with_columns(
        pl.Series("rank", np.where(found, rank_arr, 32000).astype(np.int32)),
        pl.Series(score_col, np.where(found, score_arr, -1e9).astype(np.float32)))
    sizes = cand.group_by("s1_idx", "src").agg(pl.col("rank").max().cast(pl.Int32).alias("m"))
    res = {}
    for k in ks:
        counts = (sizes.with_columns(pl.min_horizontal(pl.col("m"), pl.lit(k)).alias("c"))
                  .group_by("s1_idx").agg((pl.col("c") * (pl.col("src") == 2)).sum().alias("n2"),
                                          (pl.col("c") * (pl.col("src") == 3)).sum().alias("n3")))
        res[str(k)] = _metrics(t.with_columns((pl.col("rank") <= k).alias("hit")), counts, s1)
    for k, thr in (thresholds or []):
        kept = cand.filter((pl.col("rank") <= k) & (pl.col(score_col) >= thr))
        counts = kept.group_by("s1_idx").agg((pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3"))
        res[f"k{k}_t{thr}"] = _metrics(t.with_columns(((pl.col("rank") <= k) & (pl.col(score_col) >= thr)).alias("hit")),
                                       counts, s1)
        del kept
    return res
