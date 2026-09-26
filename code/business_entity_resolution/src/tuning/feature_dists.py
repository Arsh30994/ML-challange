"""Quantiles of the pool-dependent features (rec_gap, rec_rank, rec_n_close, s1_n) on the floored (7.966) scored sets:
test France/India/US (work/test_run_v3/cand_*.parquet), small-pool val (candidates_val_k10.parquet) and large-pool val
(work/largepool/cand, rows of val S1 only; windows computed over the whole universe). Same formulas as pair_features."""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
W = Path("/workspace/amlc/work"); FLOOR = 7.966
REPO = Path(__file__).resolve().parents[4]
Q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def win(c):
    c = c.select("s1_idx", "src", "cand_idx", "score").with_columns(
        pl.len().over("s1_idx", "src").cast(pl.Float32).alias("s1_n"),
        pl.col("score").rank("ordinal", descending=True).over("src", "cand_idx").cast(pl.Float32).alias("rec_rank"),
        (pl.col("score").max().over("src", "cand_idx") - pl.col("score")).alias("rec_gap"))
    return c.with_columns((pl.col("rec_gap") <= 1.0).cast(pl.Float32).sum().over("src", "cand_idx").alias("rec_n_close"))


def qs(c):
    out = {"rows": c.height}
    for f in ("rec_gap", "rec_rank", "rec_n_close", "s1_n"):
        x = c[f].to_numpy()
        out[f] = {"mean": round(float(x.mean()), 3), **{f"p{int(q * 100)}": round(float(np.quantile(x, q)), 3) for q in Q}}
    out["share_rec_rank1"] = round(float((c["rec_rank"] == 1).mean()), 4)
    return out


res = {}
for c in ("France", "India", "US"):
    res[f"test_{c}"] = qs(win(pl.read_parquet(W / f"test_run_v3/cand_{c}.parquet")))
    print(c, res[f"test_{c}"]["rec_n_close"], flush=True)
nv = pl.read_parquet(W / "norm_val_s1.parquet", columns=["idx", "country"])
sv = win(pl.read_parquet(W / "candidates_val_k10.parquet").filter(pl.col("score") >= FLOOR)).join(
    nv.rename({"idx": "s1_idx"}), on="s1_idx")
val_ids = pl.Series("entity_id", (W / "split/val_s1_ids.txt").read_text().split()).to_frame()
for c in ("India", "US"):
    res[f"smallpool_val_{c}"] = qs(sv.filter(pl.col("country") == c))
    n1 = pl.scan_parquet(W / "norm_trainall_s1.parquet").filter(pl.col("country") == c).select("idx", "entity_id").collect()
    v = n1.join(val_ids, on="entity_id", how="semi").select(pl.col("idx").alias("s1_idx"))
    lp = win(pl.scan_parquet(str(W / f"largepool/cand/{c}/*.parquet")).filter(pl.col("score") >= FLOOR)
             .select("s1_idx", "src", "cand_idx", "score").collect())
    res[f"largepool_val_{c}"] = qs(lp.join(v, on="s1_idx", how="semi"))
    del lp
    print(c, "done", flush=True)
(REPO / "reports/largepool_val/feature_dists.json").write_text(json.dumps(res, indent=1))
