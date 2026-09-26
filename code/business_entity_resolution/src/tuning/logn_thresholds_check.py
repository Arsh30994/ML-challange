import sys, json
sys.path.insert(0, "/workspace/amlc/gt_checks/code")
from pathlib import Path
import polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
W = Path("/workspace/amlc/work"); gt = load_ground_truth("/workspace/amlc/dataset/dataset")
val = pl.Series("entity_id", (W / "split/val_s1_ids.txt").read_text().split()).to_frame()
out = {}
for c, ths in (("India", [(0.66, 0.80), (0.62, 0.71), (0.63, 0.72)]), ("US", [(0.66, 0.80), (0.59, 0.68)])):
    a1, a2, a3 = (pl.scan_parquet(f"{W}/norm_trainall_s{n}.parquet").filter(pl.col("country") == c).select("idx", "entity_id", "script").collect() for n in (1, 2, 3))
    tr = truth_pairs(gt, a1, a2, a3).select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y")); tc = tr.group_by("s1_idx").agg(pl.len().alias("n_true"))
    v = a1.join(val, on="entity_id", how="semi").filter(pl.col("entity_id").hash(7) % 2 == 1).select(pl.col("idx").alias("s1_idx"))
    b = v.join(tc, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0))
    sc = pl.read_parquet(W / f"largepool/scored_{c}_f7.966_v2ALL.parquet").join(tr, on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
    k, _ = prepare(sc, tc, v); k = k.join(v, on="s1_idx", how="semi")
    for t, te in ths:
        out[f"{c} {t}/{te}"] = round(macro_f05(k, b, t, te)["macro_f05"], 5)
print(json.dumps(out))
Path("/workspace/amlc/gt_checks/reports/largepool_val/logN_thresholds_largepool_halfB.json").write_text(json.dumps(out, indent=1))
