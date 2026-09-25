"""k10 vs k10 + score floor 7.966 with the locked matcher (v1 ALL model, t=0.70, t_empty=0.70).
Features are recomputed on the floored candidate set (it is the scored set). Decide on half A, report half B."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb, numpy as np, polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
from business_entity_resolution.src.pair_features import FEATURES, build_features
W = Path("/workspace/amlc/work"); T, TE, FLOOR = 0.70, 0.70, 7.966
n1, n2, n3 = (pl.read_parquet(W / f"norm_val_s{n}.parquet") for n in (1, 2, 3))
tr = truth_pairs(load_ground_truth("/workspace/amlc/dataset/dataset"), n1, n2, n3)
tc = tr.group_by("s1_idx").agg(pl.len().alias("n_true"))
s1 = n1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").hash(7).alias("h")).with_columns(
    pl.when(pl.col("h") % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("half"))
m = lgb.Booster(model_file=str(W / "loco_model_ALL.txt"))
cand = pl.read_parquet(W / "candidates_val_k10.parquet")
res = {}
for name, c in (("k10", cand), ("k10_floor7.966", cand.filter(pl.col("score") >= FLOOR))):
    f = build_features(c, n1, n2, n3).join(tr.select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y")),
                                           on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
    f = f.with_columns(pl.Series("p", m.predict(f.select(FEATURES).to_numpy(), num_threads=6).astype(np.float32)))
    f = f.join(s1.select("s1_idx", "half"), on="s1_idx", how="left")
    r = {"pairs": c.height, "cand_per_s1": round(c.height / n1.height, 3)}
    for h in ("A", "B"):
        r[h] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in
                macro_f05(*prepare(f.filter(pl.col("half") == h), tc, s1.filter(pl.col("half") == h)), T, TE).items()}
    res[name] = r; print(name, r, flush=True)
dA = res["k10"]["A"]["macro_f05"] - res["k10_floor7.966"]["A"]["macro_f05"]
res["decision"] = {"half_A_diff_k10_minus_floor": round(dA, 5),
                   "chosen": "k10_floor7.966" if dA <= 0.001 else "k10", "rule": "prefer smaller set if within 0.001 on half A"}
print(res["decision"])
Path("/workspace/amlc/gt_checks/reports/model_baseline").mkdir(parents=True, exist_ok=True)
Path("/workspace/amlc/gt_checks/reports/model_baseline/floor_decision.json").write_text(json.dumps(res, indent=1))
