import sys, json, resource, polars as pl, numpy as np
sys.path.insert(0, '/workspace/amlc/gt_checks/code')
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
n1, n2, n3 = (pl.read_parquet(f'/workspace/amlc/work/norm_val_s{n}.parquet') for n in (1, 2, 3))
tr = truth_pairs(load_ground_truth('/workspace/amlc/dataset/dataset'), n1, n2, n3).select("s1_idx", "src", "cand_idx")
c = pl.read_parquet('/workspace/amlc/work/candidates_val_k15.parquet', columns=["s1_idx", "src", "cand_idx", "rank", "score"])
c = c.with_columns((pl.col("score").max().over("s1_idx", "src") - pl.col("score")).alias("gap"))
c = c.join(tr.with_columns(pl.lit(True).alias("y")), on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
T, N = tr.height, n1.height
out = []
for k in (10, 15):
    for t in np.arange(6.5, 10.01, 0.25):
        m = (c["rank"] <= k) & (c["score"] >= t)
        out.append(("abs", k, round(float(t), 2), round(int((m & c["y"]).sum()) / T, 5), round(int(m.sum()) / N, 3)))
    for dlt in (2, 3, 4, 5, 6):
        m = (c["rank"] <= k) & (c["gap"] <= dlt)
        out.append(("gap", k, dlt, round(int((m & c["y"]).sum()) / T, 5), round(int(m.sum()) / N, 3)))
for o in out: print(o)
print("peak_rss_gb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
json.dump(out, open('/workspace/amlc/work/tmp/curve.json', 'w'))
