"""Submission-2 matcher: trained on the large-pool candidate set it will predict on (same floor), so s1_n and the
record-competition features have the same distribution at train and inference time.
Training rows: 40% of train-side S1 (hash(entity_id, 11) % 10 < 4; val S1 never used), all their candidates,
from `largepool.py score --dump-train`. Early stopping on 10% of those S1 (hash(s1_idx, 5) % 10 == 0).
  python train_v2.py --floor 7.966 --features v2   -> models/v2_model_ALL_f7.966.txt + log json
"""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb, numpy as np, polars as pl
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
from business_entity_resolution.src.pair_features import FEATURES, FEATURES_V2

W = Path("/workspace/amlc/work"); REPO = Path(__file__).resolve().parents[4]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=6)
ap = argparse.ArgumentParser(); ap.add_argument("--floor", type=float, required=True)
ap.add_argument("--features", default="v2", choices=["v1", "v2"]); ap.add_argument("--tag", default="")
ap.add_argument("--exclude-s1", default="", help="parquet with entity_id of S1 to leave out (e.g. the v1 tuning subset)")
ap.add_argument("--countries", default="India,US"); ap.add_argument("--name", default="")
ap.add_argument("--s1-tenths", type=int, default=4, help="use train S1 with hash(entity_id,11)%%10 < this (dump has < 4)")
a = ap.parse_args()
F = FEATURES_V2 if a.features == "v2" else FEATURES
d = W / "largepool" / a.tag if a.tag else W / "largepool"
gt = load_ground_truth("/workspace/amlc/dataset/dataset")
parts, t0 = [], time.time()
ex = pl.read_parquet(a.exclude_s1)["entity_id"] if a.exclude_s1 else None
for c in a.countries.split(","):
    n1, n2, n3 = (pl.scan_parquet(f"{W}/norm_trainall_s{n}.parquet").filter(pl.col("country") == c)
                  .select("idx", "entity_id", "script").collect() for n in (1, 2, 3))
    tr = truth_pairs(gt, n1, n2, n3).select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y"))
    keep = n1.filter(pl.col("entity_id").hash(11) % 10 < a.s1_tenths)
    if ex is not None:
        keep = keep.filter(~pl.col("entity_id").is_in(ex))
    keep = keep["idx"]
    f = (pl.scan_parquet(d / f"trainfeat_{c}_f{a.floor:g}.parquet").filter(pl.col("s1_idx").is_in(keep))
         .select(["s1_idx", "src", "cand_idx"] + [x for x in F if x not in ("src",)]).collect()
         .join(tr, on=["s1_idx", "src", "cand_idx"], how="left"))
    del n1, n2, n3, tr
    parts.append(f.with_columns(pl.col("y").fill_null(False), pl.lit(c).alias("country")))
f = pl.concat(parts); del parts
hold = ((pl.col("country") + pl.col("s1_idx").cast(pl.Utf8)).hash(5) % 10 == 0)
hold = f.select(hold).to_series().to_numpy()
X = f.select(F).to_numpy().astype(np.float32); y = f["y"].to_numpy()
rows_by_country = dict(f.group_by("country").len().iter_rows())
train_s1 = int(f.filter(~pl.Series(hold)).select("country", "s1_idx").n_unique())
del f
ds = lgb.Dataset(X[~hold], y[~hold], feature_name=F, free_raw_data=True)
dv = lgb.Dataset(X[hold], y[hold], reference=ds)
m = lgb.train(PARAMS, ds, 1500, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
out = REPO / f"code/business_entity_resolution/models/{a.name or ('v2' + ('_v1feat' if a.features == 'v1' else '') + '_model_ALL')}_f{a.floor:g}.txt"
m.save_model(str(out))
log = {"floor": a.floor, "countries": a.countries, "excluded_s1_file": a.exclude_s1,
       "selection": f"train-side S1 (not val), hash(entity_id,11)%10 < {a.s1_tenths}" + (", minus excluded" if a.exclude_s1 else ""), "features": F, "train_rows": int((~hold).sum()), "earlystop_rows": int(hold.sum()),
       "train_s1": train_s1, "s1_tenths": a.s1_tenths,
       "positives": int(y[~hold].sum()), "rows_by_country": rows_by_country,
       "best_iter": m.best_iteration, "best_valid_logloss": m.best_score["valid_0"]["binary_logloss"],
       "seconds": round(time.time() - t0, 1), "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
       "model_file": str(out), "model_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
       "top_gain": dict(sorted(zip(F, m.feature_importance("gain").round(0).tolist()), key=lambda x: -x[1])[:10])}
(out.with_suffix(".json")).write_text(json.dumps(log, indent=1))
print(json.dumps(log), flush=True)
