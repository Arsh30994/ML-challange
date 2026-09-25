"""Leave-one-country-out check (France stand-in) with a first LightGBM matcher.

Train rows: blocking candidates (k=10) of the train-side tuning subset (25% of train-side S1 + their
matches + 25% of train-side unreferenced S2/S3). Eval rows: validation candidates (k=10).
Models: ALL (both countries), US-only, India-only. Early stopping on a 10% S1 hold-out of the training
rows (never on val). Thresholds (t, t_empty) tuned by grid on the named val population only.
"""
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb
import numpy as np
import polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare, tune
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
from business_entity_resolution.src.pair_features import FEATURES, build_features

W = Path("/workspace/amlc/work")
OUT = Path("/workspace/amlc/gt_checks/reports/loco_check")
OUT.mkdir(parents=True, exist_ok=True)
log = {}
T0 = time.time()


def feats_for(prefix, cand_path, gt, out_path):
    if Path(out_path).exists():
        return
    n1, n2, n3 = (pl.read_parquet(f"{prefix}_s{n}.parquet") for n in (1, 2, 3))
    tr = truth_pairs(gt, n1, n2, n3)
    c = pl.read_parquet(cand_path)
    f = build_features(c, n1, n2, n3)
    f = f.join(tr.select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y")), on=["s1_idx", "src", "cand_idx"], how="left")
    f = f.with_columns(pl.col("y").fill_null(False)).join(
        n1.select(pl.col("idx").alias("s1_idx"), "country", pl.col("entity_id").alias("s1_id")), on="s1_idx", how="left")
    f.write_parquet(out_path)
    tc = tr.group_by("s1_idx").agg(pl.len().alias("n_true"))
    tc.write_parquet(str(out_path).replace(".parquet", "_truecount.parquet"))
    n1.select(pl.col("idx").alias("s1_idx"), "country", pl.col("entity_id").alias("s1_id")).write_parquet(
        str(out_path).replace(".parquet", "_s1.parquet"))


t = time.time()
feats_for(str(W / "norm_tune"), W / "candidates_tune_k10.parquet", pl.read_parquet(W / "tune_gt.parquet"), W / "feat_tune_k10.parquet")
log["features_train_s"] = round(time.time() - t, 1)
t = time.time()
feats_for(str(W / "norm_val"), W / "candidates_val_k10.parquet", load_ground_truth("/workspace/amlc/dataset/dataset"), W / "feat_val_k10.parquet")
log["features_val_s"] = round(time.time() - t, 1)
print(log, flush=True)

tr = pl.read_parquet(W / "feat_tune_k10.parquet", columns=FEATURES + ["y", "country", "s1_idx"])
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=6)
models, sizes = {}, {}
for name, flt in (("ALL", None), ("US", "US"), ("India", "India")):
    d = tr if flt is None else tr.filter(pl.col("country") == flt)
    hold = (d["s1_idx"].hash(5) % 10 == 0).to_numpy()
    X = d.select(FEATURES).to_numpy(); y = d["y"].to_numpy()
    t = time.time()
    ds = lgb.Dataset(X[~hold], y[~hold], feature_name=FEATURES)
    dv = lgb.Dataset(X[hold], y[hold], reference=ds)
    m = lgb.train(PARAMS, ds, 1000, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
    models[name] = m
    sizes[name] = {"train_rows": int((~hold).sum()), "earlystop_rows": int(hold.sum()), "train_s1": int(d.filter(~pl.Series(hold))["s1_idx"].n_unique()),
                   "positives": int(y[~hold].sum()), "best_iter": m.best_iteration, "seconds": round(time.time() - t, 1)}
    m.save_model(str(W / f"loco_model_{name}.txt"))
    print(name, sizes[name], flush=True)
    del X, y, ds, dv
del tr
log["model_sizes"] = sizes

va = pl.read_parquet(W / "feat_val_k10.parquet", columns=FEATURES + ["y", "country", "s1_idx", "src", "cand_idx", "s1_id"])
tc = pl.read_parquet(W / "feat_val_k10_truecount.parquet")
s1 = pl.read_parquet(W / "feat_val_k10_s1.parquet")
Xv = va.select(FEATURES).to_numpy()
P = {n: m.predict(Xv, num_iteration=m.best_iteration, num_threads=6).astype(np.float32) for n, m in models.items()}
del Xv
imp = {n: dict(sorted(zip(FEATURES, m.feature_importance("gain").round(0).tolist()), key=lambda x: -x[1])[:8]) for n, m in models.items()}
log["top_gain_features"] = imp


def pop(model, country):
    sc = va.select("s1_idx", "src", "cand_idx", "y", "country").with_columns(pl.Series("p", P[model]))
    s1c = s1 if country == "ALL" else s1.filter(pl.col("country") == country)
    if country != "ALL":
        sc = sc.filter(pl.col("country") == country)
    return prepare(sc, tc, s1c)


res = {}
cache = {}
def get(model, country):
    if (model, country) not in cache:
        cache[(model, country)] = pop(model, country)
    return cache[(model, country)]

def run(label, model, tune_on, eval_on):
    kt, bt = get(model, tune_on)
    best = tune(kt, bt)
    ke, be = get(model, eval_on)
    r = macro_f05(ke, be, best["t"], best["t_empty"])
    res[label] = {"model": model, "thresholds_tuned_on_val": tune_on, "evaluated_on_val": eval_on,
                  "t": best["t"], "t_empty": best["t_empty"], "tune_pop_macro_f05": round(best["macro_f05"], 5),
                  **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k not in ("t", "t_empty")}}
    print(label, res[label], flush=True)

t = time.time()
run("LOCO_India: US-only model, tuned on US val -> India val", "US", "US", "India")
run("NORMAL_India_sameproc: ALL model, tuned on US val -> India val", "ALL", "US", "India")
run("NORMAL_India_allval: ALL model, tuned on all val -> India val", "ALL", "ALL", "India")
run("NORMAL_India_incountry: ALL model, tuned on India val -> India val (optimistic)", "ALL", "India", "India")
run("LOCO_US: India-only model, tuned on India val -> US val", "India", "India", "US")
run("NORMAL_US_sameproc: ALL model, tuned on India val -> US val", "ALL", "India", "US")
run("NORMAL_US_incountry: ALL model, tuned on US val -> US val (optimistic)", "ALL", "US", "US")
run("NORMAL_ALL: ALL model, tuned on all val -> all val (optimistic)", "ALL", "ALL", "ALL")
log["decode_tune_s"] = round(time.time() - t, 1)

# export the NORMAL_ALL prediction as matching_results TSV for a scorer cross-check
b = res["NORMAL_ALL: ALL model, tuned on all val -> all val (optimistic)"]
sc = va.select("s1_idx", "src", "cand_idx", "s1_id").with_columns(pl.Series("p", P["ALL"]))
n2 = pl.read_parquet(W / "norm_val_s2.parquet", columns=["idx", "entity_id"]); n3 = pl.read_parquet(W / "norm_val_s3.parquet", columns=["idx", "entity_id"])
ids = pl.concat([n2.select(pl.lit(2, dtype=pl.Int8).alias("src"), pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("mid")),
                 n3.select(pl.lit(3, dtype=pl.Int8).alias("src"), pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("mid"))])
sc = sc.with_columns((pl.col("p") == pl.col("p").max().over("src", "cand_idx")).alias("rb"), pl.col("p").max().over("s1_idx").alias("sb"))
sc = sc.with_columns((pl.col("rb") & (pl.col("s1_idx") == pl.col("s1_idx").filter(pl.col("rb")).min().over("src", "cand_idx"))).alias("rb"))
kept = sc.filter(pl.col("rb") & (pl.col("p") >= b["t"]) & (pl.col("sb") >= b["t_empty"])).join(ids, on=["src", "cand_idx"], how="left")
agg = kept.sort("p", descending=True).group_by("s1_idx").agg(pl.col("mid").str.join(",").alias("matched_entity_ids"))
pred = s1.join(agg, on="s1_idx", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).select(
    pl.col("s1_id").alias("source1_entity_id"), "matched_entity_ids").sort("source1_entity_id")
pred.write_csv(W / "loco_pred_val_ALL.tsv", separator="\t", quote_style="never")

log["results"] = res
log["total_s"] = round(time.time() - T0, 1)
log["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
(OUT / "metrics.json").write_text(json.dumps(log, indent=2))
print("done", log["total_s"], log["peak_rss_gb"])
