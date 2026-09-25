"""LOCO v2: relative name length (len_rel) instead of len_diff, honest half-split thresholds, robust t_empty.

Val S1 are split 50/50 by hash(entity_id, seed=7): thresholds are tuned on half A, reported on half B.
Robust t_empty: with the LOCO models (US-only -> India val, India-only -> US val) and t fixed at each model's
half-A best, pick the t_empty maximizing min(held-out India, held-out US) on half A; report half B and the
in-country cost for the ALL model (half B) versus its own best t_empty.
"""
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb
import numpy as np
import polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare, tune

W = Path("/workspace/amlc/work"); OUT = Path("/workspace/amlc/gt_checks/reports/loco_check")
T0 = time.time()
BASE_FEATS = ["cos_word", "cos_char", "cos_sk", "cos_addr", "addr_missing", "score", "rank", "methods", "s1_gap", "s1_n",
              "rec_rank", "rec_gap", "rec_n_close", "name_tsr", "name_jw", "name_pr", "addr_tsr", "num_jacc", "translit", "is_s3"]
FEATS = BASE_FEATS + ["len_rel"]
RELATIVE = ["rank", "s1_gap", "s1_n", "rec_rank", "rec_gap", "rec_n_close", "len_rel"]


def add_len_rel(f, prefix):
    n1 = pl.read_parquet(f"{prefix}_s1.parquet", columns=["idx", "name_n"])
    t = pl.concat([pl.read_parquet(f"{prefix}_s{s}.parquet", columns=["idx", "name_n"]).select(
        pl.lit(s, dtype=pl.Int8).alias("src"), pl.col("idx").alias("cand_idx"), pl.col("name_n").str.len_chars().alias("lb")) for s in (2, 3)])
    f = f.join(n1.select(pl.col("idx").alias("s1_idx"), pl.col("name_n").str.len_chars().alias("la")), on="s1_idx", how="left")
    f = f.join(t, on=["src", "cand_idx"], how="left")
    return f.with_columns(((pl.col("la").cast(pl.Int32) - pl.col("lb").cast(pl.Int32)).abs()
                           / pl.max_horizontal("la", "lb", pl.lit(1))).cast(pl.Float32).alias("len_rel")).drop("la", "lb")


tr = add_len_rel(pl.read_parquet(W / "feat_tune_k10.parquet", columns=BASE_FEATS + ["y", "country", "s1_idx", "src", "cand_idx"]), str(W / "norm_tune"))
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=6)
models, sizes = {}, {}
for name, flt in (("ALL", None), ("US", "US"), ("India", "India")):
    d = tr if flt is None else tr.filter(pl.col("country") == flt)
    hold = (d["s1_idx"].hash(5) % 10 == 0).to_numpy()
    X = d.select(FEATS).to_numpy(); y = d["y"].to_numpy()
    ds = lgb.Dataset(X[~hold], y[~hold], feature_name=FEATS); dv = lgb.Dataset(X[hold], y[hold], reference=ds)
    m = lgb.train(PARAMS, ds, 1000, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
    models[name] = m; sizes[name] = {"train_rows": int((~hold).sum()), "best_iter": m.best_iteration}
    m.save_model(str(W / f"loco2_model_{name}.txt")); print(name, sizes[name], flush=True)
del tr, X, y, ds, dv

va = add_len_rel(pl.read_parquet(W / "feat_val_k10.parquet", columns=BASE_FEATS + ["y", "country", "s1_idx", "src", "cand_idx", "s1_id"]), str(W / "norm_val"))
tc = pl.read_parquet(W / "feat_val_k10_truecount.parquet")
s1 = pl.read_parquet(W / "feat_val_k10_s1.parquet").with_columns(
    pl.when(pl.col("s1_id").hash(7) % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("half"))
va = va.join(s1.select("s1_idx", "half"), on="s1_idx", how="left")
Xv = va.select(FEATS).to_numpy()
P = {n: m.predict(Xv, num_iteration=m.best_iteration, num_threads=6).astype(np.float32) for n, m in models.items()}
del Xv
gain = {n: {f: round(float(g), 0) for f, g in sorted(zip(FEATS, m.feature_importance("gain")), key=lambda x: -x[1])} for n, m in models.items()}

cache = {}
def get(model, country, half):
    key = (model, country, half)
    if key not in cache:
        sc = va.select("s1_idx", "src", "cand_idx", "y", "country", "half").with_columns(pl.Series("p", P[model]))
        s = s1
        if country != "ALL":
            sc = sc.filter(pl.col("country") == country); s = s.filter(pl.col("country") == country)
        if half != "full":
            sc = sc.filter(pl.col("half") == half); s = s.filter(pl.col("half") == half)
        cache[key] = prepare(sc, tc, s)
    return cache[key]

R = lambda r: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
res = {}
def run(label, model, tune_on, eval_on):
    best = tune(*get(model, tune_on, "A"))
    r = macro_f05(*get(model, eval_on, "B"), best["t"], best["t_empty"])
    res[label] = {"model": model, "tuned_on": f"{tune_on} val half A", "evaluated_on": f"{eval_on} val half B",
                  **R(r), "tune_half_A_f05": round(best["macro_f05"], 5)}
    print(label, res[label], flush=True)
    return best

bA = run("ALL model: tuned all-A -> all-B (honest)", "ALL", "ALL", "ALL")
run("ALL model: tuned all-A -> India-B", "ALL", "ALL", "India")
run("ALL model: tuned all-A -> US-B", "ALL", "ALL", "US")
run("LOCO held-out India: US-only model, tuned US-A -> India-B", "US", "US", "India")
run("normal India: ALL model, tuned US-A -> India-B", "ALL", "US", "India")
run("LOCO held-out US: India-only model, tuned India-A -> US-B", "India", "India", "US")
run("normal US: ALL model, tuned India-A -> US-B", "ALL", "India", "US")
full = tune(*get("ALL", "ALL", "full"))
res["ALL model: tuned on same data (full val -> full val)"] = R(full)

# robust t_empty
tes = np.round(np.arange(0.1, 0.96, 0.05), 2)
tUS = tune(*get("US", "US", "A"))["t"]; tIN = tune(*get("India", "India", "A"))["t"]
rob = []
for te in tes:
    hi = macro_f05(*get("US", "India", "A"), tUS, max(float(te), tUS))["macro_f05"]
    hu = macro_f05(*get("India", "US", "A"), tIN, max(float(te), tIN))["macro_f05"]
    rob.append((float(te), round(hi, 5), round(hu, 5), round(min(hi, hu), 5)))
te_rob = max(rob, key=lambda x: x[3])[0]
tALL = bA["t"]
inB_best = macro_f05(*get("ALL", "ALL", "B"), tALL, bA["t_empty"])["macro_f05"]
inB_rob = macro_f05(*get("ALL", "ALL", "B"), tALL, max(te_rob, tALL))["macro_f05"]
heldB = {"India_heldout_at_robust": macro_f05(*get("US", "India", "B"), tUS, max(te_rob, tUS))["macro_f05"],
         "US_heldout_at_robust": macro_f05(*get("India", "US", "B"), tIN, max(te_rob, tIN))["macro_f05"]}
res["robust_t_empty"] = {"grid_half_A_(t_empty, heldout_India, heldout_US, min)": rob, "t_US_model": tUS, "t_India_model": tIN,
                         "robust_t_empty": te_rob, "ALL_model_t": tALL, "ALL_best_t_empty_halfA": bA["t_empty"],
                         "ALL_halfB_f05_at_best_t_empty": round(inB_best, 5), "ALL_halfB_f05_at_robust_t_empty": round(inB_rob, 5),
                         "in_country_cost": round(inB_best - inB_rob, 5), "use_robust_for_test": bool(inB_best - inB_rob <= 0.003),
                         **{k: round(v, 5) for k, v in heldB.items()}}
print(res["robust_t_empty"], flush=True)
out = {"features": FEATS, "relative_features": RELATIVE, "model_sizes": sizes, "results": res, "gain": gain,
       "relative_feature_gain_share": {n: round(sum(g[f] for f in RELATIVE) / sum(g.values()), 4) for n, g in gain.items()},
       "half_split": "val S1 by hash(entity_id, seed=7) % 2: A=0, B=1",
       "half_sizes": dict(s1.group_by("half").len().rows()), "total_s": round(time.time() - T0, 1),
       "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
(OUT / "metrics_v2.json").write_text(json.dumps(out, indent=2))
print("done", out["total_s"], out["peak_rss_gb"], out["relative_feature_gain_share"])
