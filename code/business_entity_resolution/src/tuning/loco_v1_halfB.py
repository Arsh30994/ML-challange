"""Evaluate the v1 (len_diff) models with the v2 honest protocol (tune half A, report half B) for a like-for-like comparison."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb, numpy as np, polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare, tune
from business_entity_resolution.src.pair_features import FEATURES
W = Path("/workspace/amlc/work")
va = pl.read_parquet(W / "feat_val_k10.parquet", columns=FEATURES + ["y", "country", "s1_idx", "src", "cand_idx"])
tc = pl.read_parquet(W / "feat_val_k10_truecount.parquet")
s1 = pl.read_parquet(W / "feat_val_k10_s1.parquet").with_columns(
    pl.when(pl.col("s1_id").hash(7) % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("half"))
va = va.join(s1.select("s1_idx", "half"), on="s1_idx", how="left")
X = va.select(FEATURES).to_numpy()
P = {}
for n in ("ALL", "US", "India"):
    m = lgb.Booster(model_file=str(W / f"loco_model_{n}.txt")); P[n] = m.predict(X, num_threads=6).astype(np.float32)
del X
def get(model, country, half):
    sc = va.select("s1_idx", "src", "cand_idx", "y", "country", "half").with_columns(pl.Series("p", P[model]))
    s = s1
    if country != "ALL":
        sc = sc.filter(pl.col("country") == country); s = s.filter(pl.col("country") == country)
    sc = sc.filter(pl.col("half") == half); s = s.filter(pl.col("half") == half)
    return prepare(sc, tc, s)
res = {}
for label, model, tu, ev in () if False else (("ALL all-A -> all-B", "ALL", "ALL", "ALL"), ("ALL all-A -> India-B", "ALL", "ALL", "India"),
                             ("LOCO India: US-only, US-A -> India-B", "US", "US", "India"), ("normal India: ALL, US-A -> India-B", "ALL", "US", "India"),
                             ("LOCO US: India-only, India-A -> US-B", "India", "India", "US"), ("normal US: ALL, India-A -> US-B", "ALL", "India", "US")):
    b = tune(*get(model, tu, "A")); r = macro_f05(*get(model, ev, "B"), b["t"], b["t_empty"])
    res[label] = {"t": b["t"], "t_empty": b["t_empty"], "macro_f05": round(r["macro_f05"], 5), "singleton_acc": round(r["singleton_acc"], 5)}
    print(label, res[label], flush=True)
Path("/workspace/amlc/gt_checks/reports/loco_check/metrics_v1_halfB.json").write_text(json.dumps(res, indent=1))

# robust t_empty for the v1 models (same procedure as loco_check2.py)
tes = np.round(np.arange(0.1, 0.96, 0.05), 2)
tUS = tune(*get("US", "US", "A"))["t"]; tIN = tune(*get("India", "India", "A"))["t"]
gA_I, gA_U = get("US", "India", "A"), get("India", "US", "A")
rob = []
for te in tes:
    hi = macro_f05(*gA_I, tUS, max(float(te), tUS))["macro_f05"]; hu = macro_f05(*gA_U, tIN, max(float(te), tIN))["macro_f05"]
    rob.append((float(te), round(hi, 5), round(hu, 5), round(min(hi, hu), 5)))
te_rob = max(rob, key=lambda x: x[3])[0]
bA = tune(*get("ALL", "ALL", "A")); gB = get("ALL", "ALL", "B")
best_B = macro_f05(*gB, bA["t"], bA["t_empty"])["macro_f05"]; rob_B = macro_f05(*gB, bA["t"], max(te_rob, bA["t"]))["macro_f05"]
out = {"grid_half_A_(t_empty, heldout_India, heldout_US, min)": rob, "robust_t_empty": te_rob,
       "effective_robust_t_empty": max(te_rob, bA["t"]), "ALL_t": bA["t"], "ALL_best_t_empty": bA["t_empty"],
       "ALL_halfB_at_best": round(best_B, 5), "ALL_halfB_at_robust": round(rob_B, 5), "in_country_cost": round(best_B - rob_B, 5),
       "use_robust_for_test": bool(best_B - rob_B <= 0.003),
       "heldout_India_halfB_at_robust": round(macro_f05(*get("US", "India", "B"), tUS, max(te_rob, tUS))["macro_f05"], 5),
       "heldout_US_halfB_at_robust": round(macro_f05(*get("India", "US", "B"), tIN, max(te_rob, tIN))["macro_f05"], 5)}
res["robust_t_empty_v1"] = out
print({k: v for k, v in out.items() if not k.startswith("grid")}); print(rob[11:])
Path("/workspace/amlc/gt_checks/reports/loco_check/metrics_v1_halfB.json").write_text(json.dumps(res, indent=1))
