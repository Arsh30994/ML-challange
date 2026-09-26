"""Pool-size check: small-pool val (val-only universe, floor 7.966; India 176,637 / US 264,726 S1) vs large-pool val
(full train universe; India 883,188 / US 1,323,633 S1) for the v1 and v2-ALL matchers.
Decoding (one S1 per record) runs over each pool's whole population. Halves: hash(entity_id, 7) % 2, A tunes, B reports."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb, numpy as np, polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
from business_entity_resolution.src.pair_features import FEATURES, FEATURES_V2, build_features

W = Path("/workspace/amlc/work"); REPO = Path(__file__).resolve().parents[4]; FLOOR = 7.966
MD = REPO / "code/business_entity_resolution/models"
MODELS = {"v1": (MD / "v1_model_ALL.txt", FEATURES, ""), "v2ALL": (MD / "v2_model_ALL_f7.966.txt", FEATURES_V2, "_v2ALL")}
TS = np.round(np.arange(0.40, 0.981, 0.02), 2)
FIXED = {"v1": [(0.70, 0.70), (0.84, 0.84), (0.90, 0.92)], "v2ALL": [(0.66, 0.80), (0.84, 0.84), (0.90, 0.92)]}
gt = load_ground_truth("/workspace/amlc/dataset/dataset")
val_ids = pl.Series("entity_id", (W / "split/val_s1_ids.txt").read_text().split()).to_frame()
half = lambda e: pl.when(e.hash(7) % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B"))  # noqa: E731


def tagc(df, c):
    return df.with_columns((pl.lit(c) + pl.col("s1_idx").cast(pl.Utf8)).alias("s1_idx"))


# ---- pools: dict pool -> dict country -> (kept per model, bases per half)
pools = {"small": {}, "large": {}}
n1, n2, n3 = (pl.read_parquet(W / f"norm_val_s{n}.parquet") for n in (1, 2, 3))
tr = truth_pairs(gt, n1, n2, n3).select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y"))
tc = tr.group_by("s1_idx").agg(pl.len().alias("n_true"))
f = build_features(pl.read_parquet(W / "candidates_val_k10.parquet").filter(pl.col("score") >= FLOOR), n1, n2, n3)
f = f.join(tr, on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
P = {m: lgb.Booster(model_file=str(p)).predict(f.select(F).to_numpy(), num_threads=6).astype(np.float32)
     for m, (p, F, _) in MODELS.items()}
s1 = n1.select(pl.col("idx").alias("s1_idx"), "country", half(pl.col("entity_id")).alias("half"))
f = f.select("s1_idx", "src", "cand_idx", "y").join(s1.select("s1_idx", "country"), on="s1_idx")
sizes = {"small": {}, "large": {}}
for c in ("India", "US"):
    fc = f.filter(pl.col("country") == c)
    sc = s1.filter(pl.col("country") == c)
    sizes["small"][c] = sc.height
    bases = {h: sc.filter(pl.col("half") == h).select("s1_idx").join(tc, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0)) for h in "AB"}
    kept = {}
    for m in MODELS:
        x = fc.with_columns(pl.Series("p", P[m][(f["country"] == c).to_numpy()]))
        kept[m], _ = prepare(x, tc, sc)
    pools["small"][c] = (kept, bases)
del f, P
for c in ("India", "US"):
    a1, a2, a3 = (pl.scan_parquet(f"{W}/norm_trainall_s{n}.parquet").filter(pl.col("country") == c)
                  .select("idx", "entity_id", "script").collect() for n in (1, 2, 3))
    sizes["large"][c] = a1.height
    trl = truth_pairs(gt, a1, a2, a3).select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y"))
    tcl = trl.group_by("s1_idx").agg(pl.len().alias("n_true"))
    v = a1.join(val_ids, on="entity_id", how="semi").select(pl.col("idx").alias("s1_idx"), half(pl.col("entity_id")).alias("half"))
    bases = {h: v.filter(pl.col("half") == h).select("s1_idx").join(tcl, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0)) for h in "AB"}
    kept = {}
    for m, (_, _, suf) in MODELS.items():
        sc = pl.read_parquet(W / "largepool" / f"scored_{c}_f{FLOOR:g}{suf}.parquet").join(
            trl, on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
        k, _ = prepare(sc, tcl, v)
        kept[m] = k.join(v.select("s1_idx"), on="s1_idx", how="semi")
    pools["large"][c] = (kept, bases)


def ev(pool, m, cs, h, t, te):
    k = pl.concat([tagc(pools[pool][c][0][m], c) for c in cs]); b = pl.concat([tagc(pools[pool][c][1][h], c) for c in cs])
    r = macro_f05(k.join(b.select("s1_idx"), on="s1_idx", how="semi"), b, t, te)
    kk = k.join(b.select("s1_idx"), on="s1_idx", how="semi").filter((pl.col("p") >= t) & (pl.col("s1_best") >= te))
    r["empty_rate"] = 1 - kk["s1_idx"].n_unique() / b.height
    return {x: round(r[x], 5) for x in ("macro_f05", "singleton_acc", "pair_precision", "pair_recall", "empty_rate")}


def grid(pool, m, cs):
    k = pl.concat([tagc(pools[pool][c][0][m], c) for c in cs]); b = pl.concat([tagc(pools[pool][c][1]["A"], c) for c in cs])
    k = k.join(b.select("s1_idx"), on="s1_idx", how="semi")
    return {(float(t), float(te)): macro_f05(k, b, float(t), float(te))["macro_f05"] for t in TS for te in TS if te >= t}


def evall(pool, m, t, te):
    return {"pooled": ev(pool, m, ("India", "US"), "B", t, te), "India": ev(pool, m, ("India",), "B", t, te),
            "US": ev(pool, m, ("US",), "B", t, te)}


res = {"pool_sizes_s1": sizes, "floor": FLOOR, "grid_t": [0.40, 0.98, 0.02], "models": {}}
for m in MODELS:
    r = {"fixed_halfB": {}}
    for pool in ("small", "large"):
        r["fixed_halfB"][pool] = {f"{t}/{te}": evall(pool, m, t, te) for t, te in FIXED[m]}
    G = {pool: grid(pool, m, ("India", "US")) for pool in ("small", "large")}
    Gc = {(pool, c): grid(pool, m, (c,)) for pool in ("small", "large") for c in ("India", "US")}
    for pool in ("small", "large"):
        (t, te), fa = max(G[pool].items(), key=lambda x: x[1])
        r[f"tuned_{pool}_A"] = {"t": t, "t_empty": te, "A": round(fa, 5),
                                "B_small": evall("small", m, t, te), "B_large": evall("large", m, t, te)}
    (t, te), mn = max(((k, min(G["small"][k], G["large"][k])) for k in G["small"]), key=lambda x: x[1])
    gap = {pool: round(max(G[pool].values()) - G[pool][(t, te)], 5) for pool in ("small", "large")}
    r["global_minmax"] = {"t": t, "t_empty": te, "A_small": round(G["small"][(t, te)], 5), "A_large": round(G["large"][(t, te)], 5),
                          "A_gap_to_pool_best": gap, "within_0.003_both": all(v <= 0.003 for v in gap.values()),
                          "B_small": evall("small", m, t, te), "B_large": evall("large", m, t, te)}
    pts = []
    for (pool, c), g in Gc.items():
        (t, te), fa = max(g.items(), key=lambda x: x[1])
        pts.append({"pool": pool, "country": c, "n_s1": sizes[pool][c], "t": t, "t_empty": te, "A": round(fa, 5)})
    x = np.log([p["n_s1"] for p in pts])
    fit = {k: np.polyfit(x, [p[k] for p in pts], 1).tolist() for k in ("t", "t_empty")}
    pred = {name: {k: round(float(np.polyval(fit[k], np.log(n))), 3) for k in fit}
            for name, n in (("France_test_259452", 259452), ("India_test_809986", 809986), ("US_test_663106", 663106))}
    r["per_country_tuned_points"] = pts
    r["logN_linear_fit"] = {"coef_[slope,intercept]": fit, "predictions": pred,
                            "note": "least squares over the 4 per-country half-A-tuned points (2 pools x 2 countries)"}
    res["models"][m] = r
    print(m, json.dumps({k: v for k, v in r.items() if k not in ("fixed_halfB",)}, default=str)[:3000], flush=True)
out = REPO / "reports/largepool_val/poolsize_check.json"
out.write_text(json.dumps(res, indent=1, default=str))
