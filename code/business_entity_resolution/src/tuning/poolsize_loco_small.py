"""Held-out-country check at SMALL pool size (the France-like case: unseen country, ~260k S1 pool).
US-only v2 large-pool model scored on small-pool val (floor 7.966), India = held-out, US = in-country.
Also re-reports the large-pool LOCO numbers from models_f7.966.json for the same thresholds."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb, numpy as np, polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
from business_entity_resolution.src.pair_features import FEATURES_V2, build_features
W = Path("/workspace/amlc/work"); REPO = Path(__file__).resolve().parents[4]; MD = REPO / "code/business_entity_resolution/models"
TS = np.round(np.arange(0.40, 0.981, 0.02), 2)
n1, n2, n3 = (pl.read_parquet(W / f"norm_val_s{n}.parquet") for n in (1, 2, 3))
tr = truth_pairs(load_ground_truth("/workspace/amlc/dataset/dataset"), n1, n2, n3).select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y"))
tc = tr.group_by("s1_idx").agg(pl.len().alias("n_true"))
f = build_features(pl.read_parquet(W / "candidates_val_k10.parquet").filter(pl.col("score") >= 7.966), n1, n2, n3)
f = f.join(tr, on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
s1 = n1.select(pl.col("idx").alias("s1_idx"), "country", pl.when(pl.col("entity_id").hash(7) % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("half"))
res = {}
for mname in ("v2_model_USonly_f7.966", "v2_model_ALL_f7.966"):
    p = lgb.Booster(model_file=str(MD / f"{mname}.txt")).predict(f.select(FEATURES_V2).to_numpy(), num_threads=6).astype(np.float32)
    x = f.select("s1_idx", "src", "cand_idx", "y").with_columns(pl.Series("p", p)).join(s1.select("s1_idx", "country"), on="s1_idx")
    r = {}
    for c in ("India", "US"):
        sc = s1.filter(pl.col("country") == c)
        k, _ = prepare(x.filter(pl.col("country") == c), tc, sc)
        b = {h: sc.filter(pl.col("half") == h).select("s1_idx").join(tc, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0)) for h in "AB"}
        kA = k.join(b["A"].select("s1_idx"), on="s1_idx", how="semi"); kB = k.join(b["B"].select("s1_idx"), on="s1_idx", how="semi")
        g = {(float(t), float(te)): macro_f05(kA, b["A"], float(t), float(te))["macro_f05"] for t in TS for te in TS if te >= t}
        (t, te), fa = max(g.items(), key=lambda z: z[1])
        rc = {"tuned_A": {"t": t, "t_empty": te, "A": round(fa, 5), "B": round(macro_f05(kB, b["B"], t, te)["macro_f05"], 5)}}
        for tt, tte in ((0.465, 0.553), (0.66, 0.80), (0.84, 0.84), (0.90, 0.92)):
            m = macro_f05(kB, b["B"], tt, tte)
            rc[f"B_at_{tt}/{tte}"] = {q: round(m[q], 5) for q in ("macro_f05", "singleton_acc", "pair_precision", "pair_recall")}
        r[c] = rc
    res[mname] = r
    print(mname, json.dumps(r), flush=True)
res["note"] = "US-only model: India rows are held-out-country; 0.465/0.553 = log-N fit value for France (259k)"
(REPO / "reports/largepool_val/poolsize_loco_small.json").write_text(json.dumps(res, indent=1))
