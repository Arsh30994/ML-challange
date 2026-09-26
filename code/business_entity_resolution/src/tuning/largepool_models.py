"""Large-pool threshold analysis for several matchers (scored files from `largepool.py score --suffix`).
Decoding over the whole country universe; only val S1 evaluated; half A tunes, half B reports.
  --pairs name=suffix ...   e.g. v1= v2all=_v2all usonly=_v2us
Outputs, per model: pooled tune (A) -> B overall/per country; per-country tuned t; for the LOCO model
(--loco usonly, trained on US only): US-A tuned -> India-B held-out; India-A tuned t (held-out-country tuned t);
robust (t, te) maximizing min(in-country US A, held-out India A) -> B."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, polars as pl
from business_entity_resolution.src.decode import macro_f05, prepare
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth

W = Path("/workspace/amlc/work"); REPO = Path(__file__).resolve().parents[4]
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", nargs="+", required=True); ap.add_argument("--loco", default="")
ap.add_argument("--floor", default="7.966"); ap.add_argument("--out", required=True)
a = ap.parse_args()
TS = np.round(np.arange(0.40, 0.981, 0.02), 2)
gt = load_ground_truth("/workspace/amlc/dataset/dataset")
val_ids = pl.Series("entity_id", (W / "split/val_s1_ids.txt").read_text().split()).to_frame()
ctx = {}
for c in ("India", "US"):
    n1, n2, n3 = (pl.scan_parquet(f"{W}/norm_trainall_s{n}.parquet").filter(pl.col("country") == c)
                  .select("idx", "entity_id", "script").collect() for n in (1, 2, 3))
    tr = truth_pairs(gt, n1, n2, n3).select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y"))
    tc = tr.group_by("s1_idx").agg(pl.len().alias("n_true"))
    v = n1.join(val_ids, on="entity_id", how="semi").select(pl.col("idx").alias("s1_idx"), pl.when(
        pl.col("entity_id").hash(7) % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("half"))
    ctx[c] = {"tr": tr, "tc": tc, "b": {h: v.filter(pl.col("half") == h).select("s1_idx").join(tc, on="s1_idx", how="left")
                              .with_columns(pl.col("n_true").fill_null(0)) for h in "AB"}, "v": v}


def kept_for(c, suffix):
    sc = pl.read_parquet(W / "largepool" / f"scored_{c}_f{a.floor}{suffix}.parquet").join(
        ctx[c]["tr"], on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
    k, _ = prepare(sc, ctx[c]["tc"], ctx[c]["v"])  # record-best over the whole universe
    return k.join(ctx[c]["v"].select("s1_idx"), on="s1_idx", how="semi")


def tag(df, c):
    return df.with_columns((pl.lit(c) + pl.col("s1_idx").cast(pl.Utf8)).alias("s1_idx"))


def f05(K, cs, h, t, te):
    k = pl.concat([tag(K[c], c) for c in cs]); b = pl.concat([tag(ctx[c]["b"][h], c) for c in cs])
    r = macro_f05(k.join(b.select("s1_idx"), on="s1_idx", how="semi"), b, t, te)
    return {x: (round(y, 5) if isinstance(y, float) else y) for x, y in r.items()}


def grid(K, cs):
    kk = pl.concat([tag(K[c], c) for c in cs]); b = pl.concat([tag(ctx[c]["b"]["A"], c) for c in cs])
    kk = kk.join(b.select("s1_idx"), on="s1_idx", how="semi")
    return {(float(t), float(te)): macro_f05(kk, b, float(t), float(te))["macro_f05"] for t in TS for te in TS if te >= t}


res = {"floor": a.floor, "grid_t": [float(TS[0]), float(TS[-1]), 0.02], "models": {}}
for pair in a.pairs:
    name, suffix = pair.split("=")
    K = {c: kept_for(c, suffix) for c in ("India", "US")}
    G = {"pooled": grid(K, ("India", "US")), "India": grid(K, ("India",)), "US": grid(K, ("US",))}
    r = {}
    for gname, g in G.items():
        (t, te), fa = max(g.items(), key=lambda x: x[1])
        r[f"tuned_on_{gname}_A"] = {"t": t, "t_empty": te, "A": round(fa, 5),
                                   "B_pooled": f05(K, ("India", "US"), "B", t, te),
                                   "B_India": f05(K, ("India",), "B", t, te), "B_US": f05(K, ("US",), "B", t, te)}
    r["at_t0.90_te0.92"] = {"B_pooled": f05(K, ("India", "US"), "B", 0.90, 0.92), "B_India": f05(K, ("India",), "B", 0.90, 0.92),
                            "B_US": f05(K, ("US",), "B", 0.90, 0.92)}
    if name == a.loco:  # trained on US only: US = in-country, India = held-out
        gi, gu = G["India"], G["US"]
        (t, te), _ = max(((k, min(gi[k], gu[k])) for k in gu), key=lambda x: x[1])
        r["robust_min_inUS_heldoutIndia"] = {"t": t, "t_empty": te, "A_US": round(gu[(t, te)], 5), "A_India": round(gi[(t, te)], 5),
                                             "B_US": f05(K, ("US",), "B", t, te), "B_India": f05(K, ("India",), "B", t, te)}
        (tu, teu), fu = max(gu.items(), key=lambda x: x[1])
        r["robust_cost_vs_US_best_on_A"] = round(fu - gu[(t, te)], 5)
    res["models"][name] = r
    print(name, json.dumps({k: (v["t"], v["t_empty"], v.get("B_pooled", {}).get("macro_f05") if isinstance(v.get("B_pooled"), dict) else None,
                                v["B_India"]["macro_f05"], v["B_US"]["macro_f05"]) for k, v in r.items() if isinstance(v, dict) and "t" in v}), flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1))
