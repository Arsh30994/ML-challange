"""Test-scale ("large-pool") validation.

Blocking + features + v1 matcher run on the ENTIRE train universe (all train S1/S2/S3, both split sides),
per country, stopwords learned on that universe (same code path as test inference). Only val S1 are scored;
train-side S1 stay in the pool so they compete for records (one-S1-per-record decoding over the whole
country population) and shape the record-side features, as on test.

  block  --country C [--exclude-s1 FILE --tag T]  -> work/largepool[/T]/cand_C/part_*.parquet (k10, no floor)
  score  --country C --floors 0,7.966 [--tag T]   -> work/largepool[/T]/scored_C_f<floor>.parquet (ids, score, p)
  eval   [--tag T]                                -> reports/largepool_val/metrics[_T].json
"""
from __future__ import annotations

import argparse, json, resource, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from business_entity_resolution.src.blocking import BlockConfig, run_blocking  # noqa: E402
from business_entity_resolution.src.decode import macro_f05, prepare  # noqa: E402
from business_entity_resolution.src.pair_features import FEATURES, FEATURES_V2, build_features  # noqa: E402

W = Path("/workspace/amlc/work")
NORM = W / "norm_trainall"
REPO = Path(__file__).resolve().parents[4]
WEIGHTS = REPO / "reports/blocking_baseline/weights.json"
MODEL = W / "loco_model_ALL.txt"
COUNTRIES = ("India", "US")


def peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)


def base_dir(tag):
    d = W / "largepool" / tag if tag else W / "largepool"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_norm(n, country, cols, exclude=None):
    lf = pl.scan_parquet(f"{NORM}_s{n}.parquet").filter(pl.col("country") == country)
    if exclude is not None:
        lf = lf.filter(~pl.col("entity_id").is_in(exclude))
    return lf.select(cols).collect()


def excl_ids(a):
    return pl.read_parquet(a.exclude_s1)["entity_id"] if a.exclude_s1 else None


def cmd_block(a):
    t0 = time.time()
    ex = excl_ids(a)
    cols = ["idx", "country", "name_n", "addr_n", "name_sk"]
    n1 = load_norm(1, a.country, cols, ex)
    n2, n3 = (load_norm(n, a.country, cols) for n in (2, 3))
    rows = {"s1": n1.height, "s2": n2.height, "s3": n3.height}
    w = json.loads(WEIGHTS.read_text())["weights"]
    cfg = BlockConfig(m_per_method=30, k_max=10, reverse_top=5, weights=w)
    d = base_dir(a.tag)
    npairs, info, tim = run_blocking(n1, n2, n3, cfg, data_tag=NORM.name + (f"/{a.tag}" if a.tag else ""),
                                     spill_dir=str(d / "cand"))
    assert list(info) == [a.country] and f"country={a.country!r}" in info[a.country]["stopwords_source"]
    log = {"country": a.country, "rows": rows, "excluded_s1": 0 if ex is None else int(len(ex)),
           "stopwords": {k: info[a.country][k] for k in ("stopwords_source", "n_stopwords", "stopwords_sample")},
           "pairs_k10": info[a.country]["pairs"], "timings": tim, "blocking_s": round(time.time() - t0, 1),
           "peak_rss_gb": peak_gb()}
    (d / f"log_block_{a.country}.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps(log, default=str), flush=True)


def cmd_score(a):
    d = base_dir(a.tag)
    ex = excl_ids(a)
    cols = ["idx", "name_n", "addr_n", "script"]
    n1 = load_norm(1, a.country, cols + ["entity_id"], ex)
    # training rows for the submission-2 matcher: 40% of train-side S1 (hash seed 11), features dumped per chunk
    val_ids = pl.Series("entity_id", Path(W / "split/val_s1_ids.txt").read_text().split())
    trs = n1.join(val_ids.to_frame(), on="entity_id", how="anti").filter(pl.col("entity_id").hash(11) % 10 < 4)["idx"]
    n1 = n1.drop("entity_id")
    n2, n3 = (load_norm(n, a.country, cols) for n in (2, 3))
    m = lgb.Booster(model_file=a.model or str(MODEL))
    feats = m.feature_name()
    assert feats in (FEATURES, FEATURES_V2), feats

    dump = []

    def red(x):
        p = m.predict(x.select(feats).to_numpy(), num_threads=6).astype(np.float32)
        if a.dump_train:
            dump.append(x.filter(pl.col("s1_idx").is_in(trs)).select(["s1_idx", "src", "cand_idx"] +
                        [f for f in FEATURES_V2 + ["len_diff"] if f not in ("s1_idx", "src", "cand_idx")]))
        return x.select("s1_idx", "src", "cand_idx", "score").with_columns(pl.Series("p", p))

    for fl in [float(v) for v in a.floors.split(",")]:
        t0 = time.time(); dump.clear()
        cand = (pl.scan_parquet(str(d / "cand" / a.country / "*.parquet")).filter(pl.col("score") >= fl)
                .select("s1_idx", "src", "cand_idx", "cos_word", "cos_char", "cos_sk", "cos_addr", "addr_missing",
                        "methods", "score", "rank").collect())
        n = cand.height
        sc = build_features(cand, n1, n2, n3, reduce=red)
        del cand
        sc.write_parquet(d / f"scored_{a.country}_f{fl:g}{a.suffix}.parquet")
        if a.dump_train:
            pl.concat(dump).write_parquet(d / f"trainfeat_{a.country}_f{fl:g}.parquet"); dump.clear()
        log = {"country": a.country, "floor": fl, "model": a.model or str(MODEL), "pairs": n, "s1": n1.height, "seconds": round(time.time() - t0, 1),
               "peak_rss_gb": peak_gb()}
        (d / f"log_score_{a.country}_f{fl:g}{a.suffix}.json").write_text(json.dumps(log, indent=1))
        print(json.dumps(log), flush=True)
        del sc


def _r(d):
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}


def cmd_eval(a):
    from business_entity_resolution.src.evaluate_blocking import truth_pairs
    from business_entity_resolution.src.loader import load_ground_truth
    d = base_dir(a.tag)
    gt = load_ground_truth("/workspace/amlc/dataset/dataset")
    val_ids = pl.Series("entity_id", Path(W / "split/val_s1_ids.txt").read_text().split())
    floors = [float(v) for v in a.floors.split(",")]
    ts = np.round(np.arange(0.40, 0.961, 0.02), 2)
    res = {"universe": "all train S1/S2/S3 (both split sides)" + (f", minus {a.exclude_s1}" if a.exclude_s1 else ""),
           "scored_on": "val S1 only; decoding (one S1 per record) over the whole country universe",
           "half_split": "val S1 by hash(entity_id, seed=7) % 2: A=0, B=1", "grid_t": [float(ts[0]), float(ts[-1]), 0.02],
           "countries": {}}
    pooled = {}
    countries = a.countries.split(",")
    res["countries_evaluated"] = countries
    for c in countries:
        ex = excl_ids(a)
        n1 = load_norm(1, c, ["idx", "entity_id", "script"], ex)
        n2, n3 = (load_norm(n, c, ["idx", "entity_id", "script"]) for n in (2, 3))
        tr = truth_pairs(gt, n1, n2, n3)
        tc = tr.group_by("s1_idx").agg(pl.len().alias("n_true"))
        # S1 names are Latin; the script group is taken from the S1's TRUE matches:
        # "Indic" = at least one true S2/S3 match in an Indic script, "Latin" = all true matches Latin, "singleton"
        rs = pl.concat([n.select(pl.lit(s, dtype=pl.Int8).alias("src"), pl.col("idx").alias("cand_idx"),
                                 (pl.col("script") != "Latin").alias("ind")) for s, n in ((2, n2), (3, n3))])
        s1ind = tr.join(rs, on=["src", "cand_idx"], how="left").group_by("s1_idx").agg(pl.col("ind").any().alias("any_ind"))
        v = n1.join(val_ids.to_frame(), on="entity_id", how="semi").select(
            pl.col("idx").alias("s1_idx"), pl.when(pl.col("entity_id").hash(7) % 2 == 0).then(pl.lit("A"))
            .otherwise(pl.lit("B")).alias("half")).join(s1ind, left_on="s1_idx", right_on="s1_idx", how="left").with_columns(
            pl.when(pl.col("any_ind").is_null()).then(pl.lit("singleton")).when(pl.col("any_ind")).then(pl.lit("Indic"))
            .otherwise(pl.lit("Latin")).alias("script")).drop("any_ind")
        rc = res["countries"][c] = {"val_s1": v.height, "universe_s1": n1.height, "floors": {}}
        for fl in floors:
            sc = pl.read_parquet(d / f"scored_{c}_f{fl:g}{a.suffix}.parquet").join(
                tr.select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y")), on=["s1_idx", "src", "cand_idx"],
                how="left").with_columns(pl.col("y").fill_null(False))
            nv = sc.join(v, on="s1_idx", how="semi")
            kept, _ = prepare(sc, tc, v)
            kept = kept.join(v.select("s1_idx"), on="s1_idx", how="semi")  # only val S1 are evaluated
            bases = {h: v.filter(pl.col("half") == h).select("s1_idx").join(tc, on="s1_idx", how="left")
                     .with_columns(pl.col("n_true").fill_null(0)) for h in ("A", "B")}
            pooled.setdefault(fl, {"kept": [], "A": [], "B": [], "v": [], "cand": 0})
            pooled[fl]["kept"].append(kept.with_columns(pl.lit(c).alias("country")))
            for h in ("A", "B"):
                pooled[fl][h].append(bases[h].with_columns(pl.lit(c).alias("country")))
            pooled[fl]["v"].append(v.with_columns(pl.lit(c).alias("country")))
            pooled[fl]["cand"] += nv.height
            rc["floors"][f"{fl:g}"] = {"cand_per_val_s1": round(nv.height / v.height, 3),
                                        "cand_per_universe_s1": round(sc.height / n1.height, 3)}
            del sc
    for fl in floors:
        kept = pl.concat(pooled[fl]["kept"]); v = pl.concat(pooled[fl]["v"])
        bA, bB = pl.concat(pooled[fl]["A"]), pl.concat(pooled[fl]["B"])
        out = {"cand_per_val_s1": round(pooled[fl]["cand"] / v.height, 3)}

        def ev(t, te, filt=None):
            r = {}
            for h, b in (("A", bA), ("B", bB)):
                bb = b if filt is None else b.join(v.filter(filt).select("s1_idx", "country"), on=["s1_idx", "country"], how="semi")
                if bb.height == 0:
                    r[h] = None; continue
                kk = kept.join(bb.select("s1_idx", "country"), on=["s1_idx", "country"], how="semi")
                x = macro_f05(kk.with_columns((pl.col("country") + pl.col("s1_idx").cast(pl.Utf8)).alias("s1_idx")),
                              bb.with_columns((pl.col("country") + pl.col("s1_idx").cast(pl.Utf8)).alias("s1_idx")), t, te)
                k2 = kk.filter((pl.col("p") >= t) & (pl.col("s1_best") >= te))
                x["empty_rate"] = 1 - k2.select(pl.col("country") + pl.col("s1_idx").cast(pl.Utf8)).n_unique() / max(bb.height, 1)
                x["mean_pred_per_s1"] = k2.height / max(bb.height, 1)
                r[h] = _r(x)
            return r

        out["locked_t0.70_te0.70"] = ev(0.70, 0.70)
        # tune on half A (pooled countries), report half B
        best = None
        grid = []
        kA = kept.join(bA.select("s1_idx", "country"), on=["s1_idx", "country"], how="semi").with_columns(
            (pl.col("country") + pl.col("s1_idx").cast(pl.Utf8)).alias("s1_idx"))
        bAk = bA.with_columns((pl.col("country") + pl.col("s1_idx").cast(pl.Utf8)).alias("s1_idx"))
        for t in ts:
            for te in ts:
                if te < t:
                    continue
                f = macro_f05(kA, bAk, float(t), float(te))["macro_f05"]
                grid.append((f, float(t), float(te)))
                if best is None or f > best[0]:
                    best = (f, float(t), float(te))
        out["tuned_halfA"] = {"t": best[1], "t_empty": best[2], "halfA_f05": round(best[0], 5)}
        out["tuned_eval"] = ev(best[1], best[2])
        # t=0.70 with t_empty tuned; t tuned with t_empty=t
        out["breakdown"] = {}
        for lab, filt in [(f"country={c}", pl.col("country") == c) for c in countries] + \
                         [(f"match_script={s}", pl.col("script") == s) for s in ("Latin", "Indic", "singleton")]:
            out["breakdown"][lab] = {"locked": ev(0.70, 0.70, filt), "tuned": ev(best[1], best[2], filt)}
        res[f"floor_{fl:g}"] = out
        print(fl, json.dumps({k: out[k] for k in ("cand_per_val_s1", "locked_t0.70_te0.70", "tuned_halfA", "tuned_eval")}), flush=True)
    rp = REPO / "reports/largepool_val"; rp.mkdir(parents=True, exist_ok=True)
    (rp / f"metrics{'_' + a.tag if a.tag else ''}{a.out_suffix}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("block", "score", "eval"):
        p = sp.add_parser(name)
        p.add_argument("--tag", default="")
        p.add_argument("--exclude-s1", default="")
        if name != "eval":
            p.add_argument("--country", required=True)
        if name != "block":
            p.add_argument("--floors", default="0,7.966")
            p.add_argument("--suffix", default="", help="scored-file suffix (e.g. _v2 for a submission-2 model)")
            p.add_argument("--model", default="")
        if name == "eval":
            p.add_argument("--countries", default="India,US")
            p.add_argument("--out-suffix", default="")
        if name == "score":
            p.add_argument("--dump-train", action="store_true")
    a = ap.parse_args()
    {"block": cmd_block, "score": cmd_score, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
