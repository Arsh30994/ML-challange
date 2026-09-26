"""Locked inference: blocking (k=10/source, reverse top-5, weights.json, floor 7.966) -> v1 LightGBM matcher
(all-country model) -> decode (t=0.70, t_empty=0.70, one S1 per record). One process per country keeps memory bounded.

    python src/infer.py country --norm-prefix /workspace/amlc/work/norm_test --country France --out-dir /workspace/amlc/work/test_run
    python src/infer.py combine --norm-prefix /workspace/amlc/work/norm_test --out-dir /workspace/amlc/work/test_run --output-dir output
    python src/infer.py valstats   (same pipeline on the val candidates, for comparison stats)
"""
from __future__ import annotations

import argparse, json, resource, sys, time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from business_entity_resolution.src.blocking import BlockConfig, run_blocking  # noqa: E402
from business_entity_resolution.src.pair_features import FEATURES, build_features  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LOCKED = {"k_per_source": 10, "m_per_method": 30, "reverse_top": 5, "score_floor": 7.966,
          "weights_file": str(REPO / "reports/blocking_baseline/weights.json"),
          "model_file": str(Path(__file__).resolve().parents[1] / "models/v1_model_ALL.txt"),
          "model_sha256": "cd16c01fc44693a20a09f27ebfc7cf92ad3b707af048ada38961ba2282bea20a", "t": 0.70, "t_empty": 0.70,
          "v2_model_file": str(Path(__file__).resolve().parents[1] / "models/v2_model_ALL_f7.966.txt"),
          "v2_model_sha256": "faa599161b075e2922af4ce324328d730c20cd99be1666068f2d4fbf3a5b4a58"}


def check_locked_model(path, digest=None) -> None:
    """Fail if an in-repo locked model (v1 default or v2 ALL, used by v3) no longer matches its pinned sha256."""
    p = Path(path).resolve()
    for key, name in (("model", "v1"), ("v2_model", "v2 ALL")):
        if p == Path(LOCKED[f"{key}_file"]).resolve():
            got = digest or sha256(path)
            assert got == LOCKED[f"{key}_sha256"], f"in-repo {name} model file changed"


def sha256(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)


def decode(sc: pl.DataFrame, t: float, t_empty: float) -> pl.DataFrame:
    """sc: (s1_idx, src, cand_idx, p) -> kept pairs after one-S1-per-record, p>=t, S1 best p>=t_empty."""
    sc = sc.with_columns((pl.col("p") == pl.col("p").max().over("src", "cand_idx")).alias("rb"),
                         pl.col("p").max().over("s1_idx").alias("sb"))
    sc = sc.with_columns((pl.col("rb") & (pl.col("s1_idx") == pl.col("s1_idx").filter(pl.col("rb")).min()
                                             .over("src", "cand_idx"))).alias("rb"))
    return sc.filter(pl.col("rb") & (pl.col("p") >= t) & (pl.col("sb") >= t_empty)).drop("rb", "sb")


def score_candidates(cand, n1, n2, n3, model_file=None):
    f = build_features(cand, n1, n2, n3)
    m = lgb.Booster(model_file=model_file or LOCKED["model_file"])
    feats = m.feature_name()  # v1 model: FEATURES (len_diff); v2 models: FEATURES_V2 (len_rel)
    assert set(feats) <= set(f.columns), feats
    p = np.empty(f.height, dtype=np.float32)
    step = 2_000_000
    for i in range(0, f.height, step):
        p[i:i + step] = m.predict(f.slice(i, step).select(feats).to_numpy(), num_threads=6)
    return cand.select("s1_idx", "src", "cand_idx", "score", "rank").with_columns(pl.Series("p", p))


def cmd_country(a):
    t0 = time.time(); log = {"country": a.country}
    n1, n2, n3 = (pl.scan_parquet(f"{a.norm_prefix}_s{n}.parquet").filter(pl.col("country") == a.country).collect() for n in (1, 2, 3))
    log["rows"] = {"s1": n1.height, "s2": n2.height, "s3": n3.height}
    w = json.loads(Path(LOCKED["weights_file"]).read_text())["weights"]
    cfg = BlockConfig(m_per_method=LOCKED["m_per_method"], k_max=LOCKED["k_per_source"], reverse_top=LOCKED["reverse_top"], weights=w)
    cand, info, tim = run_blocking(n1, n2, n3, cfg, data_tag=Path(a.norm_prefix).name)
    assert list(info) == [a.country], info.keys()
    src = info[a.country]["stopwords_source"]
    assert src.startswith(Path(a.norm_prefix).name) and f"country={a.country!r}" in src, src
    log["stopwords"] = {"source": src, "n": info[a.country]["n_stopwords"], "sample": info[a.country]["stopwords_sample"]}
    log["blocking_s"] = round(time.time() - t0, 1); log["peak_after_blocking_gb"] = peak_gb()
    log["pairs_k10"] = cand.height
    cand = cand.filter(pl.col("score") >= LOCKED["score_floor"])
    log["pairs_scored"] = cand.height
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    cand.write_parquet(Path(a.out_dir) / f"cand_{a.country}.parquet")  # blocking output kept for re-scoring
    t1 = time.time()
    log["model_file"] = a.model; log["model_sha256"] = sha256(a.model)
    check_locked_model(a.model, log["model_sha256"])
    sc = score_candidates(cand, n1, n2, n3, model_file=a.model)
    del cand
    log["features_predict_s"] = round(time.time() - t1, 1)
    pred = decode(sc.select("s1_idx", "src", "cand_idx", "p"), a.t, a.t_empty)
    log["t"], log["t_empty"] = a.t, a.t_empty
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sc.write_parquet(out / f"scored_{a.country}.parquet")
    pred.write_parquet(out / f"pred_{a.country}.parquet")
    log["pred_pairs"] = pred.height
    log["total_s"] = round(time.time() - t0, 1); log["peak_rss_gb"] = peak_gb()
    (out / f"log_{a.country}.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log), flush=True)


def _ids(prefix):
    return pl.concat([pl.read_parquet(f"{prefix}_s{s}.parquet", columns=["idx", "entity_id"]).select(
        pl.lit(s, dtype=pl.Int8).alias("src"), pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("mid")) for s in (2, 3)])


def _lists(pairs: pl.DataFrame, s1: pl.DataFrame, ids: pl.DataFrame, order_col: str, col: str) -> pl.DataFrame:
    x = pairs.join(ids, on=["src", "cand_idx"], how="left")
    assert x["mid"].null_count() == 0
    agg = x.sort(["s1_idx", order_col], descending=[False, True]).group_by("s1_idx", maintain_order=True).agg(
        pl.col("mid").str.join(",").alias(col))
    return (s1.join(agg, on="s1_idx", how="left").with_columns(pl.col(col).fill_null(""))
            .select("source1_entity_id", col).sort("source1_entity_id"))


def stats(scored: pl.DataFrame, pred: pl.DataFrame, s1: pl.DataFrame) -> dict:
    cc = scored.group_by("s1_idx").agg(pl.len().alias("nc"))
    pc = pred.group_by("s1_idx").agg(pl.len().alias("np"))
    d = s1.join(cc, on="s1_idx", how="left").join(pc, on="s1_idx", how="left").fill_null(0)
    out = {}
    for c in ["ALL"] + sorted(d["country"].unique().to_list()):
        x = d if c == "ALL" else d.filter(pl.col("country") == c)
        nc, npred = x["nc"].to_numpy(), x["np"].to_numpy()
        out[c] = {"s1": x.height, "cand_mean": round(float(nc.mean()), 3), "cand_p50": float(np.percentile(nc, 50)),
                  "cand_p95": float(np.percentile(nc, 95)), "cand_p99": float(np.percentile(nc, 99)), "cand_max": int(nc.max()),
                  "s1_without_candidates": int((nc == 0).sum()),
                  "prediction_rate": round(float((npred > 0).mean()), 5), "empty_rate": round(float((npred == 0).mean()), 5),
                  "mean_pred_per_s1": round(float(npred.mean()), 4),
                  "mean_pred_per_nonempty_s1": round(float(npred[npred > 0].mean()), 4) if (npred > 0).any() else None}
    return out


def cmd_combine(a):
    t0 = time.time()
    out = Path(a.out_dir)
    s1 = pl.read_parquet(f"{a.norm_prefix}_s1.parquet", columns=["idx", "entity_id", "country"]).select(
        pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id"), "country")
    countries = sorted(s1["country"].unique().to_list())
    scored = pl.concat([pl.read_parquet(out / f"scored_{c}.parquet", columns=["s1_idx", "src", "cand_idx", "p"]) for c in countries])
    pred = pl.concat([pl.read_parquet(out / f"pred_{c}.parquet") for c in countries])
    ids = _ids(a.norm_prefix)
    od = Path(a.output_dir); od.mkdir(parents=True, exist_ok=True)
    cand_tsv = _lists(scored, s1, ids, "p", "candidate_entity_ids")
    cand_tsv.write_csv(od / "candidate_pairs.tsv", separator="\t", quote_style="never")
    match_tsv = _lists(pred, s1, ids, "p", "matched_entity_ids")
    match_tsv.write_csv(od / "matching_results.tsv", separator="\t", quote_style="never")
    # consistency: every matched pair is in the scored candidate set
    miss = pred.join(scored, on=["s1_idx", "src", "cand_idx"], how="anti").height
    st = stats(scored, pred, s1)
    res = {"rows_candidate_pairs": cand_tsv.height, "rows_matching_results": match_tsv.height, "scored_pairs": scored.height,
           "pred_pairs": pred.height, "pred_pairs_not_in_candidates": miss, "stats": st,
           "combine_s": round(time.time() - t0, 1), "combine_peak_rss_gb": peak_gb(),
           "per_country_logs": {c: json.loads((out / f"log_{c}.json").read_text()) for c in countries}}
    (out / "combine.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_country_logs"}, indent=1))


def cmd_redecode(a):
    """Re-decode kept test scores (scored_<C>.parquet from --src-dir) at new thresholds into --out-dir
    (scored files are symlinked, logs copied with the new thresholds), ready for `combine`."""
    src, out = Path(a.src_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("scored_*.parquet")):
        c = f.stem.removeprefix("scored_")
        if a.countries and c not in a.countries.split(","):
            continue
        pred = decode(pl.read_parquet(f, columns=["s1_idx", "src", "cand_idx", "p"]), a.t, a.t_empty)
        pred.write_parquet(out / f"pred_{c}.parquet")
        (out / f.name).unlink(missing_ok=True); (out / f.name).symlink_to(f.resolve())
        log = json.loads((src / f"log_{c}.json").read_text())
        log.update({"t": a.t, "t_empty": a.t_empty, "pred_pairs": pred.height, "redecoded_from": str(f)})
        (out / f"log_{c}.json").write_text(json.dumps(log, indent=1))
        print(c, pred.height, flush=True)


def cmd_valstats(a):
    W = Path("/workspace/amlc/work")
    n1, n2, n3 = (pl.read_parquet(W / f"norm_val_s{n}.parquet") for n in (1, 2, 3))
    cand = pl.read_parquet(W / "candidates_val_k10.parquet").filter(pl.col("score") >= LOCKED["score_floor"])
    check_locked_model(a.model)
    sc = score_candidates(cand, n1, n2, n3, model_file=a.model)
    pred = decode(sc.select("s1_idx", "src", "cand_idx", "p"), LOCKED["t"], LOCKED["t_empty"])
    s1 = n1.select(pl.col("idx").alias("s1_idx"), "country")
    st = stats(sc, pred, s1)
    Path(a.out).write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("country"); c.add_argument("--norm-prefix", required=True); c.add_argument("--country", required=True); c.add_argument("--out-dir", required=True)
    c.add_argument("--model", default=LOCKED["model_file"], help="LightGBM matcher (default: in-repo v1 model)")
    c.add_argument("--t", type=float, default=LOCKED["t"]); c.add_argument("--t-empty", type=float, default=LOCKED["t_empty"])
    b = sp.add_parser("combine"); b.add_argument("--norm-prefix", required=True); b.add_argument("--out-dir", required=True); b.add_argument("--output-dir", required=True)
    v = sp.add_parser("valstats"); v.add_argument("--out", required=True)
    v.add_argument("--model", default=LOCKED["model_file"], help="LightGBM matcher (default: in-repo v1 model)")
    r = sp.add_parser("redecode"); r.add_argument("--src-dir", required=True); r.add_argument("--out-dir", required=True)
    r.add_argument("--t", type=float, required=True); r.add_argument("--t-empty", type=float, required=True)
    r.add_argument("--countries", default="", help="comma list; default all (per-country thresholds = one call per country)")
    a = ap.parse_args()
    {"country": cmd_country, "combine": cmd_combine, "valstats": cmd_valstats, "redecode": cmd_redecode}[a.cmd](a)


if __name__ == "__main__":
    main()
