"""Deterministic grouped train/val split by Source-1 entity.

* Each S1 goes to one side together with all its matched S2/S3 records.
* S1 stratified by (country value, singleton flag); in each stratum (sorted key order) the sorted
  IDs are permuted with numpy default_rng(seed) and round(val_frac * n) go to val.
* S2/S3 records unreferenced by the GT are split the same way, stratified by (source, country).
* Invariants are checked (``check_split``) and violations raise.

CLI:
    python code/business_entity_resolution/src/split.py --data-dir <dataset/dataset> \\
        --out-dir /workspace/amlc/work/split --report reports/split/split_summary.json
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from business_entity_resolution.src.loader import (  # noqa: E402
    explode_ground_truth, load_ground_truth, load_source,
)

SEED = 42
VAL_FRAC = 0.2


def _stratified_assign(df: pl.DataFrame, id_col: str, strata: list[str], val_frac: float, seed: int) -> pl.DataFrame:
    """Return (id_col, side) with round(val_frac*n) val per stratum, deterministic."""
    rng = np.random.default_rng(seed)
    df = df.sort(strata + [id_col])
    out_ids, out_side = [], []
    for key, grp in df.group_by(strata, maintain_order=True):
        ids = grp[id_col].to_numpy()
        n = len(ids)
        n_val = int(round(val_frac * n))
        perm = rng.permutation(n)
        side = np.full(n, "train", dtype=object)
        side[perm[:n_val]] = "val"
        out_ids.append(ids)
        out_side.append(side)
    if not out_ids:
        return pl.DataFrame({id_col: [], "side": []}, schema={id_col: pl.Utf8, "side": pl.Utf8})
    return pl.DataFrame({id_col: np.concatenate(out_ids).astype(str), "side": np.concatenate(out_side).astype(str)})


def make_split(s1: pl.DataFrame, gt: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
               val_frac: float = VAL_FRAC, seed: int = SEED) -> dict[str, pl.DataFrame]:
    """s1/s2/s3: (entity_id, country); gt: (source1_entity_id, matched_entity_ids).

    Returns {"s1": (entity_id, country, is_singleton, side), "s2"/"s3": (entity_id, country,
    source1_entity_id|null, referenced, side)}."""
    pairs = explode_ground_truth(gt)
    multi = pairs.group_by("matched_id").len().filter(pl.col("len") > 1)
    if multi.height:
        raise ValueError(f"{multi.height} S2/S3 IDs referenced by several S1; grouped split undefined")
    missing_s1 = gt.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti")
    if missing_s1.height:
        raise ValueError(f"{missing_s1.height} GT S1 IDs not in source1")
    s1i = (s1.select("entity_id", "country")
           .join(gt.select(pl.col("source1_entity_id").alias("entity_id"),
                           (pl.col("matched_entity_ids") == "").alias("is_singleton")),
                 on="entity_id", how="left")
           .with_columns(pl.col("is_singleton").fill_null(True)))
    s1_side = _stratified_assign(s1i, "entity_id", ["country", "is_singleton"], val_frac, seed)
    s1i = s1i.join(s1_side, on="entity_id", how="left")
    side_of_s1 = pairs.join(s1i.select(pl.col("entity_id").alias("source1_entity_id"), "side"),
                            on="source1_entity_id", how="left")
    out = {"s1": s1i}
    for name, src, pref in (("s2", s2, "S2"), ("s3", s3, "S3")):
        srci = src.select("entity_id", "country").join(
            side_of_s1.filter(pl.col("source") == pref)
            .select(pl.col("matched_id").alias("entity_id"), "source1_entity_id", "side"),
            on="entity_id", how="left")
        missing = side_of_s1.filter(pl.col("source") == pref).join(
            srci, left_on="matched_id", right_on="entity_id", how="anti")
        if missing.height:
            raise ValueError(f"{missing.height} GT {pref} IDs not in source{pref[-1]}")
        unref = srci.filter(pl.col("side").is_null())
        # distinct seed stream per source so S2/S3 draws are independent but reproducible
        unref_side = _stratified_assign(unref.select("entity_id", "country"), "entity_id", ["country"],
                                        val_frac, seed + int(pref[-1]))
        srci = (srci.join(unref_side.rename({"side": "side_u"}), on="entity_id", how="left")
                .with_columns(pl.col("source1_entity_id").is_not_null().alias("referenced"),
                              pl.coalesce("side", "side_u").alias("side"))
                .drop("side_u"))
        out[name] = srci
    check_split(out)
    return out


def check_split(out: dict[str, pl.DataFrame]) -> dict:
    """Leakage checks; raises AssertionError on violation, returns the check results."""
    s1 = out["s1"]
    res = {}
    all_ids = pl.concat([out[k].select("entity_id", "side") for k in ("s1", "s2", "s3")])
    res["ids_without_side"] = int(all_ids["side"].is_null().sum())
    res["ids_on_both_sides"] = int(all_ids.group_by("entity_id").agg(pl.col("side").n_unique())
                                   .filter(pl.col("side") > 1).height)
    res["duplicate_ids"] = int(all_ids.height - all_ids["entity_id"].n_unique())
    s1s = s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("side").alias("s1_side"))
    mism = 0
    for k in ("s2", "s3"):
        m = out[k].filter(pl.col("referenced")).join(s1s, on="source1_entity_id", how="left")
        mism += int((m["side"] != m["s1_side"]).fill_null(True).sum())
    res["matched_ids_on_other_side_than_their_s1"] = mism
    for k, v in res.items():
        assert v == 0, f"split invariant violated: {k}={v}"
    res["all_passed"] = True
    return res


def summarize(out: dict[str, pl.DataFrame]) -> dict:
    s = {}
    for side in ("train", "val"):
        s1 = out["s1"].filter(pl.col("side") == side)
        d = {"s1": s1.height, "s1_singletons": int(s1["is_singleton"].sum()),
             "s1_non_singletons": int((~s1["is_singleton"]).sum()),
             "s1_country_mix": dict(sorted(s1.group_by("country").len().rows()))}
        for k in ("s2", "s3"):
            t = out[k].filter(pl.col("side") == side)
            d[f"{k}_total"] = t.height
            d[f"{k}_matched"] = int(t["referenced"].sum())
            d[f"{k}_unreferenced"] = int((~t["referenced"]).sum())
            d[f"{k}_country_mix"] = dict(sorted(t.group_by("country").len().rows()))
        s[side] = d
    tot = {k: s["train"][k] + s["val"][k] for k in s["train"] if not isinstance(s["train"][k], dict)}
    s["val_fraction"] = {k: round(s["val"][k] / tot[k], 5) if tot[k] else None for k in tot}
    s["val_singleton_rate"] = round(s["val"]["s1_singletons"] / s["val"]["s1"], 5)
    s["train_singleton_rate"] = round(s["train"]["s1_singletons"] / s["train"]["s1"], 5)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    a = ap.parse_args(argv)
    t0 = time.time()
    s1 = load_source(a.data_dir, "train", 1, usecols=["country"])
    s2 = load_source(a.data_dir, "train", 2, usecols=["country"])
    s3 = load_source(a.data_dir, "train", 3, usecols=["country"])
    gt = load_ground_truth(a.data_dir)
    t_load = time.time() - t0
    out = make_split(s1, gt, s2, s3, a.val_frac, a.seed)
    checks = check_split(out)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for k in ("s1", "s2", "s3"):
        out[k].select("entity_id", "side").write_parquet(od / f"{k}_split.parquet")
        out[k].select("entity_id", "side").write_csv(od / f"{k}_split.tsv", separator="\t")
    summ = {"config": {"seed": a.seed, "val_frac": a.val_frac, "data_dir": str(a.data_dir),
                       "s1_strata": ["country", "is_singleton"], "unreferenced_strata": ["source", "country"],
                       "files": sorted(p.name for p in od.glob("*_split.*"))},
            "leakage_checks": checks, **summarize(out),
            "runtime_s": {"load": round(t_load, 1), "total": round(time.time() - t0, 1)},
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    rp = Path(a.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
