"""Run blocking on a normalized S1/S2/S3 set, evaluate recall@k, write reports and candidate tables.

    python code/business_entity_resolution/src/run_blocking.py --norm-prefix /workspace/amlc/work/norm_val \
        --gt /workspace/amlc/dataset/dataset/train/train_ground_truth.tsv \
        --report-dir reports/blocking_baseline --cand-dir /workspace/amlc/work [--weights w.json]
    --tune-union out.parquet  : instead keep the full union for a 20% S1 sample (weight tuning)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from business_entity_resolution.src.blocking import BlockConfig, run_blocking  # noqa: E402
from business_entity_resolution.src.evaluate_blocking import recall_at_k, truth_pairs  # noqa: E402


class Mem:
    """Per-step wall time and peak RSS (VmHWM, reset via /proc/self/clear_refs)."""

    def __init__(self):
        self.steps = {}

    @staticmethod
    def _status(key):
        for line in open("/proc/self/status"):
            if line.startswith(key):
                return int(line.split()[1]) / 1e6  # GB
        return None

    def start(self):
        try:
            open("/proc/self/clear_refs", "w").write("5")
            self.reset_ok = True
        except OSError:
            self.reset_ok = False
        self.t = time.time()

    def stop(self, name):
        self.steps[name] = {"seconds": round(time.time() - self.t, 1),
                            "peak_rss_gb": round(self._status("VmHWM:"), 3),
                            "rss_end_gb": round(self._status("VmRSS:"), 3),
                            "peak_is_per_step": self.reset_ok}
        print(f"[step] {name}: {self.steps[name]}", flush=True)


def load_gt(path):
    if str(path).endswith(".parquet"):
        return pl.read_parquet(path)
    from business_entity_resolution.src.loader import load_ground_truth
    return load_ground_truth(Path(path).parent.parent)


def write_candidate_tsv(cand: pl.DataFrame, k: int, n1, n2, n3, path):
    ids = {2: n2.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid")),
           3: n3.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid"))}
    base = n1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id")).sort("source1_entity_id")
    c = cand.select("s1_idx", "src", "cand_idx", "rank").filter(pl.col("rank") <= k)
    rows = 0
    step = 50_000  # bounded memory: write S1 in chunks (output sorted by source1_entity_id)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for b0 in range(0, base.height, step):
            b = base.slice(b0, step)
            cc = c.join(b.select("s1_idx"), on="s1_idx", how="semi")
            parts = [cc.filter(pl.col("src") == s).join(ids[s], on="cand_idx", how="left") for s in (2, 3)]
            cc = pl.concat(parts).sort(["s1_idx", "src", "rank"])
            agg = cc.group_by("s1_idx", maintain_order=True).agg(pl.col("cid").str.join(",").alias("candidate_entity_ids"))
            out = (b.join(agg, on="s1_idx", how="left").with_columns(pl.col("candidate_entity_ids").fill_null(""))
                   .select("source1_entity_id", "candidate_entity_ids"))
            fh.write(out.write_csv(separator="\t", quote_style="never", include_header=False))
            rows += out.height
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-prefix", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--report-dir")
    ap.add_argument("--cand-dir")
    ap.add_argument("--weights")
    ap.add_argument("--ks", default="5,10,15,20,30")
    ap.add_argument("--best-k", type=int, default=None)
    ap.add_argument("--export-ks", default="")
    ap.add_argument("--tune-union")
    ap.add_argument("--m-per-method", type=int, default=30)
    ap.add_argument("--reverse-top", type=int, default=0)
    ap.add_argument("--export-thresholds", default="", help="subset of --thresholds to export as TSV")
    ap.add_argument("--thresholds", default="", help="k:t,k:t pairs for top-k AND score>=t variants")
    a = ap.parse_args(argv)
    ks = [int(x) for x in a.ks.split(",")]
    cfg = BlockConfig(m_per_method=a.m_per_method, k_max=max(ks), reverse_top=a.reverse_top)
    thresholds = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in a.thresholds.split(",") if x]
    if a.weights:
        cfg.weights = json.loads(Path(a.weights).read_text())["weights"]
    mem = Mem()
    t_all = time.time()

    mem.start()
    n1, n2, n3 = (pl.read_parquet(f"{a.norm_prefix}_s{n}.parquet") for n in (1, 2, 3))
    gt = load_gt(a.gt)
    truth = truth_pairs(gt, n1, n2, n3)
    mem.stop("load_normalized_and_truth")

    if a.tune_union:
        sub = n1.filter(pl.col("entity_id").hash(11) % 5 == 0)["idx"].to_numpy()
        mem.start()
        cand, info, tim = run_blocking(n1, n2, n3, cfg, keep_all=True, s1_subset=sub)
        mem.stop("blocking_union")
        t = truth.select("s1_idx", "src", "cand_idx", pl.lit(True).alias("y"))
        cand = cand.join(t, on=["s1_idx", "src", "cand_idx"], how="left").with_columns(pl.col("y").fill_null(False))
        cand.write_parquet(a.tune_union)
        tt = truth.filter(pl.col("s1_idx").is_in(sub))
        print(json.dumps({"union_pairs": cand.height, "true_pairs_in_subset": tt.height,
                          "true_in_union": int(cand["y"].sum()), "info": info, "timings": tim}, indent=1, default=str))
        return

    mem.start()
    cand, info, tim = run_blocking(n1, n2, n3, cfg)
    mem.stop("blocking_retrieve_score_topk")

    mem.start()
    cd = Path(a.cand_dir)
    cd.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(cd / f"candidates_val_k{max(ks)}_full.parquet")
    feat_cols = [c for c in cand.columns if c not in ("s1_idx", "src", "cand_idx", "rank", "score")]
    mem.stop("write_full_candidates")

    mem.start()
    metrics = recall_at_k(cand.drop(feat_cols), truth, n1, ks, thresholds=thresholds)
    mem.stop("evaluate")

    # union ceiling (before top-k cut is not stored; report recall at k_max = union-cap ceiling)
    best_k = a.best_k
    if best_k is None:
        plateau = metrics[str(max(ks))]["pair_recall"]
        best_k = min(k for k in ks if metrics[str(k)]["pair_recall"] >= plateau - 0.005)
    mem.start()
    cand.filter(pl.col("rank") <= best_k).write_parquet(cd / f"candidates_val_k{best_k}.parquet")
    exported = {}
    for k in sorted({best_k} | {int(x) for x in a.export_ks.split(",") if x}):
        p = cd / f"candidate_pairs_val_k{k}.tsv"
        exported[k] = {"path": str(p), "rows": write_candidate_tsv(cand, k, n1, n2, n3, p)}
    for k, thr in thresholds:
        if f"{k}:{thr}" in a.export_thresholds.split(","):
            p = cd / f"candidate_pairs_val_k{k}_t{thr}.tsv"
            exported[f"k{k}_t{thr}"] = {"path": str(p), "rows": write_candidate_tsv(
                cand.filter(pl.col("score") >= thr), k, n1, n2, n3, p)}
    mem.stop("write_outputs")

    rep = {"config": asdict(cfg), "norm_prefix": a.norm_prefix, "gt": a.gt,
           "n_s1": n1.height, "n_s2": n2.height, "n_s3": n3.height,
           "k_selection_rule": "smallest k with pair recall within 0.5 pt of recall at k_max",
           "recommended_k_by_rule": best_k, "recall_at_k": metrics, "per_country_blocking": info,
           "inner_timings_s": {k: round(v, 1) for k, v in tim.items()}, "steps": mem.steps,
           "total_seconds": round(time.time() - t_all, 1), "exported_candidate_files": exported,
           "candidate_parquet": str(cd / f"candidates_val_k{best_k}.parquet")}
    rd = Path(a.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: {kk: v[kk] for kk in ("pair_recall", "entity_all_matches_kept", "entity_at_least_one_kept",
                                                "candidates_per_s1")} for k, v in metrics.items()}, indent=1))


if __name__ == "__main__":
    main()
