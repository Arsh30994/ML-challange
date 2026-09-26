"""Equivalence check for the large-pool changes (blocking.py: on-demand retrieval matrices, spill-to-disk,
single-country frame shortcut; pair_features.py: chunked `reduce` hook).
Reference = work/tmp/blocking_before_memfix.py (the blocker that produced submission v1's validation numbers).
Same slice as compare_blockers.py: 20k val S1 per country (hash seed 42), matched S2/S3 + 10% other val S2/S3.
Checks: (1) old vs new in-memory multi-country; (2) old vs new per-country pre-filtered frames + spill to disk;
(3) build_features default vs chunk=50k with reduce=identity; (4) predicted p: full table vs reduce=predict."""
import importlib.util, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lightgbm as lgb, numpy as np, polars as pl
from business_entity_resolution.src import blocking as new
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
from business_entity_resolution.src.pair_features import FEATURES, build_features
spec = importlib.util.spec_from_file_location("old_blocking", "/workspace/amlc/work/tmp/blocking_before_memfix.py")
old = importlib.util.module_from_spec(spec); sys.modules["old_blocking"] = old; spec.loader.exec_module(old)
W = Path("/workspace/amlc/work"); REPO = Path(__file__).resolve().parents[4]
n1, n2, n3 = (pl.read_parquet(W / f"norm_val_s{n}.parquet") for n in (1, 2, 3))
s1 = pl.concat([g.sort(pl.col("entity_id").hash(42)).head(20000) for _, g in n1.group_by("country", maintain_order=True)])
tr = truth_pairs(load_ground_truth("/workspace/amlc/dataset/dataset"), s1, n2, n3)
def tgt(n, src):
    m = tr.filter(pl.col("src") == src)["cand_idx"]
    return n.filter(pl.col("idx").is_in(m) | (pl.col("entity_id").hash(42) % 10 == 0))
t2, t3 = tgt(n2, 2), tgt(n3, 3)
w = json.loads((REPO / "reports/blocking_baseline/weights.json").read_text())["weights"]
key = ["s1_idx", "src", "cand_idx"]
def cmp(a, b):
    a, b = a.sort(key), b.sort(key)
    o = {"pairs_ref": a.height, "pairs_new": b.height, "only_in_ref": a.join(b, on=key, how="anti").height,
         "only_in_new": b.join(a, on=key, how="anti").height}
    j = a.join(b, on=key, suffix="_new"); d = {}
    for col in [c for c in a.columns if c not in key]:
        if a[col].dtype in (pl.Float32, pl.Float64):
            x = (j[col] - j[col + "_new"]).abs(); d[col] = {"n_diff": int((x > 0).sum()), "max_abs_diff": float(x.max() or 0)}
        else:
            d[col] = {"n_diff": int((j[col] != j[col + "_new"]).sum())}
    o["column_diffs"] = d
    o["exact_match"] = o["only_in_ref"] == 0 and o["only_in_new"] == 0 and all(v["n_diff"] == 0 for v in d.values())
    return o
out = {"slice": {"s1": s1.height, "s2": t2.height, "s3": t3.height}, "reference": "work/tmp/blocking_before_memfix.py"}
ref, _, _ = old.run_blocking(s1, t2, t3, old.BlockConfig(m_per_method=30, k_max=10, reverse_top=5, weights=w))
cfg = new.BlockConfig(m_per_method=30, k_max=10, reverse_top=5, weights=w)
mem, _, _ = new.run_blocking(s1, t2, t3, cfg)
out["1_inmemory_multicountry"] = cmp(ref, mem)
parts = []
with tempfile.TemporaryDirectory() as td:
    for c in sorted(s1["country"].unique().to_list()):
        f = lambda d: d.filter(pl.col("country") == c)  # noqa: E731
        n, info, _ = new.run_blocking(f(s1), f(t2), f(t3), cfg, spill_dir=td)
        parts.append(pl.read_parquet(f"{td}/{c}/*.parquet"))
out["2_percountry_prefiltered_spill"] = cmp(ref, pl.concat(parts))
full = build_features(mem, s1, t2, t3)
chunked = build_features(mem, s1, t2, t3, chunk=50_000, reduce=lambda x: x)
out["3_features_default_vs_chunk50k_reduce"] = cmp(full, chunked.select(full.columns))
m = lgb.Booster(model_file=str(REPO / "code/business_entity_resolution/models/v1_model_ALL.txt"))
pf = full.select(key).with_columns(pl.Series("p", m.predict(full.select(FEATURES).to_numpy(), num_threads=6).astype(np.float32)))
pr = build_features(mem, s1, t2, t3, chunk=50_000, reduce=lambda x: x.select(key).with_columns(
    pl.Series("p", m.predict(x.select(FEATURES).to_numpy(), num_threads=6).astype(np.float32))))
out["4_predict_full_vs_reduce"] = cmp(pf, pr)
out["all_exact"] = all(out[k]["exact_match"] for k in out if k[0].isdigit())
print(json.dumps(out, indent=1))
(REPO / "reports/model_baseline/blocker_equivalence_v2.json").write_text(json.dumps(out, indent=1))
