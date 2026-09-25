"""Old (work/tmp/blocking_before_memfix.py) vs new (src/blocking.py) blocker on a small val slice:
20k val S1 per country (hash seed 42), their matched S2/S3 plus 10% of the other val S2/S3 (hash seed 42).
Compares the candidate lists pair for pair, plus rank, score and all features."""
import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import polars as pl
from business_entity_resolution.src import blocking as new
from business_entity_resolution.src.evaluate_blocking import truth_pairs
from business_entity_resolution.src.loader import load_ground_truth
spec = importlib.util.spec_from_file_location("old_blocking", "/workspace/amlc/work/tmp/blocking_before_memfix.py")
old = importlib.util.module_from_spec(spec); sys.modules["old_blocking"] = old; spec.loader.exec_module(old)
W = Path("/workspace/amlc/work")
n1, n2, n3 = (pl.read_parquet(W / f"norm_val_s{n}.parquet") for n in (1, 2, 3))
s1 = pl.concat([g.sort(pl.col("entity_id").hash(42)).head(20000) for _, g in n1.group_by("country", maintain_order=True)])
tr = truth_pairs(load_ground_truth("/workspace/amlc/dataset/dataset"), s1, n2, n3)
def tgt(n, src):
    m = tr.filter(pl.col("src") == src)["cand_idx"]
    return n.filter(pl.col("idx").is_in(m) | (pl.col("entity_id").hash(42) % 10 == 0))
t2, t3 = tgt(n2, 2), tgt(n3, 3)
w = json.loads(Path("/workspace/amlc/gt_checks/reports/blocking_baseline/weights.json").read_text())["weights"]
out = {"slice": {"s1": s1.height, "s2": t2.height, "s3": t3.height}}
res = {}
for name, mod in (("old", old), ("new", new)):
    cfg = mod.BlockConfig(m_per_method=30, k_max=10, reverse_top=5, weights=w)
    c, info, _ = mod.run_blocking(s1, t2, t3, cfg)
    res[name] = c.sort(["s1_idx", "src", "cand_idx"])
a, b = res["old"], res["new"]
key = ["s1_idx", "src", "cand_idx"]
out["pairs_old"], out["pairs_new"] = a.height, b.height
out["only_in_old"] = a.join(b, on=key, how="anti").height
out["only_in_new"] = b.join(a, on=key, how="anti").height
j = a.join(b, on=key, suffix="_new")
diffs = {}
for col in [c for c in a.columns if c not in key]:
    if col in ("score",) or col.startswith("cos_") or col == "addr_missing":
        d = (j[col] - j[col + "_new"]).abs()
        diffs[col] = {"n_diff": int((d > 0).sum()), "max_abs_diff": float(d.max())}
    else:
        diffs[col] = {"n_diff": int((j[col] != j[col + "_new"]).sum())}
out["column_diffs_on_common_pairs"] = diffs
out["exact_match"] = out["only_in_old"] == 0 and out["only_in_new"] == 0 and all(v["n_diff"] == 0 for v in diffs.values())
print(json.dumps(out, indent=1))
Path("/workspace/amlc/gt_checks/reports/model_baseline/blocker_equivalence.json").write_text(json.dumps(out, indent=1))
