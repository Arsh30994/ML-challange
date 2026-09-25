"""Blocking on the train-side tuning subset with the final config (k=10, reverse top-5, weights.json)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import polars as pl
from business_entity_resolution.src.blocking import BlockConfig, run_blocking

W = "/workspace/amlc/gt_checks/reports/blocking_baseline/weights.json"
cfg = BlockConfig(m_per_method=30, k_max=10, reverse_top=5, weights=json.loads(Path(W).read_text())["weights"])
n1, n2, n3 = (pl.read_parquet(f"/workspace/amlc/work/norm_tune_s{n}.parquet") for n in (1, 2, 3))
t = time.time()
cand, info, tim = run_blocking(n1, n2, n3, cfg)
cand.write_parquet("/workspace/amlc/work/candidates_tune_k10.parquet")
print("rows", cand.height, "seconds", round(time.time() - t, 1))
