"""Normalize the TEST sources (cached as norm_test_s*.parquet) and learn per-country stopwords from them,
using the same function the blocker uses (blocking.learn_country_stopwords, frac=BlockConfig.stop_df_frac)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import polars as pl
from business_entity_resolution.src.blocking import BlockConfig, learn_country_stopwords
from business_entity_resolution.src.loader import load_source
from business_entity_resolution.src.prepare import normalize_frame

D, W = "/workspace/amlc/dataset/dataset", Path("/workspace/amlc/work")
for n in (1, 2, 3):
    p = W / f"norm_test_s{n}.parquet"
    if not p.exists():
        t = time.time()
        normalize_frame(load_source(D, "test", n)).write_parquet(p)
        print(f"test s{n} normalized in {time.time() - t:.1f}s", flush=True)
frames = [pl.read_parquet(W / f"norm_test_s{n}.parquet", columns=["country", "name_n"]) for n in (1, 2, 3)]
out = {}
cfg = BlockConfig()
for c in sorted(frames[0]["country"].unique().to_list()):
    names = [x for f in frames for x in f.filter(pl.col("country") == c)["name_n"].to_list()]
    sw = sorted(learn_country_stopwords(names, cfg.stop_df_frac))
    out[c] = {"source": f"test_source1+2+3 names with country == {c!r} ({len(names)} docs)", "n": len(sw), "stopwords": sw}
    assert len(names) > 0
extra = set(pl.concat([frames[1], frames[2]])["country"].unique().to_list()) - set(out)
out["_target_countries_without_s1"] = sorted(extra)
Path("/workspace/amlc/gt_checks/reports/loco_check/test_stopwords.json").write_text(json.dumps(out, indent=1))
for c, v in out.items():
    if not c.startswith("_"):
        print(c, v["source"], v["n"], v["stopwords"])
print("S2/S3 countries absent from test S1:", extra)
