"""Normalize a source frame in parallel (multiprocessing) and cache it as parquet.

Output columns: idx (int32 row id), entity_id, country, name_n, addr_n, name_sk, script.
Raw strings are kept in the original source files; only the normalized fields are cached.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from multiprocessing import get_context

import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from business_entity_resolution.src.normalize import normalize_text, script_of, skeleton  # noqa: E402


def _norm_chunk(args):
    names, addrs = args
    nn = [normalize_text(x) for x in names]
    return (nn, [normalize_text(x) for x in addrs], [skeleton(x) for x in nn], [script_of(x) for x in names])


def normalize_frame(df: pl.DataFrame, n_proc: int | None = None, chunk: int = 20_000) -> pl.DataFrame:
    n_proc = n_proc or max(1, min(6, (os.cpu_count() or 2) - 1))
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    cols = ([], [], [], [])
    if n_proc == 1 or len(jobs) <= 1:
        results = map(_norm_chunk, jobs)
    else:
        pool = get_context("fork").Pool(n_proc)
        results = pool.imap(_norm_chunk, jobs)
    for r in results:
        for c, v in zip(cols, r):
            c.extend(v)
    if not (n_proc == 1 or len(jobs) <= 1):
        pool.close(); pool.join()
    return pl.DataFrame({
        "idx": pl.Series(range(df.height), dtype=pl.Int32),
        "entity_id": df["entity_id"], "country": df["country"],
        "name_n": cols[0], "addr_n": cols[1], "name_sk": cols[2], "script": cols[3],
    })


def main(argv=None):
    """Normalize one side of the grouped split: writes <out-prefix>_s{1,2,3}.parquet."""
    import argparse
    import resource
    import time
    from business_entity_resolution.src.loader import load_source
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--side", default="val")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)
    for n in (1, 2, 3):
        t = time.time()
        df = load_source(a.data_dir, "train", n)
        sp = pl.read_parquet(f"{a.split_dir}/s{n}_split.parquet").filter(pl.col("side") == a.side)
        df = df.join(sp.select("entity_id"), on="entity_id", how="semi")
        t1 = time.time()
        normalize_frame(df).write_parquet(f"{a.out_prefix}_s{n}.parquet")
        print(f"s{n} rows={df.height} load_s={t1 - t:.1f} normalize_s={time.time() - t1:.1f} "
              f"peak_rss_gb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}", flush=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
