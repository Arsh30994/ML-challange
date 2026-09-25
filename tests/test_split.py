"""Split / loader invariants on tiny fixtures."""
import shutil
from pathlib import Path

import polars as pl
import pytest

from business_entity_resolution.src import loader, split

FIX = Path(__file__).parent / "fixtures" / "split"


def _synthetic(n_s1=400, seed=0):
    import random
    r = random.Random(seed)
    s1, gt, s2, s3 = [], [], [], []
    k2 = k3 = 0
    for i in range(n_s1):
        c = ["US", "India", "France"][i % 3]
        s1.append((f"S1-{i}", c))
        m = []
        if i % 7:  # non-singleton
            for _ in range(r.randint(1, 3)):
                s2.append((f"S2-{k2}", c)); m.append(f"S2-{k2}"); k2 += 1
            for _ in range(r.randint(0, 2)):
                s3.append((f"S3-{k3}", c)); m.append(f"S3-{k3}"); k3 += 1
        gt.append((f"S1-{i}", ",".join(m)))
    for j in range(300):  # unreferenced
        c = ["US", "India", "France"][j % 3]
        s2.append((f"S2-{k2}", c)); k2 += 1
        s3.append((f"S3-{k3}", c)); k3 += 1
    mk = lambda rows, cols: pl.DataFrame(rows, schema=cols, orient="row")
    return (mk(s1, ["entity_id", "country"]), mk(gt, ["source1_entity_id", "matched_entity_ids"]),
            mk(s2, ["entity_id", "country"]), mk(s3, ["entity_id", "country"]))


def test_loader_reads_fixture_and_keeps_literals():
    s1 = loader.load_source(FIX, "train", 1)
    assert s1.columns == loader.SOURCE_COLUMNS and s1.height == 4
    assert s1.filter(pl.col("entity_id") == "S1-4")["business_name"][0] == '"Quote" Inc'
    s3 = loader.load_source(FIX, "train", 3)
    assert s3.filter(pl.col("entity_id") == "S3-21")["business_name"][0] == "NA"
    gt = loader.load_ground_truth(FIX)
    assert gt.filter(pl.col("source1_entity_id") == "S1-3")["matched_entity_ids"][0] == ""


def _copy(tmp_path):
    d = tmp_path / "d"
    shutil.copytree(FIX, d)
    return d


def test_loader_rejects_bad_header(tmp_path):
    d = _copy(tmp_path)
    p = d / "train" / "train_source2.tsv"
    p.write_text(p.read_text().replace("business_name", "name", 1))
    with pytest.raises(loader.SchemaError, match="columns"):
        loader.load_source(d, "train", 2)


def test_loader_rejects_bad_prefix_and_duplicates(tmp_path):
    d = _copy(tmp_path)
    p = d / "train" / "train_source2.tsv"
    p.write_text(p.read_text().replace("S2-12", "S3-12"))
    with pytest.raises(loader.SchemaError, match="prefix"):
        loader.load_source(d, "train", 2)
    p.write_text(p.read_text().replace("S3-12", "S2-10"))
    with pytest.raises(loader.SchemaError, match="duplicate"):
        loader.load_source(d, "train", 2)


def test_loader_rejects_short_rows(tmp_path):
    d = _copy(tmp_path)
    p = d / "train" / "train_source3.tsv"
    p.write_text(p.read_text() + "S3-22\tonly name\n")
    with pytest.raises(loader.SchemaError, match="short"):
        loader.load_source(d, "train", 3)


def test_gt_token_validation():
    gt = pl.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1,S1-9"]})
    with pytest.raises(loader.SchemaError):
        loader.explode_ground_truth(gt)
    gt = pl.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1,S2-1"]})
    with pytest.raises(loader.SchemaError, match="duplicate"):
        loader.explode_ground_truth(gt)


def test_fixture_split_invariants():
    s1 = loader.load_source(FIX, "train", 1, usecols=["country"])
    s2 = loader.load_source(FIX, "train", 2, usecols=["country"])
    s3 = loader.load_source(FIX, "train", 3, usecols=["country"])
    out = split.make_split(s1, loader.load_ground_truth(FIX), s2, s3, val_frac=0.5)
    side = {r[0]: r[1] for k in out for r in out[k].select("entity_id", "side").iter_rows()}
    assert side["S2-10"] == side["S3-20"] == side["S1-1"]
    assert side["S2-11"] == side["S1-2"]
    assert split.check_split(out)["all_passed"]


def test_group_integrity_and_no_leakage():
    s1, gt, s2, s3 = _synthetic()
    out = split.make_split(s1, gt, s2, s3)
    s1side = dict(out["s1"].select("entity_id", "side").iter_rows())
    pairs = loader.explode_ground_truth(gt)
    allside = {r[0]: r[1] for k in ("s2", "s3") for r in out[k].select("entity_id", "side").iter_rows()}
    for s, m in pairs.select("source1_entity_id", "matched_id").iter_rows():
        assert allside[m] == s1side[s]
    ids = [r for k in out for r in out[k]["entity_id"].to_list()]
    assert len(ids) == len(set(ids))
    assert set(s1side.values()) | set(allside.values()) == {"train", "val"}


def test_deterministic_and_seed_sensitive():
    s1, gt, s2, s3 = _synthetic()
    a = split.make_split(s1, gt, s2, s3, seed=42)
    b = split.make_split(s1.sample(fraction=1.0, shuffle=True, seed=1), gt, s2.reverse(), s3, seed=42)
    c = split.make_split(s1, gt, s2, s3, seed=7)
    for k in ("s1", "s2", "s3"):
        assert a[k].select("entity_id", "side").sort("entity_id").equals(b[k].select("entity_id", "side").sort("entity_id"))
    assert not a["s1"].select("entity_id", "side").sort("entity_id").equals(c["s1"].select("entity_id", "side").sort("entity_id"))


def test_stratified_proportions():
    s1, gt, s2, s3 = _synthetic(n_s1=2100)
    out = split.make_split(s1, gt, s2, s3, val_frac=0.2)
    st = out["s1"].group_by("country", "is_singleton").agg((pl.col("side") == "val").mean().alias("f"), pl.len())
    for f, n in st.select("f", "len").iter_rows():
        assert abs(f - 0.2) <= 0.5 / n + 1e-9  # round() allocation per stratum
    for k in ("s2", "s3"):
        u = out[k].filter(~pl.col("referenced")).group_by("country").agg((pl.col("side") == "val").mean().alias("f"), pl.len())
        for f, n in u.select("f", "len").iter_rows():
            assert abs(f - 0.2) <= 0.5 / n + 1e-9


def test_multi_referenced_id_rejected():
    s1 = pl.DataFrame({"entity_id": ["S1-1", "S1-2"], "country": ["US", "US"]})
    gt = pl.DataFrame({"source1_entity_id": ["S1-1", "S1-2"], "matched_entity_ids": ["S2-1", "S2-1"]})
    s2 = pl.DataFrame({"entity_id": ["S2-1"], "country": ["US"]})
    s3 = pl.DataFrame({"entity_id": ["S3-1"], "country": ["US"]})
    with pytest.raises(ValueError, match="several S1"):
        split.make_split(s1, gt, s2, s3)


def test_check_split_detects_leak():
    s1, gt, s2, s3 = _synthetic()
    out = split.make_split(s1, gt, s2, s3)
    ref = out["s2"].filter(pl.col("referenced"))["entity_id"][0]
    out["s2"] = out["s2"].with_columns(
        pl.when(pl.col("entity_id") == ref).then(pl.when(pl.col("side") == "val").then(pl.lit("train")).otherwise(pl.lit("val")))
        .otherwise(pl.col("side")).alias("side"))
    with pytest.raises(AssertionError, match="other_side"):
        split.check_split(out)
