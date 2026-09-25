"""Normalization and blocking invariants on tiny inputs."""
import polars as pl

from business_entity_resolution.src import normalize as nz
from business_entity_resolution.src.blocking import BlockConfig, run_blocking
from business_entity_resolution.src.prepare import normalize_frame


def test_normalize_basic():
    assert nz.normalize_text("TSK FORTRESS (INC)") == "tsk fortress incorporated"
    assert nz.normalize_text("Crispy C0nstructions Ltd.") == "crispy constructions limited"
    assert nz.normalize_text("0046 OVERBROOK LN") == "46 overbrook lane"
    assert nz.normalize_text("(india) Teg Índustries") == "india teg industries"
    assert nz.normalize_text("www.xaipurep.com") == "xaipurep"
    assert nz.normalize_text("") == ""


def test_all_indic_scripts_transliterated_to_ascii():
    samples = ["राम मार्केटिंग", "গোল্ড প্রডিউসার", "ਸਕਾਈ ਈਸਟ", "પ્રાઇમ બાલાજી", "ଶ୍ୟାମ ଫୁଡ୍ସ୍",
               "பாலாஜி பிசினஸ்", "సన్ బెస్ట్", "ಆಲ್ ಟೆಕ್", "ലക്ഷ്മി ഫിനാൻസ്"]
    scripts = [nz.script_of(s) for s in samples]
    assert scripts == nz.SCRIPTS[1:]
    for s in samples:
        out = nz.normalize_text(s)
        assert out and out.isascii(), (s, out)
    assert nz.normalize_text("राम मार्केटिंग") == "ram marketing"
    assert nz.normalize_text("பாலாஜி") == "palaji"


def test_skeleton_cross_script():
    assert nz.skeleton(nz.normalize_text("प्राइवेट")) == nz.skeleton("private")


def _frame(rows, prefix):
    df = pl.DataFrame(rows, schema=["business_name", "business_address", "country"], orient="row")
    df = df.with_columns(pl.Series("entity_id", [f"{prefix}-{i}" for i in range(df.height)]))
    return normalize_frame(df.select("entity_id", "business_name", "business_address", "country"), n_proc=1)


def test_blocking_same_country_only_and_topk():
    s1 = _frame([("Acme Widgets", "1 Main St Austin", "US"), ("Acme Widgets", "1 Main St Austin", "France")], "S1")
    s2 = _frame([("Acme Widgets Inc", "1 Main Street Austin", "US"), ("Acme Widgets SARL", "1 Main St Austin", "France"),
                 ("ACME WIDGET", "1 Main St Austin", "India")] + [(f"Acme Widgets {i}", "1 Main St", "US") for i in range(8)], "S2")
    s3 = _frame([("acme widgets", "Main St Austin", "US")], "S3")
    # tiny data: disable the frequency caps (every token is in >1% of 3 names)
    cfg = BlockConfig(k_max=3, m_per_method=10, threshold=0.0, n_threads=1, stop_df_frac=1.0,
                      char_df_cap=1.0, addr_df_cap=1.0, word_df_cap=1.0)
    cand, info, _ = run_blocking(s1, s2, s3, cfg)
    c1 = dict(zip(s1["idx"].to_list(), s1["country"].to_list()))
    c2 = dict(zip(s2["idx"].to_list(), s2["country"].to_list()))
    c3 = dict(zip(s3["idx"].to_list(), s3["country"].to_list()))
    for s, src, j in cand.select("s1_idx", "src", "cand_idx").iter_rows():
        assert src in (2, 3)
        assert c1[s] == (c2 if src == 2 else c3)[j]
    assert cand.group_by("s1_idx", "src").len()["len"].max() <= 3
    assert set(info) == {"US", "France"}  # open set: countries taken from the data
    fr = cand.filter((pl.col("s1_idx") == 1) & (pl.col("src") == 2))
    assert fr["cand_idx"].to_list() == [1]


def test_reverse_retrieval_and_addr_missing_feature():
    s1 = _frame([("Acme Widgets", "", "US"), ("Zeta Foods", "9 Elm", "US")], "S1")
    s2 = _frame([("Acme Widgets", "1 Main", "US"), ("Zeta Food", "9 Elm", "US")], "S2")
    s3 = _frame([("Acme", "", "US")], "S3")
    cfg = BlockConfig(k_max=5, m_per_method=5, threshold=0.0, n_threads=1, stop_df_frac=1.0,
                      char_df_cap=1.0, addr_df_cap=1.0, word_df_cap=1.0, reverse_top=1)
    cand, _, _ = run_blocking(s1, s2, s3, cfg)
    row = cand.filter((pl.col("s1_idx") == 0) & (pl.col("src") == 2) & (pl.col("cand_idx") == 0))
    assert row.height == 1 and row["addr_missing"][0] == 1.0 and row["cos_addr"][0] == 0.0
    assert cand.filter((pl.col("s1_idx") == 1) & (pl.col("cand_idx") == 1) & (pl.col("src") == 2))["addr_missing"][0] == 0.0
    assert (cand["methods"] & 16).max() == 16  # reverse bit present


def test_recall_at_k_counts():
    from business_entity_resolution.src.evaluate_blocking import recall_at_k
    n1 = pl.DataFrame({"idx": [0, 1, 2], "country": ["US", "US", "India"]}, schema={"idx": pl.Int32, "country": pl.Utf8})
    cand = pl.DataFrame({"s1_idx": [0, 0, 0, 1], "src": [2, 2, 3, 2], "cand_idx": [5, 6, 7, 8],
                         "score": [3.0, 2.0, 1.0, 1.0]}, schema={"s1_idx": pl.Int32, "src": pl.Int8, "cand_idx": pl.Int32, "score": pl.Float32})
    truth = pl.DataFrame({"s1_idx": [0, 0, 2], "src": [2, 3, 2], "cand_idx": [6, 7, 9], "script": ["Latin"] * 3},
                         schema={"s1_idx": pl.Int32, "src": pl.Int8, "cand_idx": pl.Int32, "script": pl.Utf8})
    r = recall_at_k(cand, truth, n1, ks=(1, 2))
    assert r["1"]["pairs_kept"] == 1 and r["2"]["pairs_kept"] == 2  # S1-0/S2 rank 2 cut at k=1; S3 rank 1 kept
    assert r["2"]["entity_all_matches_kept"] == 0.5 and r["2"]["entity_at_least_one_kept"] == 0.5
    assert r["1"]["candidates_total"] == 3 and r["2"]["candidates_total"] == 4
    assert r["2"]["candidates_per_s1"]["max"] == 3


def test_stopwords_learned_per_country_only():
    import pytest
    from business_entity_resolution.src.blocking import build_country
    s1 = _frame([("Acme SARL", "1 rue", "France")], "S1")
    s2 = _frame([("Acme LLC", "1 Main", "US")], "S2")
    with pytest.raises(AssertionError, match="per country"):
        build_country(s1, {"s2": s2, "s3": s2}, BlockConfig())
    s2f = _frame([("Acme SAS", "1 rue", "France")], "S2")
    cfg = BlockConfig(stop_df_frac=1.0, char_df_cap=1.0, addr_df_cap=1.0, word_df_cap=1.0, threshold=0.0, n_threads=1)
    _, info, _ = run_blocking(s1, s2f, s2f.clear(), cfg, data_tag="test")
    assert "France" in info and info["France"]["stopwords_source"].startswith("test: country='France'")


def test_decode_macro_f05():
    from business_entity_resolution.src.decode import macro_f05, prepare
    sc = pl.DataFrame({"s1_idx": [0, 0, 1, 2], "src": [2, 2, 2, 2], "cand_idx": [10, 11, 10, 12],
                       "p": [0.9, 0.2, 0.6, 0.3], "y": [True, False, False, False]})
    tc = pl.DataFrame({"s1_idx": [0], "n_true": [2]}, schema={"s1_idx": pl.Int64, "n_true": pl.UInt32})
    s1 = pl.DataFrame({"s1_idx": [0, 1, 2]})
    kept, base = prepare(sc, tc, s1)
    assert kept.filter(pl.col("s1_idx") == 1).height == 0  # record 10 goes to S1 0 only
    r = macro_f05(kept, base, 0.5, 0.5)
    # S1 0: P=1, R=0.5 -> F=1.25*0.5/(0.25+0.5)=0.8333; S1 1 singleton empty ->1; S1 2 singleton, p 0.3<0.5 ->1
    assert abs(r["macro_f05"] - (0.8333333 + 1 + 1) / 3) < 1e-6
    r = macro_f05(kept, base, 0.25, 0.25)  # S1 2 now predicts record 12 -> 0
    assert abs(r["macro_f05"] - (0.8333333 + 1 + 0) / 3) < 1e-6
