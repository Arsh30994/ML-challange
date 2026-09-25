"""Pair features for the matcher (language-agnostic; no country feature).

Input: candidate table from blocking (s1_idx, src, cand_idx, cos_*, addr_missing, methods, score, rank)
plus the normalized frames. Adds:
  * S1-side group features: gap to the S1's best score per source, candidates per (S1, source)
  * record-side features: rank of this S1 among the record's candidate S1s, gap to the record's best
    score, number of S1 within 1.0 score of the record's best
  * rapidfuzz string features on normalized names/addresses (token-set ratio, Jaro-Winkler, partial
    ratio), address number-token Jaccard, name length difference, transliterated-name flag, source flag
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

BASE = ["cos_word", "cos_char", "cos_sk", "cos_addr", "addr_missing", "score", "rank", "methods"]
FEATURES = BASE + ["s1_gap", "s1_n", "rec_rank", "rec_gap", "rec_n_close", "name_tsr", "name_jw", "name_pr",
                   "addr_tsr", "num_jacc", "len_diff", "translit", "is_s3"]
# v2: absolute name-length difference replaced by relative |la-lb|/max(la,lb) (LOCO: len_diff did not transfer)
FEATURES_V2 = [f if f != "len_diff" else "len_rel" for f in FEATURES]


def _num_jacc(a: pl.Series, b: pl.Series) -> np.ndarray:
    df = pl.DataFrame({"a": a.str.extract_all(r"\d+"), "b": b.str.extract_all(r"\d+")})
    inter = df.select(pl.col("a").list.set_intersection(pl.col("b")).list.len()).to_series().to_numpy()
    union = df.select(pl.col("a").list.set_union(pl.col("b")).list.len()).to_series().to_numpy()
    return np.where(union > 0, inter / np.maximum(union, 1), -1.0).astype(np.float32)


def build_features(cand: pl.DataFrame, n1: pl.DataFrame, n2: pl.DataFrame, n3: pl.DataFrame,
                   chunk: int = 2_000_000) -> pl.DataFrame:
    c = cand.with_columns(
        (pl.col("score").max().over("s1_idx", "src") - pl.col("score")).alias("s1_gap"),
        pl.len().over("s1_idx", "src").cast(pl.Float32).alias("s1_n"),
        pl.col("score").rank("ordinal", descending=True).over("src", "cand_idx").cast(pl.Float32).alias("rec_rank"),
        pl.col("score").max().over("src", "cand_idx").alias("_rec_best"),
        (pl.col("src") == 3).cast(pl.Float32).alias("is_s3"),
    ).with_columns((pl.col("_rec_best") - pl.col("score")).alias("rec_gap"))
    c = c.with_columns((pl.col("rec_gap") <= 1.0).cast(pl.Float32).sum().over("src", "cand_idx").alias("rec_n_close")).drop("_rec_best")
    a = n1.select(pl.col("idx").alias("s1_idx"), pl.col("name_n").alias("n_a"), pl.col("addr_n").alias("a_a"))
    t = pl.concat([n.select(pl.lit(s, dtype=pl.Int8).alias("src"), pl.col("idx").alias("cand_idx"),
                            pl.col("name_n").alias("n_b"), pl.col("addr_n").alias("a_b"),
                            (pl.col("script") != "Latin").cast(pl.Float32).alias("translit"))
                   for s, n in ((2, n2), (3, n3))])
    out = []
    for c0 in range(0, c.height, chunk):
        x = c.slice(c0, chunk).join(a, on="s1_idx", how="left").join(t, on=["src", "cand_idx"], how="left")
        na, nb, aa, ab = x["n_a"].to_list(), x["n_b"].to_list(), x["a_a"].to_list(), x["a_b"].to_list()
        x = x.with_columns(
            pl.Series("name_tsr", cpdist(na, nb, scorer=fuzz.token_set_ratio, workers=6), dtype=pl.Float32),
            pl.Series("name_jw", cpdist(na, nb, scorer=JaroWinkler.similarity, workers=6), dtype=pl.Float32),
            pl.Series("name_pr", cpdist(na, nb, scorer=fuzz.partial_ratio, workers=6), dtype=pl.Float32),
            pl.Series("addr_tsr", cpdist(aa, ab, scorer=fuzz.token_set_ratio, workers=6), dtype=pl.Float32),
            pl.Series("num_jacc", _num_jacc(x["a_a"], x["a_b"])),
            (pl.col("n_a").str.len_chars().cast(pl.Int32) - pl.col("n_b").str.len_chars().cast(pl.Int32)).abs()
            .cast(pl.Float32).alias("len_diff"),
            ((pl.col("n_a").str.len_chars().cast(pl.Int32) - pl.col("n_b").str.len_chars().cast(pl.Int32)).abs()
             / pl.max_horizontal(pl.col("n_a").str.len_chars(), pl.col("n_b").str.len_chars(), pl.lit(1)))
            .cast(pl.Float32).alias("len_rel"),
        ).drop("n_a", "n_b", "a_a", "a_b")
        out.append(x)
    return pl.concat(out).with_columns(pl.col(FEATURES + ["len_rel"]).cast(pl.Float32))
