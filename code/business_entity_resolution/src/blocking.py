"""S1-centred blocking: same-country-value only, union of four retrieval methods, cheap score, top-k.

For every country value present in S1 (open set, matched by value) and every source s in (S2, S3):
  A  name word tokens, TF-IDF (idf learned on that country's names); learned stopwords removed
  B  name char 3-grams (char_wb) TF-IDF on the stopword-free name; trigrams with doc-freq above
     ``char_df_cap`` (fraction) are dropped for retrieval only
  B2 the same on the consonant skeleton (cross-script phonetic key for transliterated names)
  C  address word + number tokens TF-IDF; tokens above ``addr_df_cap`` dropped for retrieval only
Each method returns the top ``m_per_method`` targets per S1 (sparse_dot_topn). The union is scored
with the four full cosines, an address-missing flag (either side's address empty) and a linear
blocking score (weights fitted by logistic regression on a train-side subset, never on val); the top ``k_max`` per (S1, source) are kept.
Only S1-S2 and S1-S3 pairs are produced.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import scipy.sparse as sp
from numba import njit, prange
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from business_entity_resolution.src.normalize import skeleton

FEATURES = ["cos_word", "cos_char", "cos_sk", "cos_addr", "addr_missing"]
# score = sum_f w_f * f over FEATURES and miss_cos_{word,char,sk} = addr_missing * cos_*
METHOD_BITS = {"word": 1, "char": 2, "sk": 4, "addr": 8, "reverse": 16}


@dataclass
class BlockConfig:
    m_per_method: int = 30
    k_max: int = 30
    stop_df_frac: float = 0.01      # name tokens in >1% of a country's names = learned stopwords
    char_df_cap: float = 0.01       # retrieval-only cap on trigram doc frequency
    addr_df_cap: float = 0.01
    word_df_cap: float = 0.05
    threshold: float = 0.05
    chunk_rows: int = 20_000
    n_threads: int = 6
    weights: dict = field(default_factory=lambda: {"cos_word": 1.0, "cos_char": 1.0, "cos_sk": 0.5, "cos_addr": 1.0})
    methods: tuple = ("word", "char", "sk", "addr")
    reverse_top: int = 0            # >0: also add each target's top-R S1 per method (record-centred)


@njit(parallel=True, cache=True)
def _rowdot(a_ptr, a_idx, a_val, b_ptr, b_idx, b_val, ii, jj):
    out = np.zeros(ii.shape[0], dtype=np.float32)
    for p in prange(ii.shape[0]):
        i, j = ii[p], jj[p]
        x, xe = a_ptr[i], a_ptr[i + 1]
        y, ye = b_ptr[j], b_ptr[j + 1]
        s = 0.0
        while x < xe and y < ye:
            if a_idx[x] == b_idx[y]:
                s += a_val[x] * b_val[y]
                x += 1
                y += 1
            elif a_idx[x] < b_idx[y]:
                x += 1
            else:
                y += 1
        out[p] = s
    return out


def rowdot(A: sp.csr_matrix, B: sp.csr_matrix, ii, jj) -> np.ndarray:
    return _rowdot(A.indptr, A.indices, A.data, B.indptr, B.indices, B.data,
                   ii.astype(np.int64), jj.astype(np.int64))


def _sorted_csr(X):
    X = X.tocsr().astype(np.float32)
    X.sort_indices()
    return X


def _drop_cols(X: sp.csr_matrix, keep: np.ndarray) -> sp.csr_matrix:
    D = sp.diags(keep.astype(np.float32))
    return _sorted_csr(X @ D)


def learn_country_stopwords(names: list[str], frac: float) -> set[str]:
    from collections import Counter
    df = Counter()
    for t in names:
        df.update(set(t.split()))
    cap = frac * max(1, len(names))
    return {w for w, c in df.items() if c > cap}


def _core(names, stop):
    out = []
    for t in names:
        c = " ".join(w for w in t.split() if w not in stop)
        out.append(c if c else t)
    return out


def _fit(texts_by_part, **kw):
    v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, lowercase=False, **kw)
    allt = [t for part in texts_by_part for t in part]
    X = v.fit_transform(allt)
    X = _sorted_csr(X)
    df = np.bincount(X.indices, minlength=X.shape[1])
    parts, o = [], 0
    for part in texts_by_part:
        parts.append(X[o:o + len(part)])
        o += len(part)
    return parts, df, len(allt)


def build_country(n1: pl.DataFrame, targets: dict[str, pl.DataFrame], cfg: BlockConfig):
    """Vectorize one country. Returns dict: method -> {"full": [X1, X2, X3], "ret": [...]}.

    Stopwords and IDF are learned ONLY from the rows passed in (this country's S1/S2/S3 of the data
    being processed, e.g. the test sources at inference). There is no global or cross-country list to
    fall back to; the assertion below enforces a single country value in the inputs."""
    vals = set(n1["country"].unique().to_list())
    for t in targets.values():
        vals |= set(t["country"].unique().to_list())
    assert len(vals) == 1, f"stopwords/IDF must be learned per country; got country values {sorted(vals)}"
    assert n1.height > 0, "no S1 rows for this country: stopwords cannot be learned from its own data"
    names = [n1["name_n"].to_list()] + [t["name_n"].to_list() for t in targets.values()]
    stop = learn_country_stopwords([x for p in names for x in p], cfg.stop_df_frac)
    core = [_core(p, stop) for p in names]
    addrs = [n1["addr_n"].to_list()] + [t["addr_n"].to_list() for t in targets.values()]
    mats = {}
    specs = {
        "word": (core, dict(analyzer="word", token_pattern=r"\S+"), cfg.word_df_cap),
        "char": (core, dict(analyzer="char_wb", ngram_range=(3, 3)), cfg.char_df_cap),
        "sk": ([[skeleton(x) for x in p] for p in core], dict(analyzer="char_wb", ngram_range=(3, 3)), cfg.char_df_cap),
        "addr": (addrs, dict(analyzer="word", token_pattern=r"\S+"), cfg.addr_df_cap),
    }
    for m, (texts, kw, cap) in specs.items():
        parts, df, n = _fit(texts, **kw)
        keep = (df <= cap * n)
        ret = [_drop_cols(P, keep) for P in parts]
        mats[m] = {"full": parts, "ret": ret, "n_features": int(len(df)), "n_dropped": int((~keep).sum())}
    return mats, sorted(stop)


def block_country(n1: pl.DataFrame, targets: dict[str, pl.DataFrame], cfg: BlockConfig, keep_all: bool = False,
                  timings: dict | None = None):
    """Candidates for one country. n1/targets rows must already be filtered to that country.

    Returns a polars frame (s1_idx, src, cand_idx, cos_*, methods, score) with ``idx`` columns
    referring to the frames' ``idx``. keep_all=True returns the whole union (for weight tuning)."""
    t0 = time.time()
    mats, stop = build_country(n1, targets, cfg)
    if timings is not None:
        timings["vectorize_s"] = timings.get("vectorize_s", 0) + time.time() - t0
    s1_ids = n1["idx"].to_numpy()
    e1 = (n1["addr_n"] == "").to_numpy()
    out = []
    t1 = time.time()
    for si, (src, tdf) in enumerate(targets.items(), start=1):
        t_ids = tdf["idx"].to_numpy()
        e2 = (tdf["addr_n"] == "").to_numpy()
        if tdf.height == 0 or n1.height == 0:
            continue
        BT = {m: mats[m]["ret"][si].T.tocsr() for m in cfg.methods}
        rev_keys = rev_bits = None
        if cfg.reverse_top > 0:
            rk, rb = [], []
            for m in cfg.methods:
                QT = mats[m]["ret"][0].T.tocsr()
                T = mats[m]["ret"][si]
                for r0 in range(0, T.shape[0], 4 * cfg.chunk_rows):
                    R = sp_matmul_topn(T[r0:r0 + 4 * cfg.chunk_rows], QT, top_n=cfg.reverse_top,
                                       threshold=cfg.threshold, n_threads=cfg.n_threads).tocoo()
                    rk.append(R.col.astype(np.int64) * tdf.height + (R.row.astype(np.int64) + r0))
                    rb.append(np.full(R.nnz, 16, dtype=np.int8))
                del QT
            rev_keys = np.concatenate(rk)
            o = np.argsort(rev_keys, kind="stable")
            rev_keys, rev_bits = rev_keys[o], np.concatenate(rb)[o]
        for c0 in range(0, n1.height, cfg.chunk_rows):
            c1 = min(n1.height, c0 + cfg.chunk_rows)
            keys, bits = [], []
            for m in cfg.methods:
                Q = mats[m]["ret"][0][c0:c1]
                R = sp_matmul_topn(Q, BT[m], top_n=cfg.m_per_method, threshold=cfg.threshold,
                                   n_threads=cfg.n_threads).tocoo()
                keys.append((R.row.astype(np.int64) + c0) * tdf.height + R.col.astype(np.int64))
                bits.append(np.full(R.nnz, METHOD_BITS[m], dtype=np.int8))
            if rev_keys is not None:
                lo, hi = np.searchsorted(rev_keys, [c0 * tdf.height, c1 * tdf.height])
                keys.append(rev_keys[lo:hi])
                bits.append(rev_bits[lo:hi])
            k = np.concatenate(keys)
            b = np.concatenate(bits)
            if k.size == 0:
                continue
            order = np.argsort(k, kind="stable")
            k, b = k[order], b[order]
            uk, start = np.unique(k, return_index=True)
            ub = np.bitwise_or.reduceat(b, start)
            ii = (uk // tdf.height).astype(np.int64)
            jj = (uk % tdf.height).astype(np.int64)
            feats = {f"cos_{m}": rowdot(mats[m]["full"][0], mats[m]["full"][si], ii, jj) for m in ("word", "char", "sk", "addr")}
            miss = (e1[ii] | e2[jj]).astype(np.float32)
            feats["addr_missing"] = miss
            extra = {f"miss_cos_{m}": miss * feats[f"cos_{m}"] for m in ("word", "char", "sk")}
            score = np.zeros(len(ii), dtype=np.float32)
            for f, w in cfg.weights.items():
                score += np.float32(w) * (feats[f] if f in feats else extra[f])
            df = pl.DataFrame({"s1_idx": s1_ids[ii].astype(np.int32), "src": np.full(len(ii), si + 1, dtype=np.int8),
                               "cand_idx": t_ids[jj].astype(np.int32), **feats, "methods": ub, "score": score})
            if not keep_all:
                df = (df.with_columns(pl.col("score").rank("ordinal", descending=True).over("s1_idx").alias("rank"))
                      .filter(pl.col("rank") <= cfg.k_max).with_columns(pl.col("rank").cast(pl.Int16)))
            out.append(df)
    if timings is not None:
        timings["retrieve_score_s"] = timings.get("retrieve_score_s", 0) + time.time() - t1
    return (pl.concat(out) if out else None), stop, {m: (mats[m]["n_features"], mats[m]["n_dropped"]) for m in mats}


def run_blocking(n1: pl.DataFrame, n2: pl.DataFrame, n3: pl.DataFrame, cfg: BlockConfig, keep_all=False,
                 s1_subset: np.ndarray | None = None, log=print, data_tag: str = "unspecified"):
    """Loop over country values of S1 (open set). Targets with a country value absent from S1
    can never be candidates (all true pairs share the country value)."""
    res, info, timings = [], {}, {}
    for country in sorted(n1["country"].unique().to_list()):
        a = n1.filter(pl.col("country") == country)
        t = {"s2": n2.filter(pl.col("country") == country), "s3": n3.filter(pl.col("country") == country)}
        t0 = time.time()
        cand, stop, feats = block_country(a, t, cfg, keep_all=keep_all, timings=timings)
        if cand is not None:
            if s1_subset is not None:
                cand = cand.filter(pl.col("s1_idx").is_in(s1_subset))
            res.append(cand)
        info[country] = {"s1": a.height, "s2": t["s2"].height, "s3": t["s3"].height,
                         "n_stopwords": len(stop), "stopwords_sample": stop[:40],
                         "stopwords_source": f"{data_tag}: country={country!r} S1+S2+S3 names of this run only "
                                             f"({a.height + t['s2'].height + t['s3'].height} docs)",
                         "features_total_dropped": feats, "seconds": round(time.time() - t0, 1),
                         "pairs": 0 if cand is None else cand.height}
        log(f"[blocking] {country}: {info[country]['seconds']}s pairs={info[country]['pairs']}")
    return (pl.concat(res) if res else None), info, timings
