"""Ground-truth relationship checks for the Amazon ML Challenge 2026 entity-resolution data.

Checks
------
Check 1 - multi-source references:
    For every *valid* matched S2/S3 ID in ``train_ground_truth.tsv`` count the distinct
    Source-1 entities that reference it and report the IDs referenced by more than one S1.
Check 2 - records unreferenced by ground truth:
    All ``entity_id`` values of ``train_source2`` / ``train_source3`` that are not referenced
    by any valid ground-truth match. NOTE: these are "unreferenced by ground truth" records,
    they are NOT singletons. A *singleton* is only a Source-1 row whose ``matched_entity_ids``
    is the empty string. The two concepts are kept separate in every name and output.
Validation (always reported, never silently dropped):
    matched IDs missing from train S2/S3, S1 IDs inside match lists, duplicate IDs within one
    list, empty/whitespace tokens, bad prefixes, and preservation of empty lists as singletons.
    Invalid tokens are never counted as valid references.

Memory notes
------------
All TSVs are read with pandas using ``sep="\\t", dtype=str, keep_default_na=False,
quoting=csv.QUOTE_NONE``. Only ``entity_id`` is loaded for the ID sets, in chunks. IDs of the
canonical form ``S2-<digits>`` / ``S3-<digits>`` are encoded as int64 keys (one key space per
source); any non-canonical source ID gets a unique negative key, so arbitrary IDs remain exact.
The extra columns are only read chunk by chunk when writing the unreferenced rows.

CLI
---
    python code/business_entity_resolution/profiling/ground_truth_checks.py \\
        --data-dir dataset --output-dir reports/data_quality [--profile-json path]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # executed as a script: make ``business_entity_resolution`` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from business_entity_resolution.io_config import READ_KWARGS, read_tsv as _shared_read_tsv  # noqa: E402
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
DEFAULT_CHUNKSIZE = 500_000
MISSING_KEY = np.iinfo(np.int64).min
CANONICAL_ID_RE = r"S[123]-(?:0|[1-9][0-9]{0,17})"
MATCH_SOURCES = {2: "S2", 3: "S3"}

# Issue codes for matched-ID tokens. Higher-precedence issues overwrite lower ones.
ISSUE_OK = 0
ISSUE_EMPTY = 1
ISSUE_WHITESPACE = 2
ISSUE_DUPLICATE = 3
ISSUE_SOURCE1 = 4
ISSUE_BAD_PREFIX = 5
ISSUE_NOT_FOUND = 6
ISSUE_NAMES = {
    ISSUE_EMPTY: "empty_or_whitespace_token",
    ISSUE_WHITESPACE: "surrounding_whitespace_in_token",
    ISSUE_DUPLICATE: "duplicate_id_within_list",
    ISSUE_SOURCE1: "source1_id_in_matched_list",
    ISSUE_BAD_PREFIX: "bad_prefix",
    ISSUE_NOT_FOUND: "not_in_train_source2_or_source3",
}
ROW_ISSUE_GT_S1_NOT_IN_TRAIN = "source1_entity_id_not_in_train_source1"
ROW_ISSUE_GT_S1_DUPLICATE = "duplicate_source1_entity_id_in_ground_truth"

SUMMARY_FIELDS = [
    "s2_ids_referenced_by_multiple_source1",
    "s3_ids_referenced_by_multiple_source1",
    "total_duplicate_reference_ids",
    "max_source1_entities_per_referenced_id",
    "source2_unreferenced_count",
    "source3_unreferenced_count",
    "source2_unreferenced_rate",
    "source3_unreferenced_rate",
    "source1_singleton_count",
    "source1_singleton_rate",
]

MULTI_REFERENCE_FILE = "multi_source_reference_ids.csv"
UNREFERENCED_FILE = "unreferenced_source{n}_records.tsv"
UNREFERENCED_SUMMARY_FILE = "unreferenced_source_summary.csv"
SUMMARY_JSON_FILE = "ground_truth_reference_summary.json"
INVALID_FILE = "invalid_ground_truth_references.csv"
PROFILE_COPY_FILE = "profile.json"


# --------------------------------------------------------------------------- I/O helpers
def read_tsv(path, usecols=None, chunksize=None):
    """Shared ``io_config.read_tsv``. Rows with too many fields are skipped by the parser instead
    of aborting the run; every malformed row (too few or too many fields) is counted and reported
    by ``scan_tsv_structure`` and fails ``all_checks_passed``."""
    return _shared_read_tsv(path, usecols=usecols, chunksize=chunksize, on_bad_lines="skip")


def _fill_short_rows(df: pd.DataFrame) -> pd.DataFrame:
    """pandas pads short rows with NaN (genuine empties are "" because keep_default_na=False).
    Such rows are NOT hidden: they are counted per file by ``scan_tsv_structure`` and reported
    in the summary JSON; here their missing fields are only made "" so processing can continue."""
    if df.isna().to_numpy().any():
        df = df.fillna("")
    return df


MAX_EXAMPLE_LINES = 20


def scan_tsv_structure(path, expected_fields: int, max_examples: int = MAX_EXAMPLE_LINES) -> dict:
    """Raw line scan (independent of pandas) counting rows whose tab-separated field count is not
    ``expected_fields``. Line numbers are 1-based physical file lines (the header is line 1)."""
    too_few = too_many = 0
    examples = []
    data_rows = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = fh.readline()
        header_fields = header.rstrip("\r\n").count("\t") + 1
        for lineno, line in enumerate(fh, start=2):
            data_rows += 1
            n = line.rstrip("\r\n").count("\t") + 1
            if n != expected_fields:
                if n < expected_fields:
                    too_few += 1
                else:
                    too_many += 1
                if len(examples) < max_examples:
                    examples.append(lineno)
    return {
        "expected_fields": expected_fields,
        "header_fields": header_fields,
        "raw_data_lines": data_rows,
        "malformed_row_count": too_few + too_many,
        "rows_with_too_few_fields": too_few,
        "rows_with_too_many_fields": too_many,
        "example_line_numbers": examples,
        "handling": ("too-few-field rows: missing fields read as empty strings; "
                     "too-many-field rows: skipped by the pandas parser (excluded from all counts)"),
    }


def raw_empty_match_list_ids(path) -> list:
    """Independent raw read of train_ground_truth.tsv with the plain ``csv`` module (not the
    pandas path used for the counts): Source-1 IDs of rows whose 2nd field is exactly ""."""
    ids = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        col = header.index("matched_entity_ids") if header and "matched_entity_ids" in header else 1
        id_col = header.index("source1_entity_id") if header and "source1_entity_id" in header else 0
        for row in reader:
            if len(row) > max(col, id_col) and row[col] == "":
                ids.append(row[id_col])
    return ids


def check_singleton_preservation(raw_singleton_ids, source1_singleton_count: int, s1_ids_with_tokens,
                                 max_examples: int = MAX_EXAMPLE_LINES) -> dict:
    """Assert (and report) that
    (a) the independent raw count of empty ``matched_entity_ids`` equals ``source1_singleton_count``;
    (b) no singleton S1 ID appears among the S1 IDs that have any (valid or invalid) token."""
    raw_ids = pd.Series(list(raw_singleton_ids), dtype=object)
    with_tokens = pd.Series(np.asarray(s1_ids_with_tokens, dtype=object), dtype=object)
    overlap = raw_ids[raw_ids.isin(with_tokens)] if len(raw_ids) and len(with_tokens) else raw_ids.iloc[:0]
    raw_count = int(len(raw_ids))
    counts_match = raw_count == int(source1_singleton_count)
    no_overlap = len(overlap) == 0
    return {
        "raw_empty_match_list_count": raw_count,
        "source1_singleton_count": int(source1_singleton_count),
        "raw_count_equals_singleton_count": bool(counts_match),
        "singleton_ids_with_any_token_count": int(len(overlap)),
        "singleton_ids_with_any_token_examples": overlap.head(max_examples).tolist(),
        "singleton_ids_absent_from_reference_counts": bool(no_overlap),
        "passed": bool(counts_match and no_overlap),
        "method": "raw csv-module read of train_ground_truth.tsv, independent of the pandas parsing path",
    }


def all_checks_passed(validation: dict) -> bool:
    return bool(
        validation["invalid_reference_tokens"] == 0
        and validation["matched_ids_not_in_train_source2_or_source3"] == 0
        and validation["ground_truth_duplicate_source1_rows"] == 0
        and validation["ground_truth_source1_ids_not_in_train_source1"] == 0
        and validation.get("empty_lists_preserved_as_singletons") is True
        and validation.get("malformed_rows_total", 0) == 0
    )


def train_paths(data_dir) -> dict:
    train = Path(data_dir) / "train"
    return {
        "source1": train / "train_source1.tsv",
        "source2": train / "train_source2.tsv",
        "source3": train / "train_source3.tsv",
        "ground_truth": train / "train_ground_truth.tsv",
    }


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# --------------------------------------------------------------------------- ID encoding
def encode_ids(ids: pd.Series, source_number: int, noncanonical: dict, add: bool) -> np.ndarray:
    """Map entity-ID strings of one source to int64 keys.

    ``S<n>-<digits>`` (no leading zeros) -> the integer; any other string -> a negative code
    from ``noncanonical`` (created when ``add`` is True, otherwise MISSING_KEY if unknown).
    """
    s = ids.astype(object)
    keys = np.full(len(s), MISSING_KEY, dtype=np.int64)
    canon = (s.str.fullmatch(CANONICAL_ID_RE) & s.str.startswith(f"S{source_number}-")).to_numpy(bool)
    if canon.any():
        keys[canon] = s[canon].str.slice(3).astype(np.int64).to_numpy()
    if (~canon).any():
        pos = np.flatnonzero(~canon)
        for p, v in zip(pos, s.to_numpy()[pos]):
            if add:
                keys[p] = noncanonical.setdefault(v, -(len(noncanonical) + 1))
            else:
                keys[p] = noncanonical.get(v, MISSING_KEY)
    return keys


def decode_key(key: int, source_number: int, noncanonical: dict) -> str:
    if key >= 0:
        return f"S{source_number}-{key}"
    for k, v in noncanonical.items():  # only used for rare non-canonical IDs
        if v == key:
            return k
    return f"<unknown key {key}>"


@dataclass
class SourceIdIndex:
    source_number: int
    rows: int
    unique_keys: np.ndarray  # sorted unique int64 keys
    noncanonical: dict = field(default_factory=dict)

    @property
    def distinct_ids(self) -> int:
        return int(len(self.unique_keys))

    @property
    def duplicate_id_rows(self) -> int:
        return int(self.rows - len(self.unique_keys))


def load_source_ids(path, source_number: int, chunksize: int = DEFAULT_CHUNKSIZE) -> SourceIdIndex:
    """Load only entity_id of a S2/S3 file, chunk by chunk, as int64 keys."""
    noncanonical: dict = {}
    parts, rows = [], 0
    for chunk in read_tsv(path, usecols=["entity_id"], chunksize=chunksize):
        chunk = _fill_short_rows(chunk)
        rows += len(chunk)
        parts.append(encode_ids(chunk["entity_id"], source_number, noncanonical, add=True))
    keys = np.concatenate(parts) if parts else np.empty(0, np.int64)
    return SourceIdIndex(source_number, rows, np.unique(keys), noncanonical)


# --------------------------------------------------------------------------- ground truth parsing
@dataclass
class GroundTruthTokens:
    s1_ids: np.ndarray        # object array, one per GT row
    is_singleton: np.ndarray  # bool per GT row (matched_entity_ids == "")
    tok_row: np.ndarray       # int64 GT row index per token
    tok_src: np.ndarray       # int8: 2/3 for S2-/S3- prefix, 1 for S1-, 0 otherwise
    tok_key: np.ndarray       # int64 key (MISSING_KEY if not canonical / unresolved)
    tok_issue: np.ndarray     # int8 issue code
    special: dict             # token index -> raw string for non-canonical tokens

    def token_string(self, i: int) -> str:
        if i in self.special:
            return self.special[i]
        return f"S{int(self.tok_src[i])}-{int(self.tok_key[i])}"


def parse_ground_truth(path, chunksize: int = DEFAULT_CHUNKSIZE) -> GroundTruthTokens:
    s1_parts, single_parts = [], []
    rows_p, src_p, key_p, issue_p = [], [], [], []
    special: dict = {}
    row_offset = tok_offset = 0
    for chunk in read_tsv(path, chunksize=chunksize):
        chunk = _fill_short_rows(chunk)
        n = len(chunk)
        s1_parts.append(chunk["source1_entity_id"].to_numpy(dtype=object))
        matched = chunk["matched_entity_ids"].astype(object)
        empty = (matched == "").to_numpy(bool)
        single_parts.append(empty)
        nonempty_pos = np.flatnonzero(~empty)
        lists = matched.to_numpy()[nonempty_pos]
        split = [v.split(",") for v in lists]
        lens = np.fromiter((len(x) for x in split), dtype=np.int64, count=len(split))
        tokens = pd.Series(list(itertools.chain.from_iterable(split)), dtype=object)
        rows = np.repeat(nonempty_pos + row_offset, lens).astype(np.int64)
        m = len(tokens)

        stripped = tokens.str.strip()
        is_empty = (stripped == "").to_numpy(bool)
        has_ws = (~is_empty) & (stripped != tokens).to_numpy(bool)
        dup = pd.DataFrame({"r": rows, "t": tokens}).duplicated().to_numpy(bool) & ~is_empty
        prefix = tokens.str.slice(0, 3)
        src = prefix.map({"S1-": 1, "S2-": 2, "S3-": 3}).fillna(0).astype(np.int8).to_numpy()

        issue = np.zeros(m, dtype=np.int8)
        issue[src == 0] = ISSUE_BAD_PREFIX
        issue[src == 1] = ISSUE_SOURCE1
        issue[dup] = ISSUE_DUPLICATE
        issue[has_ws] = ISSUE_WHITESPACE
        issue[is_empty] = ISSUE_EMPTY

        canon = tokens.str.fullmatch(CANONICAL_ID_RE).to_numpy(bool)
        keys = np.full(m, MISSING_KEY, dtype=np.int64)
        if canon.any():
            keys[canon] = tokens[canon].str.slice(3).astype(np.int64).to_numpy()
        tok_vals = tokens.to_numpy()
        for p in np.flatnonzero(~canon):
            special[tok_offset + int(p)] = tok_vals[p]

        rows_p.append(rows); src_p.append(src); key_p.append(keys); issue_p.append(issue)
        row_offset += n
        tok_offset += m

    cat = lambda parts, dt: np.concatenate(parts) if parts else np.empty(0, dt)
    return GroundTruthTokens(
        s1_ids=cat(s1_parts, object), is_singleton=cat(single_parts, bool),
        tok_row=cat(rows_p, np.int64), tok_src=cat(src_p, np.int8),
        tok_key=cat(key_p, np.int64), tok_issue=cat(issue_p, np.int8), special=special,
    )


def load_source1_ids(path, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.Series:
    parts = [_fill_short_rows(c)["entity_id"].astype(object) for c in read_tsv(path, usecols=["entity_id"], chunksize=chunksize)]
    return pd.concat(parts, ignore_index=True) if parts else pd.Series([], dtype=object)


# --------------------------------------------------------------------------- validation
def validate_ground_truth(gt: GroundTruthTokens, indexes: dict, s1_train_ids: pd.Series):
    """Resolve token existence, finalise issue codes (in place) and build validation results.

    Returns (validation_dict, invalid_rows_dataframe).
    """
    # resolve non-canonical S2/S3 tokens against non-canonical source IDs
    for i, tok in gt.special.items():
        src = int(gt.tok_src[i])
        if src in indexes and gt.tok_issue[i] in (ISSUE_OK, ISSUE_DUPLICATE):
            gt.tok_key[i] = indexes[src].noncanonical.get(tok, MISSING_KEY)

    exists = np.zeros(len(gt.tok_key), dtype=bool)
    for src, idx in indexes.items():
        sel = (gt.tok_src == src) & (gt.tok_key != MISSING_KEY)
        exists[sel] = np.isin(gt.tok_key[sel], idx.unique_keys)
    # whitespace / empty tokens are not exact IDs
    exists &= ~np.isin(gt.tok_issue, [ISSUE_EMPTY, ISSUE_WHITESPACE])
    gt.tok_issue[(gt.tok_issue == ISSUE_OK) & ~exists] = ISSUE_NOT_FOUND

    # tokens that equal an actual train S1 entity_id (whatever their prefix)
    nonempty = gt.tok_issue != ISSUE_EMPTY
    cand_idx = np.flatnonzero(~exists & nonempty)
    cand_str = pd.Series([gt.token_string(int(i)) for i in cand_idx], dtype=object)
    in_s1 = np.zeros(len(cand_idx), dtype=bool)
    if len(cand_idx):
        s1_hits = s1_train_ids[s1_train_ids.isin(cand_str)]
        in_s1 = cand_str.isin(s1_hits).to_numpy(bool)
    tokens_equal_train_s1 = int(in_s1.sum())
    upgrade = cand_idx[in_s1 & np.isin(gt.tok_issue[cand_idx], [ISSUE_BAD_PREFIX, ISSUE_NOT_FOUND])]
    gt.tok_issue[upgrade] = ISSUE_SOURCE1

    issue_counts = {name: int((gt.tok_issue == code).sum()) for code, name in ISSUE_NAMES.items()}

    # GT source1 IDs vs train source1
    gt_s1 = pd.Series(gt.s1_ids, dtype=object)
    dup_gt_s1 = gt_s1.duplicated(keep="first").to_numpy(bool)
    gt_in_train = gt_s1.isin(s1_train_ids).to_numpy(bool) if len(gt_s1) else np.zeros(0, bool)
    train_in_gt = s1_train_ids.isin(gt_s1).to_numpy(bool) if len(s1_train_ids) else np.zeros(0, bool)

    invalid_tok = np.flatnonzero(gt.tok_issue != ISSUE_OK)
    records = [
        (int(gt.tok_row[i]), gt.s1_ids[gt.tok_row[i]], gt.token_string(int(i)), ISSUE_NAMES[int(gt.tok_issue[i])])
        for i in invalid_tok
    ]
    records += [(int(r), gt.s1_ids[r], "", ROW_ISSUE_GT_S1_NOT_IN_TRAIN) for r in np.flatnonzero(~gt_in_train)]
    records += [(int(r), gt.s1_ids[r], "", ROW_ISSUE_GT_S1_DUPLICATE) for r in np.flatnonzero(dup_gt_s1)]
    invalid_df = pd.DataFrame(records, columns=["_row", "source1_entity_id", "matched_entity_id", "issue"])
    invalid_df = invalid_df.sort_values("_row", kind="stable").drop(columns="_row").reset_index(drop=True)

    n_rows = int(len(gt.s1_ids))
    singles = int(gt.is_singleton.sum())
    rows_with_valid = np.zeros(n_rows, dtype=bool)
    rows_with_valid[gt.tok_row[gt.tok_issue == ISSUE_OK]] = True
    rows_with_invalid = np.zeros(n_rows, dtype=bool)
    rows_with_invalid[gt.tok_row[invalid_tok]] = True
    not_found_any = int((~exists & nonempty).sum())

    validation = {
        "ground_truth_rows": n_rows,
        "ground_truth_distinct_source1_ids": int(n_rows - dup_gt_s1.sum()),
        "ground_truth_duplicate_source1_rows": int(dup_gt_s1.sum()),
        "ground_truth_source1_ids_not_in_train_source1": int((~gt_in_train).sum()),
        "train_source1_ids_not_in_ground_truth": int((~train_in_gt).sum()),
        "total_matched_tokens": int(len(gt.tok_issue)),
        "valid_reference_tokens": int((gt.tok_issue == ISSUE_OK).sum()),
        "invalid_reference_tokens": int(len(invalid_tok)),
        "matched_ids_not_in_train_source2_or_source3": not_found_any,
        "source1_ids_in_matched_lists": issue_counts[ISSUE_NAMES[ISSUE_SOURCE1]],
        "matched_tokens_equal_to_a_train_source1_id": tokens_equal_train_s1,
        "duplicate_ids_within_one_list": issue_counts[ISSUE_NAMES[ISSUE_DUPLICATE]],
        "empty_or_whitespace_tokens": issue_counts[ISSUE_NAMES[ISSUE_EMPTY]],
        "tokens_with_surrounding_whitespace": issue_counts[ISSUE_NAMES[ISSUE_WHITESPACE]],
        "bad_prefix_tokens": issue_counts[ISSUE_NAMES[ISSUE_BAD_PREFIX]],
        "valid_prefix_tokens_not_found_in_train_source": issue_counts[ISSUE_NAMES[ISSUE_NOT_FOUND]],
        "issue_counts_by_primary_issue": issue_counts,
        "ground_truth_rows_with_invalid_tokens": int(rows_with_invalid.sum()),
        "nonempty_rows_without_any_valid_reference": int((~rows_with_valid & ~gt.is_singleton).sum()),
        "empty_match_lists": singles,
        # filled in by run_checks via check_singleton_preservation (independent raw read)
        "empty_lists_preserved_as_singletons": None,
        "invalid_rows_written": int(len(invalid_df)),
        "notes": ("Each invalid token is listed once under its highest-precedence issue "
                  "(empty > whitespace > duplicate > source1 > bad_prefix > not_found). "
                  "Duplicate tokens: the first occurrence stays a valid reference, repeats are flagged. "
                  "Invalid tokens are excluded from all reference counts."),
    }
    validation["all_checks_passed"] = all_checks_passed(validation)
    return validation, invalid_df


# --------------------------------------------------------------------------- check 1
def check_multi_source_references(gt: GroundTruthTokens, indexes: dict):
    """Distinct S1 per valid referenced ID. Returns (result_dict, detail_df, referenced_keys)."""
    valid = gt.tok_issue == ISSUE_OK
    s1_codes, _ = pd.factorize(pd.Series(gt.s1_ids, dtype=object))
    result, details, referenced = {}, [], {}
    overall_max = 0
    for src, label in MATCH_SOURCES.items():
        sel = valid & (gt.tok_src == src)
        pairs = pd.DataFrame({"s1": s1_codes[gt.tok_row[sel]], "key": gt.tok_key[sel],
                              "row": gt.tok_row[sel]}).drop_duplicates(["s1", "key"])
        counts = pairs["key"].value_counts()
        referenced[src] = np.sort(counts.index.to_numpy(np.int64))
        multi = counts[counts > 1]
        result[f"{label.lower()}_ids_referenced_by_multiple_source1"] = int(len(multi))
        result[f"{label.lower()}_distinct_referenced_ids"] = int(len(counts))
        result[f"{label.lower()}_max_source1_per_id"] = int(counts.max()) if len(counts) else 0
        overall_max = max(overall_max, result[f"{label.lower()}_max_source1_per_id"])
        if len(multi):
            sub = pairs[pairs["key"].isin(multi.index)]
            for key, grp in sub.groupby("key"):
                ids = sorted(set(gt.s1_ids[grp["row"].to_numpy()]))
                details.append((decode_key(int(key), src, indexes[src].noncanonical), label, len(ids), ";".join(ids)))
        result[f"{label.lower()}_multiplicity_distribution"] = {
            str(k): int(v) for k, v in counts.value_counts().sort_index().items()}
    result["total_duplicate_reference_ids"] = (result["s2_ids_referenced_by_multiple_source1"]
                                               + result["s3_ids_referenced_by_multiple_source1"])
    result["max_source1_entities_per_referenced_id"] = overall_max
    detail_df = pd.DataFrame(details, columns=["matched_entity_id", "source_prefix",
                                               "source1_entity_count", "source1_entity_ids"])
    detail_df = detail_df.sort_values(["source1_entity_count", "matched_entity_id"],
                                      ascending=[False, True]).reset_index(drop=True)
    return result, detail_df, referenced


# --------------------------------------------------------------------------- check 2
def _length_stats(lengths: np.ndarray, prefix: str) -> dict:
    if len(lengths) == 0:
        return {f"{prefix}_len_{s}": None for s in ("min", "median", "mean", "p95", "max")}
    return {
        f"{prefix}_len_min": int(lengths.min()),
        f"{prefix}_len_median": float(np.median(lengths)),
        f"{prefix}_len_mean": round(float(lengths.mean()), 3),
        f"{prefix}_len_p95": float(np.percentile(lengths, 95)),
        f"{prefix}_len_max": int(lengths.max()),
    }


def extract_unreferenced(path, index: SourceIdIndex, referenced_keys: np.ndarray, out_tsv,
                         chunksize: int = DEFAULT_CHUNKSIZE) -> dict:
    """Stream a source file, write rows unreferenced by ground truth, and collect stats."""
    src = index.source_number
    unref_count = 0
    missing = {c: 0 for c in SOURCE_COLUMNS[1:]}
    countries: dict = {}
    name_lens, addr_lens = [], []
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        header_written = False
        for chunk in read_tsv(path, chunksize=chunksize):
            chunk = _fill_short_rows(chunk)
            if not header_written:
                fh.write("\t".join(chunk.columns) + "\n")
                header_written = True
            keys = encode_ids(chunk["entity_id"], src, index.noncanonical, add=False)
            sub = chunk[~np.isin(keys, referenced_keys)].astype(object)
            if sub.empty:
                continue
            unref_count += len(sub)
            cols = list(sub.columns)
            lines = sub[cols[0]].str.cat([sub[c] for c in cols[1:]], sep="\t")
            fh.write("\n".join(lines.tolist()) + "\n")
            for c in missing:
                if c in sub:
                    missing[c] += int((sub[c].str.strip() == "").sum())
            for k, v in sub["country"].value_counts().items():
                countries[k] = countries.get(k, 0) + int(v)
            nm = sub["business_name"]
            ad = sub["business_address"]
            name_lens.append(nm[nm.str.strip() != ""].str.len().to_numpy(np.int64))
            addr_lens.append(ad[ad.str.strip() != ""].str.len().to_numpy(np.int64))
        if not header_written:
            fh.write("\t".join(SOURCE_COLUMNS) + "\n")
    name_lens = np.concatenate(name_lens) if name_lens else np.empty(0, np.int64)
    addr_lens = np.concatenate(addr_lens) if addr_lens else np.empty(0, np.int64)
    unref_distinct = int(len(np.setdiff1d(index.unique_keys, referenced_keys, assume_unique=True)))
    rate = unref_count / index.rows if index.rows else 0.0
    countries = dict(sorted(countries.items(), key=lambda kv: (-kv[1], kv[0])))
    stats = {
        "source": f"source{src}",
        "total_rows": index.rows,
        "distinct_entity_ids": index.distinct_ids,
        "duplicate_entity_id_rows": index.duplicate_id_rows,
        "noncanonical_entity_ids": len(index.noncanonical),
        "referenced_distinct_ids": int(len(referenced_keys)),
        "unreferenced_count": unref_count,
        "unreferenced_distinct_ids": unref_distinct,
        "unreferenced_rate": round(rate, 6),
        "unreferenced_pct": round(100 * rate, 4),
        "missing_business_name_count": missing["business_name"],
        "missing_business_address_count": missing["business_address"],
        "missing_country_count": missing["country"],
        "missing_business_name_rate": round(missing["business_name"] / unref_count, 6) if unref_count else 0.0,
        "missing_business_address_rate": round(missing["business_address"] / unref_count, 6) if unref_count else 0.0,
        "missing_country_rate": round(missing["country"] / unref_count, 6) if unref_count else 0.0,
    }
    stats.update(_length_stats(name_lens, "business_name"))
    stats.update(_length_stats(addr_lens, "business_address"))
    stats["country_distribution"] = {("<empty>" if k.strip() == "" else k): v for k, v in countries.items()}
    return stats


# --------------------------------------------------------------------------- orchestration
def run_checks(data_dir, output_dir, profile_json=None, chunksize: int = DEFAULT_CHUNKSIZE) -> dict:
    t0 = time.time()
    paths = train_paths(data_dir)
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing {name}: {p}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.time()
    expected = {"source1": len(SOURCE_COLUMNS), "source2": len(SOURCE_COLUMNS),
                "source3": len(SOURCE_COLUMNS), "ground_truth": len(GT_COLUMNS)}
    malformed = {f"train_{k}": scan_tsv_structure(paths[k], n) for k, n in expected.items()}
    raw_singletons = raw_empty_match_list_ids(paths["ground_truth"])
    timings["raw_scans_s"] = round(time.time() - t, 1)

    t = time.time()
    indexes = {2: load_source_ids(paths["source2"], 2, chunksize),
               3: load_source_ids(paths["source3"], 3, chunksize)}
    s1_train = load_source1_ids(paths["source1"], chunksize)
    timings["load_ids_s"] = round(time.time() - t, 1)

    t = time.time()
    gt = parse_ground_truth(paths["ground_truth"], chunksize)
    timings["parse_ground_truth_s"] = round(time.time() - t, 1)

    t = time.time()
    validation, invalid_df = validate_ground_truth(gt, indexes, s1_train)
    singleton_check = check_singleton_preservation(
        raw_singletons, int(gt.is_singleton.sum()), gt.s1_ids[np.unique(gt.tok_row)])
    del raw_singletons
    validation["singleton_preservation"] = singleton_check
    validation["empty_lists_preserved_as_singletons"] = singleton_check["passed"]
    validation["malformed_rows_total"] = int(sum(m["malformed_row_count"] for m in malformed.values()))
    validation["malformed_rows_by_file"] = {k: m["malformed_row_count"] for k, m in malformed.items()}
    validation["all_checks_passed"] = all_checks_passed(validation)
    invalid_df.to_csv(out / INVALID_FILE, index=False)
    timings["validation_s"] = round(time.time() - t, 1)

    t = time.time()
    check1, detail_df, referenced = check_multi_source_references(gt, indexes)
    detail_df.to_csv(out / MULTI_REFERENCE_FILE, index=False)
    timings["check1_s"] = round(time.time() - t, 1)

    n_gt = len(gt.s1_ids)
    singleton_count = int(gt.is_singleton.sum())
    s1_rows = int(len(s1_train))
    del gt, s1_train

    t = time.time()
    unref = {}
    for src in (2, 3):
        unref[src] = extract_unreferenced(paths[f"source{src}"], indexes[src], referenced[src],
                                          out / UNREFERENCED_FILE.format(n=src), chunksize)
    timings["check2_s"] = round(time.time() - t, 1)

    summary_rows = []
    for src in (2, 3):
        row = dict(unref[src])
        row["country_distribution"] = json.dumps(row["country_distribution"], ensure_ascii=False)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out / UNREFERENCED_SUMMARY_FILE, index=False)

    fields = {
        "s2_ids_referenced_by_multiple_source1": check1["s2_ids_referenced_by_multiple_source1"],
        "s3_ids_referenced_by_multiple_source1": check1["s3_ids_referenced_by_multiple_source1"],
        "total_duplicate_reference_ids": check1["total_duplicate_reference_ids"],
        "max_source1_entities_per_referenced_id": check1["max_source1_entities_per_referenced_id"],
        "source2_unreferenced_count": unref[2]["unreferenced_count"],
        "source3_unreferenced_count": unref[3]["unreferenced_count"],
        "source2_unreferenced_rate": unref[2]["unreferenced_rate"],
        "source3_unreferenced_rate": unref[3]["unreferenced_rate"],
        "source1_singleton_count": singleton_count,
        "source1_singleton_rate": round(singleton_count / n_gt, 6) if n_gt else 0.0,
    }
    assert list(fields) == SUMMARY_FIELDS
    summary = dict(fields)
    summary["definitions"] = {
        "source1_singleton": "train_ground_truth row whose matched_entity_ids is the empty string",
        "unreferenced_by_ground_truth": ("train_source2/3 entity_id not referenced by any valid "
                                         "matched_entity_ids token (NOT a singleton)"),
        "rates": "fractions (0-1); unreferenced rate = unreferenced rows / source rows; "
                 "singleton rate = singletons / ground-truth rows",
    }
    summary["validation"] = validation
    summary["malformed_rows"] = malformed
    summary["check1_details"] = {k: v for k, v in check1.items() if k not in fields}
    summary["unreferenced_by_ground_truth_details"] = {f"source{s}": unref[s] for s in (2, 3)}
    summary["row_counts"] = {"train_source1": s1_rows, "train_source2": indexes[2].rows,
                             "train_source3": indexes[3].rows, "train_ground_truth": n_gt}
    timings["total_s"] = round(time.time() - t0, 1)
    summary["run"] = {"data_dir": str(data_dir), "timings": timings, "peak_rss_mb": peak_rss_mb(),
                      "pandas_version": pd.__version__, "chunksize": chunksize}
    with open(out / SUMMARY_JSON_FILE, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if profile_json:
        src_profile = Path(profile_json).resolve()
        dst_profile = (out / PROFILE_COPY_FILE).resolve()
        if src_profile == dst_profile:
            raise ValueError("refusing to overwrite the original profile JSON; choose another --output-dir")
        with open(src_profile, encoding="utf-8") as fh:
            profile = json.load(fh)
        profile["ground_truth_reference"] = dict(fields)
        with open(dst_profile, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, ensure_ascii=False)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default="dataset", help="directory containing train/ (default: dataset)")
    ap.add_argument("--output-dir", default="reports/data_quality")
    ap.add_argument("--profile-json", default=None,
                    help="existing profile.json; a copy with a 'ground_truth_reference' key is written to output-dir")
    ap.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    args = ap.parse_args(argv)
    summary = run_checks(args.data_dir, args.output_dir, args.profile_json, args.chunksize)
    print(json.dumps({k: summary[k] for k in SUMMARY_FIELDS}, indent=2))
    v = summary["validation"]
    print("validation:", json.dumps({k: v[k] for k in v if k not in ("notes", "issue_counts_by_primary_issue")}, indent=2))
    print("malformed_rows:", json.dumps({k: {x: m[x] for x in ("malformed_row_count", "example_line_numbers")}
                                         for k, m in summary["malformed_rows"].items()}))
    print("run:", json.dumps(summary["run"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
