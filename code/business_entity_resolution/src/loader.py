"""Schema-validated loader for the challenge TSVs.

All reads go through ``business_entity_resolution.io_config.read_tsv`` (shared READ_KWARGS:
sep="\\t", dtype=str, keep_default_na=False, quoting=QUOTE_NONE). Files are read in pandas
chunks (bounded memory) and returned as polars DataFrames of Utf8 columns.

Validation (raises ``SchemaError`` with details):
  * header must be exactly the expected column list (same order);
  * every data line has exactly the expected number of tab-separated fields (raw line scan via
    ``profiling.ground_truth_checks.scan_tsv_structure``; pandas 3 silently reads short rows as
    "" so the parser alone cannot detect them), and the parsed row count equals the line count;
  * every ID matches ``^S<n>-\\S+$`` for the expected source prefix, no surrounding whitespace;
  * IDs are unique;
  * ground truth: matched tokens are non-empty, S2-/S3- prefixed, no duplicates within a list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from business_entity_resolution.io_config import read_tsv  # noqa: E402
from business_entity_resolution.profiling.ground_truth_checks import scan_tsv_structure  # noqa: E402

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CHUNKSIZE = 500_000


class SchemaError(ValueError):
    pass


def source_path(data_dir, split: str, n: int) -> Path:
    return Path(data_dir) / split / f"{split}_source{n}.tsv"


def gt_path(data_dir) -> Path:
    return Path(data_dir) / "train" / "train_ground_truth.tsv"


def _read(path, expected_cols, usecols=None, chunksize=CHUNKSIZE) -> pl.DataFrame:
    path = Path(path)
    header = list(read_tsv(path, nrows=0).columns)
    if header != expected_cols:
        raise SchemaError(f"{path.name}: columns {header} != expected {expected_cols}")
    struct = scan_tsv_structure(path, len(expected_cols))
    if struct["malformed_row_count"]:
        raise SchemaError(f"{path.name}: {struct['rows_with_too_few_fields']} short and "
                          f"{struct['rows_with_too_many_fields']} long rows, e.g. lines {struct['example_line_numbers'][:5]}")
    cols = usecols or expected_cols
    parts = []
    for chunk in read_tsv(path, usecols=cols, chunksize=chunksize):
        chunk = chunk[cols]
        if chunk.isna().to_numpy().any():
            bad = chunk.index[chunk.isna().any(axis=1)][:5].tolist()
            raise SchemaError(f"{path.name}: short/malformed rows (NaN padded), e.g. data rows {bad}")
        parts.append(pl.from_pandas(chunk.astype(object)).with_columns(pl.all().cast(pl.Utf8)))
    n = sum(p.height for p in parts)
    if n != struct["raw_data_lines"]:
        raise SchemaError(f"{path.name}: parsed {n} rows but file has {struct['raw_data_lines']} data lines")
    if not parts:
        return pl.DataFrame({c: [] for c in cols}, schema={c: pl.Utf8 for c in cols})
    return pl.concat(parts, rechunk=True)


def _check_ids(df: pl.DataFrame, col: str, prefix: str, name: str) -> None:
    bad = df.filter(~pl.col(col).str.contains(rf"^{prefix}-\S+$"))
    if bad.height:
        raise SchemaError(f"{name}: {bad.height} IDs without prefix '{prefix}-', e.g. {bad[col].head(5).to_list()}")
    dup = df.height - df[col].n_unique()
    if dup:
        raise SchemaError(f"{name}: {dup} duplicate {col} values")


def load_source(data_dir, split: str, n: int, usecols=None, chunksize=CHUNKSIZE) -> pl.DataFrame:
    """Load <split>_source<n>.tsv; entity_id is always loaded and validated."""
    cols = None
    if usecols is not None:
        cols = ["entity_id"] + [c for c in usecols if c != "entity_id"]
        unknown = set(cols) - set(SOURCE_COLUMNS)
        if unknown:
            raise SchemaError(f"unknown columns requested: {sorted(unknown)}")
    path = source_path(data_dir, split, n)
    df = _read(path, SOURCE_COLUMNS, cols, chunksize)
    _check_ids(df, "entity_id", f"S{n}", path.name)
    return df


def load_ground_truth(data_dir, chunksize=CHUNKSIZE) -> pl.DataFrame:
    path = gt_path(data_dir)
    df = _read(path, GT_COLUMNS, None, chunksize)
    _check_ids(df, "source1_entity_id", "S1", path.name)
    return df


def explode_ground_truth(gt: pl.DataFrame, name: str = "ground truth") -> pl.DataFrame:
    """Return (source1_entity_id, matched_id, source) pairs; validates tokens. Empty list = singleton."""
    pairs = (
        gt.filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_id"))
        .select("source1_entity_id", "matched_id")
        .explode("matched_id", empty_as_null=True)
    )
    bad = pairs.filter(~pl.col("matched_id").str.contains(r"^S[23]-\S+$"))
    if bad.height:
        raise SchemaError(f"{name}: {bad.height} matched tokens empty/whitespace/bad prefix, e.g. {bad.head(5).rows()}")
    dup = pairs.height - pairs.unique(["source1_entity_id", "matched_id"]).height
    if dup:
        raise SchemaError(f"{name}: {dup} duplicate IDs within match lists")
    return pairs.with_columns(pl.col("matched_id").str.slice(0, 2).alias("source"))
