"""Unit tests for ground-truth relationship checks (tiny fixtures under tests/fixtures/train)."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from business_entity_resolution.profiling import ground_truth_checks as gtc

FIXTURES = Path(__file__).parent / "fixtures"
READ = dict(sep="\t", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("dq")
    profile = out / "orig_profile.json"
    profile.write_text(json.dumps({"existing": 1}))
    summary = gtc.run_checks(FIXTURES, out / "reports", profile_json=profile, chunksize=2)
    return summary, out / "reports", profile


def _csv(out, name):
    return pd.read_csv(out / name, dtype=str, keep_default_na=False)


# 1) one S2 ID referenced by two S1
def test_s2_id_referenced_by_two_source1(run):
    summary, out, _ = run
    assert summary["s2_ids_referenced_by_multiple_source1"] == 1
    detail = _csv(out, gtc.MULTI_REFERENCE_FILE)
    row = detail[detail.matched_entity_id == "S2-100"].iloc[0]
    assert row.source_prefix == "S2"
    assert row.source1_entity_count == "2"
    assert row.source1_entity_ids == "S1-1;S1-2"


# 2) one S3 ID referenced by two S1
def test_s3_id_referenced_by_two_source1(run):
    summary, out, _ = run
    assert summary["s3_ids_referenced_by_multiple_source1"] == 1
    assert summary["total_duplicate_reference_ids"] == 2
    assert summary["max_source1_entities_per_referenced_id"] == 2
    detail = _csv(out, gtc.MULTI_REFERENCE_FILE)
    assert list(detail.columns) == ["matched_entity_id", "source_prefix", "source1_entity_count", "source1_entity_ids"]
    row = detail[detail.matched_entity_id == "S3-200"].iloc[0]
    assert (row.source_prefix, row.source1_entity_count, row.source1_entity_ids) == ("S3", "2", "S1-1;S1-2")
    assert len(detail) == 2


# 3) an unreferenced S2 record
def test_unreferenced_source2_record(run):
    summary, out, _ = run
    assert summary["source2_unreferenced_count"] == 1
    assert summary["source2_unreferenced_rate"] == 0.25
    rows = pd.read_csv(out / "unreferenced_source2_records.tsv", **READ)
    assert rows.to_dict("records") == [
        {"entity_id": "S2-101", "business_name": "Orphan Widgets", "business_address": "", "country": "India"}]
    s = _csv(out, gtc.UNREFERENCED_SUMMARY_FILE).set_index("source").loc["source2"]
    assert s.missing_business_address_count == "1" and s.missing_business_name_count == "0"
    assert json.loads(s.country_distribution) == {"India": 1}


# 4) an unreferenced S3 record
def test_unreferenced_source3_record(run):
    summary, out, _ = run
    assert summary["source3_unreferenced_count"] == 1
    assert summary["source3_unreferenced_rate"] == 0.25
    rows = pd.read_csv(out / "unreferenced_source3_records.tsv", **READ)
    assert rows.entity_id.tolist() == ["S3-201"]
    s = _csv(out, gtc.UNREFERENCED_SUMMARY_FILE).set_index("source").loc["source3"]
    assert s.missing_country_count == "1"
    assert float(s.business_name_len_max) == len("Unlinked Bakery")


# 5) empty S1 match list counted as singleton and preserved
def test_empty_list_is_singleton_and_preserved(run):
    summary, out, _ = run
    assert summary["source1_singleton_count"] == 1
    assert summary["source1_singleton_rate"] == 0.2
    v = summary["validation"]
    assert v["empty_match_lists"] == 1 and v["empty_lists_preserved_as_singletons"] is True
    assert v["ground_truth_rows"] == 5
    invalid = _csv(out, gtc.INVALID_FILE)
    assert "S1-3" not in invalid.source1_entity_id.tolist()  # a singleton is not an error
    # singleton terminology is never used for unreferenced S2/S3 records
    text = (out / gtc.UNREFERENCED_SUMMARY_FILE).read_text()
    assert "singleton" not in text.lower()


# 6) duplicate IDs within one label are reported
def test_duplicate_within_list_reported(run):
    summary, out, _ = run
    assert summary["validation"]["duplicate_ids_within_one_list"] == 1
    invalid = _csv(out, gtc.INVALID_FILE)
    dup = invalid[invalid.issue == "duplicate_id_within_list"]
    assert dup[["source1_entity_id", "matched_entity_id"]].values.tolist() == [["S1-4", "S2-102"]]
    # the first occurrence is still a valid reference: S2-102 is not unreferenced
    assert "S2-102" not in pd.read_csv(out / "unreferenced_source2_records.tsv", **READ).entity_id.tolist()


# 7) invalid GT IDs (not in S2/S3, and an S1 ID in the list) are reported, not dropped
def test_invalid_ids_reported_not_dropped(run):
    summary, out, _ = run
    v = summary["validation"]
    assert v["valid_prefix_tokens_not_found_in_train_source"] == 1
    assert v["source1_ids_in_matched_lists"] == 1
    assert v["matched_ids_not_in_train_source2_or_source3"] == 2  # S2-999 and S1-1
    assert v["invalid_reference_tokens"] == 3
    assert v["all_checks_passed"] is False
    invalid = _csv(out, gtc.INVALID_FILE)
    assert list(invalid.columns) == ["source1_entity_id", "matched_entity_id", "issue"]
    got = set(map(tuple, invalid.values.tolist()))
    assert ("S1-5", "S2-999", "not_in_train_source2_or_source3") in got
    assert ("S1-5", "S1-1", "source1_id_in_matched_list") in got
    # invalid IDs are not counted as valid references (S1-5 still validly references S3-203)
    assert v["valid_reference_tokens"] == 8
    assert summary["check1_details"]["s2_distinct_referenced_ids"] == 3  # S2-100, S2-102, S2-103


def test_summary_and_profile_copy(run):
    summary, out, profile = run
    data = json.loads((out / gtc.SUMMARY_JSON_FILE).read_text())
    for f in gtc.SUMMARY_FIELDS:
        assert f in data
    assert "validation" in data
    prof = json.loads((out / "profile.json").read_text())
    assert prof["existing"] == 1
    assert list(prof["ground_truth_reference"]) == gtc.SUMMARY_FIELDS
    assert json.loads(profile.read_text()) == {"existing": 1}  # original untouched


def test_whitespace_empty_and_bad_prefix_tokens(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(FIXTURES / "train", data / "train")
    (data / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-100, S2-101,,X9-5\n"
        "S1-2\tS3-200\n"
        "S1-3\t\n"
        "S1-4\tS2-102\n"
        "S1-5\tS3-203\n")
    summary = gtc.run_checks(data, tmp_path / "out", chunksize=100)
    v = summary["validation"]
    assert v["tokens_with_surrounding_whitespace"] == 1
    assert v["empty_or_whitespace_tokens"] == 1
    assert v["bad_prefix_tokens"] == 1
    assert summary["max_source1_entities_per_referenced_id"] == 1
    # " S2-101" is invalid, so S2-101 stays unreferenced by ground truth (with S2-103)
    unref = pd.read_csv(tmp_path / "out" / "unreferenced_source2_records.tsv", **READ)
    assert sorted(unref.entity_id) == ["S2-101", "S2-103"]
    detail = pd.read_csv(tmp_path / "out" / gtc.MULTI_REFERENCE_FILE, dtype=str)
    assert detail.empty and list(detail.columns) == [
        "matched_entity_id", "source_prefix", "source1_entity_count", "source1_entity_ids"]


def test_cli_runs_on_fixtures(tmp_path):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "code/business_entity_resolution/profiling/ground_truth_checks.py"
    out = tmp_path / "reports" / "data_quality"
    res = subprocess.run([sys.executable, str(script), "--data-dir", str(FIXTURES), "--output-dir", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert '"source1_singleton_count": 1' in res.stdout
    for name in [gtc.MULTI_REFERENCE_FILE, gtc.UNREFERENCED_SUMMARY_FILE, gtc.SUMMARY_JSON_FILE, gtc.INVALID_FILE,
                 "unreferenced_source2_records.tsv", "unreferenced_source3_records.tsv"]:
        assert (out / name).exists()
    assert not (out / "profile.json").exists()  # only written when --profile-json is given


def test_cli_default_data_dir(monkeypatch):
    seen = {}
    monkeypatch.setattr(gtc, "run_checks", lambda d, o, p, c: seen.update(d=d, o=o) or (_ for _ in ()).throw(SystemExit(0)))
    with pytest.raises(SystemExit):
        gtc.main([])
    assert seen == {"d": "dataset", "o": "reports/data_quality"}


# --- singleton preservation (independent raw read) ---------------------------------------------
def test_singleton_preservation_passes_on_fixture(run):
    summary, _, _ = run
    sp = summary["validation"]["singleton_preservation"]
    assert sp["raw_empty_match_list_count"] == 1 == sp["source1_singleton_count"]
    assert sp["raw_count_equals_singleton_count"] is True
    assert sp["singleton_ids_with_any_token_count"] == 0
    assert sp["passed"] is True
    assert summary["validation"]["empty_lists_preserved_as_singletons"] is True


def test_raw_empty_match_list_ids_uses_independent_read():
    assert gtc.raw_empty_match_list_ids(FIXTURES / "train" / "train_ground_truth.tsv") == ["S1-3"]


def test_singleton_preservation_fails_on_count_mismatch():
    res = gtc.check_singleton_preservation(["S1-3"], 2, ["S1-1", "S1-2"])
    assert res["raw_count_equals_singleton_count"] is False
    assert res["singleton_ids_absent_from_reference_counts"] is True
    assert res["passed"] is False


def test_singleton_preservation_fails_when_singleton_has_tokens():
    res = gtc.check_singleton_preservation(["S1-3", "S1-7"], 2, ["S1-1", "S1-3"])
    assert res["raw_count_equals_singleton_count"] is True
    assert res["singleton_ids_with_any_token_count"] == 1
    assert res["singleton_ids_with_any_token_examples"] == ["S1-3"]
    assert res["passed"] is False
    v = {"invalid_reference_tokens": 0, "matched_ids_not_in_train_source2_or_source3": 0,
         "ground_truth_duplicate_source1_rows": 0, "ground_truth_source1_ids_not_in_train_source1": 0,
         "empty_lists_preserved_as_singletons": res["passed"], "malformed_rows_total": 0}
    assert gtc.all_checks_passed(v) is False


# --- shared io config ----------------------------------------------------------------------------
def test_shared_read_kwargs_used():
    import csv
    from business_entity_resolution import io_config
    assert io_config.READ_KWARGS == dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    assert gtc.READ_KWARGS is io_config.READ_KWARGS


# --- malformed rows ------------------------------------------------------------------------------
def test_malformed_rows_counted_and_reported(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(FIXTURES / "train", data / "train")
    with open(data / "train" / "train_source2.tsv", "a") as fh:
        fh.write("S2-104\tShort Row Co\n")                     # too few fields (line 6)
    with open(data / "train" / "train_source3.tsv", "a") as fh:
        fh.write("S3-204\tLong Row Co\tSomewhere\tUS\textra\n")  # too many fields (line 6)
    with open(data / "train" / "train_ground_truth.tsv", "a") as fh:
        fh.write("S1-6\n")                                       # too few fields (line 7)
    summary = gtc.run_checks(data, tmp_path / "out", chunksize=100)
    m = summary["malformed_rows"]
    assert m["train_source2"]["malformed_row_count"] == 1
    assert m["train_source2"]["rows_with_too_few_fields"] == 1
    assert m["train_source2"]["example_line_numbers"] == [6]
    assert m["train_source3"]["rows_with_too_many_fields"] == 1
    assert m["train_source3"]["example_line_numbers"] == [6]
    assert m["train_ground_truth"]["example_line_numbers"] == [7]
    assert m["train_source1"]["malformed_row_count"] == 0
    v = summary["validation"]
    assert v["malformed_rows_total"] == 3
    assert v["malformed_rows_by_file"]["train_source2"] == 1
    assert v["all_checks_passed"] is False
    # the short GT row is read as an empty list by pandas but not by the raw read -> mismatch surfaced
    assert v["singleton_preservation"]["raw_count_equals_singleton_count"] is False
    assert v["empty_lists_preserved_as_singletons"] is False
    written = json.loads((tmp_path / "out" / gtc.SUMMARY_JSON_FILE).read_text())
    assert written["malformed_rows"]["train_source2"]["example_line_numbers"] == [6]


def test_scan_caps_examples_at_20(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("a\tb\n" + "only_one_field\n" * 25)
    res = gtc.scan_tsv_structure(p, 2)
    assert res["malformed_row_count"] == 25
    assert res["example_line_numbers"] == list(range(2, 22))
