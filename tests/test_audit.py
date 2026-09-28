import ast
import json
from pathlib import Path
import stat
import typing
import zipfile

import pytest

from audit import (OPTIONS, QUESTION, Profile, document_text, quantile, summary)
from prepare import ROOT, extract_archive
import audit


def test_percentiles_and_strict_limit():
    assert quantile([10, 20], .5) == 15
    assert quantile([10, 20], .95) == 19.5
    assert summary([511, 512, 513], [512])["over_limits"]["512"]["n"] == 1
    assert summary([], [512])["p95"] is None


def test_document_content_not_trimmed():
    assert document_text({"title": "Title", "text": " body \n"}) == "Title\n body \n"
    assert document_text({"title": "", "text": " body "}) == " body "


def test_split_audit_detects_same_text_with_different_ids(tmp_path, monkeypatch):
    folder = tmp_path / "data" / "beir" / "example"
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "title": "T", "text": "Text"}) + "\n")
    (folder / "queries.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"_id": "q1", "text": "Same   query"}, {"_id": "q2", "text": "same query"},
    ]) + "\n")
    (folder / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq2\td1\t2\n")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    _, _, _, info = audit.read_dataset("example")
    assert info["split_overlap"]["test/train"] == {"query_ids": [], "normalized_text_count": 1}
    assert info["splits"]["test"]["grades"] == {2: 1}
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq2\tmissing\t2\n")
    with pytest.raises(ValueError, match="Dangling qrel"):
        audit.read_dataset("example")


def test_report_keeps_split_table_together(tmp_path):
    profile = {"document_tokens": summary([10]), "stress_full_input": summary([60], [512, 1024, 8192, 32768]),
               "positive_full_input": summary([60], [512, 1024, 8192, 32768]),
               "profile": {"actual_header_with_final_sep": 34, "head_max_len": 192, "state_budgets": {"512": 478}}}
    dataset = {"documents": 1, "splits": {"test": {"queries": 1, "positive_pairs": 1}},
               "profiles": {alias: profile for alias in audit.LIMITS}, "empty_texts": 0,
               "duplicate_document_texts": 0,
               "split_overlap": {"test/train": {"query_ids": [], "normalized_text_count": 1}}}
    audit.render_report({"created_at": "test", "datasets": {"first": dataset, "second": dataset}}, tmp_path)
    report = (tmp_path / "report.md").read_text()
    section = report.split("## Разделения и целостность\n\n", 1)[1]
    table = section.split("\n\n", 1)[0]
    assert "| first |" in table and "| second |" in table


@pytest.mark.parametrize("member", ["../escape", "scifact/../../escape", "/tmp/escape"])
def test_archive_traversal_rejected(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(member, "bad")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out", "scifact")


def test_archive_symlink_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    entry = zipfile.ZipInfo("scifact/link")
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(entry, "../../escape")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out", "scifact")


def test_archive_replaces_partial_extraction(tmp_path):
    archive = tmp_path / "ok.zip"
    target = tmp_path / "out" / "scifact" / "corpus.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("partial")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("scifact/corpus.jsonl", "complete\n")
    extract_archive(archive, tmp_path / "out", "scifact")
    assert target.read_text() == "complete\n"


@pytest.mark.parametrize("alias", ["laya-en", "laya-multi"])
def test_laya_matches_pinned_sequence_builder(alias):
    """Execute only the inspected build_sequence function, not the remote module."""
    source = ROOT / "assets/sources/laya/laya/common.py"
    if not source.exists():
        pytest.skip("prepare.py required for real-tokenizer integration test")
    profile = Profile(alias)
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_sequence")
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(function))
    allowed_calls = {"str", "range", "list", "len", "sum", "max", "enumerate", "render_options", "encode_text", "serialize_state"}
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id in allowed_calls

    class Adapter:
        mask_token = profile.mask
        mask_token_id = profile.tok.token_to_id(profile.mask)
        cls_token_id = profile.prefix[0]
        sep_token_id = profile.prefix[-1]

    def encode_text(tok, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = profile.tok.encode(text, add_special_tokens=add_special_tokens).ids
        return {"input_ids": ids[:max_length] if truncation else ids}

    namespace = {"Union": typing.Union, "Dict": typing.Dict, "List": typing.List,
                 "Optional": typing.Optional, "encode_text": encode_text,
                 "render_options": lambda q: OPTIONS,
                 "serialize_state": lambda state: json.dumps(state, ensure_ascii=False)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    reference = namespace["build_sequence"]
    for query, doc in [("What is retrieval?", "A short document."),
                       ("Вопрос с кавычками \"тест\"", "Test. " * 2000),
                       ("mask", "text " + profile.mask + " content")]:
        state = {"query": query, "document": doc}
        ids, _ = reference(Adapter(), state, {"t": "noul", "ins": QUESTION},
                           max_len=100000, head_max_len=profile.metadata["head_max_len"])
        assert profile.full_lengths([(query, doc)])[0] == len(ids)
    assert profile.metadata["actual_header_with_final_sep"] < profile.metadata["head_max_len"]


@pytest.mark.parametrize("alias", ["laya-en", "laya-multi", "bge", "qwen-0.6b", "qwen-8b"])
def test_long_inputs_not_silently_clipped(alias):
    if not (ROOT / "assets/models" / alias).exists():
        pytest.skip("prepare.py required")
    profile = Profile(alias)
    length = profile.full_lengths([("query", "document token. " * 20000)])[0]
    assert length > max(profile.limits)
