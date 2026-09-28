"""Adapter protocol tests use saved tokenizers, not weights, torch or paid APIs."""
import ast
import hashlib
import json
from pathlib import Path
import typing

import pytest
from tokenizers import Tokenizer

from audit import INSTRUCTION, Profile, QUESTION, ROOT
from local_adapters import (bge_inputs, check_lengths, create_local_adapter, finite,
                            laya_sequence, pairs_checked, qwen_inputs, verify_manifest,
                            yes_probability)


class SavedTokenizer:
    """Minimal HF-like facade over the already pinned tokenizers Rust runtime."""
    def __init__(self, alias):
        folder = ROOT / "assets/models" / alias
        if alias == "laya-multi":
            folder /= "tokenizer"
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        config = json.loads((folder / "tokenizer_config.json").read_text())
        for name in ("mask_token", "cls_token", "sep_token"):
            value = config.get(name)
            value = value.get("content") if isinstance(value, dict) else value
            setattr(self, name, value)
            setattr(self, name + "_id", self.tokenizer.token_to_id(value) if value else None)

    def encode(self, text, add_special_tokens=True):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, texts, text_pair=None, add_special_tokens=True, truncation=False,
                 max_length=None, **kwargs):
        def one(text, other=None):
            ids = self.tokenizer.encode(text, other, add_special_tokens=add_special_tokens).ids
            return ids[:max_length] if truncation and max_length else ids
        if isinstance(texts, str):
            return {"input_ids": one(texts, text_pair)}
        return {"input_ids": [one(text, text_pair[i] if text_pair else None)
                              for i, text in enumerate(texts)]}


@pytest.fixture(scope="module")
def tokenizers():
    if not (ROOT / "assets/models/laya-multi/tokenizer/tokenizer.json").exists():
        pytest.skip("prepare.py required for real-tokenizer integration tests")
    return {alias: SavedTokenizer(alias) for alias in ("laya-multi", "bge", "qwen-0.6b", "qwen-8b")}


def test_probability_is_stable_and_not_rounded():
    assert yes_probability(0, 0) == .5
    assert yes_probability(-1000, 1000) == 1
    assert yes_probability(1000, -1000) == 0
    expected = 0.52497918747894
    assert yes_probability(0, .1) == pytest.approx(expected)
    assert yes_probability(0, .1) != round(expected, 4)
    assert yes_probability(0, .2, 2) == pytest.approx(expected)
    for value in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            finite(value)
    with pytest.raises(ValueError):
        yes_probability(0, 1, 0)


def test_pairs_preserve_every_character():
    pair = (" question \n", "title\n\nfull document\t")
    assert pairs_checked([pair]) == [pair]
    assert pairs_checked([{"query": pair[0], "document": pair[1]}]) == [pair]
    with pytest.raises(TypeError):
        pairs_checked([("query", None)])


@pytest.mark.parametrize("alias", ["qwen-0.6b", "qwen-8b"])
def test_qwen_matches_frozen_audit_and_rejects_overflow(tokenizers, alias):
    pairs = [("What is relevance?", "Title\nA document."), (" вопрос ", "正文\nконец")]
    encoded, lengths = qwen_inputs(tokenizers[alias], pairs, 32768)
    assert lengths == Profile(alias).full_lengths(pairs)
    assert len(encoded["input_ids"]) == 2
    assert all(mask == [1] * count for mask, count in zip(encoded["attention_mask"], lengths))
    with pytest.raises(ValueError, match="truncation is forbidden"):
        qwen_inputs(tokenizers[alias], pairs, min(lengths) - 1)


def test_bge_matches_frozen_audit_and_rejects_overflow(tokenizers):
    pairs = [("What is relevance?", "Title\nA document."), ("вопрос", "документ")]
    _, lengths = bge_inputs(tokenizers["bge"], pairs, 8192)
    assert lengths == Profile("bge").full_lengths(pairs)
    with pytest.raises(ValueError, match="truncation is forbidden"):
        bge_inputs(tokenizers["bge"], pairs, 2)


def official_build_sequence():
    """Execute only pinned pure formatter functions, never import torch/model code."""
    path = ROOT / "assets/sources/laya/laya/common.py"
    tree = ast.parse(path.read_text())
    names = {"serialize_state", "render_criterion", "_resolve_noul_labels", "render_options", "build_sequence"}
    subset = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                             and node.name in names], type_ignores=[])
    namespace = {"json": json, "Dict": typing.Dict, "List": typing.List, "Optional": typing.Optional,
                 "Union": typing.Union, "_DEFAULT_NOUL_LABELS": {"false": "false", "true": "true"},
                 "encode_text": lambda tok, text, **kwargs: tok(text, **kwargs)}
    exec(compile(subset, str(path), "exec"), namespace)
    return namespace["build_sequence"]


def test_laya_matches_official_full_ids_and_audit(tokenizers):
    tok = tokenizers["laya-multi"]
    state = {"query": "Что это?", "document": "document " + tok.mask_token + "\n終わり"}
    actual = laya_sequence(tok, state, QUESTION, 8192, 256)
    expected, markers = official_build_sequence()(tok, state, {"t": "noul", "ins": QUESTION, "crit": None},
                                                  max_len=8192, head_max_len=256)
    assert actual == {"ids": expected, "markers": markers, "qtype": 2}
    assert len(expected) == Profile("laya-multi").full_lengths([(state["query"], state["document"])])[0]


def test_laya_rejects_state_and_instruction_clipping(tokenizers):
    tok = tokenizers["laya-multi"]
    with pytest.raises(ValueError, match="truncation is forbidden"):
        laya_sequence(tok, {"query": "q", "document": "long text " * 200}, QUESTION, 50, 256)
    with pytest.raises(ValueError, match="instruction/options would be truncated"):
        laya_sequence(tok, {"query": "q", "document": "d"}, "long question " * 300, 8192, 256)
    with pytest.raises(ValueError):
        laya_sequence(tok, {}, QUESTION, 8192, 1)


def test_manifest_rejects_wrong_revision_missing_weights_unlisted_and_tampered_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text("{}")
    revision = "a" * 40
    manifest = tmp_path / "manifest.json"

    def save():
        manifest.write_text(json.dumps({"repo": "example/test", "revision": revision,
                                      "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in model.iterdir()}}))

    save()
    with pytest.raises(ValueError, match="no model weights"):
        verify_manifest(model, manifest, revision=revision, repo="example/test")
    (model / "model.safetensors").write_bytes(b"test placeholder only")
    save()
    assert verify_manifest(model, manifest, revision=revision, repo="example/test")["manifest_sha256"]
    with pytest.raises(ValueError, match="repo/revision"):
        verify_manifest(model, manifest, revision="b" * 40)
    extra = model / "unlisted.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="Unverified files"):
        verify_manifest(model, manifest, revision=revision)
    save()
    config.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(model, manifest, revision=revision)


def test_limits_and_alias_guard_without_optional_libraries():
    assert check_lengths([[1, 2]], 2) == [2]
    with pytest.raises(ValueError):
        check_lengths([[]], 2)
    with pytest.raises(ValueError, match="Unknown local"):
        create_local_adapter("unknown", {})
