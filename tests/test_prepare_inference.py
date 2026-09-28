import hashlib
import json
import stat
import zipfile

import pytest

import prepare_inference as p


def metadata(name, payload=b"test"):
    return {"rfilename": name, "size": len(payload),
            "blobId": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}


def test_selects_one_native_checkpoint_not_alternative_formats():
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors",
             "onnx/model.onnx", "pytorch_model.bin", "1_LogitScore/config.json", "README.md"]
    assert {row["name"] for row in p.choose_files("bge", [metadata(name) for name in names])} == set(names[:4])
    with pytest.raises(ValueError, match="both single and sharded"):
        p.choose_files("bge", [metadata(name) for name in names + ["model-00001-of-00002.safetensors"]])
    with pytest.raises(ValueError):
        p.choose_files("bge", [metadata("../escape")])


def test_plan_never_contacts_network_or_writes_files(tmp_path, monkeypatch, capsys):
    config = {"models": {"bge": {"repo": "BAAI/bge-reranker-v2-m3", "revision": "a" * 40,
                                "model_path": "weights/bge", "model_manifest": "weights/manifests/bge.json"}}}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(p.requests, "Session", lambda: pytest.fail("No network without --download"))
    p.main(["--config", str(config_file), "--model", "bge"])
    assert json.loads(capsys.readouterr().out)["download"] is False
    assert list(tmp_path.iterdir()) == [config_file]


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def raise_for_status(self):
        pass
    def iter_content(self, size):
        yield self.payload


class Session:
    def __init__(self, response):
        self.response, self.calls = response, []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_resumes_download_validates_content_and_preserves_bad_final(tmp_path):
    payload = b"abcdef"
    target = tmp_path / "weights.safetensors"
    part = tmp_path / "weights.safetensors.part"
    part.write_bytes(payload[:3])
    expected = {"size": 6, "sha256": hashlib.sha256(payload).hexdigest()}
    session = Session(Response(payload[3:], 206, {"Content-Range": "bytes 3-5/6"}))
    p.download_file("https://example.test/file", target, expected, session)
    assert session.calls[0][1]["headers"]["Range"] == "bytes=3-"
    assert target.read_bytes() == payload and not part.exists()
    target.write_bytes(b"wrong!")
    with pytest.raises(ValueError, match="SHA256"):
        p.download_file("https://example.test/file", target, expected, session)
    assert len(session.calls) == 1 and target.read_bytes() == b"wrong!"


def test_ignored_range_restarts_partial_and_bad_digest_stays_partial(tmp_path):
    target = tmp_path / "file"
    (tmp_path / "file.part").write_bytes(b"a")
    expected = {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    p.download_file("url", target, expected, Session(Response(b"abc")))
    assert target.read_bytes() == b"abc"
    other = tmp_path / "other"
    with pytest.raises(ValueError, match="SHA256"):
        p.download_file("url", other, expected, Session(Response(b"bad")))
    assert not other.exists() and (tmp_path / "other.part").read_bytes() == b"bad"


def archive(tmp_path, extra=None):
    path = tmp_path / "laya.zip"
    prefix = f"laya-{p.CODE_REVISION}/"
    with zipfile.ZipFile(path, "w") as bundle:
        for name in ("__init__.py", "agent.py", "common.py", "hooks.py", "revisions.py"):
            bundle.writestr(prefix + "laya/" + name, b"# source\n")
        if extra:
            bundle.writestr(*extra)
    return path


def test_safe_extraction_is_repeatable_but_does_not_replace_tampering(tmp_path):
    path = archive(tmp_path)
    output = tmp_path / "source"
    files = p.extract_code(path, output)
    assert len(files) == 5 and p.extract_code(path, output) == files
    (output / "laya/agent.py").write_text("tampered")
    with pytest.raises(ValueError, match="differs"):
        p.extract_code(path, output)


@pytest.mark.parametrize("danger", ["escape", "symlink"])
def test_archive_rejects_unsafe_member_before_any_extraction(tmp_path, danger):
    if danger == "escape":
        entry = f"laya-{p.CODE_REVISION}/../../escape"
    else:
        entry = zipfile.ZipInfo(f"laya-{p.CODE_REVISION}/link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    path = archive(tmp_path, (entry, "../../outside"))
    output = tmp_path / "source"
    with pytest.raises(ValueError):
        p.extract_code(path, output)
    assert not output.exists()


def test_existing_complete_manifest_tamper_never_regenerates(tmp_path):
    folder = tmp_path / "weights/bge"
    folder.mkdir(parents=True)
    weights = folder / "model.safetensors"
    weights.write_bytes(b"good")
    manifest = tmp_path / "manifest.json"
    cfg = {"repo": "BAAI/bge-reranker-v2-m3", "revision": "a" * 40,
           "model_path": "weights/bge", "model_manifest": "manifest.json"}
    manifest.write_text(json.dumps({"repo": cfg["repo"], "revision": cfg["revision"],
                                   "files_sha256": {weights.name: hashlib.sha256(b"good").hexdigest()}}))
    original_manifest = manifest.read_bytes()
    weights.write_bytes(b"evil")
    session = Session(Response(b"unused"))
    with pytest.raises(ValueError, match="hash mismatch"):
        p.prepare_model("bge", cfg, tmp_path, session)
    assert not session.calls and manifest.read_bytes() == original_manifest
