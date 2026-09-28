import pytest

import prepare


def test_initial_download_resolves_the_recorded_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    urls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"sha": prepare.MODEL_REVISIONS["bge"], "siblings": []}
    def get(url, **kwargs):
        urls.append(url)
        return Response()
    monkeypatch.setattr(prepare.requests, "get", get)
    result = prepare.prepare_model("bge")
    assert urls == ["https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3/revision/" + prepare.MODEL_REVISIONS["bge"]]
    assert result["revision"] == prepare.MODEL_REVISIONS["bge"]


def test_prior_manifest_cannot_change_recorded_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    prepare.write_json(tmp_path / "assets/manifests/model-bge.json",
                       {"repo": prepare.MODELS["bge"], "revision": "changed"})
    monkeypatch.setattr(prepare.requests, "get", lambda *a, **k: pytest.fail("must be offline"))
    with pytest.raises(ValueError, match="pinned experiment"):
        prepare.prepare_model("bge")
