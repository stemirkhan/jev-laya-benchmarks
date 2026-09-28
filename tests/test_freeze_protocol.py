import copy

import pytest

import freeze_protocol as freeze
import rerank
from test_rerank import FakeAdapter, fixture_candidates


class GPUFixture(FakeAdapter):
    def metadata(self):
        return {"model": "fake-not-a-measurement", "gpu": {"name": "fixture", "uuid": "fixture-uuid"}}


@pytest.fixture
def pilots(tmp_path):
    source = tmp_path / "source"
    bundle = fixture_candidates(source)
    config = {"status": "pilot_unverified_not_main_protocol", "warmup_lists": 1, "repeats": 1,
              "models": {model: {"batch_size": 20} for model in freeze.GPU_MODELS}}
    runs = []
    for model in sorted(freeze.GPU_MODELS):
        out = tmp_path / model
        rerank.execute(bundle, out, freeze.effective_config(config, model), lambda *a: GPUFixture())
        runs.append(out)
    return source, config, runs


def freeze_valid(tmp_path, pilots):
    source, config, runs = pilots
    return freeze.freeze(runs, source, config, tmp_path / "protocol.json", reviewed=True)


def test_review_and_all_four_gpu_pilots_required(tmp_path, pilots):
    source, config, runs = pilots
    with pytest.raises(ValueError, match="Review pilot"):
        freeze.freeze(runs, source, config, tmp_path / "protocol.json")
    with pytest.raises(ValueError, match="All four"):
        freeze.freeze(runs[:3], source, config, tmp_path / "protocol.json", reviewed=True)
    with pytest.raises(ValueError, match="Duplicate pilot"):
        freeze.freeze([*runs, runs[0]], source, config, tmp_path / "protocol.json", reviewed=True)
    assert not (tmp_path / "protocol.json").exists()


def test_freeze_and_test_run_then_resume_without_inference(tmp_path, pilots):
    source, config, _ = pilots
    protocol = freeze_valid(tmp_path, pilots)
    settings = freeze.effective_config(config, "bge")
    bundle = rerank.Candidates(source, "test")
    fake = GPUFixture()
    out = tmp_path / "main"
    result = rerank.execute(bundle, out, settings, lambda *a: fake, protocol=protocol)
    assert result["complete"] and result["split"] == "test" and fake.calls == 3
    assert rerank.read_json(out / "protocol.json") == protocol
    assert rerank.read_json(out / "manifest.json")["identity"]["protocol_sha256"] == protocol["protocol_sha256"]
    rerank.execute(bundle, out, settings, lambda *a: pytest.fail("already complete"), protocol=protocol)
    with pytest.raises(ValueError, match="never overwrite"):
        freeze_valid(tmp_path, pilots)


def test_jev_not_authorized_without_its_own_pilot(tmp_path, pilots):
    source, _, _ = pilots
    protocol = freeze_valid(tmp_path, pilots)
    assert "jev" not in protocol["models"]
    with pytest.raises(ValueError, match="no reviewed successful pilot"):
        freeze.verify_protocol(protocol, rerank.Candidates(source, "test"), {"model": "jev"})


def test_jev_authorized_only_with_successful_separate_pilot(tmp_path, pilots):
    source, config, runs = pilots
    config["models"]["jev"] = rerank.read_json(rerank.ROOT / "config/rerank-pilot.json")["models"]["jev"]
    settings = freeze.effective_config(config, "jev")
    out = tmp_path / "jev"
    rerank.execute(rerank.Candidates(source), out, settings, lambda *a: FakeAdapter(), api_limit=100)
    protocol = freeze.freeze([*runs, out], source, config, tmp_path / "protocol.json", reviewed=True)
    assert "jev" in protocol["models"]
    fake = FakeAdapter()
    result = rerank.execute(rerank.Candidates(source, "test"), tmp_path / "main-jev", settings,
                           lambda *a: fake, protocol=protocol, api_limit=100)
    assert result["complete"] and fake.calls == 3


def test_changed_pilot_artifact_rejected(tmp_path, pilots):
    source, config, runs = pilots
    with (runs[0] / "events.jsonl").open("a") as dest:
        dest.write("{}\n")
    with pytest.raises(ValueError, match="Pilot artifact changed"):
        freeze.freeze(runs, source, config, tmp_path / "protocol.json", reviewed=True)


def test_incomplete_pilot_rejected(tmp_path, pilots):
    source, config, runs = pilots
    summary = rerank.read_json(runs[0] / "summary.json")
    summary["complete"] = False
    rerank.write_json(runs[0] / "summary.json", summary)
    with pytest.raises(ValueError, match="must be complete"):
        freeze.freeze(runs, source, config, tmp_path / "protocol.json", reviewed=True)


def test_changed_settings_and_instruction_code_rejected(tmp_path, pilots, monkeypatch):
    source, config, runs = pilots
    changed = copy.deepcopy(config)
    changed["models"]["bge"]["batch_size"] = 4
    with pytest.raises(ValueError, match="settings differ"):
        freeze.freeze(runs, source, changed, tmp_path / "protocol.json", reviewed=True)
    monkeypatch.setattr(freeze, "inference_hashes", lambda: {"audit.py": "changed-instruction"})
    with pytest.raises(ValueError, match="code/instructions/dependencies changed"):
        freeze.freeze(runs, source, config, tmp_path / "protocol.json", reviewed=True)


def test_gpu_environment_hardware_and_live_writer_gate(tmp_path, pilots):
    source, config, runs = pilots
    with rerank.exclusive_run(runs[0]):
        with pytest.raises(ValueError, match="already active"):
            freeze.freeze(runs, source, config, tmp_path / "protocol.json", reviewed=True)
    adapter_path = runs[0] / "adapter.json"
    original = rerank.read_json(adapter_path)
    changed = copy.deepcopy(original)
    changed["gpu"]["uuid"] = "different-gpu"
    rerank.write_json(adapter_path, changed)
    with pytest.raises(ValueError, match="same recorded GPU"):
        freeze.freeze(runs, source, config, tmp_path / "protocol.json", reviewed=True)
    rerank.write_json(adapter_path, original)
    manifest_path = runs[0] / "manifest.json"
    manifest = rerank.read_json(manifest_path)
    manifest["identity"]["environment"]["platform"] = "different-platform"
    rerank.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="environments differ"):
        freeze.freeze(runs, source, config, tmp_path / "protocol.json", reviewed=True)


def test_frozen_settings_checksum_environment_and_code_gate(tmp_path, pilots, monkeypatch):
    source, config, _ = pilots
    protocol = freeze_valid(tmp_path, pilots)
    bundle = rerank.Candidates(source, "test")
    settings = freeze.effective_config(config, "bge")
    with pytest.raises(ValueError, match="Settings differ"):
        freeze.verify_protocol(protocol, bundle, {**settings, "repeats": 2})
    modified = copy.deepcopy(protocol)
    modified["models"]["bge"]["config"]["repeats"] = 2
    with pytest.raises(ValueError, match="checksum differs"):
        freeze.verify_protocol(modified, bundle, settings)
    with monkeypatch.context() as patch:
        patch.setattr(rerank, "environment", lambda: {"changed": True})
        with pytest.raises(ValueError, match="Environment differs"):
            freeze.verify_protocol(protocol, bundle, settings)
    with monkeypatch.context() as patch:
        patch.setattr(rerank, "source_hashes", lambda: {"changed": "code"})
        with pytest.raises(ValueError, match="Code/instructions/dependencies differ"):
            freeze.verify_protocol(protocol, bundle, settings)


def test_changed_hardware_rejected_before_score_and_adapter_closed(tmp_path, pilots):
    source, config, _ = pilots
    protocol = freeze_valid(tmp_path, pilots)
    fake = FakeAdapter()  # No pilot GPU identity.
    with pytest.raises(ValueError, match="Adapter/hardware differs"):
        rerank.execute(rerank.Candidates(source, "test"), tmp_path / "main",
                       freeze.effective_config(config, "bge"), lambda *a: fake, protocol=protocol)
    assert fake.closed and fake.calls == 0


def test_failed_attempts_remain_in_reviewed_evidence(tmp_path, pilots):
    source, config, runs = pilots
    failed = tmp_path / "bge-recovered"
    settings = freeze.effective_config(config, "bge")
    with pytest.raises(RuntimeError):
        rerank.execute(rerank.Candidates(source), failed, settings, lambda *a: GPUFixture(fail_at=2))
    rerank.execute(rerank.Candidates(source), failed, settings, lambda *a: GPUFixture(), retry_failed=True)
    selected = [failed if path.name == "bge" else path for path in runs]
    protocol = freeze.freeze(selected, source, config, tmp_path / "protocol.json", reviewed=True)
    assert protocol["models"]["bge"]["historical_list_errors"] == 1
    assert any(event["event"] == "list_error" for event in rerank.rows(failed / "events.jsonl"))
