"""Acceptance GATE units only: controller, assets and model transport are mocked.

No 24-arm execution, DSH subprocess, tokenizer/helper, HTTP server or LLM runs.
Twenty-four rows/receivers below are explicit structural gate fixtures, not
evidence that the actual controller or R0 protocol has completed.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from evaluation import bounded_acceptance as module
from evaluation.bounded_pilot import checked_seal, seal
from evaluation.bounded_preflight import file_digest
from network_runtime.contracts import sha256_json


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


def material_fixture(tmp_path):
    directory = tmp_path / "material"
    directory.mkdir()
    documents = {
        "cases.json": [{"role": "mock gate cases, not actual evaluation tasks"}],
        "references.json": [{"role": "mock gate references, not semantic Gold"}],
        "dialogues.json": {"mock": [{"tool": "inert", "arguments": {}}]},
        "support.json": {"role": "mock gate support"},
        "annotation-a.json": {"original": {"role": "a", "verdict": "fixture"}, "addendum": []},
        "annotation-b.json": {"original": {"role": "b", "verdict": "fixture"}, "addendum": []},
        "adjudication.json": {"role": "fixture adjudication"},
        "sources.json": {"source_text": "inert standalone text, never executed"},
        "metadata.json": {"role": "unit gate only"},
        "annotation-amendment-v4.json": {"role": "retained fixture amendment"},
    }
    for name, value in documents.items():
        write_json(directory / name, value)
    manifest = seal({"status": "frozen_for_bounded_development",
        "role": "AI_assisted_developer_evaluation_not_human_Gold", "unresolved_blockers": [],
        "files": {name: file_digest(directory / name) for name in module.MATERIAL_NAMES},
        "annotations": {role: file_digest(directory / name) for role, name in module.ANNOTATIONS.items()},
        "extra_files": {name: file_digest(directory / name) for name in documents
                        if name not in {*module.MATERIAL_NAMES, *module.ANNOTATIONS.values()}}})
    write_json(directory / "review-manifest.json", manifest)
    return directory, documents, manifest


def update_review(directory, manifest, mutate):
    changed = copy.deepcopy({k: v for k, v in manifest.items() if k != "digest"})
    mutate(changed)
    write_json(directory / "review-manifest.json", seal(changed))


def test_material_requires_actual_bound_documents_but_only_projects_execution_inputs(tmp_path):
    directory, documents, review = material_fixture(tmp_path)
    material, observed = module.load_material(directory)
    assert material == {name: documents[name] for name in module.MATERIAL_NAMES}
    assert observed == review
    assert not set(material) & {"annotation-a.json", "adjudication.json", "sources.json"}


@pytest.mark.parametrize("status", ["draft", "pending_root_adjudication", "frozen_for_research"])
def test_pending_and_draft_material_cannot_open_gate(tmp_path, status):
    directory, _, review = material_fixture(tmp_path)
    update_review(directory, review, lambda r: r.update(status=status))
    with pytest.raises(ValueError, match="drafts cannot run"):
        module.load_material(directory)


@pytest.mark.parametrize("field", ["blockers", "role", "one_annotation", "short_digest", "missing_extra", "path_escape"])
def test_review_gate_rejects_unresolved_unbound_or_nonlocal_claims(tmp_path, field):
    directory, _, review = material_fixture(tmp_path)
    def mutate(r):
        if field == "blockers":
            r["unresolved_blockers"] = ["unresolved fixture ambiguity"]
        elif field == "role":
            r["role"] = "independent_human_Gold"
        elif field == "one_annotation":
            r["annotations"].pop("b")
        elif field == "short_digest":
            r["annotations"]["a"] = "sha256:anything"
        elif field == "missing_extra":
            r["extra_files"].pop("adjudication.json")
        else:
            r["extra_files"]["../unbound.json"] = sha256_json("outside")
    update_review(directory, review, mutate)
    with pytest.raises(ValueError):
        module.load_material(directory)


@pytest.mark.parametrize("name", ["cases.json", "references.json", "annotation-a.json", "annotation-b.json",
                                  "adjudication.json", "sources.json", "metadata.json", "annotation-amendment-v4.json"])
def test_each_frozen_material_file_is_hash_checked(tmp_path, name):
    directory, _, _ = material_fixture(tmp_path)
    write_json(directory / name, {"changed": True})
    with pytest.raises(ValueError, match="hash mismatch"):
        module.load_material(directory)


def test_hashes_without_annotation_files_do_not_establish_two_reviews(tmp_path):
    directory, _, _ = material_fixture(tmp_path)
    (directory / "annotation-a.json").unlink()
    with pytest.raises(ValueError, match="standalone regular"):
        module.load_material(directory)


def test_material_rejects_duplicate_json_even_if_raw_hash_is_resealed(tmp_path):
    directory, _, review = material_fixture(tmp_path)
    (directory / "references.json").write_text('{"role":"first","role":"second"}')
    update_review(directory, review, lambda r: r["files"].update({
        "references.json": file_digest(directory / "references.json")}))
    with pytest.raises(ValueError, match="duplicate"):
        module.load_material(directory)


def test_material_symlink_cannot_substitute_external_original(tmp_path):
    directory, _, _ = material_fixture(tmp_path)
    original = directory / "annotation-a.json"
    outside = tmp_path / "external-annotation.json"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="standalone regular"):
        module.load_material(directory)


def gate_mocks(monkeypatch, tmp_path, *, close_results=None, run_error=None, during_run=None, run_report=None):
    """Mock execution explicitly; exercise only real hash/freeze/report gates."""
    material, _, _ = material_fixture(tmp_path)
    cache = tmp_path / "empty-bytecode-cache"
    cache.mkdir()
    monkeypatch.setattr(module.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(module.sys, "pycache_prefix", str(cache))
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(cache))
    assets = tmp_path / "assets.json"
    assets.write_text("unit mocked asset descriptor, no model")
    codec = tmp_path / "codec"
    codec.write_text("unit inert codec, never invoked")
    tree = tmp_path / "execution-tree"
    tree.mkdir()
    (tree / "source.py").write_text("# inert dependency freeze fixture\n")
    fake_assets = SimpleNamespace(inspect=lambda: {"model": {"sha256": sha256_json("not-model")}})
    monkeypatch.setattr(module.Assets, "load", lambda path: fake_assets)
    monkeypatch.setattr(module, "local_spec", lambda **kwargs: (
        {"installed_dsh_node_modules": tree}, {"unit_codec": codec}, {"scope": "unit gate only"}))
    monkeypatch.setattr(module, "make_protocol", lambda study, cases, **kwargs:
        seal({"study": study, "cases": cases, "support": kwargs["support"], "role": "unit protocol gate only"}))
    monkeypatch.setattr(module, "validate_references", lambda protocol, refs: refs)
    close_calls = []
    results = close_results if close_results is not None else [{"closed": True, "drained": True}] * 24

    class Receiver:
        def __init__(self, index, value):
            self.index, self.value = index, value

        def close(self):
            close_calls.append(self.index)
            if isinstance(self.value, Exception):
                raise self.value
            return copy.deepcopy(self.value)

    factory = SimpleNamespace(receivers=[Receiver(i, value) for i, value in enumerate(results)])
    monkeypatch.setattr(module, "broker_factory", lambda *args: factory)
    run_calls = []
    def fake_run(protocol, references, output, **kwargs):
        run_calls.append({"protocol": protocol, "references": references, **kwargs})
        if during_run is not None:
            during_run(material, assets, cache)
        if run_error:
            raise run_error
        return copy.deepcopy(run_report) if run_report is not None else seal({
            "controllerMechanicsPassed": True, "assigned_arms": 24,
            "rows": [{"role": "mock row", "mechanics_passed": True, "error": None}] * 24,
            "observations": [{"role": "mock observation", "protocol_completed": True}] * 24, "unrun": []})
    monkeypatch.setattr(module, "run", fake_run)
    return (material, assets, codec, tmp_path / "output"), close_calls, run_calls


def test_positive_gate_pins_all_material_and_does_not_claim_r0_or_model_completion(monkeypatch, tmp_path):
    args, closes, runs = gate_mocks(monkeypatch, tmp_path)
    result = module.acceptance(*args)
    assert result["mechanical_acceptance_passed"] and len(closes) == 24 and len(runs) == 1
    assert result["actualModelCalls"] == 0 and result["r0Complete"] is False
    assert result["generationParity"] == "not_tested" and not result["researchEvidenceEligible"]
    assert result["schema"] == "ensuredskill.io/r0-controller-acceptance/v2"
    assert result["claimed_arms"] == result["protocol_completed_arms"] == result["mechanics_passed_arms"] == 24
    assert result["unrun_arms"] == 0 and "completed_arms" not in result
    dependencies = checked_seal(json.loads((args[-1] / "dependencies.json").read_text()))
    _, review = module.load_material(args[0])
    assert {"material/" + name for name in module._material_names(review)} <= set(dependencies["files"])
    assert "preflight/assets_manifest" in dependencies["files"]
    assert not {"annotation-a.json", "adjudication.json", "sources.json"} & set(runs[0])
    assert checked_seal(json.loads((args[-1] / "report.json").read_text())) == result


def reacceptance_gate_fixture(monkeypatch, tmp_path, **kwargs):
    from tests.test_bounded_reacceptance import parent_fixture
    args, closes, runs = gate_mocks(monkeypatch, tmp_path, **kwargs)
    material, review = module.load_material(args[0])
    protocol = module.make_protocol("r0-reviewed-development-20260917", material["cases.json"],
                                    support=material["support.json"])
    old = tmp_path / "old-fixture"
    old.mkdir()
    directory, registry, amendment, value = parent_fixture(
        old, protocol=protocol, review=review, dialogues=material["dialogues.json"])
    monkeypatch.setattr(module, "REGISTRY", registry)
    fake_run = module.run
    def consuming_fake_run(protocol, references, output, **run_kwargs):
        from evaluation.bounded_reacceptance import consume_claim
        consume_claim(registry, run_kwargs["reacceptance_claim"], output,
                      protocol["digest"], sha256_json(run_kwargs["dialogues"]))
        return fake_run(protocol, references, output, **run_kwargs)
    monkeypatch.setattr(module, "run", consuming_fake_run)
    return args, closes, runs, directory, registry, amendment, value


def test_reacceptance_gate_claims_once_pins_parent_and_keeps_two_batch_denominator(monkeypatch, tmp_path):
    from evaluation.bounded_reacceptance import parent_snapshot
    args, _, runs, _, registry, amendment, value = reacceptance_gate_fixture(monkeypatch, tmp_path)
    before = parent_snapshot(registry)
    report = module.acceptance(*args, reacceptance=amendment)
    assert report["mechanical_acceptance_passed"] and len(runs) == 1
    linked = report["reacceptance"]
    assert linked["total_assigned_arms"] == 48 and linked["total_started_arms"] == 25
    assert linked["parent_measurement_invalid_arms"] == 1 and linked["parent_preserved"]
    assert not linked["original_r0_window_restarted"] and not report["r0Complete"]
    assert parent_snapshot(registry) == before
    frozen = json.loads((args[-1] / "dependencies.json").read_text())
    assert "reacceptance/amendment" in frozen["files"]
    assert "reacceptance/parent/controller/report.json" in frozen["files"]
    assert report["reacceptance_amendment_digest"] == value["digest"]
    again = module.acceptance(*args[:-1], tmp_path / "different-output", reacceptance=amendment)
    assert not again["mechanical_acceptance_passed"] and len(runs) == 1


def test_reacceptance_rechecks_after_freeze_before_claim(monkeypatch, tmp_path):
    args, _, runs, _, registry, amendment, _ = reacceptance_gate_fixture(monkeypatch, tmp_path)
    real_freeze = module.freeze
    def drifting_freeze(*positional, **keyword):
        result = real_freeze(*positional, **keyword)
        value = json.loads(amendment.read_text())
        value["authorization"]["statement"] += " changed"
        write_json(amendment, seal({key: item for key, item in value.items() if key != "digest"}))
        return result
    monkeypatch.setattr(module, "freeze", drifting_freeze)
    with pytest.raises(ValueError, match="changed during dependency freeze"):
        module.acceptance(*args, reacceptance=amendment)
    assert not runs and not args[-1].exists()
    import sqlite3
    with sqlite3.connect(registry) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='engineering_reacceptance_claims'").fetchone()


def test_reacceptance_parent_drift_after_mock_execution_prevents_acceptance(monkeypatch, tmp_path):
    import sqlite3
    args, _, _, _, registry, amendment, _ = reacceptance_gate_fixture(monkeypatch, tmp_path)
    original_run = module.run
    def drift_after_run(*positional, **kwargs):
        report = original_run(*positional, **kwargs)
        with sqlite3.connect(registry) as db:
            db.execute("UPDATE studies SET last_clock=last_clock+1")
        return report
    monkeypatch.setattr(module, "run", drift_after_run)
    report = module.acceptance(*args, reacceptance=amendment)
    assert not report["mechanical_acceptance_passed"] and not report["dependencies_unchanged"]
    assert not report["reacceptance"]["parent_preserved"]
    assert report["failure"]["code"] == "halted_parent_changed"


@pytest.mark.parametrize("failure", ["write_enabled", "no_env", "relative", "stale", "runtime_mismatch", "not_directory"])
def test_bytecode_isolation_gate_rejects_invalid_startup(monkeypatch, tmp_path, failure):
    args, closes, runs = gate_mocks(monkeypatch, tmp_path)
    if failure == "write_enabled":
        monkeypatch.setattr(module.sys, "dont_write_bytecode", False)
    elif failure == "no_env":
        monkeypatch.delenv("PYTHONPYCACHEPREFIX")
    elif failure == "relative":
        monkeypatch.setenv("PYTHONPYCACHEPREFIX", "relative-cache")
    elif failure == "runtime_mismatch":
        monkeypatch.setattr(module.sys, "pycache_prefix", None)
    else:
        cache = tmp_path / "empty-bytecode-cache"
        if failure == "stale":
            (cache / "stale.pyc").write_bytes(b"not imported")
        else:
            cache.rmdir()
            cache.write_text("not a directory")
    with pytest.raises(ValueError, match="isolated absolute"):
        module.acceptance(*args)
    assert not runs and not closes and not args[-1].exists()


@pytest.mark.parametrize("target", ["references.json", "annotation-a.json", "adjudication.json", "assets", "cache"])
def test_drift_during_mock_execution_fails_closed_and_still_reports(monkeypatch, tmp_path, target):
    def mutate(material, assets, cache):
        path = assets if target == "assets" else cache / "late.pyc" if target == "cache" else material / target
        path.write_text("unit drift")
    args, closes, _ = gate_mocks(monkeypatch, tmp_path, during_run=mutate)
    result = module.acceptance(*args)
    assert not result["mechanical_acceptance_passed"] and not result["dependencies_unchanged"]
    assert result["failure"]["code"] == "acceptance_failed_or_dependency_drift"
    assert len(closes) == 24 and result["assigned_arms"] == 24
    assert (args[-1] / "report.json").exists()


@pytest.mark.parametrize("bad", [RuntimeError("unit close"), None, {"closed": True, "drained": "yes"},
                                  {"closed": True, "drained": False}, {"closed": False, "drained": True}])
def test_cleanup_failure_does_not_hide_report_or_skip_other_receivers(monkeypatch, tmp_path, bad):
    results = [bad] + [{"closed": True, "drained": True}] * 23
    args, closes, _ = gate_mocks(monkeypatch, tmp_path, close_results=results)
    result = module.acceptance(*args)
    assert closes == list(range(24)) and len(result["receiver_cleanup"]) == 24
    assert not result["mechanical_acceptance_passed"] and result["failure"]
    assert result["assigned_arms"] == result["claimed_arms"] == result["protocol_completed_arms"] == 24
    assert (args[-1] / "report.json").exists()


def test_primary_controller_failure_is_preserved_when_cleanup_also_raises(monkeypatch, tmp_path):
    args, closes, _ = gate_mocks(monkeypatch, tmp_path, run_error=TimeoutError("mock execution only"),
        close_results=[OSError("unit cleanup")] + [{"closed": True, "drained": True}] * 23)
    result = module.acceptance(*args)
    assert result["failure"]["type"] == "TimeoutError"
    assert result["receiver_cleanup"][0]["error_type"] == "OSError"
    assert len(closes) == 24 and result["controller_report_digest"] is None
    assert result["assigned_arms"] == 24 and not result["mechanical_acceptance_passed"]


def test_empty_cleanup_cannot_pass_vacuously(monkeypatch, tmp_path):
    args, _, _ = gate_mocks(monkeypatch, tmp_path, close_results=[])
    result = module.acceptance(*args)
    assert not result["mechanical_acceptance_passed"]
    assert result["failure"]["code"] == "controller_receiver_count_mismatch"
    assert result["receiver_cleanup_status"] == "not_required"


def test_material_change_between_parse_and_freeze_is_not_silently_rebound(monkeypatch, tmp_path):
    args, closes, runs = gate_mocks(monkeypatch, tmp_path)
    original = module.freeze

    def races_with_freeze(trees, files, **kwargs):
        # All new hashes are internally valid, but they describe different
        # labels from the already parsed in-memory execution inputs.
        directory = args[0]
        write_json(directory / "references.json", [{"role": "different unit reference"}])
        review = json.loads((directory / "review-manifest.json").read_text())
        update_review(directory, review, lambda r: r["files"].update({
            "references.json": file_digest(directory / "references.json")}))
        return original(trees, files, **kwargs)

    monkeypatch.setattr(module, "freeze", races_with_freeze)
    with pytest.raises(ValueError, match="changed while preparing"):
        module.acceptance(*args)
    assert not runs and not closes and not args[-1].exists()


def startup_failure_fixture():
    """Shape-only gate fixture reflecting a pre-request controller rejection."""
    return seal({"assigned_arms": 24, "controllerMechanicsPassed": False,
        "rows": [{"case_id": "mock-startup", "arm": "treatment", "mechanics_passed": False,
                  "terminal": "measurement_invalid", "error": {"type": "ValueError", "stage": "dsh_startup"}}],
        "observations": [{"protocol_completed": False}], "unrun": [{"role": "mock assigned arm"}] * 23})


def test_startup_failure_counts_claimed_not_completed_and_cleanup_actual_receivers(monkeypatch, tmp_path):
    args, closes, _ = gate_mocks(monkeypatch, tmp_path, run_report=startup_failure_fixture(),
                                 close_results=[{"closed": True, "drained": True}])
    result = module.acceptance(*args)
    assert result["assigned_arms"] == 24 and result["claimed_arms"] == 1
    assert result["protocol_completed_arms"] == result["mechanics_passed_arms"] == 0
    assert result["unrun_arms"] == 23 and not result["mechanical_acceptance_passed"]
    assert result["created_receivers"] == result["closed_drained_receivers"] == 1
    assert result["receiver_cleanup_status"] == "complete" and closes == [0]
    assert result["failure"] == {"code": "controller_arm_failed", "type": "ValueError",
        "case_id": "mock-startup", "arm": "treatment", "terminal": "measurement_invalid", "stage": "dsh_startup"}


def test_reported_controller_failure_remains_primary_if_receiver_cleanup_fails(monkeypatch, tmp_path):
    args, _, _ = gate_mocks(monkeypatch, tmp_path, run_report=startup_failure_fixture(),
                           close_results=[OSError("unit receiver cleanup")])
    result = module.acceptance(*args)
    assert result["failure"]["code"] == "controller_arm_failed"
    assert result["failure"]["type"] == "ValueError"
    assert result["receiver_cleanup_status"] == "failed"
    assert result["receiver_cleanup"][0]["error_type"] == "OSError"


@pytest.mark.parametrize("mutation", ["unrun", "protocol", "mechanics"])
def test_24_claims_alone_cannot_pass_with_incomplete_or_failed_arms(monkeypatch, tmp_path, mutation):
    report = {"assigned_arms": 24, "controllerMechanicsPassed": True,
        "rows": [{"mechanics_passed": True, "error": None} for _ in range(24)],
        "observations": [{"protocol_completed": True} for _ in range(24)], "unrun": []}
    if mutation == "unrun":
        report["unrun"] = [{"role": "inconsistent extra assignment"}]
    elif mutation == "protocol":
        report["observations"][0]["protocol_completed"] = False
    else:
        report["rows"][0]["mechanics_passed"] = False
    args, _, _ = gate_mocks(monkeypatch, tmp_path, run_report=seal(report))
    result = module.acceptance(*args)
    assert not result["mechanical_acceptance_passed"] and result["failure"]
    assert result["claimed_arms"] == 24 and result["receiver_cleanup_status"] == "complete"


def test_failure_summary_does_not_copy_unstructured_prompt_or_stderr(monkeypatch, tmp_path):
    report = startup_failure_fixture()
    report.pop("digest")
    report["rows"][0]["error"] = {"type": {"untrusted": True}, "stage": "prompt\ntext",
                                     "message": "do not reproduce arbitrary source or secrets"}
    args, _, _ = gate_mocks(monkeypatch, tmp_path, run_report=seal(report),
                           close_results=[{"closed": True, "drained": True}])
    result = module.acceptance(*args)
    assert result["failure"]["type"] is None and result["failure"]["stage"] is None
    assert "message" not in result["failure"] and result["failure"]["code"] == "controller_arm_failed"


def test_existing_v1_output_bytes_are_never_rewritten(monkeypatch, tmp_path):
    args, closes, runs = gate_mocks(monkeypatch, tmp_path)
    args[-1].mkdir()
    original = b'{"schema":"ensuredskill.io/r0-controller-acceptance/v1","completed_arms":1}\n'
    (args[-1] / "report.json").write_bytes(original)
    with pytest.raises(ValueError, match="output must be new"):
        module.acceptance(*args)
    assert (args[-1] / "report.json").read_bytes() == original
    assert not runs and not closes
