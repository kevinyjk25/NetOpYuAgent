"""Read compilation is inactive, source-bound, reusable, and isolated from effects."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from network_runtime.l0 import L0Catalog, instantiate_read, validate_read_result_shape
from network_runtime.l0.compiler import L0CompileError, compile_documents, load_documents, parse_document
from network_runtime.l0.models import CompiledAtomicRead, ReadScalarSchema


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/read-contracts/health.yaml"


def _raw() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def _pin(raw: dict, role: str, value: dict) -> None:
    source = next(item for item in raw["spec"]["sources"] if item["role"] == role)
    source["text"] = json.dumps(value)
    source["sha256"] = "sha256:" + hashlib.sha256(source["text"].encode()).hexdigest()


def _compile(raw: dict | None = None) -> CompiledAtomicRead:
    result = compile_documents([parse_document(raw or _raw())])[0]
    assert isinstance(result, CompiledAtomicRead)
    return result


def _parameterized(required: bool = True) -> dict:
    raw = _raw()
    raw["spec"]["inputSchema"]["properties"] = {"device": {"type": "string"}}
    raw["spec"]["inputSchema"]["required"] = ["device"] if required else []
    tool = json.loads(raw["spec"]["sources"][1]["text"])
    tool["inputSchema"] = raw["spec"]["inputSchema"]
    _pin(raw, "tool", tool)
    return raw


def test_zero_input_compiles_in_existing_catalog_without_write_fields() -> None:
    catalog = L0Catalog.from_path(EXAMPLE)
    contract = catalog.require("fixture.health.read")
    assert isinstance(contract, CompiledAtomicRead)
    assert not contract.runtime_authority_granted
    assert "preflight" not in contract.spec.model_dump()
    assert "inactive" in catalog.explain("fixture.health.read")
    assert "no execution authority" in catalog.graph("fixture.health.read")
    assert catalog.for_capability("fixture.health.read") == ()  # Write lookup must not admit reads.
    assert json.loads(catalog.to_json())
    with pytest.raises(TypeError):
        catalog.to_saga_definition("fixture.health.read")


def test_zero_input_instantiation_and_result_check_grant_no_authority() -> None:
    contract = _compile()
    draft = instantiate_read(contract, {})
    assert draft["arguments"] == {}
    assert draft["runtimeAuthorityGranted"] is False
    shape = validate_read_result_shape(contract, {"healthy": True})
    assert shape["shapeValid"] and not shape["businessCorrectnessProven"]
    assert not shape["sourceAuthenticityVerified"]


def test_distinct_requests_reuse_one_contract_and_snapshot_arguments() -> None:
    contract = _compile(_parameterized())
    arguments = {"device": "sw1"}
    first = instantiate_read(contract, arguments)
    arguments["device"] = "sw2"
    second = instantiate_read(contract, arguments)
    assert first["arguments"] == {"device": "sw1"}
    assert first["contractHash"] == second["contractHash"] == contract.contract_hash
    assert first["requestDigest"] != second["requestDigest"]


def test_optional_argument_stays_absent_without_default_invention() -> None:
    contract = _compile(_parameterized(False))
    assert instantiate_read(contract, {})["arguments"] == {}
    assert instantiate_read(contract, {"device": "sw1"})["arguments"] == {"device": "sw1"}


@pytest.mark.parametrize("arguments", [{}, {"unknown": "sw1"}, {"device": 1},
                                       {"device": "${target}"}, {"device": "{{target}}"}])
def test_invalid_request_does_not_produce_draft(arguments: dict) -> None:
    with pytest.raises(ValueError):
        instantiate_read(_compile(_parameterized()), arguments)


@pytest.mark.parametrize("result", [{}, {"healthy": "true"}, {"healthy": 1},
                                   {"healthy": True, "secret": "hidden"}])
def test_result_shape_failures_are_not_success(result: dict) -> None:
    with pytest.raises(ValueError):
        validate_read_result_shape(_compile(), result)


@pytest.mark.parametrize("field,value", [("effect", "write"), ("tool", "other"),
                                        ("capability", "other")])
def test_adapter_declaration_mismatch_rejected(field: str, value: str) -> None:
    raw = _raw()
    adapter = json.loads(raw["spec"]["sources"][2]["text"])
    adapter[field] = value
    _pin(raw, "adapter", adapter)
    with pytest.raises(L0CompileError, match="mapping/effect"):
        _compile(raw)


@pytest.mark.parametrize("key", ["inputSchema", "outputSchema"])
def test_source_schema_cannot_be_silently_changed_or_omitted(key: str) -> None:
    raw = _raw()
    tool = json.loads(raw["spec"]["sources"][1]["text"])
    del tool[key]
    _pin(raw, "tool", tool)
    with pytest.raises(L0CompileError):
        _compile(raw)


def test_scope_declaration_mismatch_rejected() -> None:
    raw = _raw()
    raw["spec"]["access"]["requiredScopes"] = ["admin:read"]
    with pytest.raises(L0CompileError, match="access declaration"):
        _compile(raw)


@pytest.mark.parametrize("role", ["skill", "tool", "adapter"])
def test_missing_source_rejected(role: str) -> None:
    raw = _raw()
    raw["spec"]["sources"] = [item for item in raw["spec"]["sources"] if item["role"] != role]
    with pytest.raises(L0CompileError, match="exactly one"):
        _compile(raw)


def test_changed_source_bytes_and_compiled_model_tampering_rejected() -> None:
    raw = _raw()
    raw["spec"]["sources"][0]["text"] += " changed"
    with pytest.raises(L0CompileError, match="digest mismatch"):
        _compile(raw)
    contract = _compile()
    contract.spec.input_schema.properties["injected"] = ReadScalarSchema(type="string")
    with pytest.raises(ValueError):
        instantiate_read(contract, {})
    with pytest.raises(ValueError, match="hash mismatch"):
        instantiate_read(_compile().model_copy(update={"contract_hash": "bad"}), {})
    with pytest.raises(ValueError):
        instantiate_read(_compile().model_copy(update={"runtime_authority_granted": True}), {})


@pytest.mark.parametrize("schema", [
    {"type": "array"}, {"type": "string", "enum": ["sw1"]},
    {"type": "string", "default": "sw1"},
])
def test_unsupported_schema_is_not_flattened(schema: dict) -> None:
    raw = _parameterized()
    raw["spec"]["inputSchema"]["properties"]["device"] = schema
    with pytest.raises(L0CompileError):
        _compile(raw)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), True])
def test_invalid_numeric_values_rejected(value: float) -> None:
    raw = _parameterized()
    raw["spec"]["inputSchema"]["properties"]["device"] = {"type": "number"}
    tool = json.loads(raw["spec"]["sources"][1]["text"])
    tool["inputSchema"] = raw["spec"]["inputSchema"]
    _pin(raw, "tool", tool)
    with pytest.raises(ValueError):
        instantiate_read(_compile(raw), {"device": value})


def test_existing_effect_compiler_and_read_catalog_can_coexist() -> None:
    docs = load_documents(ROOT / "network_runtime/l0/examples")
    original = compile_documents(docs)
    combined = compile_documents([*docs, parse_document(_raw())])
    assert [item for item in combined if not isinstance(item, CompiledAtomicRead)] == original
    assert len(combined) == len(original) + 1


def test_write_contract_gates_cannot_be_removed() -> None:
    docs = load_documents(ROOT / "network_runtime/l0/examples")
    raw = next(item for item in docs if item.kind == "AtomicEffect").model_dump(by_alias=True)
    raw["spec"]["approval"]["required"] = False
    with pytest.raises(L0CompileError, match="cannot disable approval"):
        parse_document(raw)


def test_effect_promotion_explicitly_blocks_read_candidates() -> None:
    from network_runtime.l0.promotion import assess_promotion

    folder = ROOT / "network_runtime/l0/promotion_examples/url1-network-access"
    assessment = assess_promotion(
        skill_path=folder / "SKILL.md", candidate_path=EXAMPLE,
        capability_catalog_path=folder / "capabilities.yaml", l05_path=folder / "L0.5.yaml",
    )
    assert assessment.report["status"] == "blocked"
    assert assessment.compiled_contract is None
    assert not assessment.report["executionEligible"]
    assert any(item["code"] == "L0_COMPILE_FAILED" for item in assessment.report["findings"])


@pytest.mark.parametrize("hint", [False, "true", None, 1])
def test_contradictory_or_untyped_read_hint_rejected(hint: object) -> None:
    raw = _raw()
    tool = json.loads(raw["spec"]["sources"][1]["text"])
    tool["annotations"] = {"readOnlyHint": hint}
    _pin(raw, "tool", tool)
    with pytest.raises(L0CompileError, match="contradicts"):
        _compile(raw)


@pytest.mark.parametrize("suffix", [',"name":"health_snapshot"', ',"example":NaN', ',"example":1e999'])
def test_bad_source_json_rejected(suffix: str) -> None:
    raw = _raw()
    source = raw["spec"]["sources"][1]
    source["text"] = source["text"][:-1] + suffix + "}"
    source["sha256"] = "sha256:" + hashlib.sha256(source["text"].encode()).hexdigest()
    with pytest.raises(L0CompileError):
        _compile(raw)
