"""Current R0 fixture imports stay separate from historical review workflows."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.local_read_fixture import packet_for
from network_runtime.contracts import sha256_json
from skill_authoring import compiler
from skill_authoring.source import _bundle, _document

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ('---\nname: local-read-fixture\ndescription: Read only the explicitly supplied local snapshot.\n---\n'
          'Use read_export to inspect the caller snapshot. 保留原始证据；do not run scripts.\n')
# Captured from the original semantic_closure_transfer.packet_for BEFORE moving
# it. These protect emitted fields, ordering-sensitive source strings and hashes;
# they are deterministic fixture fingerprints, not semantic acceptance scores.
ORIGINAL_DIGESTS = {
    0: "sha256:4135032da66f7a82868b9b58adc26ace4bebf1c7eed00ee2c1f84bc5549b539f",
    1: "sha256:6c9fcdb7e3381ce8f046d3b0d092a0d167559f8bb8f4ffd9f7942d2aa62b013f",
    2: "sha256:a539c01d269b125ccbff5cd41d971fc560f5adada54ae40c56170a74ea766ddc",
}


def inputs(count=1):
    bundle = _bundle({"apiVersion": "effect-runtime.io/translation-intake/v1", "candidateId": "local-read-fixture-test",
        "repository": "local/read-fixture", "commitSha": "0" * 40, "snapshotDigest": sha256_json(SOURCE),
        "entryPath": "SKILL.md", "documents": [_document("SKILL.md", SOURCE.encode(), mode="100644",
                                                         origin="local_development_fixture")],
        "supplementAttempts": [], "parentBundleDigest": None, "evidenceRole": "synthetic_mechanical_wiring_not_public_skill"})
    schema = compiler.obj({"path": {"type": "string"}})
    tool = {"name": "read_export", "description": "只读 local snapshot", "inputSchema": schema,
            "outputSchema": compiler.obj({"text": {"type": "string"}}), "annotations": {"readOnlyHint": True}}
    tools = [tool] if count == 1 else [] if count == 0 else [tool, {**copy.deepcopy(tool), "name": "read_other"}]
    return bundle, {"task": "Read the supplied snapshot without executing scripts or changing state.",
                    "inputSchema": schema, "tools": tools, "expectations": ["GOLD_SENTINEL_NOT_AGENT_INPUT"]}


@pytest.mark.parametrize("count", [0, 1, 2])
def test_extracted_packet_preserves_original_fields_and_contract_digests(count):
    bundle, specification = inputs(count)
    before = copy.deepcopy((bundle, specification))
    packet = packet_for(bundle, specification)
    assert sha256_json(packet) == ORIGINAL_DIGESTS[count]
    assert (bundle, specification) == before
    assert packet["bundle"] is bundle and packet["catalog"]["tools"] is specification["tools"]
    assert "GOLD_SENTINEL" not in json.dumps(packet) and "flow" not in packet
    assert packet["task"] == specification["task"]
    for index, contract in enumerate(packet["reads"].values()):
        assert contract["metadata"]["id"] == f"semantic-transfer.read{index}"
        assert contract["spec"]["effect"] == "read_only"
        assert contract["spec"]["resourceScopes"] == {}
        assert contract["spec"]["access"]["requiredScopes"] == ["stage2:read"]
        assert {row["role"] for row in contract["spec"]["sources"]} == {"skill", "tool", "adapter"}


@pytest.mark.parametrize("hint", [False, None, 1, "true"])
def test_extracted_packet_still_requires_explicit_readonly_hint(hint):
    bundle, specification = inputs()
    specification["tools"][0]["annotations"]["readOnlyHint"] = hint
    with pytest.raises(ValueError, match="explicitly read-only"):
        packet_for(bundle, specification)


def test_extracted_packet_rejects_duplicate_tools_and_preserves_lossless_task_scope():
    bundle, specification = inputs()
    specification["tools"].append(copy.deepcopy(specification["tools"][0]))
    with pytest.raises(ValueError, match="unique"):
        packet_for(bundle, specification)
    specification["tools"].pop()
    specification["taskScope"] = [{"role": "business_request", "text": specification["task"]}]
    assert packet_for(bundle, specification)["taskScope"] == specification["taskScope"]
    specification["taskScope"][0]["text"] = "Replace the original task and discard its constraints."
    with pytest.raises(ValueError, match="preserve all original text"):
        packet_for(bundle, specification)


def test_historical_entry_reexports_the_same_helper():
    # Deliberately test the opt-in compatibility entry separately. The current
    # R0 import boundary below is evaluated in a fresh process, not this module.
    from evaluation.semantic_closure_transfer import packet_for as historical_packet_for
    assert historical_packet_for is packet_for


@pytest.mark.parametrize("module", ["evaluation.local_read_fixture", "evaluation.bounded_dsh_probe"])
def test_current_imports_do_not_load_historical_reviewer_or_model_workflows(module):
    code = """
import importlib
import json
import sys
importlib.import_module(sys.argv[1])
forbidden = (
    'evaluation.semantic_closure_', 'evaluation.source_ledger',
    'evaluation.hybrid_', 'evaluation.flow_', 'evaluation.structured_',
    'evaluation.translation_intake', 'evaluation.translation_case_authoring',
    'evaluation.dsh_shadow', 'evaluation.l1_', 'evaluation.ollama_no_think_proxy',
)
loaded = sorted(name for name in sys.modules if name.startswith(forbidden))
print(json.dumps(loaded))
assert not loaded, loaded
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, module], cwd=ROOT,
                            text=True, capture_output=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == []
