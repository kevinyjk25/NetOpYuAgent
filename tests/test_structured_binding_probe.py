import json

import pytest

from evaluation.structured_binding_probe import main, read_json
from network_runtime.contracts import sha256_json


def test_demo_and_cli_replay_preserve_failures_without_tool_execution(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--output", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["legacyFlatSchemasAccepted"] == 0 and report["newStructuredSchemasAccepted"] == 2
    assert len(report["checks"]) == 3 and all(row["blocked"] for row in report["checks"])
    assert report["providerCalls"] == report["newModelCalls"] == 0
    assert report["translationMetrics"] is None and not report["wholeSkillTranslationProven"]
    for name, digest in report["artifactDigests"].items():
        assert sha256_json(read_json(out / name)) == digest
    original = read_json(out / "host-binding.json")
    packet = {"catalog": read_json(out / "catalog.json"), "tool": "get_device_status",
              "sourceSchemas": original["binding"]["sourceSchemas"], "expression": original["binding"]["expression"]}
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet))
    main(["compile", str(packet_path), "--output", str(tmp_path / "compiled")])
    capsys.readouterr()
    assert read_json(tmp_path / "compiled/host-binding.json") == original
    main(["materialize", str(out / "host-binding.json"), str(out / "catalog.json"),
          str(out / "request-source.json"), "--output", str(tmp_path / "materialized")])
    capsys.readouterr()
    assert (out / "arguments.json").read_bytes() == (tmp_path / "materialized/arguments.json").read_bytes()
    with pytest.raises(SystemExit):
        main(["demo", "--output", str(out)])


def test_cli_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "packet.json"
    path.write_text('{"sourceSchemas":{},"sourceSchemas":{"changed":{}}}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)
