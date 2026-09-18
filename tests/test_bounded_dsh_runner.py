"""Runner termination checks with no DSH, sockets or model inference."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from evaluation import bounded_dsh_probe as probe


@pytest.mark.parametrize("failure", ["timeout", "spawn", "config"])
def test_runner_keeps_failure_report_and_partial_output_and_stops_batch(tmp_path, monkeypatch, failure):
    binary = tmp_path / "not-executed"
    binary.write_text("synthetic fixture executable")
    monkeypatch.setattr(probe, "_default_dsh_binary", lambda: binary)
    monkeypatch.setattr(probe.shutil, "which", lambda *_, **__: str(binary))
    monkeypatch.setattr(probe, "implementation_fingerprint", lambda: {"fixture": "test"})
    monkeypatch.setattr(probe, "sync_settings", lambda *_, **__: None)
    monkeypatch.setattr(probe, "parse_dumped_config", lambda _: [
        SimpleNamespace(entry_id=name, disabled=False)
        for name in probe.SAFE_ACTIVE_IDS | {"bounded-probe-tools"}])

    def start_broker(self):
        self.server = SimpleNamespace(server_port=1)
        self.base_url = "http://127.0.0.1:1"
        return self
    monkeypatch.setattr(probe.ModelBroker, "start", start_broker)
    def close_broker(self, timeout):
        assert timeout == 2
        self.closed = True
        return {"closed": True, "drained": True, "inflight": False, "workers": 0, "connections": 0}
    monkeypatch.setattr(probe.ModelBroker, "close", close_broker)

    class Host:
        def __init__(self, provider, *_, **__):
            self.provider = provider
            self.endpoint = "http://127.0.0.1:1/not-served"
            self.graph = self.result = None
            self.route, self.errors = "native", []
        def start(self):
            return self
        def close(self):
            self.provider.close()
            return {"drained": True}
        def attempts(self):
            return []
    monkeypatch.setattr(probe, "ToolHost", Host)
    calls = []
    def process(args, **_):
        calls.append(args)
        if "--version" in args:
            return subprocess.CompletedProcess(args, 0, "scripted-version", "")
        if "--dump-config" in args:
            return subprocess.CompletedProcess(args, 1 if failure == "config" else 0, "config", "")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 1, output=b"partial stdout", stderr=b"partial stderr")
        raise OSError("scripted process creation failure")
    monkeypatch.setattr(probe.subprocess, "run", process)
    output = tmp_path / "out"
    report = probe.run(output)
    assert not report["allPassed"] and report["actualModelCalls"] == 0
    assert len(report["cases"]) == 1
    row = report["cases"][0]
    assert not row["passed"] and row["processError"] is not None
    assert row["brokerClose"]["drained"] is True
    assert report["budget"]["study"]["status"] == "halted"
    meter = json.loads((output / "native/meter.json").read_text())
    assert meter["arm"]["status"] == ("timeout" if failure == "timeout" else "measurement_invalid")
    assert (output / "report.json").exists() and (output / "native/observation.json").exists()
    assert not (output / "compiled").exists()
    if failure == "timeout":
        assert (output / "native/dsh-stdout.txt").read_text() == "partial stdout"
        assert (output / "native/dsh-stderr.txt").read_text() == "partial stderr"


@pytest.mark.parametrize("drained", [True, False, None, "true", 1])
def test_runner_pass_requires_explicit_broker_drain_even_if_other_checks_match(tmp_path, monkeypatch, drained):
    """Mocked runner gate only: matching fixtures are not a real DSH closure."""
    binary = tmp_path / "not-executed"
    binary.write_text("synthetic fixture executable")
    monkeypatch.setattr(probe, "_default_dsh_binary", lambda: binary)
    monkeypatch.setattr(probe.shutil, "which", lambda *_, **__: str(binary))
    monkeypatch.setattr(probe, "implementation_fingerprint", lambda: {"fixture": "test"})
    monkeypatch.setattr(probe, "sync_settings", lambda *_, **__: None)
    monkeypatch.setattr(probe, "parse_dumped_config", lambda _: [
        SimpleNamespace(entry_id=name, disabled=False)
        for name in probe.SAFE_ACTIVE_IDS | {"bounded-probe-tools"}])
    monkeypatch.setattr(probe, "delivered_result", lambda *_: {"matched": True, "testMock": True})
    close_timeouts = []

    def start_broker(self):
        self.server = SimpleNamespace(server_port=1)
        self.base_url = "http://127.0.0.1:1"
        return self

    def close_broker(self, timeout):
        close_timeouts.append(timeout)
        self.closed = True
        return {"closed": True, "drained": drained, "inflight": drained is not True,
                "workers": 0 if drained is True else 1, "connections": 0}

    monkeypatch.setattr(probe.ModelBroker, "start", start_broker)
    monkeypatch.setattr(probe.ModelBroker, "close", close_broker)
    hosts = []

    class Host:
        def __init__(self, provider, broker, directory, **_):
            self.provider, self.broker, self.case = provider, broker, directory.name
            self.endpoint = "http://127.0.0.1:1/not-served"
            self.graph, self.result, self.errors = None, None, []
            self.route = {"native": "native", "compiled": "runtime_read_prefix", "fallback": "fallback"}[self.case]
            hosts.append(self)

        def start(self):
            return self

        def close(self):
            self.provider.close()
            return {"drained": True}

        def attempts(self):
            return []

    monkeypatch.setattr(probe, "ToolHost", Host)

    def process(args, **_):
        if "--version" in args or "--dump-config" in args:
            return subprocess.CompletedProcess(args, 0, "fixture configuration", "")
        host = hosts[-1]
        # These are test-owned receipts, not model inference or post-agent work.
        stages = ["agent", "agent"] if host.case == "native" else [
            "agent", "compiler", "fallback" if host.case == "fallback" else "agent"]
        for index, stage in enumerate(stages):
            host.broker.ledger.reserve_model_call(host.broker.arm_id, f"fixture-{index}", stage, 1, 1)
            host.broker.ledger.settle_call(host.broker.arm_id, f"fixture-{index}", 1, 1)
        result = host.provider.invoke("read_export", probe.ARGUMENTS, request_id="fixture-read")
        host.result = {"text": result["result"]["value"]}
        return subprocess.CompletedProcess(args, 0, "mock process completed", "")

    monkeypatch.setattr(probe.subprocess, "run", process)
    output = tmp_path / "out"
    report = probe.run(output)
    expected_pass = drained is True
    assert report["allPassed"] is expected_pass
    assert len(report["cases"]) == (3 if expected_pass else 1)
    assert close_timeouts == [2] * len(report["cases"])
    for row in report["cases"]:
        assert row["passed"] is expected_pass
        assert row["processExit"] == 0 and row["processError"] is None
        assert row["hostClose"]["drained"] is True
        assert row["brokerClose"]["drained"] == drained
        assert row["providerCalls"] == 1 and row["delivery"]["matched"] is True
        saved = json.loads((output / row["case"] / "observation.json").read_text())
        assert saved["brokerClose"] == row["brokerClose"]
    if not expected_pass:
        assert report["budget"]["study"]["status"] == "halted"
        assert not (output / "compiled").exists()
    assert report["actualModelCalls"] == 0
    assert report["pilotQualified"] is False and report["researchEvidenceEligible"] is False
