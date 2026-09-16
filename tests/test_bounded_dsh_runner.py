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
    monkeypatch.setattr(probe.ModelBroker, "close", lambda self: setattr(self, "closed", True))

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
    assert report["budget"]["study"]["status"] == "halted"
    meter = json.loads((output / "native/meter.json").read_text())
    assert meter["arm"]["status"] == ("timeout" if failure == "timeout" else "measurement_invalid")
    assert (output / "report.json").exists() and (output / "native/observation.json").exists()
    assert not (output / "compiled").exists()
    if failure == "timeout":
        assert (output / "native/dsh-stdout.txt").read_text() == "partial stdout"
        assert (output / "native/dsh-stderr.txt").read_text() == "partial stderr"
