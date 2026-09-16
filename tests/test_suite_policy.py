"""Test selection is explicit; full CI gates and historical failures stay intact."""
import json
import os
import subprocess
import sys

import pytest

from tests.conftest import POLICY, PROTECTED_CURRENT, ROOT, load_suite_policy


def write_policy(root, rows):
    path = root / "tests" / "suite_policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "netopyu.test-suites/v1", "historical_tests": rows}))
    return path


def row(name="tests/test_old.py"):
    return {"path": name, "reason": "Explicit historical fixture for selector tests only."}


def test_manifest_is_explicit_and_leaves_core_gates_current():
    historical = load_suite_policy()
    assert len(historical) == 24
    assert not PROTECTED_CURRENT.intersection(historical)
    assert all((ROOT / name).is_file() for name in PROTECTED_CURRENT)
    assert "tests/test_task_delivery_ablation.py" not in historical
    assert all((ROOT / name).is_file() and reason for name, reason in historical.items())


@pytest.mark.parametrize("name", sorted(PROTECTED_CURRENT))
def test_core_gate_cannot_be_excluded_by_manifest_edit(tmp_path, name):
    path = write_policy(tmp_path, [row(name)])
    with pytest.raises(ValueError, match="protected current gate"):
        load_suite_policy(path, tmp_path)


@pytest.mark.parametrize("name", ["../tests/test_old.py", "/tests/test_old.py", "tests/../tests/test_old.py", "tests/test_*.py", "tests/not_test.py"])
def test_invalid_or_nonexistent_manifest_paths_fail_closed(tmp_path, name):
    with pytest.raises(ValueError):
        load_suite_policy(write_policy(tmp_path, [row(name)]), tmp_path)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "no_reason", "extra_field", "empty", "wrong_schema"])
def test_invalid_policy_does_not_silently_drop_tests(tmp_path, fault):
    path = write_policy(tmp_path, [row()])
    if fault != "missing":
        (tmp_path / "tests/test_old.py").write_text("def test_old(): pass\n")
    value = json.loads(path.read_text())
    if fault == "duplicate":
        value["historical_tests"].append(row())
    elif fault == "no_reason":
        value["historical_tests"][0]["reason"] = " "
    elif fault == "extra_field":
        value["historical_tests"][0]["prefix"] = "test_flow_"
    elif fault == "empty":
        value["historical_tests"] = []
    elif fault == "wrong_schema":
        value["schema"] = "unknown"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_suite_policy(path, tmp_path)


@pytest.fixture
def selector_workspace(tmp_path):
    write_policy(tmp_path, [row()])
    (tmp_path / "tests/conftest.py").write_bytes(POLICY.with_name("conftest.py").read_bytes())
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\nmarkers =\n    historical_research: explicit historical lane\n")
    for name in ("old", "current", "flow_future_pilot"):
        (tmp_path / "tests" / f"test_{name}.py").write_text(f"def test_{name}(): pass\n")
    return tmp_path


def collect(workspace, *args):
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
                            cwd=workspace, env=env, text=True, capture_output=True, timeout=30)
    nodes = {line for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line}
    return result, nodes


@pytest.mark.parametrize("suite,expected", [(None, {"current", "flow_future_pilot"}),
    ("current", {"current", "flow_future_pilot"}), ("historical", {"old"}),
    ("all", {"old", "current", "flow_future_pilot"})])
def test_explicit_suite_selection_and_future_tests_default_current(selector_workspace, suite, expected):
    args = [] if suite is None else ["--test-suite=" + suite]
    result, nodes = collect(selector_workspace, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert nodes == {f"tests/test_{name}.py::test_{name}" for name in expected}


def test_explicit_historical_file_still_requires_historical_or_all(selector_workspace):
    result, nodes = collect(selector_workspace, "tests/test_old.py")
    assert result.returncode == 5 and not nodes
    result, nodes = collect(selector_workspace, "tests/test_old.py", "--test-suite=historical")
    assert result.returncode == 0 and nodes == {"tests/test_old.py::test_old"}


def test_retirement_and_ci_explicitly_include_historical_suite():
    script = (ROOT / "scripts/netopyu-dsh").read_text()
    retirement = script.split("run_e2e() {", 1)[1].split("case ", 1)[0]
    assert '"$PYTHON_BIN" -m pytest -q --test-suite=all' in retirement
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "PYTEST_ADDOPTS: --test-suite=all" in workflow
    assert "run: scripts/netopyu-dsh retirement" in workflow
