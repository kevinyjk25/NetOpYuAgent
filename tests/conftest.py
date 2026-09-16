"""Explicit test lanes; historical research remains mandatory in the full gate.

Membership is a reviewed file manifest, never a filename prefix or a marker
chosen by a newly added test. Keep original paths because fixtures import them.
"""
import json
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).with_name("suite_policy.json")
# Mixed product/safety files must not disappear through a manifest-only edit.
PROTECTED_CURRENT = frozenset({
    "tests/test_suite_policy.py", "tests/test_flow_admission.py", "tests/test_l0_flow.py",
    "tests/test_governed_hybrid.py", "tests/test_structured_flow.py", "tests/test_network_runtime.py",
    "tests/test_effect_runtime_p11.py", "tests/test_hybrid_session.py",
    "tests/test_translation_study.py", "tests/test_public_skill_dsh_ab.py",
    "tests/test_flow_checkpoint.py", "tests/test_flow_tree_authoring.py",
    "tests/test_bounded_budget.py", "tests/test_bounded_scoring.py", "tests/test_bounded_provider.py",
    "tests/test_bounded_transport.py", "tests/test_isolated_compiler.py", "tests/test_bounded_pilot.py",
    "tests/test_bounded_probe.py", "tests/test_bounded_dsh_probe.py",
    "tests/test_runtime_comparison.py", "tests/test_harness_skill_runtime_ab.py", "tests/test_model_endpoint.py",
    "tests/test_bounded_dsh_runner.py", "tests/test_bounded_import_boundary.py",
})


def load_suite_policy(path=POLICY, root=ROOT):
    """Reject missing, duplicate, unreasoned, escaping or protected entries."""
    root = Path(root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) != {"schema", "historical_tests"}
            or value["schema"] != "netopyu.test-suites/v1"
            or not isinstance(value["historical_tests"], list) or not value["historical_tests"]):
        raise ValueError("invalid test suite policy schema")
    historical = {}
    for row in value["historical_tests"]:
        if not isinstance(row, dict) or set(row) != {"path", "reason"}:
            raise ValueError("historical entries require an exact path and reason")
        name, reason = row["path"], row["reason"]
        if not isinstance(name, str) or not isinstance(reason, str) or not reason.strip():
            raise ValueError("historical entries require nonempty strings")
        relative = PurePosixPath(name)
        if (relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
                or relative.parent != PurePosixPath("tests") or not relative.name.startswith("test_")
                or relative.suffix != ".py"):
            raise ValueError(f"invalid exact test path: {name}")
        if name in PROTECTED_CURRENT:
            raise ValueError(f"protected current gate cannot be historical: {name}")
        if name in historical:
            raise ValueError(f"duplicate historical test: {name}")
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"historical test missing or outside repository: {name}")
        historical[name] = reason.strip()
    return historical


def pytest_addoption(parser):
    parser.getgroup("netopyu test suites").addoption(
        "--test-suite", choices=("current", "historical", "all"), default="current",
        help="current (default), explicit historical research, or all (required by CI/retirement)",
    )


def pytest_configure(config):
    try:
        config._netopyu_historical_tests = load_suite_policy()
    except (OSError, ValueError) as exc:
        raise pytest.UsageError(f"invalid test suite policy: {exc}") from exc


def pytest_report_header(config):
    return (f"NetOpYu test suite: {config.getoption('test_suite')}; "
            f"{len(config._netopyu_historical_tests)} explicitly historical files; full gate requires --test-suite=all")


def pytest_collection_modifyitems(config, items):
    suite = config.getoption("test_suite")
    historical = config._netopyu_historical_tests
    selected, deselected = [], []
    for item in items:
        path = Path(item.path).resolve()
        name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else None
        is_historical = name in historical
        if is_historical:
            item.add_marker(pytest.mark.historical_research)
        keep = suite == "all" or (is_historical if suite == "historical" else not is_historical)
        (selected if keep else deselected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
