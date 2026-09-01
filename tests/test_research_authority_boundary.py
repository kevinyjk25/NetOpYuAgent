"""Regression tests for the EnsuredSkill research-authority boundary."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ROOTS = (
    ROOT / "effect_runtime",
    ROOT / "network_runtime",
    ROOT / "dsh_adapter",
    ROOT / "l1_runtime",
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.append(node.module)
    return tuple(values)


def test_product_execution_code_does_not_import_offline_evaluator() -> None:
    offenders: list[str] = []
    for source_root in PRODUCT_ROOTS:
        for path in source_root.rglob("*.py"):
            imported = _imports(path)
            if any(
                name == "evaluation" or name.startswith("evaluation.")
                for name in imported
            ):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_dsh_retirement_gate_lives_only_in_evaluation_plane() -> None:
    assert not (ROOT / "dsh_adapter/evaluation.py").exists()
    evaluator = ROOT / "evaluation/dsh_adapter_parity.py"
    assert evaluator.exists()
    assert "offline_evaluation_only" in evaluator.read_text(encoding="utf-8")


def test_default_dsh_cli_import_does_not_load_frozen_extensions() -> None:
    script = """
import sys
import dsh_adapter.cli
blocked = {
    'dsh_adapter.a2a_provider',
    'dsh_adapter.learning',
    'l1_runtime.service',
}
loaded = sorted(blocked.intersection(sys.modules))
raise SystemExit('unexpected frozen imports: ' + ', '.join(loaded) if loaded else 0)
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_active_document_navigation_excludes_frozen_productization() -> None:
    content = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    active, frozen = content.split("#### 冻结的未来工程参考", maxsplit=1)
    assert "l1-decision-plane.md" not in active
    assert "p19-canary-runbook.md" not in active
    assert "l1-decision-plane.md" in frozen
    assert "p19-canary-runbook.md" in frozen


def test_holdout_tooling_is_not_documented_as_collected_evidence() -> None:
    convergence = (ROOT / "docs/convergence-evaluation.md").read_text(encoding="utf-8")
    decision_plane = (ROOT / "docs/l1-decision-plane.md").read_text(encoding="utf-8")
    assert "P1.9 已提供仓库外密封 holdout" not in convergence
    assert "仓库内没有真实私有用例、人工真值或独立资格结果" in convergence
    assert "仓库未包含真实私有 Case" in decision_plane
    assert "This is not an active project roadmap." in decision_plane
