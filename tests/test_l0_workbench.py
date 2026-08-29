from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from network_runtime.contracts import sha256_json
from network_runtime.l0.promotion import PromotionError, package_promotion, review_promotion
from network_runtime.l0.workbench import (
    WORKBENCH_SCHEMA,
    export_workbench_html,
    inspect_workbench,
    list_workbench,
    render_workbench_html,
)


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "network_runtime" / "l0" / "promotion_examples" / "url1-network-access"
SKILL = PROMOTION / "SKILL.md"
CAPABILITIES = PROMOTION / "capabilities.yaml"
L05 = PROMOTION / "L0.5.yaml"
CANDIDATE = ROOT / "network_runtime" / "l0" / "examples" / "s1-network-access-grant.yaml"


def _package(root: Path, name: str = "proposal") -> Path:
    proposal = root / name
    package_promotion(
        skill_path=SKILL,
        candidate_path=CANDIDATE,
        capability_catalog_path=CAPABILITIES,
        output_directory=proposal,
        l05_path=L05,
    )
    return proposal


def test_workbench_projects_integrity_trajectory_diff_and_no_authority(tmp_path: Path) -> None:
    proposal = _package(tmp_path)
    before = {path.name: path.read_bytes() for path in proposal.iterdir()}
    view = inspect_workbench(proposal)
    assert view["apiVersion"] == WORKBENCH_SCHEMA
    assert view["status"] == "ready_for_review"
    assert view["proposal"] == {
        "proposal_hash": view["proposal"]["proposal_hash"],
        "trajectory_hash": view["proposal"]["trajectory_hash"],
        "integrity_valid": True,
        "execution_eligible": False,
        "auto_activated": False,
        "activation_available": False,
    }
    assert [item["id"] for item in view["trajectory"]["nodes"]] == [
        "L1", "L0.5", "L0-authoring", "L0-compiled",
    ]
    assert view["semantic_diff"]["L1_to_L0.5"] == {
        "parameter_names_exact": True,
        "profiles_not_widened": True,
        "risk_not_weakened": True,
        "approval_not_weakened": True,
    }
    assert all(value is False for key, value in view["controls"].items() if key in {
        "same_session_approval", "runtime_registration", "execution_authority",
    })
    body = dict(view)
    body.pop("apiVersion")
    digest = body.pop("view_digest")
    assert digest == sha256_json(body)
    assert before == {path.name: path.read_bytes() for path in proposal.iterdir()}


def test_workbench_review_is_digest_minimized_and_still_not_active(tmp_path: Path) -> None:
    proposal = _package(tmp_path)
    review_promotion(
        proposal_directory=proposal,
        reviewer="secret-human-reviewer",
        decision="approve",
        reason="secret internal review reason",
    )
    view = inspect_workbench(proposal)
    assert view["status"] == "approved_not_active"
    assert view["review"]["decision"] == "approve"
    assert view["proposal"]["activation_available"] is False
    rendered = json.dumps(view, ensure_ascii=False)
    assert "secret-human-reviewer" not in rendered
    assert "secret internal review reason" not in rendered


def test_workbench_export_is_self_contained_draft_only_and_escapes_script_data(
    tmp_path: Path,
) -> None:
    proposal = _package(tmp_path)
    output = tmp_path / "workbench.html"
    result = export_workbench_html(proposal, output)
    assert result["ok"] is True
    assert result["activation_available"] is False
    rendered = output.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in rendered
    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
    assert "Download untrusted L0.5 draft" in rendered
    assert "+'\\n'" in rendered
    assert "approve" not in rendered.split("<button", 1)[-1].split("</button>", 1)[0]

    view = inspect_workbench(proposal)
    view["claim_boundary"] = "</script><script>alert('escape')</script>"
    escaped = render_workbench_html(view)
    assert "</script><script>alert('escape')</script>" not in escaped
    assert "\\u003c/script\\u003e" in escaped


def test_workbench_rejects_tampering_and_lists_invalid_entries_without_names(
    tmp_path: Path,
) -> None:
    valid = _package(tmp_path, "private-valid-name")
    invalid = _package(tmp_path, "private-invalid-name")
    with (invalid / "02-L0.5.yaml").open("a", encoding="utf-8") as stream:
        stream.write("# tampered\n")
    with pytest.raises(PromotionError, match="integrity"):
        inspect_workbench(invalid)
    listing = list_workbench(tmp_path)
    assert listing["count"] == 2
    assert {item["status"] for item in listing["proposals"]} == {
        "ready_for_review", "invalid",
    }
    rendered = json.dumps(listing)
    assert "private-valid-name" not in rendered
    assert "private-invalid-name" not in rendered
    assert inspect_workbench(valid)["proposal"]["integrity_valid"] is True


def test_workbench_rejects_symlinked_proposal_and_output(tmp_path: Path) -> None:
    proposal = _package(tmp_path)
    proposal_link = tmp_path / "proposal-link"
    proposal_link.symlink_to(proposal, target_is_directory=True)
    with pytest.raises(PromotionError, match="unsafe"):
        inspect_workbench(proposal_link)

    output_target = tmp_path / "target.html"
    output_target.write_text("do not overwrite", encoding="utf-8")
    output_link = tmp_path / "output-link.html"
    output_link.symlink_to(output_target)
    with pytest.raises(PromotionError, match="unsafe"):
        export_workbench_html(proposal, output_link)
    assert output_target.read_text(encoding="utf-8") == "do not overwrite"


def test_workbench_cli_inspect_and_export(tmp_path: Path) -> None:
    proposal = _package(tmp_path)
    output = tmp_path / "review.html"
    inspect_result = subprocess.run(
        [str(ROOT / "scripts" / "netopyu-l0"), "workbench-inspect",
         "--proposal", str(proposal)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert inspect_result.returncode == 0, inspect_result.stderr
    assert json.loads(inspect_result.stdout)["status"] == "ready_for_review"
    export_result = subprocess.run(
        [str(ROOT / "scripts" / "netopyu-l0"), "workbench-export",
         "--proposal", str(proposal), "--output", str(output)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert export_result.returncode == 0, export_result.stderr
    assert json.loads(export_result.stdout)["activation_available"] is False
    assert output.is_file()
