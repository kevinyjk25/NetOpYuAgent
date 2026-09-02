"""Deterministic drift gate for the production L1 candidate catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dsh_adapter.bridge import build_manifest
from dsh_adapter.skills import build_skill_manifest
from network_runtime.contracts import sha256_json

from .catalog import CatalogPolicy, build_catalog, catalog_digest


CATALOG_BASELINE_SCHEMA = "netopyu.io/l1-catalog-baseline/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "data" / "l1_catalog_baseline.json"
POLICY_PATH = Path(__file__).resolve().parent / "policies" / "catalog.yaml"
PROFILES = ("lan", "dc", "wan")


def _portable_skill_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove checkout-specific absolute paths while retaining semantic content."""
    skills = manifest.get("skills")
    if not isinstance(skills, list):
        raise TypeError("L1 Skill manifest skills must be an array")
    return {
        "profile": manifest.get("profile"),
        "mode": manifest.get("mode"),
        "skills": [
            {
                key: value
                for key, value in skill.items()
                if key not in {"path", "resource_base"}
            }
            for skill in skills
        ],
    }


def build_snapshot() -> dict[str, Any]:
    backend_mode = os.getenv("NETOPYU_BACKEND", "mock").strip().lower()
    if backend_mode != "mock":
        raise RuntimeError("L1 catalog drift qualification requires NETOPYU_BACKEND=mock")
    policy = CatalogPolicy(POLICY_PATH)
    profiles: dict[str, Any] = {}
    for profile in PROFILES:
        manifest = build_manifest(profile, include_destructive=False)
        tools = manifest.get("tools")
        if not isinstance(tools, list):
            raise TypeError(f"L1 {profile} manifest tools must be an array")
        catalog = build_catalog(profile, tools, policy)
        skill_manifest = build_skill_manifest(profile, backend_mode)
        profiles[profile] = {
            "catalog_digest": catalog_digest(catalog),
            "tool_declarations_digest": sha256_json(tools),
            "skill_manifest_digest": sha256_json(
                _portable_skill_manifest(skill_manifest),
            ),
            "candidate_count": len(catalog),
            "candidate_ids": [item.identity for item in catalog],
        }
    body = {
        "backend_mode": backend_mode,
        "include_destructive": False,
        "catalog_policy_digest": policy.digest,
        "profiles": profiles,
    }
    return {
        "apiVersion": CATALOG_BASELINE_SCHEMA,
        **body,
        "snapshot_digest": sha256_json(body),
    }


def check_baseline(path: Path = DEFAULT_BASELINE) -> dict[str, Any]:
    expected = json.loads(path.read_text(encoding="utf-8"))
    current = build_snapshot()
    if (
        not isinstance(expected, dict)
        or set(expected) != {
            "apiVersion", "backend_mode", "include_destructive",
            "catalog_policy_digest", "profiles", "snapshot_digest",
        }
        or expected.get("apiVersion") != CATALOG_BASELINE_SCHEMA
        or not isinstance(expected.get("profiles"), dict)
        or set(expected["profiles"]) != set(PROFILES)
    ):
        raise ValueError("L1 catalog baseline Schema is invalid")
    differences: dict[str, Any] = {}
    expected_body = {
        key: expected[key]
        for key in (
            "backend_mode", "include_destructive", "catalog_policy_digest", "profiles",
        )
    }
    computed_expected_digest = sha256_json(expected_body)
    if expected.get("snapshot_digest") != computed_expected_digest:
        differences["baseline_integrity"] = {
            "declared": expected.get("snapshot_digest"),
            "computed": computed_expected_digest,
        }
    for profile in PROFILES:
        old = expected.get("profiles", {}).get(profile, {})
        new = current["profiles"][profile]
        old_ids = set(old.get("candidate_ids", [])) if isinstance(old, dict) else set()
        new_ids = set(new["candidate_ids"])
        if old != new:
            differences[profile] = {
                "added": sorted(new_ids - old_ids),
                "removed": sorted(old_ids - new_ids),
                "expected_catalog_digest": old.get("catalog_digest") if isinstance(old, dict) else None,
                "current_catalog_digest": new["catalog_digest"],
                "schema_or_content_changed": old_ids == new_ids,
            }
    for field in ("backend_mode", "include_destructive", "catalog_policy_digest"):
        if expected.get(field) != current.get(field):
            differences[field] = {
                "expected": expected.get(field), "current": current.get(field),
            }
    return {
        "apiVersion": "netopyu.io/l1-catalog-drift-report/v1",
        "ok": not differences,
        "baseline": str(path.resolve()),
        "expected_snapshot_digest": expected.get("snapshot_digest"),
        "current_snapshot_digest": current["snapshot_digest"],
        "differences": differences,
    }


def record_baseline(path: Path = DEFAULT_BASELINE, *, replace: bool = False) -> dict[str, Any]:
    if path.exists() and not replace:
        raise FileExistsError("L1 catalog baseline exists; use --replace after manifest review")
    snapshot = build_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "record"))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.action == "record":
        result = record_baseline(arguments.baseline, replace=arguments.replace)
        ok = True
    else:
        result = check_baseline(arguments.baseline)
        ok = bool(result["ok"])
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
