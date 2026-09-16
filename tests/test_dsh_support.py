"""Neutral helpers and the old shadow module's compatibility patch points."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation import dsh_support as support


def test_support_import_does_not_load_shadow_or_l1_benchmark():
    result = subprocess.run([sys.executable, "-c", """
import sys
import evaluation.dsh_support
assert not any(name.startswith('evaluation.l1_') for name in sys.modules)
assert 'evaluation.dsh_shadow' not in sys.modules
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", [
    "ConfigEntry", "DSH_TESTED_VERSION", "SAFE_ACTIVE_IDS", "REQUIRED_DISABLED_IDS",
    "parse_dumped_config", "_default_dsh_binary", "_node_path", "shutil",
])
def test_shadow_preserves_helper_exports(name):
    from evaluation import dsh_shadow
    assert getattr(dsh_shadow, name) is getattr(support, name)


def test_parser_preserves_js_yaml_identity_and_disabled_semantics():
    result = support.parse_dumped_config(
        "- id: active\n  name: '@vendor/active'\n  config: !!js process.env.NOT_EVALUATED\n"
        '- id: stopped\n  name: "@vendor/stopped"\n  disabled: true\n'
        "- id: unnamed\n  disabled: false\n")
    assert result == (support.ConfigEntry("active", "@vendor/active", False),
                      support.ConfigEntry("stopped", "@vendor/stopped", True),
                      support.ConfigEntry("unnamed", "", False))
    with pytest.raises(FrozenInstanceError):
        result[0].disabled = True


@pytest.mark.parametrize("text,error", [
    ("", "no entries"), ("  - id: indented\n", "no entries"),
    ("- id: duplicate\n- id: duplicate\n", "duplicate entry ids"),
])
def test_parser_keeps_historical_rejections(text, error):
    with pytest.raises(ValueError, match=error):
        support.parse_dumped_config(text)


def test_binary_discovery_uses_explicit_override_or_existing_default(tmp_path, monkeypatch):
    monkeypatch.setattr(support.Path, "home", classmethod(lambda _: tmp_path))
    monkeypatch.delenv("NETOPYU_DSH_BIN", raising=False)
    assert support._default_dsh_binary() == (
        tmp_path / "Library/Application Support/NetOpYuAgent/dsh-runtime/node_modules/.bin/dsh")
    configured = tmp_path / "custom" / ".." / "fixture-dsh"
    monkeypatch.setenv("NETOPYU_DSH_BIN", str(configured))
    assert support._default_dsh_binary() == configured.resolve()


@pytest.mark.parametrize("existing_node,bundled_node", [(True, False), (False, True), (False, False)])
def test_node_discovery_does_not_change_path_precedence(tmp_path, monkeypatch, existing_node, bundled_node):
    monkeypatch.setenv("PATH", "/fixture/bin")
    monkeypatch.setattr(support.Path, "home", classmethod(lambda _: tmp_path))
    monkeypatch.setattr(support.shutil, "which", lambda name, *, path: "/fixture/bin/node" if existing_node else None)
    bundled = tmp_path / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    if bundled_node:
        (bundled / "node/bin").mkdir(parents=True)
        (bundled / "node/bin/node").touch()
    expected = f"{bundled / 'node/bin'}:{bundled / 'bin/fallback'}:/fixture/bin"
    assert support._node_path() == (expected if bundled_node and not existing_node else "/fixture/bin")


def test_shadow_audit_keeps_old_module_monkeypatch_points(monkeypatch):
    from evaluation import dsh_shadow
    monkeypatch.setattr(dsh_shadow, "SAFE_ACTIVE_IDS", frozenset({"system-prompt"}))
    monkeypatch.setattr(dsh_shadow, "REQUIRED_DISABLED_IDS", frozenset())
    monkeypatch.setattr(dsh_shadow, "DSH_TESTED_VERSION", "fixture-version")
    monkeypatch.setattr(dsh_shadow, "parse_dumped_config", lambda _: (
        support.ConfigEntry("system-prompt", "fixture", False),))
    result = dsh_shadow.audit_dumped_config("NETOPYU_L1_SHADOW_SYSTEM_PROMPT", dsh_version="fixture-version")
    assert result.active_ids == ("system-prompt",)
    assert support.DSH_TESTED_VERSION == "0.1.1-rc.2"
    assert "tool-bash" in support.REQUIRED_DISABLED_IDS


def test_shadow_adapter_keeps_old_binary_and_node_patch_points(tmp_path, monkeypatch):
    from evaluation import dsh_shadow
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "fake-dsh"
    executable.touch(mode=0o700)
    monkeypatch.setattr(dsh_shadow, "_default_dsh_binary", lambda: executable)
    monkeypatch.setattr(dsh_shadow, "_node_path", lambda: "/fixture/node-path")
    monkeypatch.setattr(dsh_shadow.DSHShadowAdapter, "_run", lambda *_args, **_kwargs: ("fixture\n", "", 0))
    monkeypatch.setattr(dsh_shadow, "audit_dumped_config", lambda *_args, **_kwargs: "fixture-audit")
    with dsh_shadow.DSHShadowAdapter(project_root=root, model="fixture", base_url="http://127.0.0.1") as adapter:
        assert adapter.dsh_binary == executable
        assert adapter.environment["PATH"] == "/fixture/node-path"
        assert adapter.audit == "fixture-audit"
