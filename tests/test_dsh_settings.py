from pathlib import Path

import yaml

from dsh_adapter.settings import sync_settings


def test_sync_preserves_unrelated_settings_and_existing_models(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "ui-onboarding:\n  welcomeNoticeVersion: keep-me\n"
        "llm-pi-ai:\n  providers:\n    netopyu-ollama:\n      models:\n"
        "        - id: custom-model\n          name: Custom\n",
        encoding="utf-8",
    )

    settings = sync_settings(
        path,
        base_url="http://127.0.0.1:11434/",
        primary_model="qwen3.5:27b",
        fast_model="qwen2.5:7b",
    )

    assert settings["ui-onboarding"]["welcomeNoticeVersion"] == "keep-me"
    provider = settings["llm-pi-ai"]["providers"]["netopyu-ollama"]
    assert provider["baseURL"] == "http://127.0.0.1:11434/v1"
    assert [model["id"] for model in provider["models"]] == [
        "custom-model", "qwen3.5:27b", "qwen2.5:7b",
    ]
    assert settings["agent-default-model"]["model"] == "qwen3.5:27b"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == settings


def test_explicit_default_switch_is_required(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    first = sync_settings(
        path,
        base_url="http://localhost:11434",
        primary_model="qwen3.5:27b",
        fast_model="qwen2.5:7b",
    )
    assert first["agent-default-model"]["model"] == "qwen3.5:27b"

    switched = sync_settings(
        path,
        base_url="http://localhost:11434",
        primary_model="qwen3.5:27b",
        fast_model="qwen2.5:7b",
        default_model="qwen2.5:7b",
    )
    assert switched["agent-default-model"]["model"] == "qwen2.5:7b"

    preserved = sync_settings(
        path,
        base_url="http://localhost:11434",
        primary_model="qwen3.5:27b",
        fast_model="qwen2.5:7b",
    )
    assert preserved["agent-default-model"]["model"] == "qwen2.5:7b"


def test_preset_switch_is_explicit_and_preserved(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    sync_settings(
        path,
        base_url="http://localhost:11434",
        primary_model="qwen3.5:27b",
        fast_model="qwen2.5:7b",
        default_preset="minimal",
    )
    preserved = sync_settings(
        path,
        base_url="http://localhost:11434",
        primary_model="qwen3.5:27b",
        fast_model="qwen2.5:7b",
    )
    assert preserved["agent-presets"]["default"] == "minimal"
