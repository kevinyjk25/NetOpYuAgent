"""Safely maintain the NetOpYu-owned portion of DSH settings."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


PROVIDER_ID = "netopyu-ollama"


def _model(model_id: str, *, thinking: bool) -> dict[str, Any]:
    model: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "contextWindow": 32768,
        "maxTokens": 4096,
    }
    if thinking:
        model["compat"] = {"thinkingFormat": "deepseek"}
    return model


def sync_settings(
    path: Path,
    *,
    base_url: str,
    primary_model: str,
    fast_model: str,
    default_model: str | None = None,
    default_preset: str | None = None,
) -> dict[str, Any]:
    """Merge NetOpYu's provider without discarding unrelated DSH settings."""
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings = loaded if isinstance(loaded, dict) else {}
        previous_mode = path.stat().st_mode & 0o777
    else:
        settings = {}
        previous_mode = 0o600

    llm = settings.setdefault("llm-pi-ai", {})
    providers = llm.setdefault("providers", {})
    provider = providers.setdefault(PROVIDER_ID, {})
    provider.update({
        "displayName": "NetOpYu Ollama",
        "apiKeyEnv": "NETOPYU_OLLAMA_API_KEY",
        "api": "openai-completions",
        "baseURL": f"{base_url.rstrip('/')}/v1",
        "compat": {
            "supportsDeveloperRole": False,
            "maxTokensField": "max_tokens",
        },
    })

    existing = {
        item.get("id"): item
        for item in provider.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    existing[primary_model] = _model(primary_model, thinking=True)
    existing[fast_model] = _model(fast_model, thinking=False)
    provider["models"] = list(existing.values())

    if default_model is not None:
        if default_model not in existing:
            raise ValueError(f"model {default_model!r} is not configured for {PROVIDER_ID}")
        settings["agent-default-model"] = {
            "provider": PROVIDER_ID,
            "model": default_model,
        }
    elif "agent-default-model" not in settings:
        settings["agent-default-model"] = {
            "provider": PROVIDER_ID,
            "model": primary_model,
        }

    if default_preset is not None:
        settings.setdefault("agent-presets", {})["default"] = default_preset

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, previous_mode)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--primary-model", required=True)
    parser.add_argument("--fast-model", required=True)
    parser.add_argument("--default-model")
    parser.add_argument("--default-preset")
    args = parser.parse_args()
    settings = sync_settings(
        args.path,
        base_url=args.base_url,
        primary_model=args.primary_model,
        fast_model=args.fast_model,
        default_model=args.default_model,
        default_preset=args.default_preset,
    )
    default = settings["agent-default-model"]
    print(f"{default['provider']}/{default['model']}")


if __name__ == "__main__":
    main()
