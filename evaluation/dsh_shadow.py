"""Fail-closed DSH-in-the-loop shadow benchmark for P1.8-B1.

The DSH headless Agent/Session/LLM loop is real.  All tool providers, Skills,
shells, filesystem tools, Web access, subagents, and NetOpYu effects are absent.
The only accepted result is a strict, non-authoritative ``L1Decision`` proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dsh_adapter.settings import sync_settings
from network_runtime.contracts import sha256_json

from .l1_adapters import AdapterResponse, SYSTEM_PROMPT, decision_prompt, prompt_fingerprint
from .l1_benchmark import (
    DEFAULT_THRESHOLDS,
    _append_checkpoint,
    _history,
    _initialize_checkpoint,
    _read_checkpoint,
    _write_report,
    aggregate,
    evaluator_fingerprint,
    qualify,
    score_case,
)
from .l1_catalog import L1CandidateRetriever, L1CatalogEntry, build_profile_catalog, catalog_digest
from .l1_contract import L1Action, L1Category, L1Decision, L1Scenario
from .l1_scenarios import build_l1_scenarios, scenario_set_digest


DSH_SHADOW_REPORT_SCHEMA = "netopyu.io/l1-dsh-shadow-report/v1"
DSH_TESTED_VERSION = "0.1.1-rc.2"
MAX_PROCESS_OUTPUT = 2_000_000

# This is deliberately exact.  A DSH upgrade or a newly activated plugin must
# be reviewed before the shadow evaluator is allowed to contact a model.
SAFE_ACTIVE_IDS = frozenset({
    "timer",
    "llm",
    "session",
    "typert",
    "typert-loader",
    "typert-gateway",
    "session-title",
    "agent",
    "agent-default-model",
    "settings",
    "credentials",
    "llm-pi-ai",
    "session-persistence-jsonl",
    "session-query-sqlite",
    "session-projection",
    "token-meter",
    "compaction-basic",
    "timeout-policy",
    "spill-local",
    "spill-policy",
    "session-checkpoint-policy",
    "repeat-tool-reminder",
    "tools",
    "system-prompt",
    "agent-loop",
    "headless-startup",
    "headless-runner",
})

REQUIRED_DISABLED_IDS = frozenset({
    "session-title-llm",
    "user-questions",
    "llm-retry",
    "attachment-local",
    "session-telemetry-otel",
    "jobs",
    "subprocess",
    "sandbox",
    "sandbox-policy",
    "bash-sandbox",
    "pwsh-sandbox",
    "approval",
    "permission",
    "shell-env",
    "fs-observation-policy",
    "agent-instructions",
    "skill",
    "skill-filesystem",
    "skill-badge",
    "commands",
    "command-feedback",
    "command-compact",
    "goal",
    "goal-round-driver",
    "command-goal",
    "plan-mode",
    "subagent",
    "subagent-spawn-in-process",
    "subagent-fork-in-process",
    "workflow-worker-thread",
    "web",
    "web-search-deepseek",
    "llm-deepseek",
    "code-runtime",
    "fs-sandbox",
    "tool-bash",
    "tool-pwsh",
    "tool-jobs",
    "tool-fs",
    "tool-fs-search",
    "tool-skill",
    "tool-subagent-control",
    "tool-subagent-list-agents",
    "tool-subagent",
    "tool-subagent-fork",
    "tool-subagent-report",
    "tool-workflow",
    "tool-result-pruner",
    "tool-todo",
    "tool-goal",
    "tool-ralph",
    "tool-str-replace-editor",
    "tool-web",
})


@dataclass(frozen=True)
class ConfigEntry:
    entry_id: str
    plugin_name: str
    disabled: bool


@dataclass(frozen=True)
class ConfigAudit:
    dsh_version: str
    config_digest: str
    active_ids: tuple[str, ...]
    disabled_ids: tuple[str, ...]


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def parse_dumped_config(text: str) -> tuple[ConfigEntry, ...]:
    """Parse only entry identity/name/disabled state from DSH's JS-YAML dump."""
    starts = list(re.finditer(r"^- id: ([^\n]+)$", text, re.MULTILINE))
    entries: list[ConfigEntry] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start():end]
        name = re.search(r"^  name: ['\"]?([^'\"\n]+)['\"]?$", block, re.MULTILINE)
        entries.append(ConfigEntry(
            entry_id=match.group(1).strip(),
            plugin_name=name.group(1).strip() if name else "",
            disabled=bool(re.search(r"^  disabled: true$", block, re.MULTILINE)),
        ))
    if not entries:
        raise ValueError("DSH shadow config contains no entries")
    if len({item.entry_id for item in entries}) != len(entries):
        raise ValueError("DSH shadow config contains duplicate entry ids")
    return tuple(entries)


def audit_dumped_config(text: str, *, dsh_version: str) -> ConfigAudit:
    if dsh_version != DSH_TESTED_VERSION:
        raise ValueError(
            f"DSH shadow requires reviewed version {DSH_TESTED_VERSION}, got {dsh_version}"
        )
    entries = parse_dumped_config(text)
    by_id = {item.entry_id: item for item in entries}
    active = {item.entry_id for item in entries if not item.disabled}
    disabled = {item.entry_id for item in entries if item.disabled}
    missing = REQUIRED_DISABLED_IDS - disabled
    if missing:
        raise ValueError(
            "DSH shadow config did not disable required entries: "
            + ", ".join(sorted(missing))
        )
    if active != SAFE_ACTIVE_IDS:
        unexpected = active - SAFE_ACTIVE_IDS
        absent = SAFE_ACTIVE_IDS - active
        details = []
        if unexpected:
            details.append("unexpected active=" + ",".join(sorted(unexpected)))
        if absent:
            details.append("missing reviewed=" + ",".join(sorted(absent)))
        raise ValueError("DSH shadow active-entry allowlist mismatch: " + "; ".join(details))
    if any(not item.disabled and item.entry_id.startswith("tool-") for item in entries):
        raise ValueError("DSH shadow config exposes an active tool provider")
    system = by_id.get("system-prompt")
    if system is None or system.disabled or "NETOPYU_L1_SHADOW_SYSTEM_PROMPT" not in text:
        raise ValueError("DSH shadow system prompt is not bound to the reviewed environment input")
    return ConfigAudit(
        dsh_version=dsh_version,
        config_digest=_digest_bytes(text.encode("utf-8")),
        active_ids=tuple(sorted(active)),
        disabled_ids=tuple(sorted(disabled)),
    )


def _loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("DSH shadow B1 accepts a loopback HTTP model endpoint only")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("DSH shadow model endpoint cannot contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("DSH shadow base URL must not include an API path")
    return base_url.rstrip("/")


def _default_dsh_binary() -> Path:
    configured = os.environ.get("NETOPYU_DSH_BIN")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / "Library/Application Support/NetOpYuAgent/dsh-runtime/node_modules/.bin/dsh"
    )


def _node_path() -> str:
    current = os.environ.get("PATH", "")
    if shutil.which("node", path=current):
        return current
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    node = bundled / "node/bin/node"
    if node.is_file():
        return f"{bundled / 'node/bin'}:{bundled / 'bin/fallback'}:{current}"
    return current


def _validated_decision(
    content: str,
    candidates: tuple[L1CatalogEntry, ...],
) -> L1Decision:
    decision = L1Decision.model_validate_json(content)
    expected_kind = (
        "skill" if decision.action == L1Action.SELECT_SKILL else
        "tool" if decision.action == L1Action.SELECT_TOOL else None
    )
    if decision.target is None:
        return decision
    matching = [
        item for item in candidates
        if item.target == decision.target
        and (expected_kind is None or item.kind == expected_kind)
    ]
    if len(matching) != 1:
        raise ValueError("DSH shadow target is outside supplied candidates")
    selected = matching[0]
    absent = {
        name for name in selected.required_parameters
        if name not in decision.arguments or decision.arguments[name] in (None, "", [])
    }
    if decision.action in {L1Action.SELECT_SKILL, L1Action.SELECT_TOOL}:
        if absent:
            raise ValueError("DSH shadow selection omitted required candidate parameters")
        if decision.workflow != selected.workflow_hint:
            raise ValueError("DSH shadow workflow differs from the candidate contract")
    elif decision.action == L1Action.CLARIFY and set(decision.missing_fields) != absent:
        raise ValueError("DSH shadow clarification does not match required parameters")
    return decision


class DSHShadowAdapter:
    """Official DSH headless loop with zero callable domain/tool surface."""

    name = "dsh-headless-shadow"

    def __init__(
        self,
        *,
        project_root: Path,
        model: str,
        base_url: str,
        dsh_binary: Path | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model.strip():
            raise ValueError("DSH shadow model id is required")
        if not 1 <= timeout_seconds <= 300:
            raise ValueError("DSH shadow timeout must be between 1 and 300 seconds")
        self.project_root = project_root.resolve()
        self.model = model
        self.base_url = _loopback_base_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.dsh_binary = (dsh_binary or _default_dsh_binary()).expanduser().resolve()
        self.patch = self.project_root / "evaluation/dsh_shadow.patch.yml"
        if not self.dsh_binary.is_file() or not os.access(self.dsh_binary, os.X_OK):
            raise ValueError(f"DSH shadow binary is not executable: {self.dsh_binary}")
        if not self.patch.is_file():
            raise ValueError("DSH shadow patch is missing")
        self._temporary_home = tempfile.TemporaryDirectory(prefix="netopyu-dsh-shadow-")
        self.home = Path(self._temporary_home.name)
        fast_model = model if "qwen3" not in model.lower() else "qwen2.5:7b"
        settings_path = self.home / "settings.yaml"
        sync_settings(
            settings_path,
            base_url=self.base_url,
            primary_model=model,
            fast_model=fast_model,
            default_model=model,
        )
        self.settings_digest = _digest_bytes(settings_path.read_bytes())
        self.environment = {
            "PATH": _node_path(),
            "HOME": str(Path.home()),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "DSH_HOME": str(self.home),
            "DSH_PERMISSION_MODE": "read-only",
            "DSH_TELEMETRY_MODE": "DISABLED",
            "DSH_TOOLS_MODE": "native",
            "NETOPYU_L1_SHADOW_SYSTEM_PROMPT": SYSTEM_PROMPT,
            # Ollama ignores the value but DSH requires the configured variable.
            "NETOPYU_OLLAMA_API_KEY": "local-loopback-no-auth",
        }
        version = self._run(("--version",), timeout=10.0)[0].strip().splitlines()[0]
        dumped = self._run((
            "--profile", "headless", "--patch", str(self.patch), "--dump-config",
        ), timeout=30.0)[0]
        self.audit = audit_dumped_config(dumped, dsh_version=version)

    def close(self) -> None:
        self._temporary_home.cleanup()

    def __enter__(self) -> "DSHShadowAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self, arguments: tuple[str, ...], *, timeout: float) -> tuple[str, str, float]:
        argv = (str(self.dsh_binary),) + arguments
        started = time.perf_counter()
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                argv,
                cwd=self.project_root,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
            )
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TimeoutError("DSH shadow process timed out") from error
            elapsed_ms = (time.perf_counter() - started) * 1000
            stdout_file.seek(0, os.SEEK_END)
            stdout_size = stdout_file.tell()
            stderr_file.seek(0, os.SEEK_END)
            stderr_size = stderr_file.tell()
            if stdout_size > MAX_PROCESS_OUTPUT or stderr_size > MAX_PROCESS_OUTPUT:
                raise ValueError("DSH shadow process output exceeds 2 MB")
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read().decode("utf-8", "strict")
            stderr = stderr_file.read().decode("utf-8", "replace")
        if return_code != 0:
            # DSH/model text is deliberately not propagated or persisted.
            raise RuntimeError(f"DSH shadow process failed with exit {return_code}")
        return stdout, stderr, elapsed_ms

    def decide(
        self,
        prompt: str,
        candidates: tuple[L1CatalogEntry, ...],
    ) -> AdapterResponse:
        started = time.perf_counter()
        content = ""
        try:
            stdout, _stderr, elapsed_ms = self._run((
                "--profile", "headless", "--patch", str(self.patch),
                decision_prompt(prompt, candidates),
            ), timeout=self.timeout_seconds)
            content = stdout.strip()
            decision = _validated_decision(content, candidates)
            return AdapterResponse(
                decision=decision,
                error_type=None,
                error=None,
                elapsed_ms=elapsed_ms,
                output_digest=_digest_bytes(content.encode("utf-8")),
            )
        except (OSError, RuntimeError, TimeoutError, TypeError, UnicodeError, ValueError) as error:
            safe_error = (
                "DSH shadow timed out" if isinstance(error, TimeoutError)
                else "DSH shadow process failed" if isinstance(error, (OSError, RuntimeError))
                else "DSH shadow response failed strict validation"
            )
            return AdapterResponse(
                decision=None,
                error_type=type(error).__name__,
                error=safe_error,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                output_digest=(
                    _digest_bytes(content.encode("utf-8")) if content else None
                ),
            )


def shadow_evaluator_fingerprint(project_root: Path) -> str:
    paths = ("evaluation/dsh_shadow.py", "evaluation/dsh_shadow.patch.yml")
    return sha256_json({
        "reference_evaluator": evaluator_fingerprint(),
        **{
            name: _digest_bytes((project_root / name).read_bytes())
            for name in paths
        },
    })


def _reference_delta(
    baseline_path: Path,
    *,
    dataset_complete: bool,
    model: str,
    model_artifact_digest: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    if not dataset_complete:
        return {
            "status": "subset-not-comparable",
            "deltas": {},
            "reference_fingerprint": None,
        }
    if not baseline_path.is_file():
        return {"status": "missing", "deltas": {}, "reference_fingerprint": None}
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    matching = [
        item for item in payload.get("baselines", [])
        if item.get("adapter") == "openai-compatible"
        and item.get("model") == model
        and item.get("model_artifact_digest") == model_artifact_digest
    ]
    if not matching:
        return {"status": "missing", "deltas": {}, "reference_fingerprint": None}
    reference = matching[-1]
    names = (
        "selection_accuracy",
        "parameter_field_f1",
        "clarification_recall",
        "missing_fields_accuracy",
        "workflow_accuracy",
        "safety_escape_rate",
        "end_to_end_accuracy",
    )
    return {
        "status": "compared",
        "deltas": {
            name: round(float(metrics[name]) - float(reference["metrics"][name]), 4)
            for name in names
        },
        "reference_fingerprint": reference.get("fingerprint"),
    }


def run_shadow_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    all_scenarios = build_l1_scenarios()
    selected = tuple(
        item for item in all_scenarios
        if (not arguments.category or item.category.value in arguments.category)
        and (not arguments.language or item.language in arguments.language)
    )
    if arguments.smoke_per_category:
        buckets: dict[L1Category, list[L1Scenario]] = defaultdict(list)
        for item in selected:
            buckets[item.category].append(item)
        selected = tuple(
            item
            for category in L1Category
            for item in buckets[category][:arguments.smoke_per_category]
        )
    if arguments.max_cases:
        selected = selected[:arguments.max_cases]
    if not selected:
        raise ValueError("P1.8-B1 filters selected no scenarios")
    full_dataset = len(selected) == len(all_scenarios) and all(
        left.scenario_id == right.scenario_id
        for left, right in zip(selected, all_scenarios, strict=True)
    )
    if arguments.record and not full_dataset:
        raise ValueError("--record requires the complete unfiltered P1.8-B1 dataset")

    project_root = Path(__file__).resolve().parents[1]
    catalogs = {profile: build_profile_catalog(profile) for profile in ("lan", "dc", "wan")}
    retrievers = {
        profile: L1CandidateRetriever(catalog) for profile, catalog in catalogs.items()
    }
    dataset_digest = scenario_set_digest(all_scenarios)
    combined_catalog_digest = sha256_json({
        profile: catalog_digest(catalog) for profile, catalog in catalogs.items()
    })
    output = Path(arguments.output_dir)
    started = time.perf_counter()
    adapter = DSHShadowAdapter(
        project_root=project_root,
        model=arguments.model,
        base_url=arguments.base_url,
        dsh_binary=Path(arguments.dsh_bin).expanduser() if arguments.dsh_bin else None,
        timeout_seconds=arguments.timeout,
    )
    try:
        fingerprint = sha256_json({
            "adapter": adapter.name,
            "model": adapter.model,
            "model_artifact_digest": arguments.model_artifact_digest or "unresolved",
            "prompt": prompt_fingerprint(),
            "dataset": dataset_digest,
            "catalog": combined_catalog_digest,
            "candidate_top_k": arguments.candidate_top_k,
            "evaluator": shadow_evaluator_fingerprint(project_root),
            "dsh_version": adapter.audit.dsh_version,
            "dsh_config": adapter.audit.config_digest,
            "settings": adapter.settings_digest,
        })
        checkpoint = output / "checkpoint.jsonl"
        cached = _read_checkpoint(checkpoint, fingerprint) if arguments.resume else {}
        selected_ids = {item.scenario_id for item in selected}
        if not set(cached) <= selected_ids:
            raise ValueError("P1.8-B1 checkpoint contains cases outside the selected run")
        if not arguments.resume or not checkpoint.is_file():
            _initialize_checkpoint(checkpoint, fingerprint)

        by_id = dict(cached)
        for scenario in selected:
            if scenario.scenario_id in by_id:
                continue
            candidates = retrievers[scenario.profile].retrieve(
                scenario.prompt, top_k=arguments.candidate_top_k,
            )
            response = adapter.decide(scenario.prompt, candidates)
            score = score_case(
                scenario, response, tuple(item.target for item in candidates),
            )
            by_id[scenario.scenario_id] = score
            _append_checkpoint(checkpoint, score)
        scores = [by_id[item.scenario_id] for item in selected]
        metrics = aggregate(scores)
        qualified, gate_failures = qualify(metrics)
        qualification_eligible = full_dataset and bool(arguments.model_artifact_digest)
        if not full_dataset:
            qualified = False
            gate_failures.insert(0, "qualification requires all 160 unfiltered scenarios")
        elif not arguments.model_artifact_digest:
            qualified = False
            gate_failures.insert(0, "model qualification requires an immutable artifact digest")
        summary = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "adapter": adapter.name,
            "model": adapter.model,
            "model_artifact_digest": arguments.model_artifact_digest or "unresolved",
            "fingerprint": fingerprint,
            "metrics": {
                name: metrics[name] for name in (
                    "strict_output_rate", "candidate_recall", "action_accuracy",
                    "selection_accuracy", "parameter_field_f1",
                    "clarification_recall", "missing_fields_accuracy",
                    "workflow_accuracy", "safety_escape_rate", "end_to_end_accuracy",
                )
            },
        }
        history = _history(output / "history.jsonl", summary, record=arguments.record)
        reference = _reference_delta(
            Path(arguments.baseline),
            dataset_complete=full_dataset,
            model=adapter.model,
            model_artifact_digest=arguments.model_artifact_digest,
            metrics=metrics,
        )
        report = {
            "apiVersion": DSH_SHADOW_REPORT_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "adapter": adapter.name,
            "model": adapter.model,
            "model_artifact_digest": arguments.model_artifact_digest or "unresolved",
            "scope": "dsh-headless-agent-session-llm-no-tools-no-effects",
            "dataset_cases": len(all_scenarios),
            "evaluated_cases": len(selected),
            "resumed_cases": len(cached),
            "dataset_complete": full_dataset,
            "qualification_eligible": qualification_eligible,
            "dataset_digest": dataset_digest,
            "catalog_digest": combined_catalog_digest,
            "evaluator_fingerprint": shadow_evaluator_fingerprint(project_root),
            "prompt_fingerprint": prompt_fingerprint(),
            "fingerprint": fingerprint,
            "candidate_top_k": arguments.candidate_top_k,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "thresholds": DEFAULT_THRESHOLDS,
            "qualified": qualified,
            "gate_failures": gate_failures,
            "metrics": metrics,
            "trend": history,
            "reference_adapter_comparison": reference,
            "dsh": {
                "version": adapter.audit.dsh_version,
                "config_digest": adapter.audit.config_digest,
                "settings_digest": adapter.settings_digest,
                "active_ids": adapter.audit.active_ids,
                "disabled_ids": adapter.audit.disabled_ids,
                "ephemeral_home_removed_after_run": True,
            },
            "cases": [asdict(item) for item in scores],
        }
        report["artifacts"] = _write_report(report, output)
        return report
    finally:
        adapter.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base-url", default="http://127.0.0.1:11434")
    value.add_argument("--model", default="qwen2.5:7b")
    value.add_argument("--model-artifact-digest", default="")
    value.add_argument("--dsh-bin", default="")
    value.add_argument("--timeout", type=float, default=60.0)
    value.add_argument("--candidate-top-k", type=int, default=12)
    value.add_argument("--max-cases", type=int, default=0)
    value.add_argument("--smoke-per-category", type=int, default=0)
    value.add_argument("--category", action="append", choices=[item.value for item in L1Category])
    value.add_argument("--language", action="append", choices=["zh", "en", "mixed"])
    value.add_argument("--output-dir", default="artifacts/l1-dsh-shadow")
    value.add_argument("--baseline", default="data/l1_model_baselines.json")
    value.add_argument("--record", action="store_true")
    value.add_argument("--gate", action="store_true")
    value.add_argument("--resume", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if not 1 <= arguments.candidate_top_k <= 32:
        raise SystemExit("--candidate-top-k must be between 1 and 32")
    if not 0 <= arguments.max_cases <= 160:
        raise SystemExit("--max-cases must be between 0 and 160")
    if not 0 <= arguments.smoke_per_category <= 10:
        raise SystemExit("--smoke-per-category must be between 0 and 10")
    if arguments.max_cases and arguments.smoke_per_category:
        raise SystemExit("--max-cases and --smoke-per-category are mutually exclusive")
    if arguments.model_artifact_digest and not re.fullmatch(
        r"sha256:[0-9a-f]{64}", arguments.model_artifact_digest
    ):
        raise SystemExit("--model-artifact-digest must be sha256:<64 lowercase hex>")
    report = run_shadow_benchmark(arguments)
    print(json.dumps({
        "ok": True,
        "qualified": report["qualified"],
        "qualification_eligible": report["qualification_eligible"],
        "gate_failures": report["gate_failures"],
        "adapter": report["adapter"],
        "model": report["model"],
        "evaluated_cases": report["evaluated_cases"],
        "resumed_cases": report["resumed_cases"],
        "metrics": report["metrics"],
        "reference_adapter_comparison": report["reference_adapter_comparison"],
        "dsh": report["dsh"],
        "artifacts": report["artifacts"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if arguments.gate and not report["qualified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfigAudit",
    "ConfigEntry",
    "DSHShadowAdapter",
    "DSH_SHADOW_REPORT_SCHEMA",
    "DSH_TESTED_VERSION",
    "REQUIRED_DISABLED_IDS",
    "SAFE_ACTIVE_IDS",
    "audit_dumped_config",
    "parse_dumped_config",
    "run_shadow_benchmark",
    "shadow_evaluator_fingerprint",
]
