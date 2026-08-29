"""P1.9-B2 DSH/Hermes adapter-hook parity over a sealed private holdout."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import Field, model_validator

from hermes_adapter.client import HermesWorkerClient
from hermes_adapter.plugin import HermesAdapterConfig, NetOpYuHermesAdapter
from network_runtime.contracts import sha256_json

from .catalog_gate import check_baseline
from .contracts import StrictModel
from .holdout import HoldoutCase, load_consensus
from .qualification import _resolve_model_artifact_digest, _rate
from .service import _selection_client


ADAPTER_QUALIFICATION_SCHEMA = "netopyu.io/l1-adapter-qualification/v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = _PROJECT_ROOT / "dsh-plugin-netopyu" / "test" / "l1-qualification-driver.mjs"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROFILES = ("lan", "dc", "wan")
_MAX_REPORT_FAILURES = 200


class AdapterReceipt(StrictModel):
    case_digest: str
    repetition: int = Field(ge=1, le=5)
    profile: Literal["lan", "dc", "wan"]
    harness: Literal["dsh", "hermes"]
    status: Literal["decided", "policy_terminal", "protocol_failure"]
    decision_digest: str | None
    prompt_digest: str
    catalog_digest: str
    candidate_digest: str
    policy_digest: str
    model: str | None
    protocol_valid: bool

    @model_validator(mode="after")
    def validate_digests(self) -> "AdapterReceipt":
        for value in (
            self.case_digest, self.prompt_digest, self.catalog_digest,
            self.candidate_digest, self.policy_digest,
        ):
            if not _DIGEST.fullmatch(value):
                raise ValueError("adapter qualification digest is invalid")
        if self.decision_digest is not None and (
            not _DIGEST.fullmatch(self.decision_digest)
        ):
            raise ValueError("adapter qualification Decision digest is invalid")
        if self.status == "protocol_failure" and self.decision_digest is not None:
            raise ValueError("protocol failure cannot carry a Decision digest")
        if self.status != "protocol_failure" and self.decision_digest is None:
            raise ValueError("valid adapter receipt requires a Decision digest")
        return self


class _HermesHookContext:
    """Minimum public PluginContext surface needed to execute official hooks."""

    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.commands: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, str] = {}
        self.hooks: dict[str, list[Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[str(definition["name"])] = definition

    def register_command(self, name: str, handler: Any, description: str = "") -> None:
        self.commands[name] = {"handler": handler, "description": description}

    def register_skill(self, name: str, path: str) -> None:
        self.skills[name] = path

    def register_hook(self, name: str, handler: Any) -> None:
        self.hooks.setdefault(name, []).append(handler)


def _case_digest(case: HoldoutCase) -> str:
    return sha256_json({"case_id": case.case_id})


def _receipt(
    *, case: HoldoutCase, repetition: int, harness: str, envelope: dict[str, Any],
) -> AdapterReceipt:
    evidence = envelope.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("adapter qualification evidence is missing")
    return AdapterReceipt(
        case_digest=_case_digest(case),
        repetition=repetition,
        profile=envelope.get("profile"),
        harness=harness,
        status=envelope.get("status"),
        decision_digest=envelope.get("decision_digest"),
        prompt_digest=evidence.get("prompt_digest"),
        catalog_digest=evidence.get("catalog_digest"),
        candidate_digest=evidence.get("candidate_digest"),
        policy_digest=evidence.get("policy_digest"),
        model=evidence.get("model"),
        protocol_valid=bool(evidence.get("protocol_valid")),
    )


def _worker_environment(root: Path, *, base_url: str, api_key: str | None, repair_limit: int) -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "NODE_OPTIONS", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONBREAKPOINT",
        "LD_PRELOAD", "DYLD_INSERT_LIBRARIES",
    ):
        environment.pop(key, None)
    environment.update({
        "NETOPYU_BACKEND": "mock",
        "NETOPYU_DSH_BACKEND": "mock",
        "NETOPYU_L1_DECISION_BASE_URL": base_url,
        "NETOPYU_L1_DECISION_REPAIR_LIMIT": str(repair_limit),
        "NETOPYU_L1_DECISION_STORE": str(root / "decisions.sqlite"),
        "NETOPYU_NETWORK_RUNTIME_STORE": str(root / "runtime.sqlite"),
        "NETOPYU_DSH_NETWORK_RUNTIME_STORE": str(root / "runtime.sqlite"),
        "NETOPYU_TOOL_RESULT_STORE": str(root / "results.sqlite"),
        "NETOPYU_DSH_TOOL_RESULT_STORE": str(root / "results.sqlite"),
        "NETOPYU_DSH_HITL_STORE": str(root / "hitl.sqlite"),
        "NETOPYU_DSH_WORKER_SOCKET": str(root / "worker.sock"),
        "PYTHONPATH": os.pathsep.join(filter(None, (
            str(_PROJECT_ROOT), os.environ.get("PYTHONPATH"),
        ))),
    })
    if api_key:
        environment["NETOPYU_L1_DECISION_API_KEY"] = api_key
    return environment


@contextmanager
def _running_worker(
    root: Path, environment: dict[str, str], *, timeout_seconds: float,
) -> Iterator[HermesWorkerClient]:
    socket_path = root / "worker.sock"
    log_path = root / "worker.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "dsh_adapter.worker", "--socket", str(socket_path)],
            cwd=_PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        client = HermesWorkerClient(socket_path, timeout_seconds=timeout_seconds)
        deadline = time.monotonic() + min(30.0, timeout_seconds)
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("qualification Worker stopped during startup")
                try:
                    client.ping()
                    break
                except Exception:
                    time.sleep(0.05)
            else:
                raise TimeoutError("qualification Worker startup timed out")
            yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def _run_dsh_hooks(
    *,
    cases: tuple[HoldoutCase, ...],
    model: str,
    repetitions: int,
    node_executable: str,
    environment: dict[str, str],
    case_timeout_seconds: float,
) -> list[AdapterReceipt]:
    receipts: list[AdapterReceipt] = []
    for profile in _PROFILES:
        selected = [case for case in cases if case.profile == profile]
        if not selected:
            continue
        case_payload = [
            {
                "case_digest": _case_digest(case),
                "prompt": case.prompt,
                "repetition": repetition,
            }
            for case in sorted(selected, key=lambda item: item.case_id)
            for repetition in range(1, repetitions + 1)
        ]
        result = subprocess.run(
            [node_executable, str(_DRIVER)],
            cwd=_PROJECT_ROOT,
            env=environment,
            input=json.dumps({
                "profile": profile, "model": model, "cases": case_payload,
            }, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=case_timeout_seconds * max(1, len(case_payload)) + 60,
            check=False,
        )
        if result.returncode != 0 or len(result.stdout.encode("utf-8")) > 4_000_000:
            raise RuntimeError("DSH adapter qualification driver failed")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("DSH adapter qualification output is invalid") from error
        raw_receipts = payload.get("receipts") if isinstance(payload, dict) else None
        if not isinstance(raw_receipts, list) or len(raw_receipts) != len(case_payload):
            raise ValueError("DSH adapter qualification receipt coverage is invalid")
        receipts.extend(AdapterReceipt.model_validate(item) for item in raw_receipts)
    return receipts


def _run_hermes_hooks(
    *,
    cases: tuple[HoldoutCase, ...],
    model: str,
    repetitions: int,
    client: HermesWorkerClient,
    socket_path: Path,
    case_timeout_seconds: float,
) -> list[AdapterReceipt]:
    receipts: list[AdapterReceipt] = []
    for profile in _PROFILES:
        selected = [case for case in cases if case.profile == profile]
        if not selected:
            continue
        adapter = NetOpYuHermesAdapter(
            client,
            HermesAdapterConfig(
                profile=profile,
                socket_path=socket_path,
                include_destructive=False,
                operator_id="qualification:hermes",
                own_agent_id=f"qualification-hermes-{profile}",
                peer_urls=(),
                timeout_seconds=case_timeout_seconds,
                decision_mode="shadow",
                decision_model=model,
            ),
        )
        context = _HermesHookContext()
        adapter.register(context)
        pre_hooks = context.hooks.get("pre_llm_call", [])
        post_hooks = context.hooks.get("post_llm_call", [])
        if len(pre_hooks) != 1 or len(post_hooks) != 1:
            raise RuntimeError("Hermes qualification hooks are missing or ambiguous")
        for case in sorted(selected, key=lambda item: item.case_id):
            for repetition in range(1, repetitions + 1):
                session_id = (
                    f"qualification:hermes:{_case_digest(case)[-20:]}:{repetition}"
                )
                pre_hooks[0](session_id=session_id, user_message=case.prompt)
                history = client.request(
                    "l1-decision-recent",
                    profile=profile,
                    args={"limit": 2, "session_id": session_id},
                )
                if (
                    not isinstance(history, dict)
                    or history.get("count") != 1
                    or not isinstance(history.get("decisions"), list)
                ):
                    raise RuntimeError("Hermes qualification Decision receipt is missing")
                envelope = history["decisions"][0].get("envelope")
                if not isinstance(envelope, dict):
                    raise ValueError("Hermes qualification envelope is invalid")
                receipts.append(_receipt(
                    case=case,
                    repetition=repetition,
                    harness="hermes",
                    envelope=envelope,
                ))
                post_hooks[0](session_id=session_id)
    return receipts


def _receipt_index(receipts: list[AdapterReceipt]) -> dict[tuple[str, int], AdapterReceipt]:
    output: dict[tuple[str, int], AdapterReceipt] = {}
    for receipt in receipts:
        key = (receipt.case_digest, receipt.repetition)
        if key in output:
            raise ValueError("adapter qualification receipt is duplicated")
        output[key] = receipt
    return output


def _build_report(
    *,
    cases: tuple[HoldoutCase, ...],
    repetitions: int,
    model: str,
    model_artifact_digest: str,
    adjudication: dict[str, Any],
    catalog_gate: dict[str, Any],
    dsh_receipts: list[AdapterReceipt],
    hermes_receipts: list[AdapterReceipt],
) -> dict[str, Any]:
    expected_keys = {
        (_case_digest(case), repetition)
        for case in cases
        for repetition in range(1, repetitions + 1)
    }
    dsh = _receipt_index(dsh_receipts)
    hermes = _receipt_index(hermes_receipts)
    coverage = set(dsh) == expected_keys and set(hermes) == expected_keys
    input_matches = 0
    semantic_matches = 0
    prompt_bindings = 0
    protocol_successes = 0
    failed_pair_count = 0
    failures: list[dict[str, Any]] = []
    case_by_digest = {_case_digest(case): case for case in cases}
    for key in sorted(expected_keys):
        left = dsh.get(key)
        right = hermes.get(key)
        if left is None or right is None:
            continue
        case = case_by_digest[key[0]]
        expected_prompt_digest = sha256_json({"direct_user_text": case.prompt})
        prompt_match = (
            left.prompt_digest == expected_prompt_digest
            and right.prompt_digest == expected_prompt_digest
        )
        model_match = (
            left.model == right.model
            and (
                left.model == model
                or (
                    left.model is None
                    and left.status == right.status == "policy_terminal"
                )
            )
        )
        input_match = (
            left.profile == right.profile == case.profile
            and left.prompt_digest == right.prompt_digest
            and left.catalog_digest == right.catalog_digest
            and left.candidate_digest == right.candidate_digest
            and left.policy_digest == right.policy_digest
            and model_match
        )
        semantic_match = (
            left.status == right.status
            and left.decision_digest == right.decision_digest
            and left.protocol_valid == right.protocol_valid
        )
        protocol_success = (
            left.status != "protocol_failure"
            and right.status != "protocol_failure"
            and left.protocol_valid
            and right.protocol_valid
        )
        prompt_bindings += int(prompt_match)
        input_matches += int(input_match)
        semantic_matches += int(semantic_match)
        protocol_successes += int(protocol_success)
        failed = [
            name for name, passed in (
                ("prompt_binding", prompt_match),
                ("input_contract", input_match),
                ("decision_digest", semantic_match),
                ("protocol", protocol_success),
            ) if not passed
        ]
        if failed and len(failures) < _MAX_REPORT_FAILURES:
            failures.append({
                "case_digest": key[0],
                "repetition": key[1],
                "profile": case.profile,
                "category": case.category,
                "language": case.language,
                "failed_gates": failed,
            })
        failed_pair_count += int(bool(failed))
    denominator = len(expected_keys)
    repeatability: dict[str, Any] = {"evaluated": repetitions >= 2}
    for harness, values in (("dsh", dsh), ("hermes", hermes)):
        if repetitions < 2:
            repeatability[harness] = None
            continue
        stable = 0
        for case_digest in sorted({_case_digest(case) for case in cases}):
            decisions = {
                (
                    values[(case_digest, repetition)].status,
                    values[(case_digest, repetition)].decision_digest,
                    values[(case_digest, repetition)].protocol_valid,
                )
                for repetition in range(1, repetitions + 1)
                if (case_digest, repetition) in values
            }
            stable += int(len(decisions) == 1)
        repeatability[harness] = _rate(stable, len(cases))
    metrics = {
        "pair_count": denominator,
        "prompt_binding_rate": _rate(prompt_bindings, denominator),
        "input_contract_parity_rate": _rate(input_matches, denominator),
        "decision_digest_parity_rate": _rate(semantic_matches, denominator),
        "protocol_success_rate": _rate(protocol_successes, denominator),
        "repeatability": repeatability,
    }
    requirements = {
        "sealed_consensus_ready": bool(adjudication.get("ready_for_holdout_run")),
        "catalog_baseline_clean": bool(catalog_gate.get("ok")),
        "immutable_model_artifact": model_artifact_digest != "unresolved",
        "exact_case_coverage": coverage,
        "prompt_binding": metrics["prompt_binding_rate"] == 1.0,
        "input_contract_parity": metrics["input_contract_parity_rate"] == 1.0,
        "decision_digest_parity": metrics["decision_digest_parity_rate"] == 1.0,
        "protocol_success": metrics["protocol_success_rate"] == 1.0,
        "repeatability_when_requested": (
            repetitions < 2
            or all(repeatability[harness] == 1.0 for harness in ("dsh", "hermes"))
        ),
    }
    passed = all(requirements.values())
    body = {
        "scope": {
            "level": "adapter_hook_to_worker",
            "dsh_javascript_agent_pre_step": True,
            "hermes_python_pre_llm_call": True,
            "persistent_owner_only_worker": True,
            "real_dsh_web_process": False,
            "real_hermes_process": False,
            "note": (
                "This executes the production adapter hook code against a real temporary Worker. "
                "It does not certify DSH Web, Hermes CLI, UI composition, or deployment identity."
            ),
        },
        "status": "adapter_parity_passed" if passed else "adapter_parity_failed",
        "passed": passed,
        "case_count": len(cases),
        "repetitions": repetitions,
        "model": model,
        "model_artifact_digest": model_artifact_digest,
        "sealed_manifest_digest": adjudication.get("sealed_manifest_digest"),
        "consensus_labels_digest": adjudication.get("consensus_labels_digest"),
        "catalog_snapshot_digest": catalog_gate.get("current_snapshot_digest"),
        "requirements": requirements,
        "metrics": metrics,
        "failures": {
            "count": failed_pair_count,
            "details_truncated": failed_pair_count > len(failures),
            "cases": failures,
        },
        "privacy": {
            "private_prompts_transport": "stdin_and_in_process_only",
            "raw_prompts_emitted": False,
            "raw_labels_emitted": False,
            "argument_values_emitted": False,
        },
        "claim_boundary": (
            "Passing proves parity for this adapter/Worker/model/catalog artifact and sealed set; "
            "it is not a production success probability or full Harness certification."
        ),
    }
    return {
        "apiVersion": ADAPTER_QUALIFICATION_SCHEMA,
        **body,
        "report_digest": sha256_json(body),
    }


def _qualify_adapter_material(
    *,
    cases: tuple[HoldoutCase, ...],
    model: str,
    model_artifact_digest: str | None,
    adjudication: dict[str, Any],
    catalog_gate: dict[str, Any],
    repetitions: int = 1,
    repair_limit: int = 1,
    case_timeout_seconds: float = 120.0,
    node_executable: str | None = None,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("adapter qualification cases are empty")
    if not 1 <= repetitions <= 5:
        raise ValueError("adapter qualification repetitions must be 1..5")
    if repair_limit not in {0, 1}:
        raise ValueError("adapter qualification repair limit must be zero or one")
    if not 1 <= case_timeout_seconds <= 600:
        raise ValueError("adapter qualification case timeout must be 1..600 seconds")
    if not adjudication.get("ready_for_holdout_run"):
        raise ValueError("adapter qualification requires resolved reviewer consensus")
    client = _selection_client(model)
    artifact_digest = _resolve_model_artifact_digest(client, model_artifact_digest)
    node = node_executable or os.getenv("NETOPYU_NODE") or shutil.which("node")
    node_path = Path(node).expanduser().resolve() if node else None
    if (
        node_path is None
        or not node_path.is_file()
        or not os.access(node_path, os.X_OK)
    ):
        raise RuntimeError("adapter qualification requires a Node executable")
    with tempfile.TemporaryDirectory(prefix="nq-", dir="/tmp") as directory:
        root = Path(directory)
        environment = _worker_environment(
            root,
            base_url=client.base_url,
            api_key=client.api_key,
            repair_limit=repair_limit,
        )
        with _running_worker(
            root, environment,
            timeout_seconds=case_timeout_seconds * (repair_limit + 1) + 10,
        ) as worker:
            dsh_receipts = _run_dsh_hooks(
                cases=cases,
                model=model,
                repetitions=repetitions,
                node_executable=str(node_path),
                environment=environment,
                case_timeout_seconds=case_timeout_seconds,
            )
            hermes_receipts = _run_hermes_hooks(
                cases=cases,
                model=model,
                repetitions=repetitions,
                client=worker,
                socket_path=root / "worker.sock",
                case_timeout_seconds=case_timeout_seconds,
            )
    return _build_report(
        cases=cases,
        repetitions=repetitions,
        model=model,
        model_artifact_digest=artifact_digest,
        adjudication=adjudication,
        catalog_gate=catalog_gate,
        dsh_receipts=dsh_receipts,
        hermes_receipts=hermes_receipts,
    )


def qualify_adapter_files(
    *,
    cases_path: Path,
    manifest_path: Path,
    first_labels_path: Path,
    second_labels_path: Path,
    model: str,
    model_artifact_digest: str | None = None,
    repetitions: int = 1,
    repair_limit: int = 1,
    case_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    if os.getenv("NETOPYU_BACKEND", "mock").strip().casefold() != "mock":
        raise RuntimeError("adapter qualification requires NETOPYU_BACKEND=mock")
    cases, _labels, adjudication = load_consensus(
        cases_path, manifest_path, first_labels_path, second_labels_path,
    )
    catalog_gate = check_baseline()
    if not catalog_gate["ok"]:
        raise RuntimeError("adapter qualification Catalog gate failed")
    return _qualify_adapter_material(
        cases=cases,
        model=model,
        model_artifact_digest=model_artifact_digest,
        adjudication=adjudication,
        catalog_gate=catalog_gate,
        repetitions=repetitions,
        repair_limit=repair_limit,
        case_timeout_seconds=case_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reviewer_one", type=Path)
    parser.add_argument("reviewer_two", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-artifact-digest")
    parser.add_argument("--repetitions", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--repair-limit", type=int, choices=(0, 1), default=1)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = qualify_adapter_files(
            cases_path=arguments.cases,
            manifest_path=arguments.manifest,
            first_labels_path=arguments.reviewer_one,
            second_labels_path=arguments.reviewer_two,
            model=arguments.model,
            model_artifact_digest=arguments.model_artifact_digest,
            repetitions=arguments.repetitions,
            repair_limit=arguments.repair_limit,
            case_timeout_seconds=arguments.case_timeout,
        )
    except Exception as error:
        print(json.dumps({
            "apiVersion": ADAPTER_QUALIFICATION_SCHEMA,
            "status": "error",
            "error_type": type(error).__name__,
            "privacy": "error output omits private input values",
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
