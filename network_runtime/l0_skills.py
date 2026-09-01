"""First-class deterministic Network L0 Skill and intent contracts.

An LLM may select one of these skills and propose arguments.  It cannot alter
the step graph, desired state, verifier, compensation policy or failure mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import PlanIntegrityError, sha256_json
from .policies import reviewed_contracts


INTENT_SCHEMA_VERSION = 1
L0_SKILL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class L0StepContract:
    step_id: str
    phase: str
    condition: str
    failure_action: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class L0SkillContract:
    skill_id: str
    version: str
    tool_name: str
    tool_contract_id: str
    intent_kind: str
    target_fields: tuple[str, ...]
    allowed_profiles: tuple[str, ...]
    steps: tuple[L0StepContract, ...]
    contract_hash: str
    schema_version: int = L0_SKILL_SCHEMA_VERSION
    compiled_contract: Any | None = field(default=None, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        skill_id: str,
        version: str,
        tool_name: str,
        tool_contract_id: str,
        intent_kind: str,
        target_fields: tuple[str, ...],
        allowed_profiles: tuple[str, ...] = ("*",),
        compensatable: bool = False,
    ) -> "L0SkillContract":
        steps = [
            L0StepContract("validate_parameters", "prepare", "always", "clarify_or_reject"),
            L0StepContract("compile_intent", "prepare", "validated_arguments", "clarify_or_reject"),
            L0StepContract("preflight", "prepare", "always", "reject"),
            L0StepContract("approval", "approve", "always", "reject"),
            L0StepContract("revalidate", "execute", "always", "abort_without_write"),
            L0StepContract("execute", "execute", "preconditions_unchanged", "reconcile"),
            L0StepContract("verify", "verify", "write_may_have_been_sent", "compensate_or_escalate"),
        ]
        if compensatable:
            steps.append(L0StepContract(
                "compensate", "compensate", "verification_failed", "manual_intervention",
            ))
        steps.append(L0StepContract("audit", "terminal", "always", "fail_closed"))
        stable = {
            "schema_version": L0_SKILL_SCHEMA_VERSION,
            "skill_id": skill_id,
            "version": version,
            "tool_name": tool_name,
            "tool_contract_id": tool_contract_id,
            "intent_kind": intent_kind,
            "target_fields": list(target_fields),
            "allowed_profiles": list(allowed_profiles),
            "steps": [step.to_dict() for step in steps],
        }
        return cls(contract_hash=sha256_json(stable), **{
            **stable,
            "target_fields": tuple(target_fields),
            "allowed_profiles": tuple(allowed_profiles),
            "steps": tuple(steps),
        })

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "version": self.version,
            "tool_name": self.tool_name,
            "tool_contract_id": self.tool_contract_id,
            "intent_kind": self.intent_kind,
            "target_fields": list(self.target_fields),
            "allowed_profiles": list(self.allowed_profiles),
            "steps": [step.to_dict() for step in self.steps],
            "contract_hash": self.contract_hash,
        }
        if self.compiled_contract is not None:
            value.update({
                "contract_format": self.compiled_contract.api_version,
                "effect_capability": self.compiled_contract.spec.effect.capability,
                "derivation": self.compiled_contract.derivation,
            })
        return value

    @classmethod
    def from_compiled(cls, compiled: Any, binding: Any) -> "L0SkillContract":
        compensatable = compiled.spec.compensation is not None
        steps = [
            L0StepContract("validate_parameters", "prepare", "always", "clarify_or_reject"),
            L0StepContract("compile_intent", "prepare", "validated_arguments", "clarify_or_reject"),
            L0StepContract("preflight", "prepare", "always", "reject"),
            L0StepContract("approval", "approve", "always", "reject"),
            L0StepContract("revalidate", "execute", "always", "abort_without_write"),
            L0StepContract("execute", "execute", "preconditions_unchanged", "reconcile"),
            L0StepContract("verify", "verify", "write_may_have_been_sent", "compensate_or_escalate"),
        ]
        if compensatable:
            steps.append(L0StepContract(
                "compensate", "compensate", "verification_failed", "manual_intervention",
            ))
        steps.append(L0StepContract("audit", "terminal", "always", "fail_closed"))
        return cls(
            skill_id=compiled.metadata.id,
            version=compiled.metadata.version,
            tool_name=binding.tool_name,
            tool_contract_id=binding.tool_contract_id,
            intent_kind=compiled.spec.intent.kind,
            target_fields=compiled.spec.intent.target_fields,
            allowed_profiles=compiled.spec.profiles,
            steps=tuple(steps),
            contract_hash=compiled.contract_hash,
            schema_version=2,
            compiled_contract=compiled,
        )

    def verify_integrity(self) -> None:
        if self.compiled_contract is not None:
            payload = self.compiled_contract.model_dump(by_alias=True, mode="json")
            observed = payload.pop("contractHash")
            if observed != self.contract_hash or sha256_json(payload) != observed:
                raise PlanIntegrityError(f"L0 v2 contract hash mismatch: {self.skill_id}")
            if (
                self.compiled_contract.metadata.id != self.skill_id
                or self.compiled_contract.metadata.version != self.version
                or self.compiled_contract.spec.effect.tool != self.tool_name
                or self.compiled_contract.spec.intent.kind != self.intent_kind
                or self.compiled_contract.spec.intent.target_fields != self.target_fields
                or self.compiled_contract.spec.profiles != self.allowed_profiles
            ):
                raise PlanIntegrityError(f"L0 v2 Runtime projection mismatch: {self.skill_id}")
            return
        payload = self.to_dict()
        payload.pop("contract_hash")
        if sha256_json(payload) != self.contract_hash:
            raise PlanIntegrityError(f"L0 Skill contract hash mismatch: {self.skill_id}")


@dataclass(frozen=True)
class IntentSpec:
    intent_kind: str
    l0_skill_id: str
    l0_skill_version: str
    profile: str
    operation: str
    targets: tuple[str, ...]
    desired_state: dict[str, Any]
    constraints: dict[str, Any]
    arguments_digest: str
    provenance: dict[str, str]
    intent_hash: str
    schema_version: int = INTENT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        contract: L0SkillContract,
        profile: str,
        arguments: dict[str, Any],
        provenance: dict[str, str],
        targets: tuple[str, ...],
    ) -> "IntentSpec":
        arguments_digest = sha256_json(arguments)
        if contract.compiled_contract is not None:
            from .l0.expressions import render_template

            desired_state = render_template(
                contract.compiled_contract.spec.intent.desired_state,
                {
                    "arguments": arguments,
                    "intent": {
                        "arguments_digest": arguments_digest,
                        "configuration_digest": sha256_json(
                            arguments.get("config_lines")
                            or arguments.get("config_text")
                            or arguments.get("changes")
                        ),
                    },
                },
            )
        else:
            desired_state = _desired_state(contract.tool_name, arguments)
        stable = {
            "schema_version": INTENT_SCHEMA_VERSION,
            "intent_kind": contract.intent_kind,
            "l0_skill_id": contract.skill_id,
            "l0_skill_version": contract.version,
            "profile": profile,
            "operation": contract.tool_name,
            "targets": list(targets),
            "desired_state": desired_state,
            "constraints": _constraints(arguments),
            "arguments_digest": arguments_digest,
            "provenance": dict(sorted(provenance.items())),
        }
        return cls(intent_hash=sha256_json(stable), **{
            **stable,
            "targets": tuple(targets),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent_kind": self.intent_kind,
            "l0_skill_id": self.l0_skill_id,
            "l0_skill_version": self.l0_skill_version,
            "profile": self.profile,
            "operation": self.operation,
            "targets": list(self.targets),
            "desired_state": self.desired_state,
            "constraints": self.constraints,
            "arguments_digest": self.arguments_digest,
            "provenance": self.provenance,
            "intent_hash": self.intent_hash,
        }

    def verify_integrity(self) -> None:
        payload = self.to_dict()
        payload.pop("intent_hash")
        if sha256_json(payload) != self.intent_hash:
            raise PlanIntegrityError("IntentSpec hash mismatch")


class L0SkillRegistry:
    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], L0SkillContract] = {}
        self._by_tool: dict[str, list[L0SkillContract]] = {}
        self._by_id: dict[str, list[L0SkillContract]] = {}

    @staticmethod
    def _version_key(value: str) -> tuple[int, int, int]:
        parts = value.split(".")
        if len(parts) != 3 or any(not item.isdigit() for item in parts):
            raise RuntimeError(f"L0 Skill version must be MAJOR.MINOR.PATCH: {value!r}")
        return tuple(int(item) for item in parts)  # type: ignore[return-value]

    def register(self, contract: L0SkillContract) -> None:
        key = (contract.skill_id, contract.version)
        if key in self._by_key:
            raise RuntimeError(
                f"duplicate L0 Skill contract {contract.skill_id}@{contract.version}"
            )
        self._version_key(contract.version)
        self._by_key[key] = contract
        self._by_tool.setdefault(contract.tool_name, []).append(contract)
        self._by_id.setdefault(contract.skill_id, []).append(contract)
        self._by_tool[contract.tool_name].sort(
            key=lambda item: (item.skill_id, self._version_key(item.version)),
        )
        self._by_id[contract.skill_id].sort(key=lambda item: self._version_key(item.version))

    def candidates_for_tool(
        self, profile: str, tool_name: str,
    ) -> tuple[L0SkillContract, ...]:
        return tuple(
            contract for contract in self._by_tool.get(tool_name, ())
            if "*" in contract.allowed_profiles or profile in contract.allowed_profiles
        )

    def for_tool(
        self,
        profile: str,
        tool_name: str,
        *,
        skill_id: str | None = None,
        version: str | None = None,
    ) -> L0SkillContract | None:
        candidates = self.candidates_for_tool(profile, tool_name)
        if skill_id is not None:
            candidates = tuple(item for item in candidates if item.skill_id == skill_id)
        if version is not None:
            candidates = tuple(item for item in candidates if item.version == version)
        if not candidates:
            return None
        if skill_id is None and len({item.skill_id for item in candidates}) > 1:
            # Ambiguous semantic entrypoints must never be chosen from tool name alone.
            return None
        return max(candidates, key=lambda item: self._version_key(item.version))

    def get(self, skill_id: str, version: str | None = None) -> L0SkillContract | None:
        values = tuple(self._by_id.get(skill_id, ()))
        if version is not None:
            return self._by_key.get((skill_id, version))
        return max(values, key=lambda item: self._version_key(item.version)) if values else None

    def contracts(self) -> tuple[L0SkillContract, ...]:
        return tuple(self._by_key[key] for key in sorted(self._by_key))

    def unregister(self, skill_id: str, version: str) -> None:
        """Remove one exact adapter-owned contract from a temporary registration."""
        contract = self._by_key.pop((skill_id, version), None)
        if contract is None:
            return
        self._by_tool[contract.tool_name] = [
            item for item in self._by_tool[contract.tool_name]
            if (item.skill_id, item.version) != (skill_id, version)
        ]
        self._by_id[skill_id] = [
            item for item in self._by_id[skill_id] if item.version != version
        ]
        if not self._by_tool[contract.tool_name]:
            del self._by_tool[contract.tool_name]
        if not self._by_id[skill_id]:
            del self._by_id[skill_id]


REGISTRY = L0SkillRegistry()


def _register_production_v2() -> None:
    from .l0.production import BINDINGS, contracts

    for compiled in contracts():
        binding = BINDINGS[(compiled.metadata.id, compiled.metadata.version)]
        REGISTRY.register(L0SkillContract.from_compiled(compiled, binding))


_register_production_v2()


def compile_intent(
    contract: L0SkillContract,
    *,
    profile: str,
    tool_name: str,
    arguments: dict[str, Any],
    provenance: dict[str, str],
    targets: tuple[str, ...],
) -> IntentSpec:
    if contract.tool_name != tool_name:
        raise PlanIntegrityError("L0 Skill tool binding mismatch")
    if set(provenance) != set(arguments):
        raise PlanIntegrityError("every compiled argument must have provenance")
    missing_targets = [field for field in contract.target_fields if arguments.get(field) in (None, "")]
    if missing_targets:
        raise PlanIntegrityError(
            "L0 Skill intent target is unresolved: " + ", ".join(missing_targets)
        )
    if not targets:
        raise PlanIntegrityError("L0 Skill intent has no resolved target")
    return IntentSpec.create(
        contract=contract,
        profile=profile,
        arguments=arguments,
        provenance=provenance,
        targets=targets,
    )


def validate_registry() -> None:
    tool_contracts = reviewed_contracts()
    missing = sorted(set(tool_contracts) - {item.tool_name for item in REGISTRY.contracts()})
    if missing:
        raise RuntimeError("reviewed write tools without L0 Skill contracts: " + ", ".join(missing))
    for contract in REGISTRY.contracts():
        contract.verify_integrity()
        tool_contract = tool_contracts.get(contract.tool_name)
        if tool_contract is None or tool_contract.contract_id != contract.tool_contract_id:
            raise RuntimeError(f"stale L0 Skill tool contract {contract.skill_id}")


def _constraints(arguments: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "environment", "reason", "force", "dry_run", "to_transport", "version", "role",
        "change_id", "expected_revision", "correlation_id",
    )
    return {field: arguments[field] for field in fields if field in arguments}


def _desired_state(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    fixed: dict[str, dict[str, Any]] = {
        "restart_service": {"service_health": "healthy", "rollout": "complete"},
        "rollback_service": {"service_health": "healthy", "rollback": "complete"},
        "rollback_deploy": {"rolled_back": True, "services_healthy": True},
        "drain_node": {"drained": True, "schedulable": False, "pending": 0, "failed": 0},
        "delete_resource": {"exists": False},
        "grant_user_access": {"admitted": True},
        "revoke_user_access": {"admitted": False},
        "dc_grant_app_access": {"allowed": True},
        "dc_revoke_app_access": {"allowed": False},
        "access_policy_grant_entitlement": {
            "allowed": True, "role": arguments.get("role"),
        },
        "access_policy_revoke_entitlement": {"allowed": False, "roles": []},
        "platform_restart_service": {"status": "healthy", "rollout": "complete"},
        "platform_rollback_service": {
            "status": "healthy", "version": arguments.get("version"),
        },
        "network_apply_app_enforcement": {"allowed": True},
        "network_revoke_app_enforcement": {"allowed": False},
    }
    if tool_name in fixed:
        return fixed[tool_name]
    if tool_name == "failover":
        return {"primary": arguments.get("target"), "healthy": True}
    if tool_name == "wan_failover_path":
        return {"transport": arguments.get("to_transport"), "state": "up"}
    if tool_name == "fabric_set_access_vlan":
        return {"vlan_id": arguments.get("vlan_id")}
    if tool_name in {"edit_device_config", "push_config", "dc_config_push"}:
        return {"requested_configuration_digest": sha256_json(
            arguments.get("config_lines") or arguments.get("config_text") or arguments.get("changes")
        )}
    raise PlanIntegrityError(f"L0 Skill {tool_name} has no desired-state compiler")


validate_registry()
