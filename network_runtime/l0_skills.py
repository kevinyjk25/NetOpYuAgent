"""First-class deterministic Network L0 Skill and intent contracts.

An LLM may select one of these skills and propose arguments.  It cannot alter
the step graph, desired state, verifier, compensation policy or failure mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        return {
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

    def verify_integrity(self) -> None:
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
        stable = {
            "schema_version": INTENT_SCHEMA_VERSION,
            "intent_kind": contract.intent_kind,
            "l0_skill_id": contract.skill_id,
            "l0_skill_version": contract.version,
            "profile": profile,
            "operation": contract.tool_name,
            "targets": list(targets),
            "desired_state": _desired_state(contract.tool_name, arguments),
            "constraints": _constraints(arguments),
            "arguments_digest": sha256_json(arguments),
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
        self._by_tool: dict[str, L0SkillContract] = {}
        self._by_id: dict[str, L0SkillContract] = {}

    def register(self, contract: L0SkillContract) -> None:
        if contract.tool_name in self._by_tool or contract.skill_id in self._by_id:
            raise RuntimeError(f"duplicate L0 Skill contract {contract.skill_id}")
        self._by_tool[contract.tool_name] = contract
        self._by_id[contract.skill_id] = contract

    def for_tool(self, profile: str, tool_name: str) -> L0SkillContract | None:
        contract = self._by_tool.get(tool_name)
        if contract is None:
            return None
        if "*" not in contract.allowed_profiles and profile not in contract.allowed_profiles:
            return None
        return contract

    def get(self, skill_id: str) -> L0SkillContract | None:
        return self._by_id.get(skill_id)

    def contracts(self) -> tuple[L0SkillContract, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))


REGISTRY = L0SkillRegistry()


def _register(
    skill_id: str,
    version: str,
    tool_name: str,
    intent_kind: str,
    target_fields: tuple[str, ...],
    *,
    profiles: tuple[str, ...] = ("*",),
) -> None:
    tool_contract = reviewed_contracts()[tool_name]
    REGISTRY.register(L0SkillContract.create(
        skill_id=skill_id,
        version=version,
        tool_name=tool_name,
        tool_contract_id=tool_contract.contract_id,
        intent_kind=intent_kind,
        target_fields=target_fields,
        allowed_profiles=profiles,
        compensatable=tool_contract.compensator is not None,
    ))


_register("network.device.config.edit", "1.0.0", "edit_device_config", "configure_device", ("device_id",), profiles=("lan",))
_register("network.device.config.push", "1.0.0", "push_config", "configure_device", ("device_id",))
_register("network.service.restart", "1.0.0", "restart_service", "restart_service", ("service",))
_register("network.service.rollback", "1.0.0", "rollback_service", "rollback_service", ("service",))
_register("network.deploy.rollback", "1.0.0", "rollback_deploy", "rollback_deployment", ("deploy_id",))
_register("network.node.drain", "1.0.0", "drain_node", "drain_node", ("node_id",))
_register("network.resource.failover", "1.0.0", "failover", "failover_resource", ("resource_id",))
_register("network.resource.delete", "1.0.0", "delete_resource", "delete_resource", ("resource_id",))
_register("network.lan.user-access.grant", "1.0.0", "grant_user_access", "grant_network_access", ("user_id",), profiles=("lan",))
_register("network.lan.user-access.revoke", "1.0.0", "revoke_user_access", "revoke_network_access", ("user_id",), profiles=("lan",))
_register("network.dc.fabric-config.push", "1.0.0", "dc_config_push", "configure_fabric", ("node",), profiles=("dc",))
_register("network.dc.app-access.grant", "1.0.0", "dc_grant_app_access", "grant_application_access", ("user_id", "app_id"), profiles=("dc",))
_register("network.dc.app-access.revoke", "1.0.0", "dc_revoke_app_access", "revoke_application_access", ("user_id", "app_id"), profiles=("dc",))
_register("network.wan.path.failover", "1.0.0", "wan_failover_path", "failover_wan_path", ("tunnel",), profiles=("wan",))


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
    fields = ("environment", "reason", "force", "dry_run", "to_transport", "version", "role")
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
    }
    if tool_name in fixed:
        return fixed[tool_name]
    if tool_name == "failover":
        return {"primary": arguments.get("target"), "healthy": True}
    if tool_name == "wan_failover_path":
        return {"transport": arguments.get("to_transport"), "state": "up"}
    if tool_name in {"edit_device_config", "push_config", "dc_config_push"}:
        return {"requested_configuration_digest": sha256_json(
            arguments.get("config_lines") or arguments.get("config_text") or arguments.get("changes")
        )}
    raise PlanIntegrityError(f"L0 Skill {tool_name} has no desired-state compiler")


validate_registry()
