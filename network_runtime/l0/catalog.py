"""Multi-version lookup, explanation and Saga projection for L0 v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from effect_runtime.saga import SagaDefinition

from .compiler import L0CompileError, compile_documents, load_documents
from .models import (
    CompiledAtomicEffect,
    CompiledCompositeEffect,
    CompiledContract,
    SEMVER,
)


def _version_key(value: str) -> tuple[int, int, int]:
    match = SEMVER.fullmatch(value)
    if match is None:
        raise L0CompileError(f"invalid semantic version {value!r}")
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


class L0Catalog:
    def __init__(self, contracts: Iterable[CompiledContract]):
        self._by_key: dict[tuple[str, str], CompiledContract] = {}
        self._by_capability: dict[str, list[CompiledAtomicEffect]] = {}
        for contract in contracts:
            key = (contract.metadata.id, contract.metadata.version)
            if key in self._by_key:
                raise L0CompileError(f"duplicate compiled contract {key[0]}@{key[1]}")
            self._by_key[key] = contract
            if isinstance(contract, CompiledAtomicEffect):
                self._by_capability.setdefault(contract.spec.effect.capability, []).append(contract)
        for values in self._by_capability.values():
            values.sort(key=lambda item: (item.metadata.id, _version_key(item.metadata.version)))

    @classmethod
    def from_path(cls, path: str | Path) -> "L0Catalog":
        return cls(compile_documents(load_documents(path)))

    def contracts(self) -> tuple[CompiledContract, ...]:
        return tuple(self._by_key[key] for key in sorted(self._by_key))

    def get(self, skill_id: str, version: str | None = None) -> CompiledContract | None:
        if version is not None:
            return self._by_key.get((skill_id, version))
        versions = [
            value for (candidate, _version), value in self._by_key.items()
            if candidate == skill_id
        ]
        return max(versions, key=lambda item: _version_key(item.metadata.version)) if versions else None

    def require(self, skill_id: str, version: str | None = None) -> CompiledContract:
        value = self.get(skill_id, version)
        if value is None:
            suffix = f"@{version}" if version else ""
            raise KeyError(f"unknown L0 contract {skill_id}{suffix}")
        return value

    def for_capability(self, capability: str) -> tuple[CompiledAtomicEffect, ...]:
        return tuple(self._by_capability.get(capability, ()))

    def explain(self, skill_id: str, version: str | None = None) -> str:
        contract = self.require(skill_id, version)
        lines = [
            f"{contract.metadata.id}@{contract.metadata.version}",
            f"Owner: {contract.metadata.owner}",
            f"Kind: {contract.kind}",
        ]
        if isinstance(contract, CompiledAtomicEffect):
            lines.extend([
                f"Derivation: {contract.derivation}",
                f"Effect: {contract.spec.effect.capability}",
                f"Targets: {', '.join(contract.spec.intent.target_fields)}",
                f"Parameters: {', '.join(sorted(contract.spec.parameters))}",
                "Preflight: " + ", ".join(item.capability for item in contract.spec.preflight),
                f"Verification: {contract.spec.verification.capability} "
                f"({len(contract.spec.verification.predicates)} predicates)",
                "Compensation: " + (
                    contract.spec.compensation.capability
                    if contract.spec.compensation else "none -> manual intervention"
                ),
                f"Approval: {contract.spec.approval.risk}/{contract.spec.approval.mode}",
                f"Contract hash: {contract.contract_hash}",
            ])
            if contract.lineage:
                lines.append("Lineage: " + " -> ".join(
                    f"{item.id}@{item.version}" for item in contract.lineage
                ))
        else:
            lines.extend([
                "Steps: " + " -> ".join(item.id for item in contract.steps),
                f"Approval: {contract.approval.risk}/{contract.approval.mode}",
                f"Compensation order: {contract.compensation_order}",
                f"Definition hash: {contract.definition_hash}",
            ])
            for step in contract.steps:
                lines.append(
                    f"  - {step.id}: {step.skill_ref.id}@{step.skill_ref.version} "
                    f"[{step.contract_hash}] depends={list(step.depends_on)}"
                )
        return "\n".join(lines)

    def graph(self, skill_id: str, version: str | None = None) -> str:
        contract = self.require(skill_id, version)
        lines = ["flowchart TD"]
        if isinstance(contract, CompiledAtomicEffect):
            nodes = ["validate", "preflight", "approve", "revalidate", "execute", "verify", "audit"]
            if contract.spec.compensation:
                nodes.insert(-1, "compensate")
            for left, right in zip(nodes, nodes[1:]):
                lines.append(f"  {left} --> {right}")
            lines.append(f'  execute["{contract.spec.effect.capability}"]')
            lines.append(f'  verify["verify: {contract.spec.verification.capability}"]')
        else:
            for step in contract.steps:
                lines.append(f'  {step.id}["{step.id}\\n{step.skill_ref.id}@{step.skill_ref.version}"]')
                for dependency in step.depends_on:
                    lines.append(f"  {dependency} --> {step.id}")
        return "\n".join(lines)

    def diff(
        self,
        left_id: str,
        right_id: str,
        left_version: str | None = None,
        right_version: str | None = None,
    ) -> dict[str, Any]:
        """Return a small, review-oriented semantic diff between two contracts."""
        left = self.require(left_id, left_version)
        right = self.require(right_id, right_version)
        result: dict[str, Any] = {
            "left": {
                "ref": f"{left.metadata.id}@{left.metadata.version}",
                "kind": left.kind,
                "hash": getattr(left, "contract_hash", getattr(left, "definition_hash", None)),
            },
            "right": {
                "ref": f"{right.metadata.id}@{right.metadata.version}",
                "kind": right.kind,
                "hash": getattr(right, "contract_hash", getattr(right, "definition_hash", None)),
            },
        }
        if isinstance(left, CompiledAtomicEffect) and isinstance(right, CompiledAtomicEffect):
            left_parameters = left.spec.parameters
            right_parameters = right.spec.parameters
            shared = sorted(set(left_parameters) & set(right_parameters))
            result.update({
                "relationship": right.derivation,
                "same_effect_capability": (
                    left.spec.effect.capability == right.spec.effect.capability
                ),
                "parameters": {
                    "added": sorted(set(right_parameters) - set(left_parameters)),
                    "removed": sorted(set(left_parameters) - set(right_parameters)),
                    "changed": {
                        name: {
                            "from": left_parameters[name].model_dump(mode="json"),
                            "to": right_parameters[name].model_dump(mode="json"),
                        }
                        for name in shared
                        if left_parameters[name] != right_parameters[name]
                    },
                },
                "preflight_capabilities": {
                    "from": [item.capability for item in left.spec.preflight],
                    "to": [item.capability for item in right.spec.preflight],
                },
                "verification_predicates": {
                    "from": len(left.spec.verification.predicates),
                    "to": len(right.spec.verification.predicates),
                },
                "approval": {
                    "from": left.spec.approval.model_dump(mode="json"),
                    "to": right.spec.approval.model_dump(mode="json"),
                },
                "desired_state": {
                    "from": left.spec.intent.desired_state,
                    "to": right.spec.intent.desired_state,
                },
            })
            return result
        if isinstance(left, CompiledCompositeEffect) and isinstance(right, CompiledCompositeEffect):
            left_steps = {item.id: item for item in left.steps}
            right_steps = {item.id: item for item in right.steps}
            shared = sorted(set(left_steps) & set(right_steps))
            result.update({
                "relationship": "composite",
                "steps": {
                    "added": sorted(set(right_steps) - set(left_steps)),
                    "removed": sorted(set(left_steps) - set(right_steps)),
                    "changed": {
                        name: {
                            "from": left_steps[name].model_dump(mode="json"),
                            "to": right_steps[name].model_dump(mode="json"),
                        }
                        for name in shared
                        if left_steps[name] != right_steps[name]
                    },
                },
                "approval": {
                    "from": left.approval.model_dump(mode="json"),
                    "to": right.approval.model_dump(mode="json"),
                },
                "compensation_order": {
                    "from": left.compensation_order,
                    "to": right.compensation_order,
                },
            })
            return result
        result["relationship"] = "incompatible-kinds"
        return result

    def to_saga_definition(
        self, skill_id: str, version: str | None = None,
    ) -> "SagaDefinition":
        from effect_runtime.saga import SagaDefinition, SagaStepSpec

        contract = self.require(skill_id, version)
        if not isinstance(contract, CompiledCompositeEffect):
            raise TypeError("only a compiled CompositeEffect can become a SagaDefinition")
        return SagaDefinition.create(
            contract.metadata.id,
            contract.metadata.version,
            tuple(SagaStepSpec(
                step.id,
                step.domain,
                step.capability,
                depends_on=step.depends_on,
                compensation_capability_id=step.compensation_capability,
            ) for step in contract.steps),
        )

    def to_json(self) -> str:
        return json.dumps(
            [item.model_dump(by_alias=True, mode="json") for item in self.contracts()],
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n"
