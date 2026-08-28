"""Runtime projection and parity checks for compiled production L0 v2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from network_runtime.contracts import PlanIntegrityError

from .expressions import ExpressionError, render_effect_request, render_template
from .models import CompiledAtomicEffect, ParameterSpec, value_matches_type


Resolver = Callable[[Any, dict[str, Any]], Any]


class ResolverRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Resolver] = {}

    def register(self, name: str) -> Callable[[Resolver], Resolver]:
        def decorator(handler: Resolver) -> Resolver:
            if name in self._handlers:
                raise RuntimeError(f"duplicate L0 resolver {name!r}")
            self._handlers[name] = handler
            return handler
        return decorator

    def resolve(self, name: str, value: Any, context: dict[str, Any]) -> Any:
        handler = self._handlers.get(name)
        if handler is None:
            raise PlanIntegrityError(f"unregistered L0 parameter resolver {name!r}")
        return handler(value, context)

    def names(self) -> frozenset[str]:
        return frozenset(self._handlers)


RESOLVERS = ResolverRegistry()


@RESOLVERS.register("identity")
def _identity(value: Any, _context: dict[str, Any]) -> Any:
    return value


@dataclass(frozen=True)
class RuntimeParity:
    ok: bool
    arguments: dict[str, Any]
    effect_arguments: dict[str, Any]
    desired_state: dict[str, Any]
    errors: tuple[str, ...]


def _validate_value(name: str, value: Any, spec: ParameterSpec) -> list[str]:
    errors: list[str] = []
    if not value_matches_type(value, spec.type):
        return [f"{name} does not satisfy L0 v2 type={spec.type}"]
    if spec.enum and value not in spec.enum:
        errors.append(f"{name} is outside L0 v2 enum")
    if spec.minimum is not None and value < spec.minimum:
        errors.append(f"{name} is below L0 v2 minimum")
    if spec.maximum is not None and value > spec.maximum:
        errors.append(f"{name} exceeds L0 v2 maximum")
    if spec.min_length is not None and len(value) < spec.min_length:
        errors.append(f"{name} is shorter than L0 v2 minLength")
    if spec.max_length is not None and len(value) > spec.max_length:
        errors.append(f"{name} exceeds L0 v2 maxLength")
    if spec.pattern is not None and not re.fullmatch(spec.pattern, value):
        errors.append(f"{name} does not match L0 v2 pattern")
    if spec.fixed is not None and value != spec.fixed:
        errors.append(f"{name} differs from the L0 v2 fixed value")
    return errors


def validate_runtime_projection(
    *,
    compiled: CompiledAtomicEffect,
    tool_name: str,
    tool_contract_id: str,
    verifier_id: str,
    compensator_id: str | None,
    profile: str,
    arguments: dict[str, Any],
    intent: dict[str, Any],
    resolver_context: dict[str, Any] | None = None,
) -> RuntimeParity:
    errors: list[str] = []
    spec = compiled.spec
    labels = compiled.metadata.labels
    if spec.effect.tool != tool_name:
        errors.append("L0 v2 effect tool differs from Runtime tool")
    if labels.get("runtime-tool-contract") != tool_contract_id:
        errors.append("L0 v2 ToolContract binding differs from Runtime")
    if labels.get("runtime-verifier") != verifier_id:
        errors.append("L0 v2 verifier binding differs from Runtime")
    if (spec.compensation is not None) != (compensator_id is not None):
        errors.append("L0 v2 compensation presence differs from Runtime")
    if profile not in spec.profiles:
        errors.append(f"profile {profile!r} is not allowed by L0 v2")
    unknown = sorted(set(arguments) - set(spec.parameters))
    missing = sorted(
        name for name, parameter in spec.parameters.items()
        if parameter.required and parameter.fixed is None and name not in arguments
    )
    if unknown:
        errors.append("L0 v2 has no parameters: " + ", ".join(unknown))
    if missing:
        errors.append("L0 v2 required parameters are missing: " + ", ".join(missing))
    resolved = dict(arguments)
    for name, value in arguments.items():
        parameter = spec.parameters.get(name)
        if parameter is None:
            continue
        if parameter.resolver:
            resolved[name] = RESOLVERS.resolve(
                parameter.resolver, value, resolver_context or {},
            )
        errors.extend(_validate_value(name, resolved[name], parameter))
    try:
        effect_arguments = render_effect_request(spec.effect.request, resolved)
    except ExpressionError as error:
        errors.append(str(error))
        effect_arguments = {}
    if effect_arguments != resolved:
        errors.append("L0 v2 effect request is not parity-equivalent to normalized Runtime arguments")
    desired_state = dict(intent.get("desired_state") or {})
    configuration_digest = intent.get("desired_state", {}).get(
        "requested_configuration_digest",
    )
    try:
        rendered_desired = render_template(spec.intent.desired_state, {
            "arguments": resolved,
            "intent": {
                "arguments_digest": intent.get("arguments_digest"),
                "configuration_digest": configuration_digest,
            },
        })
    except ExpressionError as error:
        errors.append(str(error))
        rendered_desired = {}
    if rendered_desired != desired_state:
        errors.append("L0 v2 desired state differs from compiled Runtime Intent")
    expected_preflight = {
        field: resolved[field]
        for field in labels.get("runtime-preflight-fields", "").split(",")
        if field and field in resolved
    }
    try:
        observed_preflight = render_effect_request(spec.preflight[0].arguments, resolved)
    except ExpressionError as error:
        errors.append(str(error))
        observed_preflight = {}
    if observed_preflight != expected_preflight:
        errors.append("L0 v2 preflight projection differs from ToolContract")
    return RuntimeParity(
        not errors, resolved, effect_arguments, rendered_desired, tuple(errors),
    )


def require_effect_arguments(contract: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    compiled = getattr(contract, "compiled_contract", None)
    if not isinstance(compiled, CompiledAtomicEffect):
        raise PlanIntegrityError("Runtime L0 contract is not a compiled AtomicEffect v2")
    try:
        effect = render_effect_request(compiled.spec.effect.request, arguments)
    except ExpressionError as error:
        raise PlanIntegrityError(str(error)) from error
    if effect != arguments:
        raise PlanIntegrityError("approved arguments differ from the L0 v2 effect request")
    return effect
