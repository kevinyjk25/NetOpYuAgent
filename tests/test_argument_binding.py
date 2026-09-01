from __future__ import annotations

from network_runtime.argument_binding import validate_exact_argument_binding


FIELDS = ("entity_id", "desired_value", "expected_revision", "reason")


def test_exact_argument_binding_accepts_identity_mapping() -> None:
    values = {
        "entity_id": "edge-01", "desired_value": "vlan-120",
        "expected_revision": 1, "reason": "approved change",
    }
    result = validate_exact_argument_binding(
        authoritative=values, invocation=dict(values),
        declared_fields=FIELDS, required_fields=FIELDS,
    )
    assert result.ok is True
    assert result.code == "argument_binding_verified"
    assert result.source_digest == result.invocation_digest


def test_exact_argument_binding_reports_loss_injection_drift_and_unknowns() -> None:
    result = validate_exact_argument_binding(
        authoritative={
            "entity_id": "edge-01", "desired_value": "vlan-120",
            "expected_revision": 1, "unsupported_scope": "all",
        },
        invocation={
            "entity_id": "edge-02", "desired_value": "vlan-120",
            "reason": "invented",
        },
        declared_fields=FIELDS, required_fields=FIELDS,
    )
    assert result.ok is False
    assert result.unknown_source_fields == ("unsupported_scope",)
    assert result.missing_source_fields == ("reason",)
    assert result.dropped_fields == ("expected_revision",)
    assert result.injected_fields == ("reason",)
    assert result.changed_fields == ("entity_id",)
