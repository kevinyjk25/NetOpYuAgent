"""Official-SDK MCP servers for the local enterprise Service Layer.

Each ``--domain`` value starts an independent MCP server process.  All
processes share the same transactional SQLite database, matching the failure
and ownership boundaries of separate enterprise systems while keeping local
setup deterministic.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.server.mcpserver import MCPServer

from .models import (
    ApplicationRecord,
    ApplicationResult,
    ApplicationsResult,
    ChangeResult,
    ChangeValidationResult,
    EndpointBindingResult,
    EntitlementMutationResult,
    EntitlementResult,
    PolicyEvaluationResult,
    ServiceHealthResult,
    ServiceMutationResult,
    UserRecord,
    UserResult,
    UsersResult,
)
from .store import ServiceStore, default_store_path, utc_now


SERVER_VERSION = "1.0.0"
DOMAINS = {"identity", "application", "access-policy", "change", "cmdb", "platform", "all"}


def _meta(
    domain: str,
    action_type: str,
    contract_id: str | None = None,
    *,
    internal_only: bool = False,
) -> dict[str, Any]:
    sensitivity = {
        "identity": "restricted",
        "access-policy": "confidential",
        "change": "confidential",
        "cmdb": "confidential",
    }.get(domain, "internal")
    return {
        "netopyu": {
            "domain": "service",
            "service_domain": domain,
            "action_type": action_type,
            "requires_approval": action_type != "read_only",
            "contract_id": contract_id,
            "result_contract": "structured-content-required-v1",
            "internal_only": internal_only,
            "sensitivity": sensitivity,
            "required_roles": ["operations-reader"] if action_type == "read_only" else [],
            "freshness_limit_seconds": 300,
        }
    }


def _base(store: ServiceStore, correlation_id: str | None) -> dict[str, Any]:
    return {
        "ok": True,
        "code": "ok",
        "correlation_id": store.correlation_id(correlation_id),
        "observed_at": utc_now(),
        "simulation": True,
    }


def build_server(domain: str, store_path: str | Path | None = None) -> MCPServer:
    if domain not in DOMAINS:
        raise ValueError(f"unknown service MCP domain {domain!r}")
    store = ServiceStore(store_path)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[ServiceStore]:
        try:
            yield store
        finally:
            store.close()

    server = MCPServer(
        f"netopyu.{domain}",
        version=SERVER_VERSION,
        description=f"NetOpYu deterministic {domain} service-system simulator",
        lifespan=lifespan,
    )

    if domain in {"identity", "all"}:
        @server.tool(meta=_meta("identity", "read_only"), structured_output=True)
        def identity_list_users(
            department: str | None = None,
            correlation_id: str | None = None,
        ) -> UsersResult:
            """List authoritative enterprise identities, optionally by department."""
            users = [UserRecord(**item) for item in store.list_users(department)]
            return UsersResult(**_base(store, correlation_id), users=users)

        @server.tool(meta=_meta("identity", "read_only"), structured_output=True)
        def identity_get_user(user_id: str, correlation_id: str | None = None) -> UserResult:
            """Return one authoritative enterprise identity and lifecycle status."""
            return UserResult(
                **_base(store, correlation_id),
                user=UserRecord(**store.get_user(user_id)),
            )

    if domain in {"application", "all"}:
        @server.tool(meta=_meta("application", "read_only"), structured_output=True)
        def application_list(
            tier: str | None = None,
            correlation_id: str | None = None,
        ) -> ApplicationsResult:
            """List applications from the authoritative local application catalog."""
            apps = [ApplicationRecord(**item) for item in store.list_applications(tier)]
            return ApplicationsResult(**_base(store, correlation_id), applications=apps)

        @server.tool(meta=_meta("application", "read_only"), structured_output=True)
        def application_get(
            app_id: str,
            correlation_id: str | None = None,
        ) -> ApplicationResult:
            """Return one application, its owner, endpoint, tier, and valid roles."""
            return ApplicationResult(
                **_base(store, correlation_id),
                application=ApplicationRecord(**store.get_application(app_id)),
            )

        @server.tool(meta=_meta("application", "read_only"), structured_output=True)
        def application_check_access(
            user_id: str,
            app_id: str,
            correlation_id: str | None = None,
        ) -> EntitlementResult:
            """Read the business entitlement; this is not network enforcement evidence."""
            return EntitlementResult(
                **_base(store, correlation_id), **store.entitlement(user_id, app_id),
            )

    if domain in {"access-policy", "all"}:
        @server.tool(meta=_meta("access-policy", "read_only"), structured_output=True)
        def access_policy_evaluate(
            user_id: str,
            app_id: str,
            correlation_id: str | None = None,
        ) -> PolicyEvaluationResult:
            """Evaluate deterministic business eligibility without changing state."""
            return PolicyEvaluationResult(
                **_base(store, correlation_id), **store.evaluate_policy(user_id, app_id),
            )

        @server.tool(meta=_meta("access-policy", "read_only"), structured_output=True)
        def access_policy_get_entitlement(
            user_id: str,
            app_id: str,
            correlation_id: str | None = None,
        ) -> EntitlementResult:
            """Read exact desired roles and optimistic-concurrency revision."""
            return EntitlementResult(
                **_base(store, correlation_id), **store.entitlement(user_id, app_id),
            )

        @server.tool(
            meta=_meta("access-policy", "reversible", "service-entitlement-grant-v1"),
            structured_output=True,
        )
        def access_policy_grant_entitlement(
            user_id: str,
            app_id: str,
            role: str,
            change_id: str,
            expected_revision: int,
            reason: str,
            correlation_id: str | None = None,
        ) -> EntitlementMutationResult:
            """Grant one reviewed application role with revision and change checks."""
            corr = store.correlation_id(correlation_id)
            value = store.mutate_entitlement(
                operation="grant", user_id=user_id, app_id=app_id, role=role, roles=None,
                change_id=change_id, reason=reason, expected_revision=expected_revision,
                correlation_id=corr,
            )
            return EntitlementMutationResult(**_base(store, corr), **value)

        @server.tool(
            meta=_meta("access-policy", "reversible", "service-entitlement-revoke-v1"),
            structured_output=True,
        )
        def access_policy_revoke_entitlement(
            user_id: str,
            app_id: str,
            change_id: str,
            expected_revision: int,
            reason: str,
            correlation_id: str | None = None,
        ) -> EntitlementMutationResult:
            """Revoke all application roles with revision and change checks."""
            corr = store.correlation_id(correlation_id)
            value = store.mutate_entitlement(
                operation="revoke", user_id=user_id, app_id=app_id, role=None, roles=None,
                change_id=change_id, reason=reason, expected_revision=expected_revision,
                correlation_id=corr,
            )
            return EntitlementMutationResult(**_base(store, corr), **value)

        @server.tool(
            meta=_meta(
                "access-policy", "reversible", "service-entitlement-restore-internal-v1",
                internal_only=True,
            ),
            structured_output=True,
        )
        def access_policy_restore_entitlement(
            user_id: str,
            app_id: str,
            roles: list[str],
            change_id: str,
            expected_revision: int,
            reason: str,
            correlation_id: str | None = None,
        ) -> EntitlementMutationResult:
            """Runtime-only compensator that restores the exact approved role snapshot."""
            corr = store.correlation_id(correlation_id)
            value = store.mutate_entitlement(
                operation="restore", user_id=user_id, app_id=app_id, role=None, roles=roles,
                change_id=change_id, reason=reason, expected_revision=expected_revision,
                correlation_id=corr,
            )
            return EntitlementMutationResult(**_base(store, corr), **value)

    if domain in {"change", "all"}:
        @server.tool(meta=_meta("change", "read_only"), structured_output=True)
        def change_get(change_id: str, correlation_id: str | None = None) -> ChangeResult:
            """Read one simulated enterprise change record."""
            return ChangeResult(**_base(store, correlation_id), **store.change(change_id))

        @server.tool(meta=_meta("change", "read_only"), structured_output=True)
        def change_validate_window(
            change_id: str,
            correlation_id: str | None = None,
        ) -> ChangeValidationResult:
            """Validate business approval and execution window; Runtime approval remains separate."""
            return ChangeValidationResult(
                **_base(store, correlation_id), **store.validate_change(change_id),
            )

    if domain in {"cmdb", "all"}:
        @server.tool(meta=_meta("cmdb", "read_only"), structured_output=True)
        def cmdb_get_endpoint_binding(
            subject_type: str,
            subject_id: str,
            correlation_id: str | None = None,
        ) -> EndpointBindingResult:
            """Map a business identity/application to a declared network endpoint."""
            return EndpointBindingResult(
                **_base(store, correlation_id),
                **store.endpoint_binding(subject_type, subject_id),
            )

    if domain in {"platform", "all"}:
        @server.tool(meta=_meta("platform", "read_only"), structured_output=True)
        def platform_get_service_health(
            service: str,
            environment: str,
            correlation_id: str | None = None,
        ) -> ServiceHealthResult:
            """Read health and revision from the simulated service platform."""
            return ServiceHealthResult(
                **_base(store, correlation_id), **store.service_health(service, environment),
            )

        @server.tool(
            meta=_meta("platform", "reversible", "service-platform-restart-v1"),
            structured_output=True,
        )
        def platform_restart_service(
            service: str,
            environment: str,
            change_id: str,
            expected_revision: int,
            reason: str,
            correlation_id: str | None = None,
        ) -> ServiceMutationResult:
            """Restart one service through a reviewed change and revision check."""
            corr = store.correlation_id(correlation_id)
            value = store.mutate_service(
                operation="restart", service=service, environment=environment, version=None,
                change_id=change_id, reason=reason, expected_revision=expected_revision,
                correlation_id=corr,
            )
            return ServiceMutationResult(**_base(store, corr), **value)

        @server.tool(
            meta=_meta("platform", "reversible", "service-platform-rollback-v1"),
            structured_output=True,
        )
        def platform_rollback_service(
            service: str,
            environment: str,
            version: str,
            change_id: str,
            expected_revision: int,
            reason: str,
            correlation_id: str | None = None,
        ) -> ServiceMutationResult:
            """Roll back one service version through the deterministic platform simulator."""
            corr = store.correlation_id(correlation_id)
            value = store.mutate_service(
                operation="rollback", service=service, environment=environment, version=version,
                change_id=change_id, reason=reason, expected_revision=expected_revision,
                correlation_id=corr,
            )
            return ServiceMutationResult(**_base(store, corr), **value)

        @server.tool(
            meta=_meta(
                "platform", "reversible", "service-platform-restore-internal-v1",
                internal_only=True,
            ),
            structured_output=True,
        )
        def platform_restore_service(
            service: str,
            environment: str,
            version: str,
            change_id: str,
            expected_revision: int,
            reason: str,
            correlation_id: str | None = None,
        ) -> ServiceMutationResult:
            """Runtime-only compensator restoring the approved service snapshot."""
            corr = store.correlation_id(correlation_id)
            value = store.mutate_service(
                operation="restore", service=service, environment=environment, version=version,
                change_id=change_id, reason=reason, expected_revision=expected_revision,
                correlation_id=corr,
            )
            return ServiceMutationResult(**_base(store, corr), **value)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    parser.add_argument("--store", default=str(default_store_path()))
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = build_server(args.domain, args.store)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(
            "streamable-http", host=args.host, port=args.port,
            stateless_http=True, json_response=True,
        )


if __name__ == "__main__":
    main()
