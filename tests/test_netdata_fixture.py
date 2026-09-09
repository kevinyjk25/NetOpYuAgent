"""Isolated host/domain/egress mechanics, not live Netdata or translation Gold."""

import copy
import json
import socket
from dataclasses import replace

import pytest

from evaluation.netdata_fixture import IsolatedNetdataHost, query_arguments
from evaluation.netdata_task_demo import context, execute_task, run_demo
from network_runtime.capabilities import DataSensitivity
from network_runtime.l0.structured_schema import DataBindingError

TEXT = "Read the explicitly scoped synthetic page; preserve privacy and unknown completeness."


def test_demo_rejects_existing_output_before_reading_any_source(tmp_path):
    with pytest.raises(FileExistsError):
        run_demo("nonexistent-source", tmp_path)


def host():
    return IsolatedNetdataHost(
        node="fixture-node", listener="local", device_ip="10.0.0.8"
    )


def test_real_gateway_info_query_projection_summary_without_network_or_raw_egress(
    monkeypatch,
):
    monkeypatch.setattr(
        socket, "socket", lambda *a, **k: pytest.fail("no network allowed")
    )
    h = host()
    result = execute_task(h, TEXT)
    assert [call["operation"] for call in h.calls] == ["info", "query"]
    assert result["summary"]["returnedPageRows"] == 3
    assert result["summary"]["severityCounts"] == {"crit": 1, "warning": 2}
    assert result["summary"]["wholeWindowCount"] is None
    assert result["trace"]["networkCalls"] == 0
    rendered = json.dumps(result)
    assert "SYNTHETIC-PRIVATE-MESSAGE" not in rendered and "10.0.0.8" not in rendered


def test_permuted_columns_empty_page_and_pagination_never_claim_full_window():
    h = host()
    baseline = execute_task(h, TEXT)["summary"]
    h = host()
    width = len(h._response["columns"])
    for meta in h._response["columns"].values():
        meta["index"] = width - 1 - meta["index"]
    h._response["data"] = [list(reversed(row)) for row in h._response["data"]]
    h._response["pagination"]["anchor"] = 1788911900000000
    assert execute_task(h, TEXT)["summary"] == baseline
    h._response["data"] = []
    result = execute_task(h, TEXT)["summary"]
    assert result["returnedPageRows"] == 0 and result["emptyPageProvesNoTraps"] is False


@pytest.mark.parametrize(
    "issue", ["role", "scope", "node", "function", "clearance", "anonymous", "wildcard"]
)
def test_access_failure_occurs_before_any_host_callback(issue):
    h = host()
    c = context(h)
    if issue == "role":
        c = replace(c, roles=frozenset({"guest"}))
    elif issue == "scope":
        c = replace(c, scopes=c.scopes - {"netdata:logs:read"})
    elif issue == "node":
        c = replace(c, scopes=c.scopes - {"node_id:fixture-node"})
    elif issue == "function":
        c = replace(c, scopes=c.scopes - {"function_id:snmp:traps"})
    elif issue == "clearance":
        c = replace(c, clearance=DataSensitivity.PUBLIC)
    elif issue == "anonymous":
        c = replace(c, authenticated=False)
    else:
        c = replace(c, scopes=c.scopes | {"*"})
    with pytest.raises(PermissionError):
        execute_task(h, TEXT, access_context=c)
    assert h.calls == []


@pytest.mark.parametrize(
    "issue",
    [
        "parameter",
        "source",
        "duplicate_source",
        "unknown_widget",
        "info_error",
        "version",
    ],
)
def test_discovery_failure_prevents_query(issue):
    h = host()
    if issue == "parameter":
        h._info["accepted_params"].remove("selections")
    elif issue == "source":
        h._info["required_params"][0]["options"][0]["id"] = "unavailable"
    elif issue == "duplicate_source":
        h._info["required_params"][0]["options"] *= 2
    elif issue == "unknown_widget":
        h._info["required_params"].append(copy.deepcopy(h._info["required_params"][0]))
    elif issue == "info_error":
        h._info["status"] = 503
    else:
        h._info["v"] = 99
    with pytest.raises(DataBindingError):
        execute_task(h, TEXT)
    assert [call["operation"] for call in h.calls] == ["info"]


@pytest.mark.parametrize(
    "issue,code",
    [
        ("partial", "netdata_partial_or_unknown"),
        ("unknown_partial", "netdata_partial_or_unknown"),
        ("error", "netdata_query_error"),
        ("unit_ms", "netdata_time_unit_or_window"),
        ("outside_window", "netdata_time_unit_or_window"),
        ("wrong_device", "netdata_filter_mismatch"),
        ("wrong_category", "netdata_filter_mismatch"),
        ("wrong_job", "netdata_filter_mismatch"),
        ("duplicate_index", "column_index_duplicate"),
        ("extra_row", "netdata_page_bound"),
        ("unknown_severity", "value_constraint"),
        ("query_version", "netdata_query_version"),
    ],
)
def test_response_failures_never_yield_summary_or_export_payload(issue, code):
    h = host()
    r = h._response
    if issue == "partial":
        r["partial"] = True
    elif issue == "unknown_partial":
        del r["partial"]
    elif issue == "error":
        r.update(status=503, errorMessage="SYNTHETIC-PRIVATE-MESSAGE")
    elif issue == "unit_ms":
        r["data"][0][3] //= 1000
    elif issue == "outside_window":
        r["data"][0][3] = (h.now_seconds - 90000) * 1000000
    elif issue == "wrong_device":
        r["data"][0][1] = "10.0.0.9"
    elif issue == "wrong_category":
        r["data"][0][5] = "availability"
    elif issue == "wrong_job":
        r["data"][0][4] = "not-local"
    elif issue == "duplicate_index":
        r["columns"]["MESSAGE"]["index"] = 2
    elif issue == "unknown_severity":
        r["data"][0][2] = "SYNTHETIC-PRIVATE-MESSAGE"
    elif issue == "query_version":
        r["v"] = 99
    with pytest.raises(DataBindingError) as error:
        execute_task(h, TEXT, last=2 if issue == "extra_row" else 200)
    assert error.value.code == code
    assert "SYNTHETIC-PRIVATE-MESSAGE" not in str(error.value)
    assert len(h.calls) == 2


@pytest.mark.parametrize("seconds", [86400000, 86400000000, True, "86400", -60, 0])
def test_query_units_and_bounds_do_not_coerce_or_guess(seconds):
    h = host()
    with pytest.raises(DataBindingError):
        query_arguments(
            h._info,
            node=h.node,
            listener=h.listener,
            device_ip=h.device_ip,
            window_seconds=seconds,
        )


def test_host_primitive_forbids_mixed_info_query_and_unscoped_operations():
    h = host()
    args = query_arguments(
        h._info, node=h.node, listener=h.listener, device_ip=h.device_ip
    )
    args["body"]["info"] = True
    with pytest.raises(ValueError):
        h.observe(args)
    del args["body"]["info"]
    args["body"]["selections"]["TRAP_SOURCE_IP"] = ["10.0.0.9"]
    with pytest.raises(PermissionError):
        h.observe(args)
    args["function"] = "arbitrary-function"
    with pytest.raises(DataBindingError):
        h.observe(args)
    assert h.calls == []
