"""Finite waiting/unknown outcomes on the scripted transport, never live LLMs."""
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import threading
import time
from urllib.parse import urlsplit

import pytest

from evaluation import bounded_transport as transport
from evaluation.bounded_budget import BudgetError, CAPS
from tests.test_bounded_transport import http_handler, request, setup


@pytest.mark.parametrize("mutation", ["pause", "close", "deadline"])
def test_journal_to_backend_gap_is_guarded_without_new_backend_call(tmp_path, monkeypatch, mutation):
    clock = [1000.0]
    broker, ledger = setup(tmp_path, clock=lambda: clock[0])
    write = transport.write_new

    def after_journal(path, value):
        write(path, value)
        if path.name == "request.json":
            if mutation == "pause":
                ledger.pause_study("fixture", "operator_stop")
            elif mutation == "close":
                broker.close(timeout=0)
            else:
                clock[0] += 420

    monkeypatch.setattr(transport, "write_new", after_journal)
    with pytest.raises((BudgetError, PermissionError)):
        broker.dispatch(broker.tokens["agent"], "native", request())
    assert not broker.backend.calls
    call = ledger.snapshot("fixture")["calls"][0]
    assert call["status"] in {"unknown", "budget_exceeded"}
    assert call["charged_input"] == 64 and call["charged_output"] == 100
    broker.close()


def test_blocked_callback_expires_without_release_and_late_success_cannot_resettle(tmp_path, monkeypatch):
    monkeypatch.setitem(CAPS, "arm_seconds", 0.25)
    entered, release = threading.Event(), threading.Event()

    def block(*_):
        entered.set()
        assert release.wait(3)
        return {"content": "late fixture result, never delivered"}

    broker, ledger = setup(tmp_path, response=block, clock=time.time)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(broker.dispatch, broker.tokens["agent"], "native", request())
        try:
            assert entered.wait(1)
            second = pool.submit(broker.dispatch, broker.tokens["runtime"], "native", request())
            with pytest.raises((TimeoutError, BudgetError)):
                first.result(timeout=1)
            with pytest.raises((TimeoutError, BudgetError)):
                second.result(timeout=1)
            call = ledger.snapshot("fixture")["calls"][0]
            assert call["status"] != "reserved" and call["charged_output"] == 100
            assert not broker.close(timeout=0.03)["drained"]
        finally:
            release.set()
    assert broker.close(timeout=1)["drained"]
    assert ledger.snapshot("fixture")["calls"][0] == call
    with pytest.raises(BudgetError, match="already settled"):
        ledger.settle_call(broker.arm_id, call["request_id"], 64, 16)
    completion = json.loads(next(broker.output.glob("*/backend-completion.json")).read_text())
    assert completion["completed_after_abort"] and not completion["delivery_authorized"]
    assert not list(broker.output.glob("*/response.json"))
    assert len(broker.backend.calls) == 1


def test_close_deadline_remains_bounded_while_another_thread_holds_lifecycle_lock(tmp_path):
    broker, _ = setup(tmp_path)
    broker.start()
    holding, release = threading.Event(), threading.Event()

    def hold():
        with broker.lock:
            holding.set()
            assert release.wait(2)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert holding.wait(1)
        started = time.monotonic()
        assert broker.close(timeout=0.03)["drained"] is False
        assert broker.closed and time.monotonic() - started < 0.5
    finally:
        release.set()
        thread.join(1)
    # Cleanup was scheduled by the only close call, despite its timed-out lock.
    assert broker.shutdown_done.wait(1)
    assert broker.server.socket.fileno() == -1
    assert not broker.thread.is_alive() and not broker.shutdown_failed


@pytest.mark.parametrize("mutation", ["close", "deadline"])
def test_backend_guard_returning_after_revocation_cannot_start_callback(tmp_path, monkeypatch, mutation):
    clock = [1000.0]
    broker, ledger = setup(tmp_path, clock=lambda: clock[0])
    guard = ledger.guard_reserved_call
    entered, release = threading.Event(), threading.Event()
    count = 0

    def block_guard(*args):
        nonlocal count
        count += 1
        result = guard(*args)
        if count == 2:  # Actual worker guard, after foreground guard.
            entered.set()
            assert release.wait(2)
        return result

    monkeypatch.setattr(ledger, "guard_reserved_call", block_guard)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(broker.dispatch, broker.tokens["agent"], "native", request())
        try:
            assert entered.wait(1)
            if mutation == "close":
                assert not broker.close(timeout=0.03)["drained"]
            else:
                clock[0] = broker.deadline
        finally:
            release.set()
        with pytest.raises((PermissionError, BudgetError, TimeoutError)):
            future.result(timeout=1)
    assert broker.close(timeout=1)["drained"]
    assert not broker.backend.calls
    assert ledger.snapshot("fixture")["calls"][0]["status"] in {"unknown", "budget_exceeded"}
    assert list(broker.output.glob("*/settlement.json"))


def test_lifecycle_watchdog_remaining_time_never_waits_for_ledger(tmp_path, monkeypatch):
    clock = [1000.0]
    broker, ledger = setup(tmp_path, clock=lambda: clock[0])

    def no_db(*_):
        raise AssertionError("watchdog must use the immutable arm deadline, not acquire a DB lock")

    monkeypatch.setattr(ledger, "inspect_arm", no_db)
    assert broker._remaining() == 420
    clock[0] = broker.deadline
    assert broker._remaining() == 0
    assert broker.close(timeout=1)["drained"]


@pytest.mark.parametrize("phase", ["encoding", "headers", "body", "flush"])
def test_egress_rechecks_absolute_deadline_between_phases(tmp_path, monkeypatch, phase):
    clock = [1000.0]
    broker, ledger = setup(tmp_path, clock=lambda: clock[0])
    handler = http_handler(monkeypatch, broker, request())
    response = broker.dispatch(broker.tokens["agent"], "native", request())
    handler.model_claimed = True
    encode = transport.json.dumps
    end_headers, write, flush = handler.end_headers, handler.wfile.write, handler.wfile.flush
    calls = []

    def step(name, callback, *args, **kwargs):
        calls.append(name)
        result = callback(*args, **kwargs)
        if phase == name:
            clock[0] += 420
        return result

    monkeypatch.setattr(transport.json, "dumps", lambda *a, **k: step("encoding", encode, *a, **k))
    handler.end_headers = lambda: step("headers", end_headers)
    handler.wfile.write = lambda *a: step("body", write, *a)
    handler.wfile.flush = lambda: step("flush", flush)
    handler.reply(200, response, False)
    assert handler.close_connection
    assert ledger.snapshot("fixture")["study"]["status"] == "halted"
    assert ledger.snapshot("fixture")["calls"][0]["status"] == "settled"
    if phase == "encoding":
        assert handler.statuses == []
    if phase in {"encoding", "headers"}:
        assert "body" not in calls
    if phase == "body":
        assert "flush" not in calls
    broker.close()


def test_real_http_backpressure_hits_absolute_egress_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "EGRESS_SECONDS", 0.15)
    broker, ledger = setup(tmp_path, response=lambda *_: {"content": "x" * 4_000_000})
    broker.start()
    original = broker.server.RequestHandlerClass.reply

    def small_send_buffer(handler, *args):
        handler.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        return original(handler, *args)

    monkeypatch.setattr(broker.server.RequestHandlerClass, "reply", small_send_buffer)
    address = urlsplit(broker.base_url)
    connection = socket.create_connection((address.hostname, address.port), timeout=1)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    raw = json.dumps(request()).encode()
    path = "/r/" + broker.tokens["agent"] + "/api/chat"
    try:
        connection.sendall(f"POST {path} HTTP/1.1\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw)
        # Do not consume the response. A real socket write must be interrupted.
        end = time.monotonic() + 2
        while broker.connections or not broker.backend.calls:
            assert time.monotonic() < end
            time.sleep(0.01)
        assert ledger.snapshot("fixture")["study"]["status"] == "halted"
        assert ledger.snapshot("fixture")["calls"][0]["status"] == "settled"
        assert any(event["code"] in {"http_egress_deadline", "http_delivery_failed"} for event in broker.errors)
    finally:
        connection.close()
        assert broker.close(timeout=1)["drained"]


def test_shutdown_failure_cannot_be_reported_drained(tmp_path, monkeypatch):
    broker, _ = setup(tmp_path)
    http_handler(monkeypatch, broker, request())

    def fail():
        raise OSError("synthetic close failure")

    monkeypatch.setattr(broker.server, "server_close", fail)
    result = broker.close(timeout=1)
    assert result["cleanup_failed"] and not result["drained"]
    assert any(event["code"] == "transport_shutdown_failed" for event in broker.errors)


def test_valid_json_with_short_http_body_is_not_executed(tmp_path, monkeypatch):
    broker, ledger = setup(tmp_path)
    handler = http_handler(monkeypatch, broker, request())
    handler.headers["Content-Length"] = str(int(handler.headers["Content-Length"]) + 1)
    handler.do_POST()
    assert handler.statuses == [409] and not broker.backend.calls
    assert not ledger.snapshot("fixture")["calls"]
    assert len(list(broker.output.glob("http-*.json"))) == 1
    broker.close()


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_real_http_slow_drip_has_absolute_ingress_deadline(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(transport, "INGRESS_SECONDS", 0.2)
    broker, ledger = setup(tmp_path)
    broker.start()
    address = urlsplit(broker.base_url)
    connection = socket.create_connection((address.hostname, address.port), timeout=1)
    path = "/r/" + broker.tokens["agent"] + "/api/chat"
    prefix = f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
    if phase == "body":
        prefix += "Content-Length: 10000\r\n\r\n"
    try:
        connection.sendall(prefix.encode())
        # Drips are more frequent than the socket idle timeout, but the whole
        # header/body attempt must be cut off by the absolute watchdog.
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            try:
                connection.sendall(b"x")
            except OSError:
                break
            time.sleep(0.03)
        assert not broker.backend.calls and not ledger.snapshot("fixture")["calls"]
        assert any(event["code"] == "http_ingress_deadline" for event in broker.errors)
        assert list(broker.output.glob("http-*.json"))
    finally:
        connection.close()
        assert broker.close(timeout=1)["drained"]


def test_close_interrupts_real_http_body_and_reports_connection_drain(tmp_path):
    broker, ledger = setup(tmp_path)
    broker.start()
    address = urlsplit(broker.base_url)
    connection = socket.create_connection((address.hostname, address.port), timeout=1)
    path = "/r/" + broker.tokens["agent"] + "/api/chat"
    try:
        connection.sendall(f"POST {path} HTTP/1.1\r\nContent-Length: 10000\r\n\r\n{{".encode())
        end = time.monotonic() + 1
        while not broker.connections and time.monotonic() < end:
            time.sleep(0.005)
        assert broker.connections
        started = time.monotonic()
        broker.close(timeout=0.03)
        assert time.monotonic() - started < 0.5
        assert broker.close(timeout=1)["drained"]
        assert not broker.backend.calls and not ledger.snapshot("fixture")["calls"]
    finally:
        connection.close()
        broker.close()
