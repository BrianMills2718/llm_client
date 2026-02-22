from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import llm_client.__main__ as cli


def _make_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute(
        """
        CREATE TABLE foundation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            project TEXT,
            run_id TEXT,
            trace_id TEXT,
            event_id TEXT,
            event_type TEXT,
            payload TEXT NOT NULL,
            caller TEXT,
            task TEXT
        )
        """
    )
    return db


def _telemetry_payload(*, caller: str, source: str, mode: str, operation: str = "result_model_semantics_adoption") -> str:
    payload = {
        "event_id": "evt_test",
        "event_type": "ConfigChanged",
        "timestamp": "2026-02-22T00:00:00Z",
        "run_id": "run_test",
        "session_id": "sess_test",
        "actor_id": "llm_client:semantics_telemetry",
        "operation": {"name": operation, "version": "0.6.1"},
        "inputs": {
            "artifact_ids": [],
            "params": {
                "caller": caller,
                "config_source": source,
                "result_model_semantics": mode,
                "observed_count": 1,
            },
            "bindings": {},
        },
        "outputs": {"artifact_ids": [], "payload_hashes": []},
    }
    return json.dumps(payload)


def test_cmd_semantics_json_aggregates_adoption(monkeypatch, capsys) -> None:
    db = _make_db()
    db.execute(
        "INSERT INTO foundation_events (timestamp, project, event_type, payload, caller, task) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-02-22T00:00:00Z", "proj", "ConfigChanged", _telemetry_payload(caller="call_llm", source="explicit_config", mode="requested"), "call_llm", "test"),
    )
    db.execute(
        "INSERT INTO foundation_events (timestamp, project, event_type, payload, caller, task) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-02-22T00:01:00Z", "proj", "ConfigChanged", _telemetry_payload(caller="call_llm", source="explicit_config", mode="requested"), "call_llm", "test"),
    )
    db.execute(
        "INSERT INTO foundation_events (timestamp, project, event_type, payload, caller, task) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-02-22T00:02:00Z", "proj", "ConfigChanged", _telemetry_payload(caller="acall_llm", source="env_or_default", mode="legacy"), "acall_llm", "test"),
    )
    # Non-telemetry event should be ignored.
    db.execute(
        "INSERT INTO foundation_events (timestamp, project, event_type, payload, caller, task) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-02-22T00:03:00Z", "proj", "ConfigChanged", _telemetry_payload(caller="call_llm", source="explicit_config", mode="requested", operation="unrelated"), "call_llm", "test"),
    )
    db.commit()

    monkeypatch.setattr(cli, "_connect", lambda: db)
    args = SimpleNamespace(
        project=None,
        caller=None,
        task=None,
        days=None,
        limit=1000,
        format="json",
    )
    cli.cmd_semantics(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total_events"] == 3
    assert any(
        row["caller"] == "call_llm"
        and row["config_source"] == "explicit_config"
        and row["semantics"] == "requested"
        and row["count"] == 2
        for row in data["rows"]
    )
    assert any(
        row["caller"] == "acall_llm"
        and row["config_source"] == "env_or_default"
        and row["semantics"] == "legacy"
        and row["count"] == 1
        for row in data["rows"]
    )


def test_cmd_semantics_table_respects_caller_filter(monkeypatch, capsys) -> None:
    db = _make_db()
    db.execute(
        "INSERT INTO foundation_events (timestamp, project, event_type, payload, caller, task) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-02-22T00:00:00Z", "proj", "ConfigChanged", _telemetry_payload(caller="call_llm", source="explicit_config", mode="requested"), "call_llm", "test"),
    )
    db.execute(
        "INSERT INTO foundation_events (timestamp, project, event_type, payload, caller, task) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-02-22T00:01:00Z", "proj", "ConfigChanged", _telemetry_payload(caller="acall_llm", source="env_or_default", mode="legacy"), "acall_llm", "test"),
    )
    db.commit()

    monkeypatch.setattr(cli, "_connect", lambda: db)
    args = SimpleNamespace(
        project=None,
        caller="call_llm",
        task=None,
        days=None,
        limit=1000,
        format="table",
    )
    cli.cmd_semantics(args)
    out = capsys.readouterr().out
    assert "Semantics Adoption:" in out
    assert "call_llm" in out
    assert "requested" in out
    assert "acall_llm" not in out
