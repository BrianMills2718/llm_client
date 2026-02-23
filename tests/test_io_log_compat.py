from __future__ import annotations

from pathlib import Path

from llm_client import io_log
from llm_client.observability import experiments as obs_experiments
from llm_client.observability import query as obs_query


def test_io_log_start_run_delegates(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_start_run(**kwargs):  # type: ignore[no-untyped-def]
        called.update(kwargs)
        return "rid123"

    monkeypatch.setattr(obs_experiments, "start_run", fake_start_run)

    rid = io_log.start_run(dataset="D", model="M")

    assert rid == "rid123"
    assert called["dataset"] == "D"
    assert called["model"] == "M"


def test_io_log_get_cost_delegates(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_get_cost(**kwargs):  # type: ignore[no-untyped-def]
        called.update(kwargs)
        return 1.23

    monkeypatch.setattr(obs_query, "get_cost", fake_get_cost)

    total = io_log.get_cost(project="proj")

    assert total == 1.23
    assert called["project"] == "proj"


def test_io_log_get_background_mode_adoption_delegates(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_get_background_mode_adoption(**kwargs):  # type: ignore[no-untyped-def]
        called.update(kwargs)
        return {"records_considered": 7}

    monkeypatch.setattr(obs_query, "get_background_mode_adoption", fake_get_background_mode_adoption)

    summary = io_log.get_background_mode_adoption(
        experiments_path="/tmp/e.jsonl",
        run_id_prefix="alpha.",
    )

    assert summary["records_considered"] == 7
    assert called["experiments_path"] == "/tmp/e.jsonl"
    assert called["run_id_prefix"] == "alpha."


def test_io_log_import_jsonl_still_works(monkeypatch, tmp_path: Path) -> None:
    calls_file = tmp_path / "calls.jsonl"
    calls_file.write_text(
        '{"timestamp":"2026-01-01T00:00:00+00:00","project":"p","model":"m","caller":"c"}\n'
    )

    old_enabled = io_log._enabled
    old_root = io_log._data_root
    old_project = io_log._project
    old_db_path = io_log._db_path
    old_db_conn = io_log._db_conn
    try:
        io_log._enabled = True
        io_log._data_root = tmp_path
        io_log._project = "p"
        io_log._db_path = tmp_path / "t.db"
        io_log._db_conn = None
        imported = io_log.import_jsonl(calls_file, table="llm_calls")
        assert imported == 1
    finally:
        io_log._enabled = old_enabled
        io_log._data_root = old_root
        io_log._project = old_project
        io_log._db_path = old_db_path
        if io_log._db_conn is not None:
            io_log._db_conn.close()
        io_log._db_conn = old_db_conn
