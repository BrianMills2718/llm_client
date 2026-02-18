"""Persistent I/O logging for LLM calls and embeddings.

Appends one JSONL record per LLM call to:
    {DATA_ROOT}/{PROJECT}/{PROJECT}_llm_client_data/calls.jsonl
Appends one JSONL record per embedding call to:
    {DATA_ROOT}/{PROJECT}/{PROJECT}_llm_client_data/embeddings.jsonl

Optionally writes both to a SQLite database at LLM_CLIENT_DB_PATH
(default: ~/projects/data/llm_observability.db).

Configured via env vars (library convention — llm_client already auto-loads
from ~/.secrets/api_keys.env):

    LLM_CLIENT_LOG_ENABLED  — "1" (default) or "0" to disable
    LLM_CLIENT_DATA_ROOT    — base dir (default: ~/projects/data)
    LLM_CLIENT_PROJECT      — project name (default: basename(os.getcwd()))
    LLM_CLIENT_DB_PATH      — SQLite DB path (default: ~/projects/data/llm_observability.db)

Or override at runtime via configure().
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_enabled: bool = os.environ.get("LLM_CLIENT_LOG_ENABLED", "1") == "1"
_data_root: Path = Path(os.environ.get("LLM_CLIENT_DATA_ROOT", str(Path.home() / "projects" / "data")))
_project: str | None = os.environ.get("LLM_CLIENT_PROJECT")
_db_path: Path = Path(os.environ.get("LLM_CLIENT_DB_PATH", str(Path.home() / "projects" / "data" / "llm_observability.db")))
_db_conn: sqlite3.Connection | None = None
_db_lock = threading.Lock()


def _get_project() -> str:
    """Get project name, lazily resolving cwd if not configured."""
    if _project is not None:
        return _project
    return Path.cwd().name


def _log_dir() -> Path:
    return _data_root / _get_project() / f"{_get_project()}_llm_client_data"


def configure(
    *,
    enabled: bool | None = None,
    data_root: str | Path | None = None,
    project: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Override logging config at runtime."""
    global _enabled, _data_root, _project, _db_path, _db_conn
    if enabled is not None:
        _enabled = enabled
    if data_root is not None:
        _data_root = Path(data_root)
    if project is not None:
        _project = project
    if db_path is not None:
        with _db_lock:
            if _db_conn is not None:
                _db_conn.close()
                _db_conn = None
            _db_path = Path(db_path)


def log_call(
    *,
    model: str,
    messages: list[dict[str, Any]] | None = None,
    result: Any = None,
    error: Exception | None = None,
    latency_s: float | None = None,
    caller: str = "call_llm",
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Append one JSONL record. Never raises — logging must not break calls."""
    if not _enabled:
        return
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)

        # Extract fields from result if available
        response_content = None
        usage = None
        cost = None
        finish_reason = None
        if result is not None:
            response_content = getattr(result, "content", None)
            usage = getattr(result, "usage", None)
            cost = getattr(result, "cost", None)
            finish_reason = getattr(result, "finish_reason", None)

        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "model": model,
            "messages": _truncate_messages(messages),
            "response": response_content,
            "usage": usage,
            "cost": cost,
            "finish_reason": finish_reason,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "error": str(error) if error else None,
            "caller": caller,
            "task": task,
            "trace_id": trace_id,
        }
        with open(d / "calls.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # SQLite dual-write
        _write_call_to_db(
            timestamp=timestamp,
            model=model,
            messages=_truncate_messages(messages),
            response=response_content,
            usage=usage,
            cost=cost,
            finish_reason=finish_reason,
            latency_s=round(latency_s, 3) if latency_s is not None else None,
            error=str(error) if error else None,
            caller=caller,
            task=task,
            trace_id=trace_id,
        )
    except Exception:
        # Never break LLM calls for logging
        logger.debug("io_log.log_call failed", exc_info=True)


def log_embedding(
    *,
    model: str,
    input_count: int,
    input_chars: int,
    dimensions: int | None = None,
    usage: dict | None = None,
    cost: float | None = None,
    latency_s: float | None = None,
    error: Exception | None = None,
    caller: str = "embed",
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Append one JSONL record for an embedding call. Never raises."""
    if not _enabled:
        return
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "model": model,
            "input_count": input_count,
            "input_chars": input_chars,
            "dimensions": dimensions,
            "usage": usage,
            "cost": cost,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "error": str(error) if error else None,
            "caller": caller,
            "task": task,
            "trace_id": trace_id,
        }
        with open(d / "embeddings.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # SQLite dual-write
        _write_embedding_to_db(
            timestamp=timestamp,
            model=model,
            input_count=input_count,
            input_chars=input_chars,
            dimensions=dimensions,
            usage=usage,
            cost=cost,
            latency_s=round(latency_s, 3) if latency_s is not None else None,
            error=str(error) if error else None,
            caller=caller,
            task=task,
            trace_id=trace_id,
        )
    except Exception:
        logger.debug("io_log.log_embedding failed", exc_info=True)


# ---------------------------------------------------------------------------
# SQLite observability database
# ---------------------------------------------------------------------------

_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    project TEXT,
    model TEXT NOT NULL,
    messages TEXT,
    response TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost REAL,
    finish_reason TEXT,
    latency_s REAL,
    error TEXT,
    caller TEXT,
    task TEXT,
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    project TEXT,
    model TEXT NOT NULL,
    input_count INTEGER,
    input_chars INTEGER,
    dimensions INTEGER,
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    cost REAL,
    latency_s REAL,
    error TEXT,
    caller TEXT,
    task TEXT,
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS task_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    project TEXT,
    task TEXT,
    trace_id TEXT,
    rubric TEXT NOT NULL,
    method TEXT NOT NULL,
    overall_score REAL NOT NULL,
    dimensions TEXT,
    reasoning TEXT,
    output_model TEXT,
    judge_model TEXT,
    agent_spec TEXT,
    prompt_id TEXT,
    cost REAL,
    latency_s REAL,
    git_commit TEXT
);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_calls_model ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_calls_task ON llm_calls(task);
CREATE INDEX IF NOT EXISTS idx_calls_project ON llm_calls(project);
CREATE INDEX IF NOT EXISTS idx_calls_trace_id ON llm_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_emb_timestamp ON embeddings(timestamp);
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
CREATE INDEX IF NOT EXISTS idx_emb_task ON embeddings(task);
CREATE INDEX IF NOT EXISTS idx_emb_project ON embeddings(project);
CREATE INDEX IF NOT EXISTS idx_emb_trace_id ON embeddings(trace_id);
CREATE INDEX IF NOT EXISTS idx_scores_task ON task_scores(task);
CREATE INDEX IF NOT EXISTS idx_scores_rubric ON task_scores(rubric);
CREATE INDEX IF NOT EXISTS idx_scores_project ON task_scores(project);
CREATE INDEX IF NOT EXISTS idx_scores_trace_id ON task_scores(trace_id);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON task_scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_scores_git_commit ON task_scores(git_commit);
"""


def _migrate_db(conn: sqlite3.Connection) -> None:
    """Add missing columns (idempotent). For DBs created before these columns existed."""
    for table in ("llm_calls", "embeddings"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "trace_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN trace_id TEXT")
            prefix = table[:3]
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{prefix}_trace_id ON {table}(trace_id)")

    # task_scores: add git_commit if missing
    scores_cols = {r[1] for r in conn.execute("PRAGMA table_info(task_scores)").fetchall()}
    if scores_cols and "git_commit" not in scores_cols:
        conn.execute("ALTER TABLE task_scores ADD COLUMN git_commit TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_git_commit ON task_scores(git_commit)")

    conn.commit()


def _get_db() -> sqlite3.Connection:
    """Lazy singleton DB connection. Creates tables on first call."""
    global _db_conn
    with _db_lock:
        if _db_conn is not None:
            return _db_conn
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        _db_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
        _db_conn.executescript(_TABLES_SQL)
        _migrate_db(_db_conn)
        _db_conn.executescript(_INDEXES_SQL)
        return _db_conn


def _write_call_to_db(
    *,
    timestamp: str,
    model: str,
    messages: list[dict[str, Any]] | None,
    response: str | None,
    usage: dict | None,
    cost: float | None,
    finish_reason: str | None,
    latency_s: float | None,
    error: str | None,
    caller: str,
    task: str | None,
    trace_id: str | None = None,
) -> None:
    """Insert a call record into SQLite. Never raises."""
    try:
        db = _get_db()
        prompt_tokens = (usage or {}).get("prompt_tokens")
        completion_tokens = (usage or {}).get("completion_tokens")
        total_tokens = (usage or {}).get("total_tokens")
        db.execute(
            """INSERT INTO llm_calls
               (timestamp, project, model, messages, response,
                prompt_tokens, completion_tokens, total_tokens,
                cost, finish_reason, latency_s, error, caller, task, trace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, _get_project(), model,
                json.dumps(messages, default=str) if messages else None,
                response[:5000] if isinstance(response, str) and len(response) > 5000 else response,
                prompt_tokens, completion_tokens, total_tokens,
                cost, finish_reason, latency_s, error, caller, task, trace_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log._write_call_to_db failed", exc_info=True)


def _write_embedding_to_db(
    *,
    timestamp: str,
    model: str,
    input_count: int,
    input_chars: int,
    dimensions: int | None,
    usage: dict | None,
    cost: float | None,
    latency_s: float | None,
    error: str | None,
    caller: str,
    task: str | None,
    trace_id: str | None = None,
) -> None:
    """Insert an embedding record into SQLite. Never raises."""
    try:
        db = _get_db()
        prompt_tokens = (usage or {}).get("prompt_tokens")
        total_tokens = (usage or {}).get("total_tokens")
        db.execute(
            """INSERT INTO embeddings
               (timestamp, project, model, input_count, input_chars, dimensions,
                prompt_tokens, total_tokens, cost, latency_s, error, caller, task, trace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, _get_project(), model,
                input_count, input_chars, dimensions,
                prompt_tokens, total_tokens,
                cost, latency_s, error, caller, task, trace_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log._write_embedding_to_db failed", exc_info=True)


def import_jsonl(jsonl_path: str | Path, table: str = "llm_calls") -> int:
    """Import existing JSONL records into SQLite. Returns count imported.

    Args:
        jsonl_path: Path to the JSONL file (calls.jsonl or embeddings.jsonl).
        table: Target table — "llm_calls" or "embeddings".

    Returns:
        Number of records imported.
    """
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if table not in ("llm_calls", "embeddings"):
        raise ValueError(f"table must be 'llm_calls' or 'embeddings', got {table!r}")

    db = _get_db()
    count = 0

    # Infer project from path: .../data/{project}/{project}_llm_client_data/calls.jsonl
    project = None
    parts = path.parts
    for i, part in enumerate(parts):
        if part.endswith("_llm_client_data") and i > 0:
            project = parts[i - 1]
            break

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        if table == "llm_calls":
            usage = r.get("usage") or {}
            db.execute(
                """INSERT INTO llm_calls
                   (timestamp, project, model, messages, response,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost, finish_reason, latency_s, error, caller, task, trace_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.get("timestamp"), project, r.get("model"),
                    json.dumps(r.get("messages"), default=str) if r.get("messages") else None,
                    r.get("response"),
                    usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                    r.get("cost"), r.get("finish_reason"),
                    r.get("latency_s"), r.get("error"),
                    r.get("caller"), r.get("task"), r.get("trace_id"),
                ),
            )
        else:  # embeddings
            usage = r.get("usage") or {}
            db.execute(
                """INSERT INTO embeddings
                   (timestamp, project, model, input_count, input_chars, dimensions,
                    prompt_tokens, total_tokens, cost, latency_s, error, caller, task, trace_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.get("timestamp"), project, r.get("model"),
                    r.get("input_count"), r.get("input_chars"), r.get("dimensions"),
                    usage.get("prompt_tokens"), usage.get("total_tokens"),
                    r.get("cost"), r.get("latency_s"), r.get("error"),
                    r.get("caller"), r.get("task"), r.get("trace_id"),
                ),
            )
        count += 1

    db.commit()
    return count


def log_score(
    *,
    rubric: str,
    method: str,
    overall_score: float,
    dimensions: dict[str, Any] | None = None,
    reasoning: str | None = None,
    output_model: str | None = None,
    judge_model: str | None = None,
    agent_spec: str | None = None,
    prompt_id: str | None = None,
    cost: float | None = None,
    latency_s: float | None = None,
    task: str | None = None,
    trace_id: str | None = None,
    git_commit: str | None = None,
) -> None:
    """Write a rubric score to the observability DB. Never raises."""
    if not _enabled:
        return
    try:
        # Auto-capture git commit if not provided
        if git_commit is None:
            from llm_client.git_utils import get_git_head

            git_commit = get_git_head()

        timestamp = datetime.now(timezone.utc).isoformat()

        # JSONL append
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": timestamp,
            "rubric": rubric,
            "method": method,
            "overall_score": overall_score,
            "dimensions": dimensions,
            "reasoning": reasoning,
            "output_model": output_model,
            "judge_model": judge_model,
            "agent_spec": agent_spec,
            "prompt_id": prompt_id,
            "cost": cost,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "task": task,
            "trace_id": trace_id,
            "git_commit": git_commit,
        }
        with open(d / "scores.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # SQLite write
        db = _get_db()
        db.execute(
            """INSERT INTO task_scores
               (timestamp, project, task, trace_id, rubric, method,
                overall_score, dimensions, reasoning, output_model,
                judge_model, agent_spec, prompt_id, cost, latency_s,
                git_commit)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, _get_project(), task, trace_id, rubric, method,
                overall_score,
                json.dumps(dimensions, default=str) if dimensions else None,
                reasoning, output_model, judge_model, agent_spec, prompt_id,
                cost, round(latency_s, 3) if latency_s is not None else None,
                git_commit,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.log_score failed", exc_info=True)


def _truncate_messages(
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Deep-copy messages for storage (full content, no truncation)."""
    if messages is None:
        return None
    return [dict(m) for m in messages]


# ---------------------------------------------------------------------------
# Resume / lookup functions
# ---------------------------------------------------------------------------


def lookup_result(trace_id: str) -> dict[str, Any] | None:
    """Look up a successful LLM call by trace_id.

    Returns a dict with response, model, cost, latency_s, finish_reason,
    prompt_tokens, completion_tokens — or None if no successful call found.
    """
    db = _get_db()
    if db is None:
        return None
    row = db.execute(
        "SELECT response, model, cost, latency_s, finish_reason, "
        "prompt_tokens, completion_tokens, timestamp "
        "FROM llm_calls WHERE trace_id = ? AND error IS NULL "
        "ORDER BY timestamp DESC LIMIT 1",
        (trace_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "response": row[0],
        "model": row[1],
        "cost": row[2],
        "latency_s": row[3],
        "finish_reason": row[4],
        "prompt_tokens": row[5],
        "completion_tokens": row[6],
        "timestamp": row[7],
    }


def get_completed_traces(
    *,
    project: str | None = None,
    task: str | None = None,
) -> set[str]:
    """Return all trace_ids that have at least one successful call.

    Filter by project and/or task to scope results.
    """
    db = _get_db()
    if db is None:
        return set()
    clauses = ["error IS NULL", "trace_id IS NOT NULL"]
    params: list[str] = []
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if task is not None:
        clauses.append("task = ?")
        params.append(task)
    where = " AND ".join(clauses)
    rows = db.execute(
        f"SELECT DISTINCT trace_id FROM llm_calls WHERE {where}",  # noqa: S608
        params,
    ).fetchall()
    return {r[0] for r in rows}
