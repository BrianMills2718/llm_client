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
import uuid
from datetime import date, datetime, timedelta, timezone
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

CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    project TEXT,
    dataset TEXT NOT NULL,
    model TEXT NOT NULL,
    config TEXT,
    metrics_schema TEXT,
    n_items INTEGER DEFAULT 0,
    n_completed INTEGER DEFAULT 0,
    n_errors INTEGER DEFAULT 0,
    summary_metrics TEXT,
    total_cost REAL DEFAULT 0.0,
    wall_time_s REAL,
    git_commit TEXT,
    status TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS experiment_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metrics TEXT NOT NULL,
    predicted TEXT,
    gold TEXT,
    latency_s REAL,
    cost REAL,
    n_tool_calls INTEGER,
    error TEXT,
    extra TEXT,
    UNIQUE(run_id, item_id)
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
CREATE INDEX IF NOT EXISTS idx_expr_dataset ON experiment_runs(dataset);
CREATE INDEX IF NOT EXISTS idx_expr_model ON experiment_runs(model);
CREATE INDEX IF NOT EXISTS idx_expr_project ON experiment_runs(project);
CREATE INDEX IF NOT EXISTS idx_expr_timestamp ON experiment_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_expr_git_commit ON experiment_runs(git_commit);
CREATE INDEX IF NOT EXISTS idx_expri_run_id ON experiment_items(run_id);
CREATE INDEX IF NOT EXISTS idx_expri_item_id ON experiment_items(item_id);
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
                response,
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


def get_cost(
    *,
    trace_id: str | None = None,
    trace_prefix: str | None = None,
    task: str | None = None,
    project: str | None = None,
    since: str | date | None = None,
) -> float:
    """Query cumulative cost from the observability DB.

    At least one filter must be provided. Combines LLM call costs and
    embedding costs. Returns total cost in USD (0.0 if no records found).

    Args:
        trace_id: Sum cost for this exact trace_id.
        trace_prefix: Sum cost for this trace_id and all children
            (matched with ``trace_id = prefix OR trace_id LIKE prefix/%``).
            Mutually exclusive with trace_id.
        task: Sum cost for this task name.
        project: Sum cost for this project.
        since: Only include records on or after this date (ISO string or date).
    """
    if trace_id is not None and trace_prefix is not None:
        raise ValueError("trace_id and trace_prefix are mutually exclusive.")
    if not any([trace_id, trace_prefix, task, project, since]):
        raise ValueError("At least one filter (trace_id, trace_prefix, task, project, since) is required.")
    try:
        db = _get_db()
    except Exception:
        return 0.0

    total = 0.0
    for table in ("llm_calls", "embeddings"):
        clauses: list[str] = ["error IS NULL"]
        params: list[str] = []
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if trace_prefix is not None:
            clauses.append("(trace_id = ? OR trace_id LIKE ?)")
            params.extend([trace_prefix, trace_prefix + "/%"])
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        if project is not None:
            clauses.append("project = ?")
            params.append(project)
        if since is not None:
            since_str = since.isoformat() if isinstance(since, date) else since
            clauses.append("timestamp >= ?")
            params.append(since_str)
        where = " AND ".join(clauses)
        row = db.execute(
            f"SELECT COALESCE(SUM(cost), 0) FROM {table} WHERE {where}",  # noqa: S608
            params,
        ).fetchone()
        total += row[0] if row else 0.0
    return total


def get_trace_tree(
    trace_prefix: str,
    *,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Roll up child traces under a parent prefix.

    Hierarchical trace_ids use ``/`` as separator:
    ``"openclaw.morning_brief/sam_gov_research_abc"``

    Given prefix ``"openclaw.morning_brief"``, returns every distinct
    trace_id starting with that prefix, with cost/call/error rollup.

    Args:
        trace_prefix: The parent trace_id prefix (matched with ``LIKE prefix/%``
            and also the exact prefix itself).
        days: Look-back window (default 7).

    Returns:
        List of dicts sorted by last_seen descending. Each dict has:
        trace_id, project, task, total_cost_usd, call_count, error_count,
        first_seen, last_seen, models_used, depth (0 = exact match, 1+ = children).
    """
    db = _get_db()
    if db is None:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()

    # Match exact prefix OR anything under it (prefix/...)
    like_pattern = trace_prefix + "/%"

    rows = db.execute(
        """SELECT
            trace_id,
            COALESCE(project, 'unknown') as project,
            COALESCE(task, 'untagged') as task,
            ROUND(SUM(CASE WHEN error IS NULL THEN cost ELSE 0 END), 6) as total_cost,
            COUNT(*) as call_count,
            SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
            MIN(timestamp) as first_seen,
            MAX(timestamp) as last_seen,
            GROUP_CONCAT(DISTINCT model) as models_used
        FROM llm_calls
        WHERE trace_id IS NOT NULL
          AND (trace_id = ? OR trace_id LIKE ?)
          AND timestamp >= ?
        GROUP BY trace_id
        ORDER BY MAX(timestamp) DESC""",
        (trace_prefix, like_pattern, cutoff_iso),
    ).fetchall()

    prefix_len = len(trace_prefix)
    result = []
    for r in rows:
        tid = r[0]
        # depth: 0 = exact match, 1 = direct child, 2+ = deeper
        if tid == trace_prefix:
            depth = 0
        else:
            # tid starts with prefix/ — count remaining slashes + 1
            suffix = tid[prefix_len + 1:]  # skip the /
            depth = suffix.count("/") + 1

        result.append({
            "trace_id": tid,
            "depth": depth,
            "project": r[1],
            "task": r[2],
            "total_cost_usd": r[3],
            "call_count": r[4],
            "error_count": r[5],
            "first_seen": r[6],
            "last_seen": r[7],
            "models_used": r[8].split(",") if r[8] else [],
        })

    return result


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def start_run(
    *,
    dataset: str,
    model: str,
    config: dict[str, Any] | None = None,
    metrics_schema: list[str] | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    project: str | None = None,
) -> str:
    """Register an experiment run. Returns run_id.

    Args:
        dataset: Dataset name (e.g. "HotpotQA", "MuSiQue").
        model: Model used (e.g. "gemini/gemini-3-flash").
        config: Run configuration (backend, mode, timeout, etc.).
        metrics_schema: Metric names items will report (e.g. ["em", "f1", "llm_em"]).
        run_id: Explicit run ID. Auto-generated UUID if None.
        git_commit: Git SHA. Auto-captured from HEAD if None.
        project: Project name override. Uses _get_project() if None.

    Returns:
        The run_id string.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    if git_commit is None:
        from llm_client.git_utils import get_git_head
        git_commit = get_git_head()

    timestamp = datetime.now(timezone.utc).isoformat()
    proj = project or _get_project()

    # JSONL append
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "run_start",
            "run_id": run_id,
            "timestamp": timestamp,
            "project": proj,
            "dataset": dataset,
            "model": model,
            "config": config,
            "metrics_schema": metrics_schema,
            "git_commit": git_commit,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("io_log.start_run JSONL write failed", exc_info=True)

    # SQLite write
    try:
        db = _get_db()
        db.execute(
            """INSERT INTO experiment_runs
               (run_id, timestamp, project, dataset, model, config,
                metrics_schema, git_commit, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')""",
            (
                run_id, timestamp, proj, dataset, model,
                json.dumps(config, default=str) if config else None,
                json.dumps(metrics_schema) if metrics_schema else None,
                git_commit,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.start_run DB write failed", exc_info=True)

    return run_id


def log_item(
    *,
    run_id: str,
    item_id: str,
    metrics: dict[str, Any],
    predicted: str | None = None,
    gold: str | None = None,
    latency_s: float | None = None,
    cost: float | None = None,
    n_tool_calls: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log one item result. Dual-write SQLite + JSONL. Never raises."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # JSONL append
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "item",
            "run_id": run_id,
            "item_id": item_id,
            "timestamp": timestamp,
            "metrics": metrics,
            "predicted": predicted,
            "gold": gold,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "cost": cost,
            "n_tool_calls": n_tool_calls,
            "error": error,
            "extra": extra,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("io_log.log_item JSONL write failed", exc_info=True)

    # SQLite write
    try:
        db = _get_db()
        db.execute(
            """INSERT OR REPLACE INTO experiment_items
               (run_id, item_id, timestamp, metrics, predicted, gold,
                latency_s, cost, n_tool_calls, error, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, item_id, timestamp,
                json.dumps(metrics, default=str),
                predicted, gold,
                round(latency_s, 3) if latency_s is not None else None,
                cost, n_tool_calls, error,
                json.dumps(extra, default=str) if extra else None,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.log_item DB write failed", exc_info=True)


def finish_run(
    *,
    run_id: str,
    summary_metrics: dict[str, Any] | None = None,
    status: str = "completed",
    wall_time_s: float | None = None,
) -> dict[str, Any]:
    """Finalize run. Auto-aggregates metrics from items if summary_metrics not provided.

    Returns the final run record as a dict.
    """
    db = _get_db()

    # Fetch items for this run
    item_rows = db.execute(
        "SELECT metrics, cost, error FROM experiment_items WHERE run_id = ?",
        (run_id,),
    ).fetchall()

    n_items = len(item_rows)
    n_errors = sum(1 for r in item_rows if r[2] is not None)
    n_completed = n_items - n_errors
    total_cost = sum(r[1] or 0.0 for r in item_rows)

    # Auto-aggregate metrics if not provided
    if summary_metrics is None:
        # Get metrics_schema from run
        schema_row = db.execute(
            "SELECT metrics_schema FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        schema: list[str] = []
        if schema_row and schema_row[0]:
            schema = json.loads(schema_row[0])

        summary_metrics = {}
        for metric_name in schema:
            values = []
            for r in item_rows:
                m = json.loads(r[0])
                v = m.get(metric_name)
                if v is not None:
                    values.append(float(v))
            if values:
                summary_metrics[f"avg_{metric_name}"] = round(
                    100.0 * sum(values) / len(values), 2
                )

    # Update run record
    try:
        db.execute(
            """UPDATE experiment_runs
               SET n_items = ?, n_completed = ?, n_errors = ?,
                   summary_metrics = ?, total_cost = ?,
                   wall_time_s = ?, status = ?
               WHERE run_id = ?""",
            (
                n_items, n_completed, n_errors,
                json.dumps(summary_metrics, default=str) if summary_metrics else None,
                round(total_cost, 6),
                round(wall_time_s, 1) if wall_time_s is not None else None,
                status, run_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.finish_run DB update failed", exc_info=True)

    # JSONL append
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "run_finish",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_items": n_items,
            "n_completed": n_completed,
            "n_errors": n_errors,
            "summary_metrics": summary_metrics,
            "total_cost": round(total_cost, 6),
            "wall_time_s": round(wall_time_s, 1) if wall_time_s is not None else None,
            "status": status,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("io_log.finish_run JSONL write failed", exc_info=True)

    # Return the full run record
    row = db.execute(
        """SELECT run_id, timestamp, project, dataset, model, config,
                  metrics_schema, n_items, n_completed, n_errors,
                  summary_metrics, total_cost, wall_time_s, git_commit, status
           FROM experiment_runs WHERE run_id = ?""",
        (run_id,),
    ).fetchone()

    if row is None:
        return {"run_id": run_id, "status": status}

    return {
        "run_id": row[0],
        "timestamp": row[1],
        "project": row[2],
        "dataset": row[3],
        "model": row[4],
        "config": json.loads(row[5]) if row[5] else None,
        "metrics_schema": json.loads(row[6]) if row[6] else None,
        "n_items": row[7],
        "n_completed": row[8],
        "n_errors": row[9],
        "summary_metrics": json.loads(row[10]) if row[10] else None,
        "total_cost": row[11],
        "wall_time_s": row[12],
        "git_commit": row[13],
        "status": row[14],
    }


def get_runs(
    *,
    dataset: str | None = None,
    model: str | None = None,
    project: str | None = None,
    since: str | date | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query experiment runs, newest first."""
    db = _get_db()

    clauses: list[str] = []
    params: list[Any] = []

    if dataset is not None:
        clauses.append("dataset = ?")
        params.append(dataset)
    if model is not None:
        clauses.append("model = ?")
        params.append(model)
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if since is not None:
        since_str = since.isoformat() if isinstance(since, date) else since
        clauses.append("timestamp >= ?")
        params.append(since_str)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    rows = db.execute(
        f"""SELECT run_id, timestamp, project, dataset, model,
                   n_items, n_completed, n_errors,
                   summary_metrics, total_cost, wall_time_s,
                   git_commit, status
            FROM experiment_runs
            {where}
            ORDER BY timestamp DESC
            LIMIT ?""",  # noqa: S608
        params,
    ).fetchall()

    results = []
    for r in rows:
        results.append({
            "run_id": r[0],
            "timestamp": r[1],
            "project": r[2],
            "dataset": r[3],
            "model": r[4],
            "n_items": r[5],
            "n_completed": r[6],
            "n_errors": r[7],
            "summary_metrics": json.loads(r[8]) if r[8] else None,
            "total_cost": r[9],
            "wall_time_s": r[10],
            "git_commit": r[11],
            "status": r[12],
        })
    return results


def get_run_items(run_id: str) -> list[dict[str, Any]]:
    """All items for a run, ordered by timestamp."""
    db = _get_db()

    rows = db.execute(
        """SELECT item_id, timestamp, metrics, predicted, gold,
                  latency_s, cost, n_tool_calls, error, extra
           FROM experiment_items
           WHERE run_id = ?
           ORDER BY timestamp""",
        (run_id,),
    ).fetchall()

    results = []
    for r in rows:
        results.append({
            "item_id": r[0],
            "timestamp": r[1],
            "metrics": json.loads(r[2]) if r[2] else {},
            "predicted": r[3],
            "gold": r[4],
            "latency_s": r[5],
            "cost": r[6],
            "n_tool_calls": r[7],
            "error": r[8],
            "extra": json.loads(r[9]) if r[9] else None,
        })
    return results


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """Side-by-side summary_metrics for 2+ runs, with deltas from first run."""
    if len(run_ids) < 2:
        raise ValueError("compare_runs requires at least 2 run_ids.")

    db = _get_db()
    runs = []
    for rid in run_ids:
        row = db.execute(
            """SELECT run_id, dataset, model, n_items, n_completed, n_errors,
                      summary_metrics, total_cost, wall_time_s, status, timestamp
               FROM experiment_runs WHERE run_id = ?""",
            (rid,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {rid}")
        runs.append({
            "run_id": row[0],
            "dataset": row[1],
            "model": row[2],
            "n_items": row[3],
            "n_completed": row[4],
            "n_errors": row[5],
            "summary_metrics": json.loads(row[6]) if row[6] else {},
            "total_cost": row[7],
            "wall_time_s": row[8],
            "status": row[9],
            "timestamp": row[10],
        })

    # Compute deltas from first run (baseline)
    baseline = runs[0]["summary_metrics"]
    deltas = []
    for run in runs[1:]:
        d: dict[str, float] = {}
        for k, v in run["summary_metrics"].items():
            if k in baseline and isinstance(v, (int, float)) and isinstance(baseline[k], (int, float)):
                d[k] = round(v - baseline[k], 2)
        deltas.append(d)

    return {
        "runs": runs,
        "deltas_from_first": deltas,
    }
