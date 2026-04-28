"""Shared helpers for reproducible experiment scripts."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.runnables import RunnableConfig

from src.db import EVENT_STATUS_COMPLETED


@dataclass
class RunMetrics:
    """Normalized metrics for one run or replay attempt."""

    completed_steps: int
    llm_calls: int
    actual_tool_calls: int
    steps_reexecuted: int = 0
    wall_clock_seconds: float = 0.0
    storage_overhead_bytes: int = 0
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def storage_overhead_bytes(db_path: str) -> int:
    """Return SQLite storage including sidecar files."""
    total = 0
    for suffix in ("", "-wal", "-shm"):
        path = f"{db_path}{suffix}"
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total


def ensure_clean_db(db_path: str) -> None:
    """Remove a SQLite database and its sidecars if they exist."""
    for suffix in ("", "-wal", "-shm"):
        path = f"{db_path}{suffix}"
        if os.path.exists(path):
            os.remove(path)


def count_completed_events(conn: sqlite3.Connection, run_id: str, step_type: Optional[str] = None) -> int:
    """Count completed events for one run, optionally filtered by step type."""
    if step_type is None:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ? AND status = ?",
            (run_id, EVENT_STATUS_COMPLETED),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ? AND step_type = ? AND status = ?",
            (run_id, step_type, EVENT_STATUS_COMPLETED),
        ).fetchone()
    return int(row[0]) if row else 0


def count_completed_tool_events(
    conn: sqlite3.Connection,
    run_id: str,
    tool_name: Optional[str] = None,
) -> int:
    """Count completed act events for one run."""
    if tool_name is None:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ? AND step_type = 'act' AND status = ?",
            (run_id, EVENT_STATUS_COMPLETED),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT COUNT(*) FROM events
               WHERE run_id = ? AND step_type = 'act' AND tool_name = ? AND status = ?""",
            (run_id, tool_name, EVENT_STATUS_COMPLETED),
        ).fetchone()
    return int(row[0]) if row else 0


def count_completed_tool_intents(conn: sqlite3.Connection, tool_name: Optional[str] = None) -> int:
    """Count completed tool intents in the current database."""
    if tool_name is None:
        row = conn.execute(
            "SELECT COUNT(*) FROM tool_intents WHERE status = 'COMPLETED'"
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM tool_intents WHERE tool_name = ? AND status = 'COMPLETED'",
            (tool_name,),
        ).fetchone()
    return int(row[0]) if row else 0


def collect_run_metrics(conn: sqlite3.Connection, run_id: str, durable: bool) -> RunMetrics:
    """Collect normalized metrics from the experiment database."""
    completed_steps = count_completed_events(conn, run_id)
    llm_calls = count_completed_events(conn, run_id, step_type="think")
    actual_tool_calls = (
        count_completed_tool_intents(conn) if durable else count_completed_tool_events(conn, run_id)
    )
    return RunMetrics(
        completed_steps=completed_steps,
        llm_calls=llm_calls,
        actual_tool_calls=actual_tool_calls,
    )


def baseline_rerun_thread_id(run_id: str, attempt: int) -> str:
    """Return a unique rerun thread id for a baseline retry."""
    return f"{run_id}::baseline-rerun::{attempt:02d}::{uuid.uuid4().hex[:8]}"


def _state_token_usage(state_values: Any) -> tuple[int, int, int]:
    """Extract token totals from the messages stored in one graph state."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if not isinstance(state_values, dict):
        return 0, 0, 0

    for message in state_values.get("messages", []):
        usage_metadata = getattr(message, "usage_metadata", None)
        if not usage_metadata and isinstance(message, dict):
            usage_metadata = message.get("usage_metadata")
        if not usage_metadata:
            continue

        prompt_tokens += int(usage_metadata.get("input_tokens", 0))
        completion_tokens += int(usage_metadata.get("output_tokens", 0))
        total_tokens += int(usage_metadata.get("total_tokens", 0)) or (
            int(usage_metadata.get("input_tokens", 0)) + int(usage_metadata.get("output_tokens", 0))
        )

    return prompt_tokens, completion_tokens, total_tokens


def token_usage_for_threads(graph: Any, thread_ids: Iterable[str]) -> tuple[int, int, int]:
    """Return aggregate prompt/completion/total tokens across several thread ids."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for thread_id in thread_ids:
        config = RunnableConfig(configurable={"thread_id": thread_id}, recursion_limit=25)
        try:
            state = graph.get_state(config)
        except Exception:
            continue
        state_prompt, state_completion, state_total = _state_token_usage(getattr(state, "values", None))
        prompt_tokens += state_prompt
        completion_tokens += state_completion
        total_tokens += state_total

    return prompt_tokens, completion_tokens, total_tokens


def run_agent_subprocess(
    run_id: str,
    db_path: str,
    *,
    prompt: str,
    baseline: bool = False,
    extra_env: Optional[dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run the project CLI in a subprocess with a prompt and database path."""
    env = os.environ.copy()
    env["DB_PATH"] = db_path
    env["AGENT_PROMPT"] = prompt
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "-m", "src", "run", run_id]
    if baseline:
        cmd.append("True")

    return subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def write_results_json(path: str, payload: dict[str, Any]) -> None:
    """Write pretty JSON results."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def metrics_to_dict(metrics: RunMetrics) -> dict[str, Any]:
    """Convert metrics to a JSON-ready dict."""
    return asdict(metrics)


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
