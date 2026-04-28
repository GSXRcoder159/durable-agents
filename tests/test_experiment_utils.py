"""Tests for shared experiment helpers."""

from pathlib import Path

from src.db import EVENT_STATUS_COMPLETED
from src.experiment_utils import collect_run_metrics, storage_overhead_bytes


def test_storage_overhead_bytes_includes_sidecar_files(tmp_path: Path) -> None:
    """SQLite overhead should include the main db and WAL/SHM sidecars."""
    db_path = tmp_path / "exp.sqlite"
    db_path.write_bytes(b"a" * 10)
    (tmp_path / "exp.sqlite-wal").write_bytes(b"b" * 7)
    (tmp_path / "exp.sqlite-shm").write_bytes(b"c" * 3)

    assert storage_overhead_bytes(str(db_path)) == 20


def test_collect_run_metrics_uses_events_for_baseline(mem_conn) -> None:
    """Baseline actual tool calls should come from completed act events."""
    mem_conn.execute(
        "INSERT INTO events (run_id, step_id, step_type, status) VALUES (?, ?, ?, ?)",
        ("baseline-run", 1, "think", EVENT_STATUS_COMPLETED),
    )
    mem_conn.execute(
        "INSERT INTO events (run_id, step_id, step_type, tool_name, status) VALUES (?, ?, ?, ?, ?)",
        ("baseline-run", 2, "act", "write_to_database", EVENT_STATUS_COMPLETED),
    )
    mem_conn.execute(
        "INSERT INTO events (run_id, step_id, step_type, tool_name, status) VALUES (?, ?, ?, ?, ?)",
        ("baseline-run", 3, "act", "web_search", EVENT_STATUS_COMPLETED),
    )
    mem_conn.commit()

    metrics = collect_run_metrics(mem_conn, "baseline-run", durable=False)

    assert metrics.llm_calls == 1
    assert metrics.actual_tool_calls == 2
    assert metrics.completed_steps == 3


def test_collect_run_metrics_uses_tool_intents_for_durable(mem_conn) -> None:
    """Durable actual tool calls should come from completed tool intents, not act events."""
    mem_conn.execute(
        "INSERT INTO events (run_id, step_id, step_type, status) VALUES (?, ?, ?, ?)",
        ("durable-run", 1, "think", EVENT_STATUS_COMPLETED),
    )
    mem_conn.execute(
        "INSERT INTO events (run_id, step_id, step_type, tool_name, status) VALUES (?, ?, ?, ?, ?)",
        ("durable-run", 2, "act", "write_to_database", EVENT_STATUS_COMPLETED),
    )
    mem_conn.execute(
        "INSERT INTO tool_intents (intent_hash, tool_name, status) VALUES (?, ?, ?)",
        ("hash-1", "write_to_database", "COMPLETED"),
    )
    mem_conn.commit()

    metrics = collect_run_metrics(mem_conn, "durable-run", durable=True)

    assert metrics.llm_calls == 1
    assert metrics.actual_tool_calls == 1
    assert metrics.completed_steps == 2
