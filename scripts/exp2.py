"""Experiment 2: idempotency replay vs duplicate tool execution baseline."""

from __future__ import annotations

import argparse
import sys
import time
import uuid

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import create_shared_connection, setup_aer_tables
from src.experiment_utils import (
    collect_run_metrics,
    count_completed_tool_events,
    count_completed_tool_intents,
    ensure_clean_db,
    metrics_to_dict,
    run_agent_subprocess,
    storage_overhead_bytes,
    write_results_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "exp2_results.json"
TMP_DIR = PROJECT_ROOT / "results" / "tmp"

REPLAY_PROMPT_TEMPLATE = """
You are a deterministic benchmark agent.
Call [write_to_database] exactly once with:
- record_id: "{record_id}"
- data: "{data}"

Do not call any other tools.
After the write succeeds, output exactly: "IDEMPOTENCY_COMPLETE"
""".strip()


def _durable_run(db_path: str, run_id: str, prompt: str) -> dict:
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    before = count_completed_tool_intents(conn, tool_name="write_to_database")
    conn.close()

    start = time.perf_counter()
    completed = run_agent_subprocess(run_id, db_path, prompt=prompt, baseline=False, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    elapsed = time.perf_counter() - start

    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    metrics = collect_run_metrics(conn, run_id, durable=True)
    after = count_completed_tool_intents(conn, tool_name="write_to_database")
    conn.close()
    metrics.actual_tool_calls = after - before
    metrics.wall_clock_seconds = elapsed
    metrics.storage_overhead_bytes = storage_overhead_bytes(db_path)
    return metrics_to_dict(metrics)


def _baseline_run(db_path: str, run_id: str, prompt: str) -> dict:
    start = time.perf_counter()
    completed = run_agent_subprocess(run_id, db_path, prompt=prompt, baseline=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    elapsed = time.perf_counter() - start

    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    metrics = collect_run_metrics(conn, run_id, durable=False)
    metrics.actual_tool_calls = count_completed_tool_events(conn, run_id, tool_name="write_to_database")
    conn.close()
    metrics.wall_clock_seconds = elapsed
    metrics.storage_overhead_bytes = storage_overhead_bytes(db_path)
    return metrics_to_dict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", default="idempotency_report_001")
    parser.add_argument("--data", default="cached replay payload")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    durable_db = str(TMP_DIR / f"exp2-durable-{uuid.uuid4().hex[:8]}.sqlite")
    baseline_db = str(TMP_DIR / f"exp2-baseline-{uuid.uuid4().hex[:8]}.sqlite")
    ensure_clean_db(durable_db)
    ensure_clean_db(baseline_db)

    prompt = REPLAY_PROMPT_TEMPLATE.format(record_id=args.record_id, data=args.data)

    durable_first = _durable_run(durable_db, f"exp2-durable-original-{uuid.uuid4().hex[:6]}", prompt)
    durable_replay = _durable_run(durable_db, f"exp2-durable-replay-{uuid.uuid4().hex[:6]}", prompt)
    durable_replay["steps_reexecuted"] = durable_replay["completed_steps"]

    baseline_first = _baseline_run(baseline_db, f"exp2-baseline-original-{uuid.uuid4().hex[:6]}", prompt)
    baseline_replay = _baseline_run(baseline_db, f"exp2-baseline-replay-{uuid.uuid4().hex[:6]}", prompt)
    baseline_replay["steps_reexecuted"] = baseline_replay["completed_steps"]

    payload = {
        "experiment": "exp2_idempotency_replay",
        "record_id": args.record_id,
        "data": args.data,
        "baseline": {
            "recovery_mode": "replay_reexecutes_tool",
            "original_run": baseline_first,
            "replay_run": baseline_replay,
        },
        "durable": {
            "recovery_mode": "replay_uses_intent_cache",
            "original_run": durable_first,
            "replay_run": durable_replay,
        },
    }
    write_results_json(str(args.results), payload)
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
