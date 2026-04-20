"""Crash-at-step-9 comparison: baseline (no recovery) vs durable (with recovery).

Runs the standard benchmark task twice, injecting a SIGKILL crash at step 9 of 12
via `inject_crash_at_step`. The baseline run stops there; the durable run resumes
from the SqliteSaver checkpoint via `cmd_recover`. Prints the step path and a
small stats table for side-by-side comparison.

Usage:
    python scripts/crash_at_9.py
"""
import os
import sys
import time
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.harness import inject_crash_at_step
from src.recovery import cmd_recover, cmd_recover_baseline
from src.db import create_shared_connection, setup_aer_tables, get_db_size_kb
from src.cli import cmd_inspect

CRASH_STEP = 9
BASELINE_DB = os.path.join(PROJECT_ROOT, "crash9_baseline.sqlite")
DURABLE_DB = os.path.join(PROJECT_ROOT, "crash9_durable.sqlite")

BENCHMARK_PROMPT = """
You are a benchmark testing agent running at temperature=0.
You must complete the following objective.
CRITICAL RULE: You are forbidden from calling multiple tools at the same time. You must wait for the exact result of the previous tool before calling the next one.

Primary Tasks:
1. Use [web_search] with the exact query: "Find the official URL for Project Nexus".
2. Use [extract_data] with the URL you just found, and request the field: "annual_report_link".
3. Use [extract_data] again with the "annual_report_link", requesting the field: "executive_summary_text".
4. Use [summarize] to summarize the "executive_summary_text".
5. Use [web_search] again with the query: "Market reactions to " followed by the summarized text.
6. Use [write_to_database] to save the market reactions. Use the record_id: "Nexus_Report_001".

After the database write is successfully completed, output the exact string: "BENCHMARK_COMPLETE: Nexus_Report_001 saved."
"""


def remove_db(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)


def completed_event_count(db_path: str, run_id: str) -> int:
    if not os.path.exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ? AND status = 'COMPLETED'",
            (run_id,),
        ).fetchone()
    return row[0] if row else 0


def benchmark_complete(db_path: str, run_id: str) -> bool:
    """True iff a write_to_database call for record_id Nexus_Report_001 completed."""
    if not os.path.exists(db_path):
        return False
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """SELECT 1 FROM tool_intents
               WHERE tool_name = 'write_to_database' AND status = 'COMPLETED'
               LIMIT 1""",
        ).fetchone()
    return bool(row)


def show_path(db_path: str, run_id: str) -> None:
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    cmd_inspect(run_id, conn)
    conn.close()


def run_baseline() -> dict:
    run_id = "crash9-baseline"
    remove_db(BASELINE_DB)

    print("\n" + "=" * 60)
    print(f"[BASELINE] No recovery. Crashing at step {CRASH_STEP}.")
    print("=" * 60)

    os.environ["AGENT_PROMPT"] = BENCHMARK_PROMPT
    t0 = time.perf_counter()
    rc = inject_crash_at_step(run_id, step_id=CRASH_STEP, db_path=BASELINE_DB, baseline=True)
    inject_crash_at_step(run_id, step_id=99, db_path=BASELINE_DB, baseline=True)
    elapsed = time.perf_counter() - t0
    print(f"[BASELINE] subprocess returncode={rc} (SIGKILL expected)")

    print("\n[BASELINE] Step path after crash:")
    show_path(BASELINE_DB, run_id)

    return {
        "label": "baseline (no recovery)",
        "elapsed_s": elapsed,
        "completed_events": completed_event_count(BASELINE_DB, run_id),
        "db_size_kb": get_db_size_kb(BASELINE_DB),
        "task_complete": benchmark_complete(BASELINE_DB, run_id),
    }


def run_durable() -> dict:
    run_id = "crash9-durable"
    remove_db(DURABLE_DB)

    print("\n" + "=" * 60)
    print(f"[DURABLE] With recovery. Crashing at step {CRASH_STEP}, then resuming.")
    print("=" * 60)

    os.environ["AGENT_PROMPT"] = BENCHMARK_PROMPT
    t0 = time.perf_counter()
    rc = inject_crash_at_step(run_id, step_id=CRASH_STEP, db_path=DURABLE_DB)
    print(f"[DURABLE] subprocess returncode={rc} (SIGKILL expected)")
    print(f"[DURABLE] Events at crash: {completed_event_count(DURABLE_DB, run_id)}")

    print("\n[DURABLE] Resuming from checkpoint...")
    cmd_recover(run_id, DURABLE_DB)
    elapsed = time.perf_counter() - t0

    print("\n[DURABLE] Step path after recovery:")
    show_path(DURABLE_DB, run_id)

    return {
        "label": "durable (with recovery)",
        "elapsed_s": elapsed,
        "completed_events": completed_event_count(DURABLE_DB, run_id),
        "db_size_kb": get_db_size_kb(DURABLE_DB),
        "task_complete": benchmark_complete(DURABLE_DB, run_id),
    }


def print_comparison(baseline: dict, durable: dict) -> None:
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    rows = [
        ("Task completed", str(baseline["task_complete"]), str(durable["task_complete"])),
        ("Completed events", str(baseline["completed_events"]), str(durable["completed_events"])),
        ("DB size (KB)", f"{baseline['db_size_kb']:.2f}", f"{durable['db_size_kb']:.2f}"),
        ("Wall-clock (s)", f"{baseline['elapsed_s']:.2f}", f"{durable['elapsed_s']:.2f}"),
    ]
    print(f"| {'Metric':<20} | {'Baseline':<22} | {'Durable':<22} |")
    print(f"|{'-' * 22}|{'-' * 24}|{'-' * 24}|")
    for name, b, d in rows:
        print(f"| {name:<20} | {b:<22} | {d:<22} |")


def main() -> None:
    baseline = run_baseline()
    durable = run_durable()
    print_comparison(baseline, durable)


if __name__ == "__main__":
    main()
