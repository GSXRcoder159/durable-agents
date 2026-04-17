"""Experiment 4: Correctness (Non-Determinism After Recovery)
How to run: python scripts/exp4.py

Run the benchmark 5 times cleanly (no crash) and 5 times with a SIGKILL crash
followed by checkpoint recovery. Compare the final write_to_database outputs.
"""

import os
import sys
import json
import sqlite3

from typing import Optional

from src.harness import inject_crash_at_step
from src.recovery import cmd_recover

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

N_RUNS = 5
CRASH_STEP = 8 # crash after summarize
DB_RECORD_ID = "Nexus_Report_001"

BENCHMARK_PROMPT = """
You are a benchmark testing agent running at temperature=0.
You MUST execute the following instructions strictly in sequential order.
CRITICAL RULE: You are forbidden from calling multiple tools at the same time. You must wait for the exact result of the previous tool before calling the next one.

Task Sequence:
1. Use [web_search] with the exact query: "Find the official URL for Project Nexus".
2. Wait for the result. Then, use [extract_data] with the URL you just found, and request the field: "annual_report_link".
3. Wait for the result. Then, use [extract_data] again with the "annual_report_link" you just found, and request the field: "executive_summary_text".
4. Wait for the result. Then, use [summarize] to summarize the "executive_summary_text" you extracted.
5. Wait for the result. Then, use [web_search] again with the query: "Market reactions to " followed by the summarized text.
6. Wait for the result. Finally, use [write_to_database] to save the market reactions you found. Use the record_id: "Nexus_Report_001".
7. After the database write is complete, output the exact string: "BENCHMARK_COMPLETE: Nexus_Report_001 saved."
"""

def db_path_for(label: str, idx: int) -> str:
    return os.path.join(PROJECT_ROOT, f"exp4_{label}_{idx:02d}.sqlite")

def remove_db(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)

def get_write_to_database_call(db_path: str) -> Optional[dict]:
    if not os.path.exists(db_path):
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """SELECT args_json, result
               FROM tool_intents
               WHERE tool_name = 'write_to_database' AND status = 'COMPLETED'
               ORDER BY completed_at DESC
               LIMIT 1"""
        ).fetchone()
    if not row:
        return None
    return {"args": json.loads(row[0]), "result": row[1]}

def count_completed_events(db_path: str, run_id: str) -> int:
    if not os.path.exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        res = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ? AND status = 'COMPLETED'",
            (run_id,)
        ).fetchone()
    return res[0] if res else 0


def run_clean(idx: int) -> Optional[dict]:
    """Run the benchmark task with no crash. Return write_to_database call info."""
    run_id = f"exp4-clean-{idx:02d}"
    path = db_path_for("clean", idx)
    remove_db(path)

    print(f"\n[Run {run_id}] Starting clean run...")
    inject_crash_at_step(run_id, step_id=999, db_path=path) # no crash
    
    result = get_write_to_database_call(path)
    if result:
        record_id = result["args"].get("record_id", "N/A")
        data_preview = str(result["args"].get("data", "N/A"))[:100]
        print(f"[Run {run_id}] OK - write_to_database called with record_id={record_id}, data_preview={data_preview}...")
    else:
        print(f"[Run {run_id}] ERROR - write_to_database call not found in DB.")
    return result

def run_crash(idx: int) -> Optional[dict]:
    """Run the benchmark task with a crash at CRASH_STEP. Return write_to_database call info after recovery."""
    run_id = f"exp4-crash-{idx:02d}"
    path = db_path_for("crash", idx)
    remove_db(path)

    # run until crash
    print(f"\n[Run {run_id}] Starting crash run (will crash at step {CRASH_STEP})...")
    returncode = inject_crash_at_step(run_id, step_id=CRASH_STEP, db_path=path)
    completed_events = count_completed_events(path, run_id)
    print(f"[Run {run_id}] SIGKILL sent (returncode={returncode}), events at crash: {completed_events}")

    # recover
    print(f"[Run {run_id}] RECOVERING from checkpoint after crash...")
    cmd_recover(run_id, path)

    result = get_write_to_database_call(path)
    if result:
        record_id = result["args"].get("record_id", "N/A")
        data_preview = str(result["args"].get("data", "N/A"))[:100]
        print(f"[Run {run_id}] OK - after recovery, write_to_database called with record_id={record_id}, data_preview={data_preview}...")
    else:
        print(f"[Run {run_id}] ERROR - after recovery, write_to_database call not found in DB.")
    return result


def main() -> None:
    os.environ["AGENT_PROMPT"] = BENCHMARK_PROMPT

    print("Experiment 4: Correctness (Non-Determinism After Recovery)")
    print(f"    Running {N_RUNS} clean runs and {N_RUNS} crash+recover runs (crash at step {CRASH_STEP})...\n")

    # clean runs
    print("=== CLEAN RUNS ===")
    crash_results: list[Optional[dict]] = []
    for i in range(1, N_RUNS + 1):
        crash_results.append(run_clean(i))
    
    # crash + recover runs
    print("\n=== CRASH + RECOVER RUNS ===")
    for i in range(1, N_RUNS + 1):
        crash_results.append(run_crash(i)) 

    # analysis
    clean_data = [res["args"]["data"] if res else None for res in crash_results[:N_RUNS]]
    crash_data = [res["args"]["data"] if res else None for res in crash_results[N_RUNS:]]
    all_data = clean_data + crash_data
    exact_match = len(set(all_data)) == 1 if all_data and any(all_data) else False

    if exact_match:
        print(f"\n✅ All runs ({N_RUNS} clean and {N_RUNS} crash+recover (step {CRASH_STEP})) produced the same final data in write_to_database.")
        print(f"    - clean record_id: {crash_results[0]['args']['record_id'] if crash_results[0] else 'N/A'}")
        print(f"    - clean data: {str(crash_results[0]['args']['data']) if crash_results[0] else 'N/A'}")
        print(f"    - crash record_id: {crash_results[-1]['args']['record_id'] if crash_results[-1] else 'N/A'}")
        print(f"    - crash data: {str(crash_results[-1]['args']['data']) if crash_results[-1] else 'N/A'}")
    else:
        print(f"\n⚠️ ERROR: Not all runs produced the same final data in write_to_database.")
        for idx, (clean, crash) in enumerate(zip(clean_data, crash_data), start=1):
            print(f"    - Run {idx:02d} | Clean Data: {str(clean)} | Crash+Recover Data: {str(crash)}")

    # save results to a JSON file
    with open(os.path.join(PROJECT_ROOT, "results/exp4_results.json"), "w") as f:
        json.dump({
            "clean_runs": clean_data,
            "crash_recover_runs": crash_data,
            "exact_match": exact_match
        }, f, indent=2)

    # clean up
    for i in range(1, N_RUNS + 1):
        remove_db(db_path_for("clean", i))
        remove_db(db_path_for("crash", i))


if __name__ == "__main__":
    main()
