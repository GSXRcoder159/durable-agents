"""Benchmark evaluation for durable-agents.

This script runs the deterministic multi-step benchmark task to:
1. Verify the exact step count (at least 12 steps).
2. Measure the storage overhead (SQLite DB size) of the durable graph.
3. Serve as the foundation for the fault injection experiments.

Usage:
    `python scripts/benchmark.py`
"""
import os
import sys
import uuid
import subprocess
import sqlite3
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.db import create_shared_connection, setup_aer_tables
from src.cli import cmd_inspect
from src.harness import inject_crash_at_step
from src.recovery import cmd_recover


load_dotenv()

DB_PATH = "benchmark_db.sqlite"

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

def get_db_size_kb(path: str) -> float:
    """Helper to record db size in KB"""
    if os.path.exists(path):
        return os.path.getsize(path) / 1024.0
    return 0.0

def main() -> None:
    # Generate a unique run_id for this benchmark run (start with "bench-" prefix for easy identification in the DB)
    run_id = f"bench-{uuid.uuid4().hex[:8]}"
    
    # Making sure we start with a clean slate for the benchmark
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("==================================================")
    print(f"Starting Benchmark Baseline Test")
    print(f"Run ID:  {run_id}")
    print(f"DB Path: {DB_PATH}")
    print("==================================================\n")

    print(f"[Storage] Initial DB Size: {get_db_size_kb(DB_PATH):.2f} KB")

    # 1. Send the prompt via environment variable to __main__.py
    env = os.environ.copy()
    env["AGENT_PROMPT"] = BENCHMARK_PROMPT
    env["DB_PATH"] = DB_PATH  

    print("\n[Agent] Starting agent subprocess (Baseline run, no crash)...")
    
    # 2. Run the Agent
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src", "run", run_id],
            env=env,
            check=True
        )
        print("\n[Agent] Subprocess finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Agent crashed unexpectedly with code {e.returncode}.")
        sys.exit(1)

    # 3. Get the final DB size after the run to measure storage overhead
    final_size = get_db_size_kb(DB_PATH)
    print(f"\n[Storage] Final DB Size: {final_size:.2f} KB")
    
    # Verifying step count
    print(f"\n[Inspect] Verifying step count for run {run_id}...\n")
    conn = create_shared_connection(DB_PATH)
    setup_aer_tables(conn)
    cmd_inspect(run_id, conn)
    conn.close()

    print("\n==================================================")
    print("Benchmark Baseline Complete.")
    print("If you see 12+ steps above, your benchmark task is ready!")
    print("==================================================")

    
    print("\n[Agent] Starting fault injection test at step 6...")
    return_code = inject_crash_at_step(run_id, step_id=6, db_path=DB_PATH)
    print(f"Crash return code: {return_code}")
    cmd_recover(run_id, DB_PATH)
    final_size_recovered = get_db_size_kb(DB_PATH)
    print(f"[Storage] DB Size after recovery: {final_size_recovered:.2f} KB")

if __name__ == "__main__":
    main()