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
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.db import create_shared_connection, setup_aer_tables
from src.cli import cmd_inspect
from src.tools import get_call_counts, reset_call_counts
from src.db import get_db_size_kb

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "benchmark_db.sqlite")

BENCHMARK_PROMPT = SHARED_PROMPT = """
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

FAULT RECOVERY RULE (AUTONOMOUS RE-PLANNING):
You are a highly intelligent agent. If a tool encounters a critical error (such as a TimeoutError), it means your current execution path is blocked. 
DO NOT blindly retry the exact same tool immediately. You MUST autonomously figure out a different path or perform a different logical tool action to reset your state before trying the failed objective again.

After the database write is successfully completed, output the exact string: "BENCHMARK_COMPLETE: Nexus_Report_001 saved."
"""


def main() -> None:
    # Generate a unique run_id for this benchmark run (start with "bench-" prefix for easy identification in the DB)
    run_id_base = f"bench-base-{uuid.uuid4().hex[:6]}"
    
    # Making sure we start with a clean slate for the benchmark
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("==================================================")
    print(f"Starting Benchmark Baseline Test")
    print(f"Run ID:  {run_id_base}")
    print(f"DB Path: {DB_PATH}")
    print("==================================================\n")

    print(f"[Storage] Initial DB Size: {get_db_size_kb(DB_PATH):.2f} KB")

    # 1. Send the prompt via environment variable to __main__.py
    env = os.environ.copy()
    env["AGENT_PROMPT"] = BENCHMARK_PROMPT
    env["DB_PATH"] = DB_PATH  
    
    reset_call_counts()

    # 2. Run the Agent
    print("\n[Agent] Starting agent subprocess (Baseline run, no crash)...")
    start_time = time.perf_counter()
    try:
        subprocess.run(
            [sys.executable, "-m", "src", "run", run_id_base],
            env=env,
            cwd=PROJECT_ROOT,
            check=True
        )
        print("\n[Agent] Subprocess finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Agent crashed unexpectedly with code {e.returncode}.")
        sys.exit(1)
    
    end_time = time.perf_counter()
    wall_clock_time = end_time - start_time

    # Get the final DB size after the run to measure storage overhead
    final_size = get_db_size_kb(DB_PATH)
    print(f"\n[Storage] Final DB Size: {final_size:.2f} KB")
    print(f"[Time] Wall-clock time: {wall_clock_time:.2f} seconds")
    
    # Verifying step count
    print(f"\n[Inspect] Verifying step count for run {run_id_base}...\n")
    conn = create_shared_connection(DB_PATH)
    setup_aer_tables(conn)
    cmd_inspect(run_id_base, conn)
    conn.close()

    print("\n==================================================")
    print("✅ Benchmark Baseline Complete.")

if __name__ == "__main__":
    main()