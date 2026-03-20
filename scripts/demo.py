"""Showcase crash recovery.

Show the crash recovery process:
1. Start agent as subprocess
2. SIGKILL agent subprocess at a specific step
3. Inspect the incomplete run to see where it left off
4. Run recovery to resume from the last completed step
5. Inspect the recovered run to see the completed steps and final result

Usage:
    `python scripts/demo.py` - runs the demo
"""
import os
import uuid
import sys
from dotenv import load_dotenv

# add src/ to path so we can import modules from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.harness import inject_crash_at_step
from src.recovery import cmd_recover
from src.db import create_shared_connection, setup_aer_tables
from src.cli import cmd_inspect

load_dotenv()  # load environment variables

CRASH_AT_STEP = 2
DB_PATH = "demo_db.sqlite"

def main() -> None:
    run_id = str(uuid.uuid4())

    print(f"Starting demo: {run_id}, {CRASH_AT_STEP}, {DB_PATH}\n")

    # Inject crash
    print(f"Starting agent subprocess and injecting crash at step {CRASH_AT_STEP}...")
    return_code = inject_crash_at_step(run_id, CRASH_AT_STEP, DB_PATH)
    print(f"Subprocess exited with return code: {return_code} (expected -9 for SIGKILL)\n")
    assert return_code == -9, f"Expected SIGKILL (-9), but got {return_code}"
    print()

    # Inspect incomplete run
    print(f"Inspecting incomplete run {run_id}...")
    conn = create_shared_connection(DB_PATH)
    setup_aer_tables(conn)
    cmd_inspect(run_id, conn)
    conn.close()
    print()

    # Recover run
    print(f"Recovering run {run_id}...")
    final_result = cmd_recover(run_id, DB_PATH)
    print(f"Final result after recovery: {final_result}\n")
    print()

    # Inspect recovered run
    print(f"Inspecting recovered run {run_id}...")
    conn = create_shared_connection(DB_PATH)
    setup_aer_tables(conn)
    cmd_inspect(run_id, conn)
    conn.close()
    print()

if __name__ == "__main__":
    main()
