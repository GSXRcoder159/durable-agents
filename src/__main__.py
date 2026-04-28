"""Project entry point.

Usage:
    Run agent: `python -m src`
    Run agent with a specific run_id (for testing): `python -m src run <run_id>`
    Inspect a previous run: `python -m src inspect <run_id>`
    Recover a crashed run: `python -m src recover <run_id>`
"""
import os
import sys
import uuid

from typing import Optional
from dotenv import load_dotenv

from src.db import create_shared_connection, setup_aer_tables
from src.agent import build_graph, build_baseline_graph
from src.logger import StepLogger
from src.cli import cmd_inspect
from src.recovery import cmd_recover, cmd_recover_baseline
from src.tools import get_call_counts, reset_call_counts

load_dotenv() # load environment variables from .env file, if it exists

DB_PATH = os.getenv("DB_PATH") or "db.sqlite"

_DEFAULT_MESSAGE = os.getenv("AGENT_PROMPT") or (
    "Search for 'LangGraph ReAct agent' and write a summary of the top result. Then write the summary to the database with record ID 'summary-001'."
)

def cmd_run(input_message: Optional[str] = None, run_id: Optional[str] = None, baseline: bool = False) -> None:
    """Run an agent and print step history.

    Args:
        input_message Optional[str]: The message to process. Defaults to None.
        run_id Optional[str]: The ID of the run. Defaults to None.
        baseline (bool, optional): Whether to run in baseline mode. Defaults to False.
    """
    if run_id is None:
        run_id = str(uuid.uuid4()) # generate short random run ID
    if input_message is None:
        input_message = _DEFAULT_MESSAGE
    
    conn = create_shared_connection(DB_PATH)
    setup_aer_tables(conn)
    graph = build_graph(conn) if not baseline else build_baseline_graph(conn)
    logger = StepLogger(conn)

    print(f"=== Running agent with run_id: {run_id} ===")
    result = logger.run(graph, input_message, run_id)

    # Calculate total token usage across all messages in the run
    total_in = 0
    total_out = 0
    messages = result.get("messages", [])
        
    for msg in messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total_in += msg.usage_metadata.get("input_tokens", 0)
            total_out += msg.usage_metadata.get("output_tokens", 0)
    total_tokens = total_in + total_out

    print("\n--- Run Metrics ---")
    print(f"Total Tokens: {total_tokens} ({total_in} in, {total_out} out)")
    
    print("\n[Detailed Token Breakdown per LLM Call]")
    call_num = 1
    for msg in messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            step_in = msg.usage_metadata.get("input_tokens", 0)
            step_out = msg.usage_metadata.get("output_tokens", 0)
            step_total = step_in + step_out
            
            action_hint = "Final Response / Thought"
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = ", ".join([tc["name"] for tc in msg.tool_calls])
                action_hint = f"Action: {tool_names}"
                
            print(f"  - Call {call_num:02d} | {action_hint:<25} | Tokens: {step_total:<4} ({step_in} in, {step_out} out)")
            call_num += 1


    print("\n[Metrics] Tools actually executed in this run:")
    counts = get_call_counts()
    tools_run = False
    for tool, count in counts.items():
        if count > 0:
            print(f"  - {tool}: {count} times")
            tools_run = True

    if not tools_run:
        print("  - No tools were actually executed (All from Cache/Idempotency).")

    print(f"Use `python -m src inspect {run_id}` to see step logs for this run.")
    print(f"\n=== Run Complete ===\n")
    cmd_inspect(run_id, conn)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "inspect":
        run_id = sys.argv[2]
        conn = create_shared_connection(DB_PATH)
        setup_aer_tables(conn)
        cmd_inspect(run_id, conn)
    elif len(sys.argv) == 3 and sys.argv[1] == "recover":
        run_id = sys.argv[2]
        cmd_recover(run_id, DB_PATH)
    elif len(sys.argv) == 3 and sys.argv[1] == "recover-baseline":
        run_id = sys.argv[2]
        cmd_recover_baseline(run_id, DB_PATH, rerun_thread_id=os.getenv("BASELINE_RERUN_THREAD_ID"))
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        run_id = sys.argv[2]
        cmd_run(run_id=run_id)
    elif len(sys.argv) == 4 and sys.argv[1] == "run" and sys.argv[3] == "True":
        run_id = sys.argv[2]
        cmd_run(run_id=run_id, baseline=True)
    elif len(sys.argv) == 1:
        cmd_run()
    else:
        print(
            "Usage: python -m src "
            "[run <run_id> | inspect <run_id> | recover <run_id> | recover-baseline <run_id>]"
        )
        sys.exit(1)
    
    
