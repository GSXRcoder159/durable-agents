"""Project entry point.

Usage:
    Run agent: `python -m src`
    Inspect a previous run: `python -m src inspect <run_id>`
"""
import sys
import uuid

from typing import Optional
from dotenv import load_dotenv

from db import create_shared_connection, setup_aer_tables
from agent import build_graph
from logger import StepLogger
from cli import cmd_inspect

load_dotenv() # load environment variables from .env file, if it exists

DB_PATH = "db.sqlite"

_DEFAULT_MESSAGE = (
    "Search for 'LangGraph crash recover' and write a summary of the top result. Then write the summary to the database with record ID 'summary-001'."
)

def cmd_run(input_message: Optional[str] = None) -> None:
    """Run an agent and print step history.

    Args:
        input_message Optional[str]: The message to process. Defaults to None.
    """
    run_id = str(uuid.uuid4()) # generate short random run ID
    if input_message is None:
        input_message = _DEFAULT_MESSAGE
    
    conn = create_shared_connection(DB_PATH)
    setup_aer_tables(conn)
    graph = build_graph(conn)
    logger = StepLogger(conn)

    print(f"=== Running agent with run_id: {run_id} ===")
    result = logger.run(graph, input_message, run_id)
    print(f"\n=== Run Complete ===\nResult: {result}\n")
    print(f"Use `python -m src inspect {run_id}` to see step logs for this run.")

    cmd_inspect(run_id, conn)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "inspect":
        run_id = sys.argv[2]
        conn = create_shared_connection(DB_PATH)
        setup_aer_tables(conn)
        cmd_inspect(run_id, conn)
    elif len(sys.argv) == 1:
        cmd_run()
    else:
        print("Usage:")
        print("  Run agent: `python -m src`")
        print("  Inspect a previous run: `python -m src inspect <run_id>`")
        sys.exit(1)