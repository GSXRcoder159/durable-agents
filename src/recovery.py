"""Crash recovery entry point."""
from langchain_core.runnables import RunnableConfig
from typing import Optional

from src.db import create_shared_connection, setup_aer_tables
from src.agent import build_graph
from src.logger import StepLogger
from src.cli import cmd_inspect

def cmd_recover(run_id: str, db_path: str = "db.sqlite", _model=None) -> Optional[str]:
    """Recover a crashed agent run from its last SqliteSaver checkpoint.

    Args:
        run_id (str): The ID of the run to recover
        db_path (str, optional): The path to the SQLite database (cannot be :memory:). Defaults to "db.sqlite".
        _model (_type_, optional): A LangChain chat model to use in the `build_graph`. Defaults to `ChatOpenAI`.

    Returns:
        Optional[str]: The final output of the recovered run, or `None` if already completed.
    """
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    graph = build_graph(conn, model=_model)
    logger = StepLogger(conn)

    # config = {"configurable": {"thread_id": run_id}}

    state = graph.get_state(RunnableConfig(configurable={"thread_id": run_id}))
    if not state.next:
        print(f"Run {run_id} is already completed. Nothing to recover.")
        return None
    
    print(f"=== Recovering run with run_id: {run_id} ===")
    print(f"Resuming from step_id: {state.next} ...")

    events = graph.stream(None, config=RunnableConfig(configurable={"thread_id": run_id}), stream_mode="debug", durability="sync")
    result = logger.process_events(events, run_id)

    print(f"\n=== Recovery Complete ===\nResult: {result}\n")
    cmd_inspect(run_id, conn)
    return result