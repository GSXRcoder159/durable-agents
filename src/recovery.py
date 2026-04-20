"""Crash recovery entry point."""
import os

from langchain_core.runnables import RunnableConfig
from typing import Any, Optional

from src.db import create_shared_connection, setup_aer_tables
from src.agent import build_graph, build_baseline_graph
from src.logger import StepLogger
from src.cli import cmd_inspect

_CONTINUE_PROMPT = (
    "Continue from where you left off and complete any remaining required work. "
    "If tools are needed, call exactly one tool at a time until the task is done."
)

def _looks_incomplete_terminal_state(values: Any) -> bool:
    """Return True if the latest assistant message looks like an empty early stop."""
    if not isinstance(values, dict):
        return False
    messages = values.get("messages")
    if not isinstance(messages, list) or not messages:
        return False

    last = messages[-1]
    if isinstance(last, dict):
        content = last.get("content")
        tool_calls = last.get("tool_calls") or []
        invalid_tool_calls = last.get("invalid_tool_calls") or []
        additional_kwargs = last.get("additional_kwargs") or {}
        response_metadata = last.get("response_metadata") or {}
    else:
        content = getattr(last, "content", None)
        tool_calls = getattr(last, "tool_calls", None) or []
        invalid_tool_calls = getattr(last, "invalid_tool_calls", None) or []
        additional_kwargs = getattr(last, "additional_kwargs", None) or {}
        response_metadata = getattr(last, "response_metadata", None) or {}

    finish_reason = response_metadata.get("finish_reason")
    has_function_call = bool(additional_kwargs.get("function_call"))
    has_content = bool(str(content).strip()) if content is not None else False

    return (
        not has_content
        and not tool_calls
        and not has_function_call
        and (finish_reason in {"STOP", "MALFORMED_FUNCTION_CALL"} or bool(invalid_tool_calls))
    )

def cmd_recover_baseline(run_id: str, db_path: str = "db.sqlite", input_message: Optional[str] = None, _model=None):
    """Recover by rerunning the baseline graph from scratch."""
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    graph = build_baseline_graph(conn, model=_model)
    logger = StepLogger(conn)

    if input_message is None:
        input_message = os.getenv("AGENT_PROMPT") or (
            "Search for 'LangGraph ReAct agent' and write a summary of the top result. "
            "Then write the summary to the database with record ID 'summary-001'."
        )

    # Use a fresh thread id so the rerun starts from scratch, but keep logging under
    # the original run_id so the baseline path shows crash + full rerun together.
    rerun_thread_id = f"{run_id}::baseline-rerun"
    config = RunnableConfig(configurable={"thread_id": rerun_thread_id}, recursion_limit=25)

    print(f"=== Baseline Recovery (re-run) with run_id: {run_id} ===")

    result: Optional[str] = None
    stream_input: Any = {"messages": [("user", input_message)]}
    previous_next: Optional[tuple] = None
    max_recovery_loops = 64
    max_continuation_nudges = 2
    continuation_nudges = 0

    for _ in range(max_recovery_loops):
        events = graph.stream(stream_input, config=config, stream_mode="debug", durability="sync")
        stream_input = None
        loop_result = logger.process_events(events, run_id)
        if loop_result is not None:
            result = loop_result

        state = graph.get_state(config)
        if not state.next:
            state_values = getattr(state, "values", None)
            if continuation_nudges < max_continuation_nudges and _looks_incomplete_terminal_state(state_values):
                continuation_nudges += 1
                print("[WARN] Baseline rerun reached empty terminal state; nudging model to continue.")
                stream_input = {"messages": [("user", _CONTINUE_PROMPT)]}
                previous_next = ()
                continue
            break

        current_next = tuple(state.next)
        if previous_next is not None and current_next == previous_next:
            print(f"[WARN] Baseline rerun made no progress; still waiting on: {state.next}")
            break
        previous_next = current_next

    print("\n=== Baseline Recovery Complete ===\n")
    cmd_inspect(run_id, conn)
    return result

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

    config = RunnableConfig(configurable={"thread_id": run_id}, recursion_limit=25)
    state = graph.get_state(config)
    if not state.next:
        print(f"Run {run_id} is already completed. Nothing to recover.")
        return None
    
    print(f"=== Recovering run with run_id: {run_id} ===")
    print(f"Resuming from step_id: {state.next} ...")

    result: Optional[str] = None
    previous_next = tuple(state.next)
    max_recovery_loops = 64
    max_continuation_nudges = 2
    continuation_nudges = 0
    stream_input: Any = None

    for _ in range(max_recovery_loops):
        events = graph.stream(stream_input, config=config, stream_mode="debug", durability="sync")
        stream_input = None
        loop_result = logger.process_events(events, run_id)
        if loop_result is not None:
            result = loop_result

        state = graph.get_state(config)
        if not state.next:
            state_values = getattr(state, "values", None)
            if continuation_nudges < max_continuation_nudges and _looks_incomplete_terminal_state(state_values):
                continuation_nudges += 1
                print("[WARN] Recovery reached empty terminal state; nudging model to continue.")
                stream_input = {"messages": [("user", _CONTINUE_PROMPT)]}
                previous_next = ()
                continue
            break

        current_next = tuple(state.next)
        if current_next == previous_next:
            print(f"[WARN] Recovery made no progress; still waiting on: {state.next}")
            break
        previous_next = current_next

    print(f"\n=== Recovery Complete ===\nResult: {result}\n")
    cmd_inspect(run_id, conn)
    return result
