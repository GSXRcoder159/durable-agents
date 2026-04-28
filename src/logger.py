"""Write-ahead step logger for the database."""
import hashlib
import json
import sqlite3

from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from typing import Any, Dict, Optional
from src.db import EVENT_STATUS_PENDING, EVENT_STATUS_COMPLETED, EVENT_STATUS_ERROR
from src.callbacks import AgentMetricsHandler

def compute_input_hash(inputs: Any) -> str:
    """Compute a hash of the inputs serialized as JSON.

    Args:
        inputs (Any): Any JSON-serializable value (dict, list, str, ...)

    Returns:
        str: First 16 hex characters of the SHA-256 hash
    """
    input_json = json.dumps(inputs, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(input_json.encode()).hexdigest()[:16]

def _extract_tool_name(payload: Dict[str, Any]) -> Optional[str]:
    """Extract the tool name from the a tools-node event payload if it exists.

    Args:
        payload (Dict[str, Any]): event payload

    Returns:
        Optional[str]: tool name of `None` if no tool calls are found
    """
    # either extract from `tool_call` (for tool calls made directly in the input)
    tool_call = payload.get("tool_call")
    if isinstance(tool_call, dict) and "name" in tool_call:
        return tool_call.get("name")
    
    # or from the last message with a tool call (for tool calls made by the model in a message)
    messages = payload.get("messages", [])
    for message in reversed(messages): # find last message with a tool call
        tool_calls = getattr(message, "tool_calls", None) if not isinstance(message, dict) else message.get("tool_calls", [])
        if tool_calls:
            first = tool_calls[0]
            if isinstance(first, dict):
                return first.get("name")
            return getattr(first, "name", None)
    return None

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

class StepLogger:
    """Write-ahead step logger for the database.
    
    Uses the write-ahead pattern:
    1. A ``PENDING`` row is inserted before the step is exectued in `events` table.
    2. After the step is executed, the row is updated to ``COMPLETED`` or ``ERROR``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
    
    def process_events(self, events, thread_id: str) -> Optional[str]:
        """Process `events` iterator and save each execution node as a row.

        Args:
            events: Iterator of debug events from LangGraph graph
            thread_id (str): The run (thread) ID

        Returns:
            Optional[str]: The result of the last executed graph node, or `None`
        """
        pending: Dict[str, int] = {} # maps node id to database row id for pending steps
        last_result: Optional[str] = None

        for event in events:
            event_type = event.get("type")
            payload = event.get("payload", {})

            if event_type == "task":
                node_name = payload.get("name", "unknown")
                step_type = "act" if node_name == "tools" else "think"
                tool_name = _extract_tool_name(payload.get("input", {})) if step_type == "act" else None
                input_hash = compute_input_hash(payload.get("input"))
                # If this exact step already has a pending row (e.g., process crashed after
                # writing PENDING but before task_result), reuse it on recovery.
                step_id = event.get("step")
                existing_pending_row = self.conn.execute(
                    """SELECT id FROM events
                       WHERE run_id = ? AND step_id = ? AND status = ?
                       ORDER BY id DESC LIMIT 1""",
                    (thread_id, step_id, EVENT_STATUS_PENDING),
                ).fetchone()

                if existing_pending_row is not None:
                    row_id = existing_pending_row[0]
                else:
                    # If step_id already exists for this run (completed/error rows), the
                    # graph is retrying, so allocate the next available step id.
                    while self.conn.execute(
                        "SELECT 1 FROM events WHERE run_id = ? AND step_id = ?",
                        (thread_id, step_id),
                    ).fetchone():
                        step_id += 1
                    cursor = self.conn.execute(
                        "INSERT INTO events (run_id, step_id, step_type, tool_name, input_hash, status) VALUES (?, ?, ?, ?, ?, ?)",
                        (thread_id, step_id, step_type, tool_name, input_hash, EVENT_STATUS_PENDING)
                    )
                    self.conn.commit()
                    row_id = cursor.lastrowid  # pyright: ignore[reportArgumentType]

                task_id = payload.get("id", "")
                pending[task_id] = row_id
            
            elif event_type == "task_result":
                task_id = payload.get("id", "")
                row_id = pending.pop(task_id, None)
                if row_id is not None:
                    result = payload.get("result")
                    output_str = json.dumps(result, default=str)[:4096] # truncate to fit in database
                    new_status = EVENT_STATUS_ERROR if payload.get("error") else EVENT_STATUS_COMPLETED
                    self.conn.execute(
                        """UPDATE events SET output = ?, status = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE id = ?""",
                        (output_str, new_status, row_id)
                    )
                    self.conn.commit()
                    last_result = output_str
        
        return last_result

    def run(self, graph: CompiledStateGraph, input_message: str, thread_id: str) -> dict:
        """Stream `graph` with debug events and log step.

        Args:
            graph (CompiledStateGraph): A compiled LangGraph graph
            input_message (str): The initial input message to the graph
            thread_id (str): The run (thread) ID

        Returns:
            Optional[str]: Serialized result of the last executed graph node, or `None` if no nodes were executed
        """
        metrics = AgentMetricsHandler()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "callbacks": [metrics],
                                  "recursion_limit": 25}

        stream_input: Any = {"messages": [("user", input_message)]}
        previous_next: Optional[tuple] = None
        max_run_loops = 64
        max_continuation_nudges = 2
        continuation_nudges = 0

        for _ in range(max_run_loops):
            events = graph.stream(stream_input, config, stream_mode="debug", durability="sync")
            stream_input = None
            self.process_events(events, thread_id)

            state = graph.get_state(config)
            if not state.next:
                state_values = getattr(state, "values", None)
                if continuation_nudges < max_continuation_nudges and _looks_incomplete_terminal_state(state_values):
                    continuation_nudges += 1
                    print("[WARN] Run reached empty terminal state; nudging model to continue.")
                    stream_input = {
                        "messages": [(
                            "user",
                            "Continue from where you left off and complete any remaining required work. "
                            "If tools are needed, call exactly one tool at a time until the task is done.",
                        )]
                    }
                    previous_next = ()
                    continue
                return state_values

            current_next = tuple(state.next)
            if previous_next is not None and current_next == previous_next:
                return getattr(state, "values", None)
            previous_next = current_next

        return graph.get_state(config).values
