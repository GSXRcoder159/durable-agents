"""Write-ahead step logger for the database."""
import hashlib
import json
import sqlite3

from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from typing import Any, Dict, Optional

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
                cursor = self.conn.execute(
                    "INSERT INTO events (run_id, step_id, step_type, tool_name, input_hash, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (thread_id, payload.get("step"), step_type, tool_name, input_hash, "PENDING")
                )
                self.conn.commit()
                task_id = payload.get("id", "")
                pending[task_id] = cursor.lastrowid # pyright: ignore[reportArgumentType]
            
            elif event_type == "task_result":
                task_id = payload.get("id", "")
                row_id = pending.pop(task_id, None)
                if row_id is not None:
                    result = payload.get("result")
                    output_str = json.dumps(result, default=str)[:4096] # truncate to fit in database
                    new_status = "ERROR" if payload.get("error") else "COMPLETE"
                    self.conn.execute(
                        """UPDATE events SET output = ?, status = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE id = ?""",
                        (output_str, new_status, row_id)
                    )
                    self.conn.commit()
                    last_result = output_str
        
        return last_result

    def run(self, graph: CompiledStateGraph, input_message: str, thread_id: str) -> Optional[str]:
        """Stream `graph` with debug events and log step.

        Args:
            graph (CompiledStateGraph): A compiled LangGraph graph
            input_message (str): The initial input message to the graph
            thread_id (str): The run (thread) ID

        Returns:
            Optional[str]: Serialized result of the last executed graph node, or `None` if no nodes were executed
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        events = graph.stream({"messages": [("user", input_message)]}, config, stream_mode="debug", durability="sync")
        return self.process_events(events, thread_id)
    