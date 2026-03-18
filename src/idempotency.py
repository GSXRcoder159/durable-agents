"""Semantic idempotency for tool executions."""
import hashlib
import json

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

def compute_intent_hash(tool_name: str, args: Dict[str, Any]) -> str:
    """Return a deterministic hash for a tool call. The hash is key-order invariant and unique per (tool_name, args) pair.

    Args:
        tool_name (str): The name of the tool being called
        args (Dict[str, Any]): The arguments being passed to the tool

    Returns:
        str: The deterministic hash for the tool call
    """
    intent = {"tool_name": tool_name, "args": args}
    canonical_json = json.dumps(intent, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical_json.encode()).hexdigest()

class IdempotencyToolWrapper(BaseTool):
    """A LangChain BaseTool wrapper that adds semantic idempotency to tool calls.
    
    It does the following:
    1. Computes the intent hash for a tool call.
    2. Queries the `tool_intents` table for a completed intent with the same hash.
    3. If found, returns the cached result immediatelly.
    4. If not found, inserts a new `PENDING` intent row, executes the tool, and updates the row.
    """

    model_config = {"arbitrary_types_allowed": True}
    wrapped_tool: BaseTool
    conn: Any

    def __init__(self, wrapped_tool: BaseTool, conn: Any, **kwargs: Any) -> None:
        super().__init__(name=wrapped_tool.name, desscription=wrapped_tool.description,
                         args_schema=wrapped_tool.args_schema, wrapped_tool=wrapped_tool,
                         conn=conn, **kwargs)

    def _run(self, *args: Any, config: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        """Execute the tool with semantic idempotency."""
        intent_hash = compute_intent_hash(self.name, kwargs)

        # Cache hit - return chached result immediately
        row = self.conn.execute(
            "SELECT result FROM tool_intents WHERE intent_hash = ? AND status = 'COMPLETED'", (intent_hash,)
        ).fetchone()
        if row is not None:
            return row[0]
        
        # Cache miss - insert PENDING intent, execute tool, and update intent row
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%f") + "Z"
        args_json = json.dumps(kwargs, sort_keys=True, separators=(',', ':'), default=str)
        self.conn.execute(
            "INSERT OR IGNORE INTO tool_intents (intent_hash, tool_name, args_json, status, created_at) VALUES (?, ?, ?, 'PENDING', ?)",
            (intent_hash, self.name, args_json, now_iso)
        )
        self.conn.commit()

        result = self.wrapped_tool._run(*args, config=config, run_manager=run_manager, **kwargs)

        completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%f") + "Z"
        # TODO: str(result) is lossy for non-string return types. For tools that return other types
        # serialize with json.dumps and deserialize on cache hit.
        self.conn.execute(
            """UPDATE tool_intents SET result = ?, status = 'COMPLETED', completed_at = ?
            WHERE intent_hash = ?""",
            (str(result), completed_at, intent_hash)
        )
        self.conn.commit()

        return result
