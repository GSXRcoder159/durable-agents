"""Reference ReAct LangGraph agent implementation."""

import os
import sqlite3

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver, SqliteSaverBase

from src.tools import ALL_TOOLS
from src.idempotency import IdempotencyToolWrapper
from src.harness import FaultInjectionWrapper

def _apply_fault_wrapper(base_tools):
    """Wrap one configured tool with the fault injector when experiment 3 is enabled."""
    target_fault = os.environ.get("EXP3_TARGET_TOOL")
    if not target_fault:
        return base_tools

    fault_type = os.environ.get("EXP3_FAULT_TYPE", "timeout")
    fault_call = int(os.environ.get("EXP3_FAULT_CALL", "1"))

    for i, tool in enumerate(base_tools):
        if tool.name == target_fault:
            print(
                f"[Agent Builder] Wrapping '{tool.name}' with FaultInjectionWrapper "
                f"(type={fault_type}, call={fault_call}) for testing!"
            )
            base_tools[i] = FaultInjectionWrapper(tool, fault_type=fault_type, call_number=fault_call)
            break
    return base_tools

def build_graph(conn: sqlite3.Connection, model=None):
    """Build and return a compiled LangGraph ReAct agent graph.

    Args:
        conn (sqlite3.Connection): an open sqlite3.Connection
        model (Unknown, optional): a LangChain chat model. Defaults to None.
            If None, `ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)` will be used.
    """
    if model is None:
        from langchain_google_genai import ChatGoogleGenerativeAI 
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    
    checkpointer = SqliteSaver(conn)
    checkpointer.setup() # creates `checkpoints` and `writes` tables

    base_tools = _apply_fault_wrapper(list(ALL_TOOLS))

    wrapped_tools = [IdempotencyToolWrapper(tool, conn) for tool in base_tools]
    return create_react_agent(model=model, tools=wrapped_tools, checkpointer=checkpointer)

def build_baseline_graph(conn: sqlite3.Connection, model=None):
    """Build a baseline ReAct agent graph with no checkpointing or idempotency.

    Args:
        conn (sqlite3.Connection): an open sqlite3.Connection
        model (Unknown, optional): a LangChain chat model. Defaults to None.
            If None, `ChatOpenAI(model="gpt-4.1-mini", temperature=0)` will be used.
    """
    if model is None:
        from langchain_google_genai import ChatGoogleGenerativeAI 
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    checkpointer = SqliteSaverBase(conn)
    checkpointer.setup() # creates `checkpoints` and `writes` tables

    base_tools = _apply_fault_wrapper(list(ALL_TOOLS))

    return create_react_agent(model=model, tools=base_tools, checkpointer=checkpointer)
