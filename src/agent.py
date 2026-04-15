"""Reference ReAct LangGraph agent implementation."""

import os
import sqlite3

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from src.tools import ALL_TOOLS
from src.idempotency import IdempotencyToolWrapper
from src.harness import FaultInjectionWrapper

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

    base_tools = list(ALL_TOOLS)

    target_fault = os.environ.get("EXP3_TARGET_TOOL")
    if target_fault:
        fault_call = int(os.environ.get("EXP3_FAULT_CALL", "1"))
        for i, tool in enumerate(base_tools):
            if tool.name == target_fault:
                print(f"[Agent Builder] Wrapping '{tool.name}' with FaultInjectionWrapper for testing!")
                base_tools[i] = FaultInjectionWrapper(tool, fault_type="timeout", call_number=1)

    wrapped_tools = [IdempotencyToolWrapper(tool, conn) for tool in base_tools]
    return create_react_agent(model=model, tools=wrapped_tools, checkpointer=checkpointer)

def build_baseline_graph(model=None):
    """Build a baseline ReAct agent graph with no checkpointing or idempotency.

    Args:
        model (Unknown, optional): a LangChain chat model. Defaults to None.
            If None, `ChatOpenAI(model="gpt-4.1-mini", temperature=0)` will be used.
    """
    if model is None:
        from langchain_google_genai import ChatGoogleGenerativeAI 
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    base_tools = list(ALL_TOOLS)
    
    target_fault = os.environ.get("EXP3_TARGET_TOOL")
    if target_fault:
        for i, tool in enumerate(base_tools):
            if tool.name == target_fault:
                base_tools[i] = FaultInjectionWrapper(tool, fault_type="timeout", call_number=1)

    return create_react_agent(model=model, tools=base_tools)
