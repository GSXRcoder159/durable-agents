"""Reference ReAct LangGraph agent implementation."""
import sqlite3

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from src.tools import ALL_TOOLS
from src.idempotency import IdempotencyToolWrapper

def build_graph(conn: sqlite3.Connection, model=None):
    """Build and return a compiled LangGraph ReAct agent graph.

    Args:
        conn (sqlite3.Connection): an open sqlite3.Connection
        model (_type_, optional): a LangChain chat model. Defaults to None.
            If None, `ChatOpenAI(model="gpt-4.1-mini", temperature=0)` will be used.
    """
    if model is None:
        from langchain_openai import ChatOpenAI # requires Openai API key in environment
        model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    checkpointer = SqliteSaver(conn)
    checkpointer.setup() # creates `checkpoints` and `writes` tables

    wrapped_tools = [IdempotencyToolWrapper(tool, conn) for tool in ALL_TOOLS]
    return create_react_agent(model=model, tools=wrapped_tools, checkpointer=checkpointer)