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
        model (Unknown, optional): a LangChain chat model. Defaults to None.
            If None, `ChatOpenAI(model="gpt-4.1-mini", temperature=0)` will be used.
    """
    if model is None:

        #from langchain_openai import ChatOpenAI # requires Openai API key in environment
        #model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        from langchain_google_genai import ChatGoogleGenerativeAI 
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    
    checkpointer = SqliteSaver(conn)
    checkpointer.setup() # creates `checkpoints` and `writes` tables

    wrapped_tools = [IdempotencyToolWrapper(tool, conn) for tool in ALL_TOOLS]
    return create_react_agent(model=model, tools=wrapped_tools, checkpointer=checkpointer)

def build_baseline_graph(model=None):
    """Build a baseline ReAct agent graph with no checkpointing or idempotency.

    Args:
        model (Unknown, optional): a LangChain chat model. Defaults to None.
            If None, `ChatOpenAI(model="gpt-4.1-mini", temperature=0)` will be used.
    """
    if model is None:
        from langchain_openai import ChatOpenAI # requires Openai API key in environment
        model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    return create_react_agent(model=model, tools=ALL_TOOLS)
