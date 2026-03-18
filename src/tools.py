"""Mock tools for the LangGraph agent."""
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Mock web search tool."""
    return f"[MOCK SEARCH] Search results for '{query}'"

@tool
def extract_data(url: str, field: str) -> str:
    """Mock data extraction tool."""
    return f"[MOCK EXTRACT] Extracted '{field}' from '{url}'"

@tool
def summarize(text: str) -> str:
    """Mock summarization tool."""
    return f"[MOCK SUMMARIZE] Summary of '{text}'"

@tool
def write_to_database(record_id: str, data: str) -> str:
    """Mock database writing tool."""
    print(f"[MOCK DB WRITE] Writing '{data}' to record '{record_id}'")
    return f"[MOCK DB WRITE] Written '{data}' to record '{record_id}' successfully"

ALL_TOOLS = [web_search, extract_data, summarize, write_to_database]
