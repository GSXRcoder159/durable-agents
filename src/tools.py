"""Mock tools for the LangGraph agent."""
from langchain_core.tools import tool

_call_counts: dict[str, int] = {}

def reset_call_counts() -> None:
    """Clear all tracked call counts."""
    _call_counts.clear()

def get_call_counts() -> dict[str, int]:
    """Get a copy of the current per-tool call counts.

    Returns:
        dict[str, int]: A copy of the current per-tool call counts.
    """
    return dict(_call_counts)

@tool
def web_search(query: str) -> str:
    """Mock web search tool."""
    _call_counts[web_search.__name__] = _call_counts.get(web_search.__name__, 0) + 1
    return f"[MOCK SEARCH] Search results for '{query}'"

@tool
def extract_data(url: str, field: str) -> str:
    """Mock data extraction tool."""
    _call_counts[extract_data.__name__] = _call_counts.get(extract_data.__name__, 0) + 1
    return f"[MOCK EXTRACT] Extracted '{field}' from '{url}'"

@tool
def summarize(text: str) -> str:
    """Mock summarization tool."""
    _call_counts[summarize.__name__] = _call_counts.get(summarize.__name__, 0) + 1
    return f"[MOCK SUMMARIZE] Summary of '{text}'"

@tool
def write_to_database(record_id: str, data: str) -> str:
    """Mock database writing tool."""
    _call_counts[write_to_database.__name__] = _call_counts.get(write_to_database.__name__, 0) + 1
    print(f"[MOCK DB WRITE] Writing '{data}' to record '{record_id}'")
    return f"[MOCK DB WRITE] Written '{data}' to record '{record_id}' successfully"

ALL_TOOLS = [web_search, extract_data, summarize, write_to_database]
