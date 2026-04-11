"""Callback handlers for LLM and tool usage metrics."""

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any

class LLMCallCounter(BaseCallbackHandler):
    """Track the number of LLM calls."""

    def __init__(self) -> None:
        super().__init__()
        self._call_count: int = 0
    
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        """Increment the call count when an LLM call starts."""
        self._call_count += 1
        return super().on_llm_start(serialized, prompts, **kwargs)
    
    def get_call_count(self) -> int:
        """Get the total number of LLM calls."""
        return self._call_count
    
    def reset(self) -> None:
        """Reset the call count."""
        self._call_count = 0