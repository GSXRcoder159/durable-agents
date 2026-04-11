"""Callback handlers for LLM and tool usage metrics."""

from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any

class AgentMetricsHandler(BaseCallbackHandler):
    """Track the number of LLM calls."""

    def __init__(self) -> None:
        super().__init__()
        self._call_count: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
    
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        """Increment the call count when an LLM call starts."""
        self._call_count += 1
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Update token usage from the LLM response."""
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            self.prompt_tokens += token_usage.get("prompt_tokens", 0)
            self.completion_tokens += token_usage.get("completion_tokens", 0)
            self.total_tokens += token_usage.get("total_tokens", 0)
    
    def get_summary(self) -> dict[str, int]:
        """Get a summary of the metrics."""
        return {
            "call_count": self._call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }
    
    def reset(self) -> None:
        """Reset the call count."""
        self._call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
