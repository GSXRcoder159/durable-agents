"""Local model factory for vLLM models."""

from langchain_openai import ChatOpenAI

from pydantic import SecretStr
from typing import Optional

def build_local_model(model_name: str, base_url: str, api_key: str = "EMPTY", temperature: float = 0, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """Build a ChatOpenAI client from a local vLLM OpenAI-compatible model.

    Args:
        model_name (str): HuggingFace model ID served by vLLM, e.g. `Qwen/Qwen3.5-4B` (must support OpenAI-compatible tool/function calling via vLLM)
        base_url (str): URL of the vLLM server, e.g. `http://localhost:8000/v1` (on GreatLaes use the comput-node IP address)
        api_key (str, optional): dummy API key. Defaults to "EMPTY".
        temperature (float, optional): sampling temperature. Defaults to 0.
        max_tokens (Optional[int], optional): maximum number of tokens. Defaults to None.
            If set to None, the default value is used.

    Returns:
        ChatOpenAI: a `ChatOpenAI` instance that can be passed to `build_graph` or `build_baseline_graph`
    """
    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    return ChatOpenAI(model=model_name, base_url=base_url, api_key=SecretStr(api_key), temperature=temperature, **kwargs)
