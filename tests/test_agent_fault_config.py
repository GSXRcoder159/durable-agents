"""Tests for fault injection wrapper configuration."""

from src.agent import build_baseline_graph, build_graph
from src.harness import FaultInjectionWrapper
from src.idempotency import IdempotencyToolWrapper


def test_build_graph_uses_fault_env_configuration(monkeypatch, mem_conn) -> None:
    """Durable graph should honor configured fault type and call number."""
    captured = {}

    def fake_create_react_agent(*, model, tools, checkpointer):
        captured["tools"] = tools
        return object()

    monkeypatch.setattr("src.agent.create_react_agent", fake_create_react_agent)
    monkeypatch.setenv("EXP3_TARGET_TOOL", "write_to_database")
    monkeypatch.setenv("EXP3_FAULT_TYPE", "rate_limit")
    monkeypatch.setenv("EXP3_FAULT_CALL", "7")

    build_graph(mem_conn, model=object())

    wrapped = next(tool for tool in captured["tools"] if tool.name == "write_to_database")
    assert isinstance(wrapped, IdempotencyToolWrapper)
    assert isinstance(wrapped.wrapped_tool, FaultInjectionWrapper)
    assert wrapped.wrapped_tool.fault_type == "rate_limit"
    assert wrapped.wrapped_tool.call_number == 7


def test_build_baseline_graph_uses_fault_env_configuration(monkeypatch, mem_conn) -> None:
    """Baseline graph should honor configured fault type and call number."""
    captured = {}

    def fake_create_react_agent(*, model, tools, checkpointer):
        captured["tools"] = tools
        return object()

    monkeypatch.setattr("src.agent.create_react_agent", fake_create_react_agent)
    monkeypatch.setenv("EXP3_TARGET_TOOL", "write_to_database")
    monkeypatch.setenv("EXP3_FAULT_TYPE", "tool_error")
    monkeypatch.setenv("EXP3_FAULT_CALL", "9")

    build_baseline_graph(mem_conn, model=object())

    wrapped = next(tool for tool in captured["tools"] if tool.name == "write_to_database")
    assert isinstance(wrapped, FaultInjectionWrapper)
    assert wrapped.fault_type == "tool_error"
    assert wrapped.call_number == 9
