"""Regression tests for StepLogger.run loop behavior."""

import sqlite3

from types import SimpleNamespace

from src.logger import StepLogger


def test_run_resumes_existing_thread_without_reinjecting_prompt() -> None:
    """Subsequent stream calls should resume the thread with None input."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    logger = StepLogger(conn)

    class FakeGraph:
        def __init__(self) -> None:
            self.stream_inputs = []
            self.stream_calls = 0
            self._states = [
                SimpleNamespace(next=("agent",), values={"messages": [{"content": "partial"}]}),
                SimpleNamespace(next=(), values={"messages": [{"content": "done"}]}),
            ]

        def get_state(self, _config):
            return self._states[min(self.stream_calls - 1, len(self._states) - 1)]

        def stream(self, stream_input, config, stream_mode, durability):
            self.stream_inputs.append(stream_input)
            self.stream_calls += 1
            return iter([])

    graph = FakeGraph()
    logger.process_events = lambda events, thread_id: None

    logger.run(graph, "hello", "run-1")

    assert graph.stream_inputs == [{"messages": [("user", "hello")]}, None]


def test_run_nudges_on_empty_terminal_stop() -> None:
    """An empty terminal stop should inject a continuation prompt once."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    logger = StepLogger(conn)

    class FakeGraph:
        def __init__(self) -> None:
            self.stream_inputs = []
            self.stream_calls = 0
            self._states = [
                SimpleNamespace(
                    next=(),
                    values={"messages": [{
                        "content": "",
                        "tool_calls": [],
                        "additional_kwargs": {},
                        "response_metadata": {"finish_reason": "STOP"},
                    }]},
                ),
                SimpleNamespace(next=(), values={"messages": [{"content": "done"}]}),
            ]

        def get_state(self, _config):
            return self._states[min(self.stream_calls - 1, len(self._states) - 1)]

        def stream(self, stream_input, config, stream_mode, durability):
            self.stream_inputs.append(stream_input)
            self.stream_calls += 1
            return iter([])

    graph = FakeGraph()
    logger.process_events = lambda events, thread_id: None

    logger.run(graph, "hello", "run-2")

    assert graph.stream_inputs[0] == {"messages": [("user", "hello")]}
    assert graph.stream_inputs[1] is not None
    assert graph.stream_inputs[1]["messages"][0][0] == "user"
    assert "Continue from where you left off" in graph.stream_inputs[1]["messages"][0][1]
