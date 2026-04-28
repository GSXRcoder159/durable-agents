"""Experiment 3: fault injection resilience."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid

from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import ToolMessage

from src.agent import build_baseline_graph, build_graph
from src.db import create_shared_connection, setup_aer_tables
from src.experiment_utils import (
    baseline_rerun_thread_id,
    collect_run_metrics,
    metrics_to_dict,
    storage_overhead_bytes,
    token_usage_for_threads,
    write_results_json,
)
from src.recovery import cmd_recover, cmd_recover_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "exp3_results.json"
TMP_DIR = PROJECT_ROOT / "results" / "tmp"

FAULT_PROMPT = """
You are a benchmark testing agent running at temperature=0.
You must complete the following objective.
CRITICAL RULES:
- You are forbidden from calling multiple tools at the same time.
- You must wait for the exact result of the previous tool before calling the next one.
- Stopping early is a failed run. The task is not complete until the database write succeeds.

Primary tasks:
1. Use [web_search] with query = "Project Nexus overview".
2. Use [extract_data] with url = "https://project-nexus.example/overview", field = "description".
3. Use [summarize] with text = "Summarize the Project Nexus description in one sentence."
4. Use [web_search] with query = "Project Nexus executive summary".
5. Use [extract_data] with url = "https://project-nexus.example/executive-summary", field = "executive_summary".
6. Use [summarize] with text = "Summarize the Project Nexus executive summary in two sentences."
7. Use [web_search] with query = "Project Nexus market reaction".
8. Use [extract_data] with url = "https://project-nexus.example/market-reaction", field = "market_reaction".
9. Use [summarize] with text = "Summarize the Project Nexus market reaction in one sentence."
10. Use [write_to_database] to save "overview summary; executive summary; market reaction summary" with record_id "Nexus_Report_001".

Fault-recovery rule:
If the first attempt to use [write_to_database] fails with a tool error, your current execution path is blocked.
Do not retry [write_to_database] immediately.
Instead, do this fallback path in order:
11. Use [web_search] with query = "Project Nexus recovery context".
12. Use [summarize] with text = "Summarize the Project Nexus recovery context in one sentence."
13. Then use [write_to_database] again with record_id "Nexus_Report_001" and data = "overview summary; executive summary; market reaction summary; recovery context".

After the database write succeeds, output exactly: "BENCHMARK_COMPLETE: Nexus_Report_001 saved."
""".strip()


@contextmanager
def patched_environ(**updates: str):
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def inject_error_state(run_id: str, db_path: str) -> None:
    """Inject an error tool message into the checkpoint to force replanning."""
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    graph = build_graph(conn)
    config = {"configurable": {"thread_id": run_id}}
    state = graph.get_state(config)
    messages = state.values.get("messages", [])

    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        tool_call = messages[-1].tool_calls[0]
        error_msg = ToolMessage(
            content="TimeoutError: Database connection failed. Execution path blocked.",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        graph.update_state(config, {"messages": [error_msg]}, as_node="tools")

    conn.close()


def _token_metrics(db_path: str, thread_ids: list[str], durable: bool) -> tuple[int | None, int | None, int | None]:
    try:
        conn = create_shared_connection(db_path)
        setup_aer_tables(conn)
        graph = build_graph(conn) if durable else build_baseline_graph(conn)
        prompt_tokens, completion_tokens, total_tokens = token_usage_for_threads(graph, thread_ids)
        conn.close()
        return prompt_tokens, completion_tokens, total_tokens
    except Exception:
        return None, None, None


def _run_initial_attempt(run_id: str, db_path: str, baseline: bool, env_updates: dict[str, str]) -> bool:
    cmd = [sys.executable, "-m", "src", "run", run_id]
    if baseline:
        cmd.append("True")
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, **env_updates, "DB_PATH": db_path, "AGENT_PROMPT": FAULT_PROMPT},
        text=True,
        capture_output=True,
    )
    return completed.returncode == 0


def durable_trial(run_id: str, db_path: str, fault_type: str, fault_call: int, max_recoveries: int) -> dict:
    """Run one durable fault-injection trial."""
    start = time.perf_counter()
    env_updates = {
        "CURRENT_RUN_ID": run_id,
        "EXP3_POSITION_MODE": "true",
        "EXP3_TARGET_TOOL": "write_to_database",
        "EXP3_FAULT_TYPE": fault_type,
        "EXP3_FAULT_CALL": str(fault_call),
    }
    attempts = 1
    recovery_attempts = 0
    success = _run_initial_attempt(run_id, db_path, baseline=False, env_updates=env_updates)

    with patched_environ(**env_updates):
        while not success and recovery_attempts < max_recoveries:
            recovery_attempts += 1
            attempts += 1
            inject_error_state(run_id, db_path)
            try:
                cmd_recover(run_id, db_path)
                success = True
            except Exception:
                success = False

    elapsed = time.perf_counter() - start
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    metrics = collect_run_metrics(conn, run_id, durable=True)
    conn.close()
    metrics.wall_clock_seconds = elapsed
    metrics.storage_overhead_bytes = storage_overhead_bytes(db_path)
    prompt_tokens, completion_tokens, total_tokens = _token_metrics(db_path, [run_id], durable=True)
    metrics.prompt_tokens = prompt_tokens
    metrics.completion_tokens = completion_tokens
    metrics.total_tokens = total_tokens

    return {
        "run_id": run_id,
        "fault_type": fault_type,
        "fault_call": fault_call,
        "attempts": attempts,
        "success": success,
        "metrics": metrics_to_dict(metrics),
    }


def baseline_trial(run_id: str, db_path: str, fault_type: str, fault_call: int, max_recoveries: int) -> dict:
    """Run one baseline fault-injection trial with full reruns."""
    start = time.perf_counter()
    env_updates = {
        "CURRENT_RUN_ID": run_id,
        "EXP3_POSITION_MODE": "true",
        "EXP3_TARGET_TOOL": "write_to_database",
        "EXP3_FAULT_TYPE": fault_type,
        "EXP3_FAULT_CALL": str(fault_call),
    }

    attempts = 1
    success = _run_initial_attempt(run_id, db_path, baseline=True, env_updates=env_updates)
    rerun_threads: list[str] = []

    with patched_environ(**env_updates):
        while not success and attempts <= max_recoveries + 1:
            rerun_thread = baseline_rerun_thread_id(run_id, attempts - 1)
            rerun_threads.append(rerun_thread)
            attempts += 1
            try:
                cmd_recover_baseline(
                    run_id,
                    db_path,
                    input_message=FAULT_PROMPT,
                    rerun_thread_id=rerun_thread,
                )
                success = True
            except Exception:
                success = False

    elapsed = time.perf_counter() - start
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    metrics = collect_run_metrics(conn, run_id, durable=False)
    conn.close()
    metrics.wall_clock_seconds = elapsed
    metrics.storage_overhead_bytes = storage_overhead_bytes(db_path)
    prompt_tokens, completion_tokens, total_tokens = _token_metrics(
        db_path,
        [run_id, *rerun_threads],
        durable=False,
    )
    metrics.prompt_tokens = prompt_tokens
    metrics.completion_tokens = completion_tokens
    metrics.total_tokens = total_tokens

    return {
        "run_id": run_id,
        "fault_type": fault_type,
        "fault_call": fault_call,
        "attempts": attempts,
        "success": success,
        "rerun_thread_ids": rerun_threads,
        "metrics": metrics_to_dict(metrics),
    }


def _summarize(trials: list[dict]) -> dict:
    success_count = sum(1 for trial in trials if trial["success"])
    average_time = sum(trial["metrics"]["wall_clock_seconds"] for trial in trials) / len(trials)
    average_attempts = sum(trial["attempts"] for trial in trials) / len(trials)
    return {
        "success_rate": success_count / len(trials),
        "average_wall_clock_seconds": average_time,
        "average_attempts": average_attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-types", nargs="+", default=["timeout", "tool_error", "rate_limit"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fault-call", type=int, default=19)
    parser.add_argument("--max-recoveries", type=int, default=3)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    baseline_trials = []
    durable_trials = []

    for fault_type in args.fault_types:
        for repeat in range(1, args.repeats + 1):
            baseline_run_id = f"exp3-baseline-{fault_type}-{repeat}-{uuid.uuid4().hex[:6]}"
            durable_run_id = f"exp3-durable-{fault_type}-{repeat}-{uuid.uuid4().hex[:6]}"
            baseline_db = str(TMP_DIR / f"{baseline_run_id}.sqlite")
            durable_db = str(TMP_DIR / f"{durable_run_id}.sqlite")

            baseline_trials.append(
                baseline_trial(baseline_run_id, baseline_db, fault_type, args.fault_call, args.max_recoveries)
            )
            durable_trials.append(
                durable_trial(durable_run_id, durable_db, fault_type, args.fault_call, args.max_recoveries)
            )

    payload = {
        "experiment": "exp3_fault_injection_resilience",
        "fault_types": args.fault_types,
        "repeats": args.repeats,
        "fault_call": args.fault_call,
        "baseline": {
            "recovery_mode": "full_rerun_from_start",
            "trials": baseline_trials,
            "summary": _summarize(baseline_trials),
        },
        "durable": {
            "recovery_mode": "resume_from_last_checkpoint",
            "trials": durable_trials,
            "summary": _summarize(durable_trials),
        },
    }
    write_results_json(str(args.results), payload)
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
