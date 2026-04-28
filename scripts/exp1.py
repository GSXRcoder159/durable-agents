"""Experiment 1: crash recovery vs full rerun baseline."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import build_baseline_graph, build_graph
from src.db import create_shared_connection, setup_aer_tables
from src.experiment_utils import (
    baseline_rerun_thread_id,
    collect_run_metrics,
    count_completed_events,
    ensure_clean_db,
    metrics_to_dict,
    storage_overhead_bytes,
    token_usage_for_threads,
    write_results_json,
)
from src.harness import inject_crash_at_step
from src.recovery import cmd_recover, cmd_recover_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "exp1_results.json"
TMP_DIR = PROJECT_ROOT / "results" / "tmp"

BENCHMARK_PROMPT = """
You are a benchmark testing agent running at temperature=0.
You MUST execute the following instructions strictly in sequential order.
CRITICAL RULES:
- You are forbidden from calling multiple tools at the same time.
- You must wait for the exact result of the previous tool before calling the next one.
- Stopping early is a failed run. The task is not complete until step 10 is executed.

Perform the following steps exactly as written, using the exact literal arguments shown:

1. Use [web_search] with query = "Project Nexus overview"
2. Use [extract_data] with url = "https://project-nexus.example/overview", field = "description"
3. Use [summarize] with text = "Summarize the Project Nexus description in one sentence."
4. Use [web_search] with query = "Project Nexus executive summary"
5. Use [extract_data] with url = "https://project-nexus.example/executive-summary", field = "executive_summary"
6. Use [summarize] with text = "Summarize the Project Nexus executive summary in two sentences."
7. Use [web_search] with query = "Project Nexus market reaction"
8. Use [extract_data] with url = "https://project-nexus.example/market-reaction", field = "market_reaction"
9. Use [summarize] with text = "Summarize the Project Nexus market reaction in one sentence."
10. Use [write_to_database] with record_id = "crash_recovery_report_001", data = "overview summary; executive summary; market reaction summary"

After writing to the database, output exactly: "BENCHMARK_COMPLETE"
""".strip()


@contextmanager
def patched_environ(**updates: str):
    """Temporarily patch environment variables used by the subprocess harness."""
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


def run_baseline_trial(run_id: str, db_path: str, crash_step: int) -> dict:
    """Crash a baseline run, then recover by rerunning from scratch."""
    ensure_clean_db(db_path)
    rerun_thread = baseline_rerun_thread_id(run_id, attempt=1)

    start = time.perf_counter()
    with patched_environ(AGENT_PROMPT=BENCHMARK_PROMPT):
        rc = inject_crash_at_step(run_id, crash_step, db_path, baseline=True)
        if rc != -9:
            raise RuntimeError(f"Expected SIGKILL (-9) during baseline crash injection, got {rc}")
        conn = create_shared_connection(db_path)
        setup_aer_tables(conn)
        pre_crash_steps = count_completed_events(conn, run_id)
        conn.close()
        cmd_recover_baseline(
            run_id,
            db_path,
            input_message=BENCHMARK_PROMPT,
            rerun_thread_id=rerun_thread,
        )
    elapsed = time.perf_counter() - start

    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    metrics = collect_run_metrics(conn, run_id, durable=False)
    conn.close()
    metrics.steps_reexecuted = metrics.completed_steps - pre_crash_steps
    metrics.wall_clock_seconds = elapsed
    metrics.storage_overhead_bytes = storage_overhead_bytes(db_path)
    prompt_tokens, completion_tokens, total_tokens = _token_metrics(
        db_path,
        [run_id, rerun_thread],
        durable=False,
    )
    metrics.prompt_tokens = prompt_tokens
    metrics.completion_tokens = completion_tokens
    metrics.total_tokens = total_tokens
    return {
        "run_id": run_id,
        "crash_step": crash_step,
        "pre_crash_completed_steps": pre_crash_steps,
        "rerun_thread_id": rerun_thread,
        "metrics": metrics_to_dict(metrics),
    }


def run_durable_trial(run_id: str, db_path: str, crash_step: int) -> dict:
    """Crash a durable run, then recover from the last checkpoint."""
    ensure_clean_db(db_path)

    start = time.perf_counter()
    with patched_environ(AGENT_PROMPT=BENCHMARK_PROMPT):
        rc = inject_crash_at_step(run_id, crash_step, db_path, baseline=False)
        if rc != -9:
            raise RuntimeError(f"Expected SIGKILL (-9) during durable crash injection, got {rc}")
        conn = create_shared_connection(db_path)
        setup_aer_tables(conn)
        pre_crash_steps = count_completed_events(conn, run_id)
        conn.close()
        cmd_recover(run_id, db_path)
    elapsed = time.perf_counter() - start

    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    metrics = collect_run_metrics(conn, run_id, durable=True)
    conn.close()
    metrics.steps_reexecuted = metrics.completed_steps - pre_crash_steps
    metrics.wall_clock_seconds = elapsed
    metrics.storage_overhead_bytes = storage_overhead_bytes(db_path)
    prompt_tokens, completion_tokens, total_tokens = _token_metrics(db_path, [run_id], durable=True)
    metrics.prompt_tokens = prompt_tokens
    metrics.completion_tokens = completion_tokens
    metrics.total_tokens = total_tokens
    return {
        "run_id": run_id,
        "crash_step": crash_step,
        "pre_crash_completed_steps": pre_crash_steps,
        "metrics": metrics_to_dict(metrics),
    }


def _average_metrics(trials: list[dict]) -> dict:
    keys = [
        "completed_steps",
        "llm_calls",
        "actual_tool_calls",
        "steps_reexecuted",
        "wall_clock_seconds",
        "storage_overhead_bytes",
    ]
    averages = {}
    for key in keys:
        averages[key] = sum(trial["metrics"][key] for trial in trials) / len(trials)

    token_values = [trial["metrics"]["total_tokens"] for trial in trials if trial["metrics"]["total_tokens"] is not None]
    averages["total_tokens"] = (sum(token_values) / len(token_values)) if token_values else None
    return averages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--crash-step", type=int, default=9)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    os.environ["AGENT_PROMPT"] = BENCHMARK_PROMPT

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    baseline_trials = []
    durable_trials = []

    for idx in range(1, args.runs + 1):
        baseline_run_id = f"exp1-baseline-{idx}-{uuid.uuid4().hex[:8]}"
        durable_run_id = f"exp1-durable-{idx}-{uuid.uuid4().hex[:8]}"
        baseline_db = str(TMP_DIR / f"{baseline_run_id}.sqlite")
        durable_db = str(TMP_DIR / f"{durable_run_id}.sqlite")

        baseline_trials.append(run_baseline_trial(baseline_run_id, baseline_db, args.crash_step))
        durable_trials.append(run_durable_trial(durable_run_id, durable_db, args.crash_step))

    payload = {
        "experiment": "exp1_crash_recovery",
        "runs": args.runs,
        "crash_step": args.crash_step,
        "baseline": {
            "recovery_mode": "full_rerun_from_start",
            "trials": baseline_trials,
            "averages": _average_metrics(baseline_trials),
        },
        "durable": {
            "recovery_mode": "resume_from_last_checkpoint",
            "trials": durable_trials,
            "averages": _average_metrics(durable_trials),
        },
    }
    write_results_json(str(args.results), payload)
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
