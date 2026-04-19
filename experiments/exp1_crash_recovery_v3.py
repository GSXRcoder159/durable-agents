"""
Experiment 1: Crash Recovery

Compares:
- Durable system (SqliteSaver + IdempotencyToolWrapper) with crash and recovery
- Baseline system (no checkpoint, no idempotency) that must restart from scratch after crash

Metrics for baseline are obtained via callbacks (in-process).
Metrics for durable are obtained from the database (events and tool_intents tables)
to account for both crashed subprocess and recovery phase.

Usage:
    python3 experiments/exp1_crash_recovery_v3.py

Output:
    results/exp1_results.txt
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph.state import RunnableConfig

from src.agent import build_graph, build_baseline_graph
from src.db import create_shared_connection, setup_aer_tables, EVENT_STATUS_COMPLETED
from src.callbacks import AgentMetricsHandler
from src.harness import inject_crash_at_step
from src.tools import reset_call_counts, get_call_counts

load_dotenv()

# ========== Configuration ==========
EXPERIMENT_DB = os.path.join(os.path.dirname(__file__), "exp1_crash_recovery.sqlite")
RESULTS_TXT = os.path.join(os.path.dirname(__file__), "..", "results", "exp1_results.txt")
NUM_RUNS = 3
CRASH_STEP = 9                     # crash after this many completed steps

MODEL = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# ========== Benchmark Task (≥12 steps, deterministic) ==========
BENCHMARK_PROMPT = """
You are a benchmark testing agent running at temperature=0.
You MUST execute the following instructions strictly in sequential order.
CRITICAL RULE: You are forbidden from calling multiple tools at the same time. You must wait for the exact result of the previous tool before calling the next one.

Perform the following steps exactly as written:

1. Use [web_search] with the query: "LangGraph agent framework overview"
2. From the result, use [extract_data] with url = the URL you just found, field = "description"
3. Use [web_search] with the query: "ReAct pattern for LLM agents"
4. From the result, use [extract_data] with url = the URL you just found, field = "key features"
5. Use [summarize] on the text: "Combine the description and key features into one paragraph."
6. Use [web_search] with the query: "Latest trends in AI agents 2025"
7. From the result, use [extract_data] with url = the URL you just found, field = "trends"
8. Use [summarize] on the text: "Summarize the trends in two sentences."
9. Use [web_search] with the query: "Market impact of AI agents"
10. From the result, use [extract_data] with url = the URL you just found, field = "market_impact"
11. Use [summarize] on the text: "Summarize the market impact in one sentence."
12. Finally, use [write_to_database] with record_id = "crash_recovery_report_001", data = the three summaries combined.

After writing to the database, output exactly: "BENCHMARK_COMPLETE"
""".strip()


# ---------- Helper: query database for durable metrics ----------
def get_metrics_from_db(db_path: str, run_id: str) -> Dict[str, float]:
    """
    Extract metrics from the database for a given run_id.
    Returns:
        - llm_calls: number of 'think' steps (each corresponds to one LLM call)
        - tool_actual_executions: number of unique tool intents completed (from tool_intents)
    """
    conn = create_shared_connection(db_path)
    # Count LLM calls = number of 'think' steps that completed
    row = conn.execute(
        "SELECT COUNT(*) FROM events WHERE run_id = ? AND step_type = 'think' AND status = ?",
        (run_id, EVENT_STATUS_COMPLETED)
    ).fetchone()
    llm_calls = row[0] if row else 0

    # Count actual tool executions = number of completed tool_intents
    row = conn.execute(
        "SELECT COUNT(*) FROM tool_intents WHERE status = 'COMPLETED'"
    ).fetchone()
    tool_actual = row[0] if row else 0

    conn.close()
    return {"llm_calls": llm_calls, "tool_actual_executions": tool_actual}


# ---------- Baseline: use invoke + callbacks (no debug mode) ----------
def run_baseline_complete(run_id: str, db_path: str) -> Tuple[Optional[str], float, float, int, int]:
    """
    Run benchmark on baseline agent using invoke (no checkpoint, no debug stream).
    Returns (final_output, elapsed_time_s, db_size_mb, llm_calls, tool_actual_executions).
    """
    # Clean any previous DB (baseline doesn't use it, but we create it for consistency)
    if os.path.exists(db_path):
        os.remove(db_path)
    # Create empty DB with tables (so we can measure storage overhead)
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    conn.close()

    graph = build_baseline_graph(model=MODEL)
    metrics = AgentMetricsHandler()
    config = RunnableConfig(callbacks=[metrics])

    start_time = time.perf_counter()
    result = graph.invoke({"messages": [("user", BENCHMARK_PROMPT)]}, config)
    elapsed = time.perf_counter() - start_time

    # Extract final output
    final_output = result["messages"][-1].content if result.get("messages") else None

    # Get metrics
    llm_calls = metrics.get_summary()["call_count"]
    tool_counts = get_call_counts()
    tool_actual = sum(tool_counts.values())
    db_size = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0.0

    return final_output, elapsed, db_size, llm_calls, tool_actual


# ---------- Durable: crash subprocess + recovery ----------
def run_durable_with_crash_and_recovery(run_id: str, crash_step: int, db_path: str) -> Tuple[Optional[str], float, float]:
    """
    Start agent subprocess, crash at crash_step, then recover.
    Returns (final_output, elapsed_recovery_time_s, db_size_mb_after_recovery).
    """
    # Clean previous DB
    if os.path.exists(db_path):
        os.remove(db_path)

    # 1. Inject crash (subprocess runs and is killed)
    returncode = inject_crash_at_step(run_id, crash_step, db_path)
    if returncode != -9:
        print(f"Warning: inject_crash_at_step returned {returncode}, expected -9")

    # 2. Recover using the same run_id
    conn = create_shared_connection(db_path)
    setup_aer_tables(conn)
    graph = build_graph(conn, model=MODEL)
    # We need a StepLogger to process events, but we cannot use debug mode due to potential issues.
    # Instead, we can use graph.stream with standard mode and manually record? However, we rely on
    # StepLogger's process_events which expects debug events. But in recovery, we can use
    # graph.invoke(None, config) to resume, but that won't give us step-by-step events.
    # Simpler: use graph.stream with None input and stream_mode="values" and ignore logging?
    # But we need to write events to DB for metrics. Since we only need final metrics (from DB)
    # and we don't need the intermediate outputs, we can just invoke the graph to resume,
    # and the existing checkpointer will cause it to continue. However, we also need to record
    # the recovery steps into events table for accurate think/tool counts. The StepLogger would
    # normally do that. Let's use a workaround: after recovery, the events table already contains
    # steps from the subprocess; recovery steps will be added by the agent's internal machinery?
    # Actually, the checkpoint recovery does not automatically write to events table; only StepLogger
    # does. So we must use StepLogger with debug events. But debug events caused issues in baseline.
    # In durable, we have a checkpointer, so debug mode should work. Let's use StepLogger.run
    # for recovery, but we need to call it with the same graph and run_id.
    from src.logger import StepLogger
    logger = StepLogger(conn)
    # Reset call counts for recovery phase (we don't need them, metrics from DB)
    reset_call_counts()

    start_time = time.perf_counter()
    # Resume by streaming None with debug mode (requires checkpointer)
    # Use the same method as in src/recovery.py
    config = RunnableConfig(configurable={"thread_id": run_id})
    events = graph.stream(None, config, stream_mode="debug", durability="sync")
    result = logger.process_events(events, run_id)
    elapsed = time.perf_counter() - start_time

    conn.close()
    db_size = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0.0
    return result, elapsed, db_size


def main():
    print("Experiment 1: Crash Recovery")
    print(f"Crash step: {CRASH_STEP}, runs per condition: {NUM_RUNS}\n")

    baseline_results = []   # each: (time_s, db_size_mb, llm_calls, tool_actual)
    durable_results = []    # each: (time_s, db_size_mb, llm_calls, tool_actual)

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run_idx} / {NUM_RUNS} ---")

        # ---------- Baseline ----------
        baseline_run_id = f"baseline_{uuid.uuid4()}"
        baseline_db = f"baseline_{run_idx}.sqlite"
        print(f"Baseline run with run_id={baseline_run_id}")
        reset_call_counts()
        _, time_b, db_b, llm_b, tool_b = run_baseline_complete(baseline_run_id, baseline_db)
        baseline_results.append((time_b, db_b, llm_b, tool_b))
        print(f"  Baseline: time={time_b:.2f}s, LLM calls={llm_b}, tool actual={tool_b}")

        # ---------- Durable (crash + recovery) ----------
        durable_run_id = f"durable_{uuid.uuid4()}"
        print(f"Durable crash+recovery with run_id={durable_run_id}")
        reset_call_counts()  # not used for durable metrics
        _, time_d, db_d = run_durable_with_crash_and_recovery(durable_run_id, CRASH_STEP, EXPERIMENT_DB)
        metrics_d = get_metrics_from_db(EXPERIMENT_DB, durable_run_id)
        durable_results.append((time_d, db_d, metrics_d["llm_calls"], metrics_d["tool_actual_executions"]))
        print(f"  Durable recovery time: {time_d:.2f}s, LLM calls={metrics_d['llm_calls']}, tool actual={metrics_d['tool_actual_executions']}")

        # Cleanup baseline DB
        if os.path.exists(baseline_db):
            os.remove(baseline_db)

    # Compute averages
    avg_b = (
        sum(r[0] for r in baseline_results) / NUM_RUNS,
        sum(r[1] for r in baseline_results) / NUM_RUNS,
        sum(r[2] for r in baseline_results) / NUM_RUNS,
        sum(r[3] for r in baseline_results) / NUM_RUNS,
    )
    avg_d = (
        sum(r[0] for r in durable_results) / NUM_RUNS,
        sum(r[1] for r in durable_results) / NUM_RUNS,
        sum(r[2] for r in durable_results) / NUM_RUNS,
        sum(r[3] for r in durable_results) / NUM_RUNS,
    )

    # Print table
    headers = ["Metric", "Baseline (full restart)", "Durable (crash+recovery)"]
    rows = [
        ["Time (s)", f"{avg_b[0]:.3f}", f"{avg_d[0]:.3f}"],
        ["LLM calls", f"{avg_b[2]:.1f}", f"{avg_d[2]:.1f}"],
        ["Actual tool executions", f"{avg_b[3]:.1f}", f"{avg_d[3]:.1f}"],
        ["DB size (MB)", f"{avg_b[1]:.2f}", f"{avg_d[1]:.2f}"],
    ]

    print("\n" + "=" * 65)
    print(f"Experiment 1 Results (averaged over {NUM_RUNS} runs)")
    print("=" * 65)
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))

    # Save results
    os.makedirs(os.path.dirname(RESULTS_TXT), exist_ok=True)
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("Experiment 1: Crash Recovery\n")
        f.write(f"generated_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%f')}Z\n")
        f.write(f"crash_step: {CRASH_STEP}\n")
        f.write(f"num_runs: {NUM_RUNS}\n")
        f.write(f"model: gpt-4.1-mini (temperature=0)\n\n")
        f.write("Per-run metrics:\n")
        for i, (t, db, llm, tool) in enumerate(baseline_results):
            f.write(f"  baseline run{i+1}: time={t:.3f}s, db={db:.2f}MB, llm_calls={llm:.0f}, tool_actual={tool:.0f}\n")
        for i, (t, db, llm, tool) in enumerate(durable_results):
            f.write(f"  durable run{i+1}: time={t:.3f}s, db={db:.2f}MB, llm_calls={llm:.0f}, tool_actual={tool:.0f}\n")
        f.write("\nAverages:\n")
        f.write(f"  Baseline : time={avg_b[0]:.3f}s, db={avg_b[1]:.2f}MB, llm_calls={avg_b[2]:.1f}, tool_actual={avg_b[3]:.1f}\n")
        f.write(f"  Durable : time={avg_d[0]:.3f}s, db={avg_d[1]:.2f}MB, llm_calls={avg_d[2]:.1f}, tool_actual={avg_d[3]:.1f}\n")
        f.write("\nTable:\n")
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        for row in rows:
            f.write(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + "\n")

    print(f"\nResults saved to {RESULTS_TXT}")


if __name__ == "__main__":
    main()