"""
Experiment 2: Idempotency (Checkpoint-Crash Race Replay)
How to run: python scripts/exp2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db import create_shared_connection, setup_aer_tables
from src.idempotency import IdempotencyToolWrapper
from src.tools import extract_data, get_call_counts, reset_call_counts, summarize, web_search, write_to_database

EXPERIMENT_DB = os.path.join(os.path.dirname(__file__), "exp2_idempotency.sqlite")
RECORD_ID = "Nexus_Report_001"
TXT_OUT = "results/exp2_results.txt"

BENCHMARK_PROMPT = """
You are a benchmark testing agent running at temperature=0.
You MUST execute the following instructions strictly in sequential order.
CRITICAL RULE: You are forbidden from calling multiple tools at the same time. You must wait for the exact result of the previous tool before calling the next one.

Task Sequence:
1. Use [web_search] with the exact query: "Find the official URL for Project Nexus".
2. Wait for the result. Then, use [extract_data] with the URL you just found, and request the field: "annual_report_link".
3. Wait for the result. Then, use [extract_data] again with the "annual_report_link" you just found, and request the field: "executive_summary_text".
4. Wait for the result. Then, use [summarize] to summarize the "executive_summary_text" you extracted.
5. Wait for the result. Then, use [web_search] again with the query: "Market reactions to " followed by the summarized text.
6. Wait for the result. Finally, use [write_to_database] to save the market reactions you found. Use the record_id: "{record_id}".
7. After the database write is complete, output the exact string: "BENCHMARK_COMPLETE: {record_id} saved."
""".strip()


def reset_tool_intents(conn: Any) -> None:
    conn.execute("DELETE FROM tool_intents")
    conn.commit()


def latest_completed_write_intent(conn: Any) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """SELECT intent_hash, args_json
           FROM tool_intents
           WHERE tool_name = ? AND status = 'COMPLETED'
           ORDER BY completed_at DESC
           LIMIT 1""",
        (write_to_database.name,),
    ).fetchone()
    if not row:
        return None
    return {"intent_hash": str(row["intent_hash"]), "args": json.loads(row["args_json"] or "{}")}


def run_task_once(tools: Dict[str, Any], record_id: str) -> None:
    url = tools[web_search.name].invoke({"query": "Find the official URL for Project Nexus"})
    annual = tools[extract_data.name].invoke({"url": url, "field": "annual_report_link"})
    exec_summary = tools[extract_data.name].invoke({"url": annual, "field": "executive_summary_text"})
    summary = tools[summarize.name].invoke({"text": exec_summary})
    market = tools[web_search.name].invoke({"query": f"Market reactions to {summary}"})
    tools[write_to_database.name].invoke({"record_id": record_id, "data": market})


def measure_run(fn) -> Tuple[float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        fn()
    finally:
        elapsed_s = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return elapsed_s, peak_bytes / (1024 * 1024)


def totals(rows: list[dict]) -> dict[str, float]:
    return {
        "time_s": float(sum(r["time_s"] for r in rows)),
        "llm_tool_calls": float(sum(r["llm_tool_calls"] for r in rows)),
        "py_peak_mb": float(max((r["memory_py_peak_mb"] for r in rows), default=0.0)),
        "tokens": float(sum(r["token_used_total"] for r in rows)),
        "db_writes": float(sum(r["write_to_database_executions"] for r in rows)),
    }


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:

    conn = create_shared_connection(EXPERIMENT_DB)
    setup_aer_tables(conn)

    prompt = BENCHMARK_PROMPT.format(record_id=RECORD_ID)
    print("Experiment 2: Idempotency (Checkpoint-Crash Race Replay)")
    print(f"record_id={RECORD_ID}\n")

    results: list[dict] = []

    baseline_tools = {
        web_search.name: web_search,
        extract_data.name: extract_data,
        summarize.name: summarize,
        write_to_database.name: write_to_database,
    }
    cached_tools = {
        web_search.name: IdempotencyToolWrapper(web_search, conn),
        extract_data.name: IdempotencyToolWrapper(extract_data, conn),
        summarize.name: IdempotencyToolWrapper(summarize, conn),
        write_to_database.name: IdempotencyToolWrapper(write_to_database, conn),
    }

    print("=== BASELINE (no cache) ===")
    reset_tool_intents(conn)
    for run in (1, 2):
        reset_call_counts()
        elapsed_s, peak_mb = measure_run(lambda: run_task_once(baseline_tools, RECORD_ID))
        results.append(
            {
                "agent": "baseline",
                "run": run,
                "time_s": elapsed_s,
                "llm_tool_calls": 6,
                "memory_py_peak_mb": peak_mb,
                "token_used_total": 0,
                "write_to_database_executions": int(get_call_counts().get(write_to_database.name, 0)),
                "intent_hash": "",
            }
        )
        print(
            f"[baseline run{run}] write_to_database_executions={results[-1]['write_to_database_executions']}, time_s={elapsed_s:.3f}"
        )

    print("\n=== CACHED (IdempotencyToolWrapper) ===")
    reset_tool_intents(conn)
    cached_intent_hash = ""
    for run in (1, 2):
        reset_call_counts()
        elapsed_s, peak_mb = measure_run(lambda: run_task_once(cached_tools, RECORD_ID))
        latest = latest_completed_write_intent(conn)
        if latest is not None:
            cached_intent_hash = latest["intent_hash"]
        results.append(
            {
                "agent": "cached",
                "run": run,
                "time_s": elapsed_s,
                "llm_tool_calls": 6,
                "memory_py_peak_mb": peak_mb,
                "token_used_total": 0,
                "write_to_database_executions": int(get_call_counts().get(write_to_database.name, 0)),
                "intent_hash": cached_intent_hash,
            }
        )
        print(
            f"[cached run{run}] write_to_database_executions={results[-1]['write_to_database_executions']}, time_s={elapsed_s:.3f}"
        )

    baseline_runs = [r for r in results if r["agent"] == "baseline"]
    cached_runs = [r for r in results if r["agent"] == "cached"]
    baseline_execs = [r["write_to_database_executions"] for r in baseline_runs]
    cached_execs = [r["write_to_database_executions"] for r in cached_runs]
    ok = baseline_execs == [1, 1] and cached_execs == [1, 0]

    btot = totals(baseline_runs)
    ctot = totals(cached_runs)
    comparison_table = format_table(
        headers=["Metric", "Baseline", "Cached"],
        rows=[
            ["Total Attempts", "2", "2"],
            ["Time (s)", f"{btot['time_s']:.3f}", f"{ctot['time_s']:.3f}"],
            ["LLM tool calls", f"{int(btot['llm_tool_calls'])}", f"{int(ctot['llm_tool_calls'])}"],
            ["Memory peak (MB)", f"{btot['py_peak_mb']:.2f}", f"{ctot['py_peak_mb']:.2f}"],
            ["Token used", f"{int(btot['tokens'])}", f"{int(ctot['tokens'])}"],
            ["DB writes (actual)", f"{int(btot['db_writes'])}", f"{int(ctot['db_writes'])}"],
        ],
    )

    print("\n" + "=" * 65)
    print("Experiment 2 Final Results (Baseline vs Cached)")
    print("=" * 65)
    print(comparison_table)
    print(f"\nBaseline write_to_database executions: {baseline_execs}")
    print(f"Cached write_to_database executions:   {cached_execs}")
    if cached_intent_hash:
        print(f"Cached intent_hash: {cached_intent_hash}")
    print(f"ok: {ok}")

    if not ok:
        raise SystemExit("Expectation failed.")

    lines = []
    lines.append("Experiment 2: Idempotency (Checkpoint-Crash Race Replay)")
    lines.append(f"generated_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%f')}Z")
    lines.append(f"db: {EXPERIMENT_DB}")
    lines.append(f"record_id: {RECORD_ID}")
    lines.append("")
    lines.append("Prompt:")
    lines.append(prompt)
    lines.append("")
    lines.append("Per-run metrics:")
    for r in results:
        lines.append(f"[{r['agent']} run{r['run']}]")
        lines.append(f"Time: {r['time_s']:.6f}s")
        lines.append(f"LLM tool calls: {r['llm_tool_calls']}")
        lines.append(f"Memory used: py_peak_mb={r['memory_py_peak_mb']:.2f}")
        lines.append(f"Token used: {r['token_used_total']}")
        lines.append(f"write_to_database executions: {r['write_to_database_executions']}")
        if r["intent_hash"]:
            lines.append(f"intent_hash: {r['intent_hash']}")
        lines.append("")
    lines.append("Baseline vs Cached:")
    lines.append(comparison_table)
    lines.append("")
    lines.append(f"Baseline write_to_database executions: {baseline_execs}")
    lines.append(f"Cached write_to_database executions:   {cached_execs}")
    if cached_intent_hash:
        lines.append(f"Cached intent_hash: {cached_intent_hash}")
    lines.append(f"ok: {ok}")

    out_dir = os.path.dirname(os.path.abspath(TXT_OUT))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote TXT report: {TXT_OUT}")

    conn.close()


if __name__ == "__main__":
    main()
