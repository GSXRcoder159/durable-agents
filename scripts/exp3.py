"""Experiment 3: Fault Injection Resilience (The Position N Trap)"""

import os
import sys
import time
import sqlite3
import subprocess
from langchain_core.messages import ToolMessage

from src.db import create_shared_connection, get_db_size_kb
from src.agent import build_graph

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DURABLE_DB_PATH = os.path.join(PROJECT_ROOT, "exp3_durable_db.sqlite")
BASELINE_DB_PATH = os.path.join(PROJECT_ROOT, "exp3_baseline_db.sqlite")

ESTIMATED_TOKENS_PER_STEP = 1200 

SHARED_PROMPT = """
You are a benchmark testing agent running at temperature=0. 
You must complete the following objective. 
CRITICAL RULE: You are forbidden from calling multiple tools at the same time. You must wait for the exact result of the previous tool before calling the next one.

Primary Tasks:
1. Use [web_search] with the exact query: "Find the official URL for Project Nexus".
2. Use [extract_data] with the URL you just found, and request the field: "annual_report_link".
3. Use [extract_data] again with the "annual_report_link", requesting the field: "executive_summary_text".
4. Use [summarize] to summarize the "executive_summary_text".
5. Use [web_search] again with the query: "Market reactions to " followed by the summarized text.
6. Use [write_to_database] to save the market reactions. Use the record_id: "Nexus_Report_001".

FAULT RECOVERY RULE:
You are a highly intelligent agent. If you are informed of a previous critical error, you MUST autonomously figure out a different path or perform a different logical tool action to reset your state before trying the failed objective again.

After the database write is successfully completed, output the exact string: "BENCHMARK_COMPLETE: Nexus_Report_001 saved."
"""

def setup_fault_environment(db_path: str, run_id: str) -> dict:
    """Configures the environment variables for the Position N Trap."""
    env = os.environ.copy()
    env["DB_PATH"] = db_path
    env["CURRENT_RUN_ID"] = run_id       
    env["EXP3_POSITION_MODE"] = "true"   
    env["EXP3_TARGET_TOOL"] = "write_to_database"
    env["EXP3_FAULT_TYPE"] = "timeout" 
    env["EXP3_FAULT_CALL"] = "12"        
    return env

def get_db_steps(db_path: str, run_id: str) -> int:
    """Extracts the exact number of LLM steps executed from the events table."""
    try:
        if not os.path.exists(db_path):
            return 0
        with sqlite3.connect(db_path) as conn:
            res = conn.execute("SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,)).fetchone()
            return res[0] if res else 0
    except Exception:
        return 0

def inject_error_state(run_id: str, db_path: str):
    """Injects a TimeoutError into the agent's checkpoint to force replanning."""
    conn = create_shared_connection(db_path)
    graph = build_graph(conn)
    config = {"configurable": {"thread_id": run_id}}
    state = graph.get_state(config)

    messages = state.values.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        tool_call = messages[-1].tool_calls[0]
        
        error_msg = ToolMessage(
            content="TimeoutError: Database connection failed. Execution path blocked.",
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )
        print("[Watchdog] Injecting Error Message into Checkpoint to force re-planning...")
        graph.update_state(config, {"messages": [error_msg]}, as_node="tools")
    conn.close()

def run_durable_agent_with_fault() -> tuple[bool, float, int, int]:
    print("\n" + "="*50)
    print("[Durable Agent] Starting Fault Injection Test (Autonomous Re-plan Mode)")
    print("="*50)
    
    if os.path.exists(DURABLE_DB_PATH):
        os.remove(DURABLE_DB_PATH)

    run_id = "exp3-durable-001"
    env = setup_fault_environment(DURABLE_DB_PATH, run_id)
    env["AGENT_PROMPT"] = SHARED_PROMPT
    start_time = time.perf_counter()
    
    print("\n[Attempt 1] Running Agent (Expecting Crash at Global Step 12...)")
    try:
        subprocess.run(
            [sys.executable, "-m", "src", "run", run_id],
            env=env, cwd=PROJECT_ROOT, check=True
        )
    except subprocess.CalledProcessError:
        print("[CRASH DETECTED] Agent hit the Position N trap at Step 12!")
    
    print("\n[Attempt 2] Recovering WITHOUT Removing the Trap...")
    success = False
    try:
        inject_error_state(run_id, DURABLE_DB_PATH)
        subprocess.run(
            [sys.executable, "-m", "src", "recover", run_id],
            env=env, cwd=PROJECT_ROOT, check=True
        )
        print("[RECOVERY SUCCESS] Agent successfully planned a new path and completed the task!")
        success = True
    except subprocess.CalledProcessError:
        print("[RECOVERY FAILED] Agent still hit the trap. Re-planning failed.")

    end_time = time.perf_counter()
    total_steps = get_db_steps(DURABLE_DB_PATH, run_id)
    return success, end_time - start_time, 2, total_steps

def run_baseline_agent_with_fault() -> tuple[bool, float, int, int]:
    print("\n" + "="*50)
    print("[Baseline Agent] Starting Fault Injection Test (Stateless Prompt Injection Mode)")
    print("="*50)
    
    run_id = "exp3-baseline-001"
    env = setup_fault_environment(BASELINE_DB_PATH, run_id)
    start_time = time.perf_counter()
    
    max_retries = 3 
    attempts = 0
    success = False
    total_baseline_steps = 0
    
    # This acts as the "External Memory" since Baseline has no Checkpoint
    external_memory_prompt = ""

    while attempts < max_retries:
        attempts += 1
        print(f"\n[Attempt {attempts}] Running Baseline Agent from scratch...")
        
        # Simulate statelessness: Wipe the database entirely
        if os.path.exists(BASELINE_DB_PATH):
            os.remove(BASELINE_DB_PATH)
        
        env["DB_PATH"] = BASELINE_DB_PATH
        
        # Inject the external memory into the system prompt
        current_prompt = SHARED_PROMPT
        if external_memory_prompt:
            print("[Watchdog] Injecting previous failure history into Baseline prompt...")
            current_prompt += f"\n\nCRITICAL SYSTEM WARNING: {external_memory_prompt}"
        
        env["AGENT_PROMPT"] = current_prompt

        try:
            subprocess.run(
                [sys.executable, "-m", "src", "run", run_id],
                env=env, cwd=PROJECT_ROOT, check=True
            )
            print(f"[SUCCESS] Baseline succeeded on attempt {attempts}!")
            success = True
            total_baseline_steps += get_db_steps(BASELINE_DB_PATH, run_id)
            break
        except subprocess.CalledProcessError:
            print(f"[CRASH DETECTED] Baseline crashed due to TimeoutError on attempt {attempts}.")
            
            wasted_steps = get_db_steps(BASELINE_DB_PATH, run_id)
            total_baseline_steps += wasted_steps
            print(f"[METRICS] Baseline wasted {wasted_steps} steps in this failed attempt.")
            
            # Update the external memory for the next loop
            external_memory_prompt = "Your previous attempt failed at the 'write_to_database' tool due to a TimeoutError. You MUST change your logical sequence of steps to avoid failing again."
            
            time.sleep(2) 

    end_time = time.perf_counter()
    if not success:
        print(f"[FAILED] Baseline gave up after {max_retries} attempts.")

    return success, end_time - start_time, attempts, total_baseline_steps

if __name__ == "__main__":
    print("Starting Experiment 3: Fault Injection Resilience")
    d_success, d_time, d_attempts, d_steps = run_durable_agent_with_fault()
    b_success, b_time, b_attempts, b_steps = run_baseline_agent_with_fault()

    d_est_tokens = d_steps * ESTIMATED_TOKENS_PER_STEP
    b_est_tokens = b_steps * ESTIMATED_TOKENS_PER_STEP

    print("\n" + "="*65)
    print("Experiment 3 Final Results")
    print("="*65)
    print(f"| Metric               | Durable Agent       | Baseline Agent      |")
    print(f"|----------------------|---------------------|---------------------|")
    print(f"| End-to-End Success   | {'Yes' if d_success else 'No'}{' '*16}| {'Yes' if b_success else 'No'}{' '*16}|")
    print(f"| Total Attempts       | {d_attempts}{' '*18}| {b_attempts} (Max: 3){' '*8}|")
    print(f"| Wall-clock Time      | {d_time:.2f} seconds       | {b_time:.2f} seconds       |")
    print(f"| Storage Overhead     | {get_db_size_kb(DURABLE_DB_PATH):.2f} KB            | {get_db_size_kb(BASELINE_DB_PATH):.2f} KB            |")
    print(f"| Total LLM Steps      | {d_steps} steps            | {b_steps} steps            |")
    print(f"| Est. Token Cost      | ~{d_est_tokens} tokens      | ~{b_est_tokens} tokens      |")
    print("="*65)
    print("Conclusion:")
    print("Durable Agent autonomously replanned using its internal Checkpoint history.")
    print("Baseline Agent required an external orchestrator to inject error history into a new prompt.")
    print("While prompt injection allows stateless replanning, it forces the agent to restart from step 1,")
    print("bypassing the step 12 trap by resetting the counter, rather than surviving it contextually.")