"""Fault injection harness for testing the robustness of the system."""
import os
import sqlite3
import subprocess
import sys
import time

from langchain_core.tools import BaseTool
from typing import Any, Optional

from src.db import EVENT_STATUS_COMPLETED

def inject_crash_at_step(run_id: str, step_id: int, db_path: str = "db.sqlite") -> int:
    """Spawn agent subprocess and SIGKILL it at step_id COMPLETED rows.

    Args:
        run_id (str): The run ID to target for fault injection 
        step_id (int): The step ID at which to inject the crash
        db_path (str, optional): The path to the SQLite database. Defaults to "db.sqlite".

    Returns:
        int: The subprocess return code
    """
    # Make sure the suprocess can see the src package.
    src_dir = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        new_pythonpath = f"{src_dir}{os.pathsep}{current_pythonpath}"
    else:
        new_pythonpath = src_dir
    
    env = {**os.environ, "PYTHONPATH": new_pythonpath, "DB_PATH": db_path}
    proc = subprocess.Popen([sys.executable, "-m", "src", "run", run_id], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    
    # Wait for the subprocess to create and initialize the DB
    max_wait_time = time.monotonic() + 10.0 # 10 seconds
    while not os.path.exists(db_path):
        if time.monotonic() > max_wait_time:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"Subprocess did not create database file at {db_path} within 10 seconds.")
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print("=== Subprocess exited early ===")
            print("STDOUT:", stdout.decode() if stdout else "(empty)")
            print("STDERR:", stderr.decode() if stderr else "(empty)")
            raise RuntimeError(f"Subprocess exited with code {proc.returncode} before DB was created.")
        time.sleep(0.05)

    # Poll the database for completed steps
    last_error = None
    while proc.poll() is None:
        try:
            # Use a fresh connection with a short timeout
            conn = sqlite3.connect(db_path, timeout=1.0)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE run_id = ? AND status = ?",
                (run_id, EVENT_STATUS_COMPLETED)
            ).fetchone()
            completed_steps = row[0] if row else 0
            conn.close()

            if completed_steps >= step_id:
                proc.kill()
                proc.wait()
                return proc.returncode
        except sqlite3.OperationalError as e:
            last_error = e
            time.sleep(0.1)
        except Exception as e:
            last_error = e
            time.sleep(0.1)

        time.sleep(0.01)

    # If we get here, the subprocess died on its own
    stdout, stderr = proc.communicate()
    print("=== Subprocess exited unexpectedly ===")
    print(f"Return code: {proc.returncode}")
    print("STDOUT:", stdout.decode() if stdout else "(empty)")
    print("STDERR:", stderr.decode() if stderr else "(empty)")
    if last_error:
        print(f"Last SQLite error: {last_error}")
    return proc.returncode

class FaultInjectionWrapper(BaseTool):
    """Tool wrapper that raises a fault exception at tool call N.
    
    Args:
        fault_type (Optional[str]): 'timeout' | 'tool_error' | 'rate_limit'
        call_number (int): the tool call number at which to raise the fault (1-indexed)
    """
    model_config = {"arbitrary_types_allowed": True}

    wrapped_tool: BaseTool
    fault_type: Optional[str] = None
    call_number: int = 1

    def __init__(self, wrapped_tool: BaseTool, fault_type: Optional[str] = None, call_number: int = 1, **kwargs: Any) -> None:
        super().__init__(name=wrapped_tool.name, description=wrapped_tool.description,
                         args_schema=wrapped_tool.args_schema, wrapped_tool=wrapped_tool,
                         fault_type=fault_type, call_number=call_number, **kwargs)
        object.__setattr__(self, "_call_count", 0)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        count = self._call_count + 1
        object.__setattr__(self, "_call_count", count)

        if count == self.call_number:
            if self.fault_type == "timeout":
                raise TimeoutError(f"[FAULT] Simulated API timeout at call {count} of tool {self.name}")
            elif self.fault_type == "tool_error":
                raise RuntimeError(f"[FAULT] Simulated tool error at call {count} of tool {self.name}")
            elif self.fault_type == "rate_limit":
                raise RuntimeError(f"[FAULT] Simulated rate limit at call {count} of tool {self.name}")
        
        return self.wrapped_tool.run(*args, **kwargs)
