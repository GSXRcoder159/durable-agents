"""CLI command for the project."""
import sqlite3

def cmd_inspect(run_id: str, conn: sqlite3.Connection) -> None:
    """Print a formatted table of step logs for `run_id`.

    Args:
        run_id (str): the run ID to look up
        conn (sqlite3.Connection): an open `sqlite3.Connection`
    
    Output:
        Prints a formatted table of step logs for `run_id` to stdout.
    """
    rows = conn.execute(
        """SELECT step_id, step_type, tool_name, status, created_at, completed_at
        FROM events
        WHERE run_id = ?
        ORDER BY step_id ASC""", (run_id,)).fetchall()
    
    if not rows:
        print(f"No logs found for run_id '{run_id}'")
        return
    
    print(f"{'Step':>4}  {'Type':<8}  {'Tool':<20}  {'Status':<10}  {'Started':<30}")
    print("-" * 76)
    for row in rows:
        tool = row["tool_name"] or "-"
        print(f"{row['step_id']:>4}  {row['step_type']:<8}  {tool:<20}  {row['status']:<10}  {(row['created_at'] or ''):<30}")
