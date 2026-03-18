"""Shared pytest fixtures for the project's tests."""
import pytest
import sqlite3

from typing import Generator

from src.db import create_shared_connection, setup_aer_tables


@pytest.fixture(scope="function")
def mem_conn() -> Generator[sqlite3.Connection, None, None]:
    """Set up in-memory SQLite connection with tables set up.

    Returns:
        sqlite3.Connection: A new in-memory SQLite connection scoped to the test function.
            Note: WAL-mode is not supported for in-memory databases.
    """
    conn = create_shared_connection(":memory:")
    setup_aer_tables(conn)
    yield conn
    conn.close()
