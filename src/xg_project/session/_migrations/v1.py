"""v1: Initial schema."""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
    """)
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            directory TEXT NOT NULL,
            filename TEXT NOT NULL,
            UNIQUE(directory, filename)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            message_id TEXT,
            message_type TEXT NOT NULL,
            content TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()


def down(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS messages")
    conn.execute("DROP TABLE IF EXISTS sessions")
    conn.execute("DELETE FROM schema_version")
    conn.commit()
