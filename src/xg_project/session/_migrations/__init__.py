"""Schema migrations for xg_project.session using sqlite-utils."""

from sqlite_utils import Database, Migrations

migrations = Migrations("xg_project")


@migrations()
def create_initial_schema(db: Database) -> None:
    """v1: Create sessions and messages tables."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            directory TEXT NOT NULL,
            filename TEXT NOT NULL,
            UNIQUE(directory, filename)
        )
    """)
    db.execute("""
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
