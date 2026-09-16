"""Integration tests for xg_project.session._migrations."""

import pytest
from sqlite_utils import Database

from xg_project.session._migrations import migrations


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path):
    """Provide a sqlite-utils Database instance."""
    return Database(str(db_path))


# --- migrations.apply() tests ---


@pytest.mark.integration
def test_migrate_creates_database(db_path):
    """migrate() creates the database file."""
    db = Database(str(db_path))
    migrations.apply(db)
    assert db_path.exists()


@pytest.mark.integration
def test_migrate_creates_sessions_table(db):
    """migrate() creates the sessions table."""
    migrations.apply(db)
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "sessions" in table_names


@pytest.mark.integration
def test_migrate_creates_messages_table(db):
    """migrate() creates the messages table."""
    migrations.apply(db)
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "messages" in table_names


@pytest.mark.integration
def test_migrate_creates_migrations_table(db):
    """migrate() creates the _sqlite_migrations tracking table."""
    migrations.apply(db)
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "_sqlite_migrations" in table_names


@pytest.mark.integration
def test_migrate_is_idempotent(db):
    """migrate() can be called multiple times safely."""
    migrations.apply(db)
    migrations.apply(db)
    migrations.apply(db)
    applied = migrations.applied(db)
    assert len(applied) == 1


@pytest.mark.integration
def test_migrate_preserves_existing_data(db):
    """migrate() preserves data in existing tables."""
    # Manually create schema and add data
    db.execute("""
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            directory TEXT NOT NULL,
            filename TEXT NOT NULL,
            UNIQUE(directory, filename)
        )
    """)
    db.execute(
        "INSERT INTO sessions (directory, filename) VALUES (?, ?)",
        ("/test/dir", "123456.jsonl"),
    )
    db.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            message_id TEXT,
            message_type TEXT NOT NULL,
            content TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    db.execute(
        "INSERT INTO messages (session_id, message_id, message_type, content, data)"
        " VALUES (?, ?, ?, ?, ?)",
        (1, "msg-1", "human", "hello", '{"content": "hello"}'),
    )

    # Run migrate (should not change anything since tables already exist)
    migrations.apply(db)

    # Verify data is preserved
    sessions = db.execute("SELECT * FROM sessions").fetchall()
    messages = db.execute("SELECT * FROM messages").fetchall()

    assert len(sessions) == 1
    assert sessions[0][1] == "/test/dir"
    assert sessions[0][2] == "123456.jsonl"
    assert len(messages) == 1
    assert messages[0][2] == "msg-1"
    assert messages[0][4] == "hello"


@pytest.mark.integration
def test_migrate_new_db_has_correct_schema(db):
    """migrate() creates database with correct schema."""
    migrations.apply(db)

    # Check sessions table structure
    sessions_info = db.execute("PRAGMA table_info(sessions)").fetchall()
    sessions_columns = [col[1] for col in sessions_info]
    assert "id" in sessions_columns
    assert "directory" in sessions_columns
    assert "filename" in sessions_columns

    # Check messages table structure
    messages_info = db.execute("PRAGMA table_info(messages)").fetchall()
    messages_columns = [col[1] for col in messages_info]
    assert "id" in messages_columns
    assert "session_id" in messages_columns
    assert "message_id" in messages_columns
    assert "message_type" in messages_columns
    assert "content" in messages_columns
    assert "data" in messages_columns


@pytest.mark.integration
def test_migrate_tracks_applied_migrations(db):
    """migrate() records which migrations were applied."""
    migrations.apply(db)
    applied = migrations.applied(db)
    assert len(applied) == 1
    assert applied[0].name == "create_initial_schema"
