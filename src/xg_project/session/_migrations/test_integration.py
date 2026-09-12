"""Integration tests for xg_project.session._migrations."""

import sqlite3

import pytest

from xg_project.session._migrations import get_version, migrate


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "test.db"


# --- get_version() tests ---


@pytest.mark.integration
def test_get_version_returns_zero_for_missing_db(db_path):
    """get_version() returns 0 when database doesn't exist."""
    assert get_version(db_path) == 0


@pytest.mark.integration
def test_get_version_returns_zero_for_empty_db(db_path):
    """get_version() returns 0 when schema_version table doesn't exist."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE other (id INTEGER)")
    conn.commit()
    conn.close()
    assert get_version(db_path) == 0


@pytest.mark.integration
def test_get_version_returns_current_version(db_path):
    """get_version() returns the stored version."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.commit()
    conn.close()
    assert get_version(db_path) == 1


# --- migrate() tests ---


@pytest.mark.integration
def test_migrate_creates_database(db_path):
    """migrate() creates the database file."""
    migrate(db_path)
    assert db_path.exists()


@pytest.mark.integration
def test_migrate_sets_version(db_path):
    """migrate() sets schema version to 1."""
    migrate(db_path)
    assert get_version(db_path) == 1


@pytest.mark.integration
def test_migrate_creates_sessions_table(db_path):
    """migrate() creates the sessions table."""
    migrate(db_path)
    conn = sqlite3.connect(str(db_path))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    table_names = [t[0] for t in tables]
    assert "sessions" in table_names


@pytest.mark.integration
def test_migrate_creates_messages_table(db_path):
    """migrate() creates the messages table."""
    migrate(db_path)
    conn = sqlite3.connect(str(db_path))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    table_names = [t[0] for t in tables]
    assert "messages" in table_names


@pytest.mark.integration
def test_migrate_creates_schema_version_table(db_path):
    """migrate() creates the schema_version table."""
    migrate(db_path)
    conn = sqlite3.connect(str(db_path))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    table_names = [t[0] for t in tables]
    assert "schema_version" in table_names


@pytest.mark.integration
def test_migrate_is_idempotent(db_path):
    """migrate() can be called multiple times safely."""
    migrate(db_path)
    migrate(db_path)
    migrate(db_path)
    assert get_version(db_path) == 1


@pytest.mark.integration
def test_migrate_preserves_existing_data(db_path):
    """migrate() preserves data in existing tables."""
    # Manually create v1 schema and add data
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute("""
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            directory TEXT NOT NULL,
            filename TEXT NOT NULL,
            UNIQUE(directory, filename)
        )
    """)
    conn.execute(
        "INSERT INTO sessions (directory, filename) VALUES (?, ?)",
        ("/test/dir", "123456.jsonl"),
    )
    conn.execute("""
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
    conn.execute(
        "INSERT INTO messages (session_id, message_id, message_type, content, data)"
        " VALUES (?, ?, ?, ?, ?)",
        (1, "msg-1", "human", "hello", '{"content": "hello"}'),
    )
    conn.commit()
    conn.close()

    # Run migrate (should not change anything)
    migrate(db_path)

    # Verify data is preserved
    conn = sqlite3.connect(str(db_path))
    sessions = conn.execute("SELECT * FROM sessions").fetchall()
    messages = conn.execute("SELECT * FROM messages").fetchall()
    conn.close()

    assert len(sessions) == 1
    assert sessions[0][1] == "/test/dir"
    assert sessions[0][2] == "123456.jsonl"
    assert len(messages) == 1
    assert messages[0][2] == "msg-1"
    assert messages[0][4] == "hello"


@pytest.mark.integration
def test_migrate_creates_backup(db_path):
    """migrate() creates a backup when upgrading."""
    # Create a v0 database (no schema_version)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE old_data (id INTEGER)")
    conn.execute("INSERT INTO old_data VALUES (42)")
    conn.commit()
    conn.close()

    # Run migrate
    migrate(db_path)

    # Check backup exists
    backup_path = db_path.with_suffix(".v0.bak")
    assert backup_path.exists()

    # Verify backup has old data
    backup_conn = sqlite3.connect(str(backup_path))
    row = backup_conn.execute("SELECT * FROM old_data").fetchone()
    backup_conn.close()
    assert row[0] == 42


@pytest.mark.integration
def test_migrate_new_db_has_correct_schema(db_path):
    """migrate() creates database with correct schema."""
    migrate(db_path)
    conn = sqlite3.connect(str(db_path))

    # Check sessions table structure
    sessions_info = conn.execute("PRAGMA table_info(sessions)").fetchall()
    sessions_columns = [col[1] for col in sessions_info]
    assert "id" in sessions_columns
    assert "directory" in sessions_columns
    assert "filename" in sessions_columns

    # Check messages table structure
    messages_info = conn.execute("PRAGMA table_info(messages)").fetchall()
    messages_columns = [col[1] for col in messages_info]
    assert "id" in messages_columns
    assert "session_id" in messages_columns
    assert "message_id" in messages_columns
    assert "message_type" in messages_columns
    assert "content" in messages_columns
    assert "data" in messages_columns

    conn.close()
