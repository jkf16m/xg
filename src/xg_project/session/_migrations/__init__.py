"""Schema migrations for xg_project.session.

Migrations work by:
- up(): create new DB with new schema, copy data from old, replace
- down(): create new DB with old schema, copy data from new, replace
"""

import shutil
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


def get_version(db_path: Path) -> int:
    """Return the current schema version from the database file."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        conn.close()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def migrate(db_path: Path) -> None:
    """Run any pending migrations to reach SCHEMA_VERSION."""
    version = get_version(db_path)
    if version == SCHEMA_VERSION:
        return
    if version < SCHEMA_VERSION:
        _up(db_path, version)
    else:
        _down(db_path, version)


def _up(db_path: Path, from_version: int) -> None:
    """Upgrade database from from_version to SCHEMA_VERSION."""
    backup_path = db_path.with_suffix(f".v{from_version}.bak")
    if db_path.exists():
        shutil.copy2(db_path, backup_path)

    tmp_path = db_path.with_suffix(".tmp")
    conn = sqlite3.connect(str(tmp_path))

    for v in range(from_version, SCHEMA_VERSION):
        _run_migration(conn, v + 1, direction="up")

    _copy_data(db_path, conn)
    conn.close()

    if db_path.exists():
        db_path.unlink()
    tmp_path.rename(db_path)


def _down(db_path: Path, from_version: int) -> None:
    """Downgrade database from from_version to SCHEMA_VERSION."""
    backup_path = db_path.with_suffix(f".v{from_version}.bak")
    if db_path.exists():
        shutil.copy2(db_path, backup_path)

    tmp_path = db_path.with_suffix(".tmp")
    conn = sqlite3.connect(str(tmp_path))

    for v in range(from_version, SCHEMA_VERSION, -1):
        _run_migration(conn, v, direction="down")

    _copy_data(db_path, conn)
    conn.close()

    if db_path.exists():
        db_path.unlink()
    tmp_path.rename(db_path)


def _run_migration(conn: sqlite3.Connection, version: int, direction: str) -> None:
    """Run a single migration step."""
    from xg_project.session._migrations import v1

    if version == 1:
        if direction == "up":
            v1.up(conn)
        else:
            v1.down(conn)


def _copy_data(src_path: Path, dst_conn: sqlite3.Connection) -> None:
    """Copy data from src database to dst connection."""
    if not src_path.exists():
        return
    src_conn = sqlite3.connect(str(src_path))

    # Copy sessions
    try:
        rows = src_conn.execute("SELECT directory, filename FROM sessions").fetchall()
        for directory, filename in rows:
            dst_conn.execute(
                "INSERT OR IGNORE INTO sessions (directory, filename) VALUES (?, ?)",
                (directory, filename),
            )
    except sqlite3.OperationalError:
        pass

    # Copy messages
    try:
        rows = src_conn.execute(
            "SELECT session_id, message_id, message_type, content, data FROM messages"
        ).fetchall()
        for session_id, message_id, message_type, content, data in rows:
            dst_conn.execute(
                "INSERT INTO messages (session_id, message_id, message_type, content, data)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, message_id, message_type, content, data),
            )
    except sqlite3.OperationalError:
        pass

    src_conn.close()
    dst_conn.commit()
