"""SQLite-backed :class:`ObservationsStorage`.

Default behavior — when no ``db_path`` is supplied — creates a fresh
file inside the OS tempdir (instead of the working directory). This
avoids the surprise of stray ``<uuid>.db`` files showing up wherever
the process happens to start.

Supports use as a context manager so the underlying connection is
closed deterministically.
"""
import os
import pickle
import sqlite3
import tempfile
from datetime import datetime
from typing import Optional, Sequence
from uuid import uuid4

from fractal.core.base.observations.observation import Observation
from fractal.core.base.observations.observations_storage import \
    ObservationsStorage


class SQLiteObservationsStorage(ObservationsStorage):

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to the SQLite file. ``None`` or empty string ⇒
                a fresh ``<uuid>.db`` is created in the OS temp directory.
        """
        if not db_path:
            db_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.db")
        self.db_path: str = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                observation BLOB NOT NULL
            )
            """
        )
        self.connection.commit()

    def write(self, observation: Observation):
        cursor = self.connection.cursor()
        obs_blob = pickle.dumps(observation)
        timestamp_str = observation.timestamp.isoformat()
        cursor.execute(
            "INSERT INTO observations (timestamp, observation) VALUES (?, ?)",
            (timestamp_str, obs_blob)
        )
        self.connection.commit()

    def read(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Sequence[Observation]:
        cursor = self.connection.cursor()
        query = "SELECT observation FROM observations"
        params = []
        conditions = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        observations = [pickle.loads(row["observation"]) for row in rows]
        return observations

    def close(self) -> None:
        """Close the SQLite connection. Idempotent."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None  # type: ignore[assignment]

    # ------------------------------------------------- context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        # Best-effort fallback — proper cleanup is via ``close()`` or
        # ``with`` block. Suppress any errors that may arise during
        # interpreter shutdown.
        try:
            self.close()
        except Exception:  # pragma: no cover  # pylint: disable=broad-exception-caught
            pass
