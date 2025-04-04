import pickle
import sqlite3
from datetime import datetime
from typing import Optional, Sequence
from uuid import uuid4

from fractal.core.base.observations.observation import Observation
from fractal.core.base.observations.observations_storage import \
    ObservationsStorage


class SQLiteObservationsStorage(ObservationsStorage):

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None or db_path == "":
            db_path: str = f'{str(uuid4())}.db'
        self.connection = sqlite3.connect(db_path)
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

    def __del__(self):
        if self.connection:
            self.connection.close()
