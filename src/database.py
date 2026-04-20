import sqlite3

from src.exceptions import DatabaseError
from src.logger import get_logger
from src.models import EventMetadata, PredictionOutput

logger = get_logger(__name__)


class Database:
    """Thin wrapper around SQLite for storing events and predictions."""

    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't already exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id               TEXT PRIMARY KEY,
                        title            TEXT,
                        description      TEXT,
                        rules            TEXT,
                        resolution_date  TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id    TEXT,
                        agent_name  TEXT,
                        prediction  TEXT,
                        probability REAL,
                        data        TEXT,
                        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (event_id) REFERENCES events (id)
                    )
                """)
            logger.info("Database initialised.", extra={"db_path": self.db_path})
        except sqlite3.Error as exc:
            logger.error("Failed to initialise database.", extra={"error": str(exc)}, exc_info=True)
            raise DatabaseError(
                message="Database initialisation failed.", details={"error": str(exc)}
            ) from exc

    def save_event(self, event: EventMetadata) -> None:
        """
        Persist an event record (INSERT OR REPLACE).

        Raises:
            DatabaseError: on any sqlite3 failure.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO events "
                    "(id, title, description, rules, resolution_date) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.title,
                        event.description,
                        event.resolution_rules,
                        event.resolution_date,
                    ),
                )
            logger.info("Event saved.", extra={"event_id": event.event_id})
        except sqlite3.Error as exc:
            logger.error(
                "Failed to save event.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            raise DatabaseError(
                message=f"Failed to save event '{event.event_id}'.",
                details={"error": str(exc)},
            ) from exc

    def save_prediction(self, agent_name: str, prediction: PredictionOutput) -> None:
        """
        Persist a prediction record.

        Raises:
            DatabaseError: on any sqlite3 failure.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO predictions "
                    "(event_id, agent_name, prediction, probability, data) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        prediction.event_id,
                        agent_name,
                        prediction.prediction.value,
                        prediction.probability,
                        prediction.model_dump_json(),
                    ),
                )
            logger.info(
                "Prediction saved.",
                extra={
                    "agent": agent_name,
                    "event_id": prediction.event_id,
                    "prediction": prediction.prediction.value,
                },
            )
        except sqlite3.Error as exc:
            logger.error(
                "Failed to save prediction.",
                extra={"agent": agent_name, "event_id": prediction.event_id, "error": str(exc)},
                exc_info=True,
            )
            raise DatabaseError(
                message=f"Failed to save prediction for agent '{agent_name}'.",
                details={"error": str(exc)},
            ) from exc

    def get_predictions_for_event(self, event_id: str) -> list[dict]:
        """
        Retrieve all predictions for a given event.

        Returns an empty list on failure (non-critical read path).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM predictions WHERE event_id = ? ORDER BY created_at DESC",
                    (event_id,),
                )
                rows = [dict(row) for row in cursor.fetchall()]
            logger.debug(
                "Predictions retrieved.",
                extra={"event_id": event_id, "count": len(rows)},
            )
            return rows
        except sqlite3.Error as exc:
            logger.error(
                "Failed to retrieve predictions.",
                extra={"event_id": event_id, "error": str(exc)},
                exc_info=True,
            )
            return []
