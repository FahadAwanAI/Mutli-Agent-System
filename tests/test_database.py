import pytest

from src.database import Database
from src.exceptions import DatabaseError
from src.models import EventMetadata, KeyFact, PredictionOutcome, PredictionOutput


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path) -> Database:
    """Fresh database backed by a temp file for each test."""
    return Database(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_event() -> EventMetadata:
    return EventMetadata(
        event_id="evt-001",
        title="Will AI replace all jobs by 2030?",
        description="A prediction market on AI job displacement.",
        resolution_rules="Resolves YES if >50% of jobs are automated by 2030.",
        resolution_date="2030-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_prediction() -> PredictionOutput:
    return PredictionOutput(
        event_id="evt-001",
        prediction=PredictionOutcome.NO,
        probability=0.15,
        key_facts=[
            KeyFact(claim="Most jobs require human creativity.", source="https://example.com"),
        ],
        rationale="Historical evidence shows technology creates more jobs than it destroys.",
    )


# ── Initialisation ────────────────────────────────────────────────────────────

class TestDatabaseInit:
    def test_tables_created_on_init(self, db):
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "events" in tables
        assert "predictions" in tables


# ── save_event ────────────────────────────────────────────────────────────────

class TestSaveEvent:
    def test_save_event_persists_record(self, db, sample_event):
        db.save_event(sample_event)

        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT id, title FROM events WHERE id = ?", (sample_event.event_id,)
            ).fetchone()

        assert row is not None
        assert row[0] == "evt-001"
        assert row[1] == sample_event.title

    def test_save_event_upserts_on_duplicate(self, db, sample_event):
        db.save_event(sample_event)
        # Modify title and save again — should not raise
        sample_event.title = "Updated title"
        db.save_event(sample_event)

        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM events WHERE id = ?", (sample_event.event_id,)
            ).fetchone()[0]

        assert count == 1  # Still only one row


# ── save_prediction ───────────────────────────────────────────────────────────

class TestSavePrediction:
    def test_save_prediction_persists_record(self, db, sample_event, sample_prediction):
        db.save_event(sample_event)
        db.save_prediction("Gemini", sample_prediction)

        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT agent_name, prediction, probability FROM predictions WHERE event_id = ?",
                (sample_prediction.event_id,),
            ).fetchone()

        assert row is not None
        assert row[0] == "Gemini"
        assert row[1] == "NO"
        assert row[2] == pytest.approx(0.15)

    def test_multiple_predictions_for_same_event(self, db, sample_event, sample_prediction):
        db.save_event(sample_event)
        db.save_prediction("ChatGPT", sample_prediction)
        db.save_prediction("Grok", sample_prediction)
        db.save_prediction("Gemini", sample_prediction)

        rows = db.get_predictions_for_event(sample_event.event_id)
        assert len(rows) == 3
        agent_names = {r["agent_name"] for r in rows}
        assert agent_names == {"ChatGPT", "Grok", "Gemini"}


# ── get_predictions_for_event ─────────────────────────────────────────────────

class TestGetPredictions:
    def test_returns_empty_list_for_unknown_event(self, db):
        rows = db.get_predictions_for_event("nonexistent-event")
        assert rows == []

    def test_returns_correct_predictions(self, db, sample_event, sample_prediction):
        db.save_event(sample_event)
        db.save_prediction("ChatGPT", sample_prediction)

        rows = db.get_predictions_for_event(sample_event.event_id)
        assert len(rows) == 1
        assert rows[0]["agent_name"] == "ChatGPT"
