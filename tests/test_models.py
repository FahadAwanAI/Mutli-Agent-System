import pytest
from pydantic import ValidationError

from src.models import (
    EventMetadata,
    KeyFact,
    PredictionOutcome,
    PredictionOutput,
    DebateTurn,
    DebateSession,
)


class TestPredictionOutput:
    def test_valid_yes_prediction(self):
        pred = PredictionOutput(
            event_id="evt-1",
            prediction=PredictionOutcome.YES,
            probability=0.85,
            key_facts=[KeyFact(claim="Strong evidence.", source="https://example.com")],
            rationale="Clear reasoning here.",
        )
        assert pred.prediction == PredictionOutcome.YES
        assert pred.probability == 0.85

    def test_valid_no_prediction(self):
        pred = PredictionOutput(
            event_id="evt-1",
            prediction=PredictionOutcome.NO,
            probability=0.2,
            key_facts=[],
            rationale="Insufficient evidence.",
        )
        assert pred.prediction == PredictionOutcome.NO

    def test_probability_below_zero_raises(self):
        with pytest.raises(ValidationError):
            PredictionOutput(
                event_id="evt-1",
                prediction="YES",
                probability=-0.1,
                key_facts=[],
                rationale="Test",
            )

    def test_probability_above_one_raises(self):
        with pytest.raises(ValidationError):
            PredictionOutput(
                event_id="evt-1",
                prediction="YES",
                probability=1.1,
                key_facts=[],
                rationale="Test",
            )

    def test_invalid_prediction_outcome_raises(self):
        with pytest.raises(ValidationError):
            PredictionOutput(
                event_id="evt-1",
                prediction="MAYBE",  # Not a valid enum value
                probability=0.5,
                key_facts=[],
                rationale="Test",
            )

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            PredictionOutput(
                # Missing event_id
                prediction="YES",
                probability=0.5,
                key_facts=[],
                rationale="Test",
            )

    def test_prediction_from_string_enum(self):
        """Pydantic should coerce string 'YES' to PredictionOutcome.YES."""
        pred = PredictionOutput(
            event_id="evt-1",
            prediction="YES",
            probability=0.6,
            key_facts=[],
            rationale="Test",
        )
        assert pred.prediction == PredictionOutcome.YES


class TestEventMetadata:
    def test_valid_event(self):
        event = EventMetadata(
            event_id="74949",
            title="Test Event",
            description="A test description.",
            resolution_rules="Resolves YES if X happens.",
            resolution_date="2025-12-31T00:00:00Z",
        )
        assert event.event_id == "74949"

    def test_optional_fields_default_to_none(self):
        event = EventMetadata(
            event_id="74949",
            title="Test",
            description="Desc",
            resolution_rules="Rules",
            resolution_date="2025-12-31",
        )
        assert event.market_probability is None
        assert event.liquidity is None


class TestKeyFact:
    def test_valid_key_fact(self):
        fact = KeyFact(claim="Some claim.", source="https://example.com")
        assert fact.claim == "Some claim."

    def test_missing_claim_raises(self):
        with pytest.raises(ValidationError):
            KeyFact(source="https://example.com")

    def test_missing_source_raises(self):
        with pytest.raises(ValidationError):
            KeyFact(claim="Some claim.")


class TestDebateModels:
    def test_debate_turn_valid(self):
        turn = DebateTurn(agent_name="ChatGPT", content="I disagree because...")
        assert turn.agent_name == "ChatGPT"
        assert turn.challenge_target is None

    def test_debate_session_valid(self):
        session = DebateSession(
            event_id="evt-1",
            transcript=[
                DebateTurn(agent_name="Grok", content="My point is..."),
                DebateTurn(agent_name="Gemini", content="Counter-point..."),
            ],
        )
        assert len(session.transcript) == 2
        assert session.summary is None
