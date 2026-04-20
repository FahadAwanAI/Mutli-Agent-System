from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionOutcome(str, Enum):
    YES = "YES"
    NO = "NO"


class KeyFact(BaseModel):
    """A single evidence claim with its source."""
    claim: str
    source: str
    source_trust_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # Trust score of the source


class PredictionOutput(BaseModel):
    """
    The structured output produced by each AI agent.

    Constraints:
      - probability must be in [0.0, 1.0]
      - prediction must be YES or NO (enforced by enum)
      - response_score tracks hallucination/bias risk
    """
    event_id: str
    prediction: PredictionOutcome
    probability: float = Field(ge=0.0, le=1.0)
    key_facts: List[KeyFact]
    rationale: str
    
    # Response scoring (optional, added after generation)
    response_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # Overall confidence score
    hallucination_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # Risk of hallucination
    bias_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # Risk of bias
    confidence_level: Optional[str] = None  # HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, UNRELIABLE


class EventMetadata(BaseModel):
    """Polymarket event data fetched from the Gamma API."""
    event_id: str
    title: str
    description: str
    resolution_rules: str
    market_probability: Optional[float] = None
    liquidity: Optional[float] = None
    resolution_date: str


# ── V1: Debate Layer Models ───────────────────────────────────────────────────

class DebateTurn(BaseModel):
    """A single turn in the debate transcript."""
    agent_name: str
    content: str
    challenge_target: Optional[str] = None   # Agent being challenged
    challenged_claim: Optional[str] = None   # Specific claim being rebutted


class DebateSession(BaseModel):
    """Full debate session record."""
    event_id: str
    transcript: List[DebateTurn]
    summary: Optional[str] = None
