class PredictionBattleError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details})"


# ── LLM / Agent Errors ───────────────────────────────────────────────────────

class AgentError(PredictionBattleError):
    """Raised when an AI agent encounters a failure."""


class LLMCallError(AgentError):
    """
    Raised when the underlying LLM API call fails.
    Covers: network errors, auth failures, rate limits, timeouts.
    """

    def __init__(self, agent_name: str, reason: str, original: Exception | None = None):
        super().__init__(
            message=f"LLM call failed for agent '{agent_name}': {reason}",
            details={"agent": agent_name, "reason": reason},
        )
        self.original = original


class LLMResponseParseError(AgentError):
    """
    Raised when the LLM returns a response that cannot be decoded as JSON.
    The raw response is preserved for debugging.
    """

    def __init__(self, agent_name: str, raw_response: str, original: Exception | None = None):
        super().__init__(
            message=f"Agent '{agent_name}' returned non-JSON response.",
            details={"agent": agent_name, "raw_response": raw_response[:500]},
        )
        self.raw_response = raw_response
        self.original = original


class LLMValidationError(AgentError):
    """
    Raised when parsed JSON fails Pydantic schema validation.
    The parsed dict is preserved so we can log exactly what was wrong.
    """

    def __init__(self, agent_name: str, parsed_data: dict, validation_errors: str):
        super().__init__(
            message=f"Agent '{agent_name}' output failed schema validation.",
            details={
                "agent": agent_name,
                "validation_errors": validation_errors,
                "parsed_data": parsed_data,
            },
        )
        self.parsed_data = parsed_data
        self.validation_errors = validation_errors


# ── Research Errors ──────────────────────────────────────────────────────────

class ResearchError(PredictionBattleError):
    """Raised when the Tavily web research step fails."""

    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Research failed for query '{query[:80]}': {reason}",
            details={"query": query, "reason": reason},
        )


# ── Polymarket / Data Errors ─────────────────────────────────────────────────

class EventFetchError(PredictionBattleError):
    """Raised when fetching event data from Polymarket fails."""

    def __init__(self, identifier: str, status_code: int | None = None, reason: str = ""):
        super().__init__(
            message=f"Failed to fetch event '{identifier}'. {reason}",
            details={"identifier": identifier, "status_code": status_code, "reason": reason},
        )


class EventNotFoundError(EventFetchError):
    """Raised when the event identifier resolves to no results."""

    def __init__(self, identifier: str):
        super().__init__(identifier=identifier, reason="No matching event found.")


# ── Database Errors ──────────────────────────────────────────────────────────

class DatabaseError(PredictionBattleError):
    """Raised for database-level failures."""


# ── Debate Errors ────────────────────────────────────────────────────────────

class DebateError(PredictionBattleError):
    """Raised when the debate engine encounters an unrecoverable error."""
