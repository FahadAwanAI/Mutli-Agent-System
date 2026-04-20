import os
from abc import ABC, abstractmethod
from typing import Optional

from tavily import TavilyClient

from src.exceptions import ResearchError
from src.logger import get_logger
from src.models import EventMetadata, PredictionOutput
from src.research import ContextChunker, ResponseScorer

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all prediction agents.

    Subclasses must implement:
        - has_valid_config() -> bool
        - generate_prediction_async(event) -> PredictionOutput  (async)
        - generate_prediction(event) -> PredictionOutput         (sync wrapper)
    """

    def __init__(self, name: str, model_name: str, archetype: str) -> None:
        self.name: str = name
        self.model_name: str = model_name
        self.archetype: str = archetype

        tavily_key: Optional[str] = os.getenv("TAVILY_API_KEY")
        self.tavily: Optional[TavilyClient] = TavilyClient(api_key=tavily_key) if tavily_key else None

        if not self.tavily:
            logger.warning(
                "Tavily client not configured — research will be skipped.",
                extra={"agent": self.name, "hint": "Set TAVILY_API_KEY in .env"},
            )

    def research(self, query: str) -> str:
        """
        Perform web research via Tavily with LLM-driven source validation and context optimization.

        Features:
          1. Fetches sources from Tavily API
          2. Optimizes context (chunking, summarization, token budgeting)
          3. Provides all sources to LLM with validation instruction
          4. LLM decides which sources to trust based on content analysis
          5. Degrades gracefully if Tavily is unavailable

        Args:
            query: Search query string.

        Returns:
            Optimized research context with all sources (LLM will filter), or fallback message.
        """
        if not self.tavily:
            logger.info(
                "Research skipped — no Tavily key.",
                extra={"agent": self.name, "query": query[:80]},
            )
            return "Research skipped: TAVILY_API_KEY not provided."

        logger.info(
            "Starting web research with source validation.",
            extra={"agent": self.name, "query": query[:120]},
        )

        try:
            response = self.tavily.search(query=query, search_depth="advanced")
            results = response.get("results", [])

            if not results:
                logger.warning(
                    "No research results found.",
                    extra={"agent": self.name, "query": query[:80]},
                )
                return "No research results found."

            # Optimize context with source validation
            optimized_context = ContextChunker.optimize_context(results, max_tokens=2000)

            logger.info(
                "Research completed with source validation.",
                extra={
                    "agent": self.name,
                    "total_results": len(results),
                    "context_length": len(optimized_context),
                },
            )
            return optimized_context

        except Exception as exc:
            err = ResearchError(query=query, reason=str(exc))
            logger.error(
                "Research failed — continuing without context.",
                extra={"agent": self.name, "error": str(err)},
                exc_info=True,
            )
            return f"Research failed: {exc}"

    def score_prediction(self, response_text: str, probability: float) -> dict:
        """
        Score a prediction response for hallucination and bias.

        Args:
            response_text: The LLM's response text
            probability: The predicted probability (0.0-1.0)

        Returns:
            Scoring dict with overall_score, confidence, risks, and recommendation
        """
        score = ResponseScorer.score_response(response_text, probability)
        logger.info(
            "Prediction scored for hallucination and bias.",
            extra={
                "agent": self.name,
                "overall_score": score["overall_score"],
                "hallucination_risk": score["hallucination_risk"],
                "bias_risk": score["bias_risk"],
                "recommendation": score["recommendation"],
            },
        )
        return score

    @abstractmethod
    def has_valid_config(self) -> bool:
        """Return True if this agent has the required API keys configured."""

    @abstractmethod
    def generate_prediction(self, event: EventMetadata) -> PredictionOutput:
        """
        Generate a locked prediction for the given event (sync entry point).

        Internally calls generate_prediction_async via asyncio.run().

        Args:
            event: Polymarket event metadata.

        Returns:
            PredictionOutput with YES/NO prediction, probability, key facts, rationale.

        Raises:
            LLMCallError:          if the API call itself fails after all retries.
            LLMResponseParseError: if the response cannot be decoded as JSON.
            LLMValidationError:    if the JSON fails Pydantic schema validation.
        """
