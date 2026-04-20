from typing import Dict, List, Tuple

from src.agents.specialized_agents import ChatGPTAgent, GeminiAgent, GrokAgent
from src.database import Database
from src.exceptions import (
    EventFetchError,
    LLMCallError,
    LLMResponseParseError,
    LLMValidationError,
)
from src.logger import get_logger
from src.models import PredictionOutput
from src.services.polymarket_service import PolymarketService
from src.utils.console import (
    console,
    print_agents_status,
    print_error,
    print_event,
    print_prediction,
    print_predictions_table,
    print_section,
)

logger = get_logger(__name__)


class PredictionService:
    """Orchestrates prediction battles across all configured agents."""

    def __init__(self):
        self.db = Database()
        self.all_agents = [
            ChatGPTAgent(),
            GrokAgent(),
            GeminiAgent(),
        ]
        logger.info("PredictionService initialised.", extra={"agent_count": len(self.all_agents)})

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_active_agents(self):
        active, inactive = [], []
        for agent in self.all_agents:
            if agent.has_valid_config():
                active.append(agent)
            else:
                inactive.append(agent.name)
        return active, inactive

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """Return a short, user-friendly error description."""
        msg = str(exc)
        if "429" in msg or "quota" in msg.lower():
            return "Rate limit exceeded — wait 1 minute."
        if "401" in msg or "unauthorized" in msg.lower() or "invalid" in msg.lower():
            return "Invalid API key."
        if "404" in msg or "not found" in msg.lower():
            return "Model or endpoint not found."
        return msg.split("\n")[0][:100]

    # ── Public API ────────────────────────────────────────────────────────────

    def run_battle(
        self, event_id: str
    ) -> Tuple[List[PredictionOutput], List[Dict]]:
        """
        Run the full prediction battle for a given event.

        Args:
            event_id: Polymarket event ID, slug, or URL.

        Returns:
            Tuple of:
              - predictions: list of PredictionOutput objects
              - agent_predictions: list of dicts ready for the debate phase
        """
        logger.info("Battle started.", extra={"event_id": event_id})

        # 1. Fetch event ──────────────────────────────────────────────────────
        try:
            event = PolymarketService.get_event_details(event_id)
        except EventFetchError as exc:
            logger.error("Event fetch failed — aborting battle.", extra={"error": str(exc)})
            print_error(f"Failed to fetch event: {exc.message}")
            return [], {}

        print_event(
            event.title,
            event.event_id,
            event.description,
            event.resolution_rules,
            event.resolution_date,
        )
        self.db.save_event(event)

        # 2. Check active agents ───────────────────────────────────────────────
        active_agents, inactive_names = self._get_active_agents()
        print_agents_status([a.name for a in active_agents], inactive_names)

        if not active_agents:
            logger.error("No agents configured — aborting battle.")
            print_error("No agents configured! Add API keys to .env")
            return [], {}

        # 3. Run predictions ───────────────────────────────────────────────────
        predictions: List[PredictionOutput] = []
        agent_predictions: List[Dict] = []

        for agent in active_agents:
            print_section(f"{agent.name} Researching")
            logger.info("Agent prediction started.", extra={"agent": agent.name, "event_id": event.event_id})

            try:
                pred = agent.generate_prediction(event)

            except LLMCallError as exc:
                logger.error(
                    "Agent API call failed — skipping.",
                    extra={"agent": agent.name, "error": exc.message},
                )
                print_error(f"{agent.name} failed: {self._classify_error(exc)}")
                continue

            except LLMResponseParseError as exc:
                logger.error(
                    "Agent returned unparseable JSON — skipping.",
                    extra={"agent": agent.name, "snippet": exc.details.get("raw_response", "")[:200]},
                )
                print_error(f"{agent.name} returned invalid JSON — skipping.")
                continue

            except LLMValidationError as exc:
                logger.error(
                    "Agent output failed schema validation — skipping.",
                    extra={"agent": agent.name, "validation_errors": exc.validation_errors},
                )
                print_error(f"{agent.name} output failed validation — skipping.")
                continue

            except Exception as exc:
                logger.error(
                    "Unexpected error during prediction — skipping.",
                    extra={"agent": agent.name, "error": str(exc)},
                    exc_info=True,
                )
                print_error(f"{agent.name} failed: {self._classify_error(exc)}")
                continue

            # Persist and collect
            self.db.save_prediction(agent.name, pred)
            predictions.append(pred)

            agent_pred = {
                "agent_name": agent.name,
                "prediction": pred.prediction.value,
                "probability": pred.probability,
                "rationale": pred.rationale,
                "key_facts": [
                    {"claim": f.claim, "source": f.source} for f in pred.key_facts
                ],
            }
            agent_predictions.append(agent_pred)

            print_prediction(
                agent.name,
                pred.prediction.value,
                pred.probability,
                pred.rationale,
                agent_pred["key_facts"],
            )

            logger.info(
                "Agent prediction completed.",
                extra={
                    "agent": agent.name,
                    "prediction": pred.prediction.value,
                    "probability": pred.probability,
                },
            )

        # 4. Summary ───────────────────────────────────────────────────────────
        if agent_predictions:
            print_section("Predictions Summary")
            print_predictions_table(agent_predictions)

        logger.info(
            "Battle complete.",
            extra={
                "event_id": event_id,
                "predictions_count": len(predictions),
                "skipped": len(active_agents) - len(predictions),
            },
        )

        return predictions, agent_predictions
