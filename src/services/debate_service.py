import os
import sqlite3
import time
from typing import Dict, List, Optional

import google.generativeai as genai

from src.database import Database
from src.exceptions import DebateError
from src.logger import get_logger
from src.utils.console import (
    console,
    print_error,
    print_header,
    print_moderator,
    print_predictions_table,
    print_section,
)

logger = get_logger(__name__)

_MAX_ATTEMPTS = 3
_INITIAL_WAIT = 10  # seconds


class DebateService:
    """Text debate using fresh predictions from the current session."""

    def __init__(self, agents):
        self.db = Database()
        self.agents = [a for a in agents if a.has_valid_config()]
        self._llm: Optional[genai.GenerativeModel] = None
        self._configure_llm()

    def _configure_llm(self) -> None:
        """Initialise the Gemini model used as debate moderator/responder."""
        for key_name in ("GEMINI_API_KEY", "CHATGPT_GEMINI_KEY", "GROK_GEMINI_KEY"):
            key = os.getenv(key_name)
            if key and len(key) > 20:
                try:
                    genai.configure(api_key=key)
                    self._llm = genai.GenerativeModel("gemini-2.0-flash")
                    logger.info("Debate LLM configured.", extra={"key_source": key_name})
                    return
                except Exception as exc:
                    logger.warning(
                        "Failed to configure debate LLM with key.",
                        extra={"key_source": key_name, "error": str(exc)},
                    )

        logger.error("No valid API key found for debate LLM — debate will be skipped.")

    def _generate_response(self, prompt: str) -> Optional[str]:
        """
        Call the debate LLM with exponential back-off on rate limits.

        Returns the response text, or None if all attempts fail.
        """
        if not self._llm:
            logger.warning("Debate LLM not configured — skipping response generation.")
            return None

        wait = _INITIAL_WAIT
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._llm.generate_content(prompt)
                text = response.text.strip()
                if text and len(text) > 20:
                    logger.debug("Debate response generated.", extra={"length": len(text)})
                    return text
                logger.warning("Debate LLM returned empty/short response.", extra={"attempt": attempt})
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "quota" in err_str.lower():
                    if attempt < _MAX_ATTEMPTS:
                        logger.warning(
                            "Rate limit hit — backing off.",
                            extra={"attempt": attempt, "wait_seconds": wait},
                        )
                        console.print(f"   [dim]⏳ Rate limit — waiting {wait}s...[/dim]")
                        time.sleep(wait)
                        wait = int(wait * 1.5)
                    else:
                        logger.error("Rate limit persisted after all retries.", extra={"error": err_str})
                else:
                    logger.error(
                        "Debate LLM call failed.",
                        extra={"attempt": attempt, "error": err_str},
                        exc_info=True,
                    )
                    time.sleep(2)

        return None

    def _get_event_title(self, event_id: str) -> str:
        """Retrieve event title from the local database."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT title FROM events WHERE id = ?", (event_id,))
                row = cursor.fetchone()
                return row[0] if row else "Unknown Event"
        except Exception as exc:
            logger.warning("Could not retrieve event title.", extra={"event_id": event_id, "error": str(exc)})
            return "Unknown Event"

    def run_debate(
        self, event_id: str, predictions: List[Dict], rounds: int = 2
    ) -> Dict:
        """
        Run a structured panel debate over the locked predictions.

        Args:
            event_id:    Polymarket event ID (used to look up the title).
            predictions: List of agent prediction dicts from PredictionService.
            rounds:      Number of debate rounds (currently used for future expansion).

        Returns:
            Dict with event_id, predictions, and transcript.

        Raises:
            DebateError: if fewer than 2 predictions are provided.
        """
        if len(predictions) < 2:
            raise DebateError(
                message="Need at least 2 predictions to run a debate.",
                details={"prediction_count": len(predictions)},
            )

        event_title = self._get_event_title(event_id)
        logger.info("Debate started.", extra={"event_id": event_id, "agents": len(predictions)})

        print_header("⚔️ PANEL DEBATE", event_title)

        console.print("\n[dim]Locked predictions:[/dim]")
        for pred in predictions:
            color = "green" if pred["prediction"] == "YES" else "red"
            console.print(
                f"   [{color}]{pred['agent_name']}: {pred['prediction']} "
                f"({pred['probability'] * 100:.0f}%)[/{color}]"
            )
        console.print()

        print_moderator("All predictions are locked. Let's begin the panel discussion.", is_intro=True)

        transcript: List[Dict] = []

        for target in predictions:
            target_name = target["agent_name"]
            claims = target.get("key_facts", [])

            if not claims:
                logger.warning("Agent has no key facts — skipping.", extra={"agent": target_name})
                continue

            print_section(f"Discussing {target_name}'s Position")

            for claim in claims:
                claim_text = claim.get("claim", "")
                source = claim.get("source", "unknown")

                console.print(f"[yellow]📌 {target_name}'s Claim:[/yellow]")
                console.print(f'   "{claim_text}"')
                console.print(f"   [dim]Source: {source}[/dim]\n")

                other_agents = [p for p in predictions if p["agent_name"] != target_name]

                if len(other_agents) >= 1:
                    challenger1 = other_agents[0]["agent_name"]
                    challenge = self._generate_response(
                        f'You are {challenger1}. Challenge {target_name}\'s claim:\n'
                        f'"{claim_text}"\n\n'
                        f'Start with "{target_name}," and explain why you disagree. 2 sentences.'
                    )
                    if challenge:
                        console.print(f"   💬 [bold]{challenger1}:[/bold]")
                        console.print(f"      {challenge}\n")
                        transcript.append({"speaker": challenger1, "text": challenge})

                        defense = self._generate_response(
                            f'You are {target_name}. {challenger1} challenged you:\n'
                            f'"{challenge}"\n\n'
                            f'Respond to {challenger1}. Defend your position. '
                            f'Start with "{challenger1}," 2 sentences.'
                        )
                        if defense:
                            console.print(f"   💬 [bold]{target_name}:[/bold]")
                            console.print(f"      {defense}\n")
                            transcript.append({"speaker": target_name, "text": defense})

                if len(other_agents) >= 2:
                    challenger2 = other_agents[1]["agent_name"]
                    addition = self._generate_response(
                        f"You are {challenger2}. {target_name} and {challenger1} are debating.\n"
                        f"Add your perspective. 2 sentences."
                    )
                    if addition:
                        console.print(f"   💬 [bold]{challenger2}:[/bold]")
                        console.print(f"      {addition}\n")
                        transcript.append({"speaker": challenger2, "text": addition})

        print_moderator(
            "This concludes our panel discussion. All predictions remain locked.",
            is_intro=False,
        )
        print_predictions_table(predictions)

        logger.info(
            "Debate complete.",
            extra={"event_id": event_id, "transcript_turns": len(transcript)},
        )

        return {"event_id": event_id, "predictions": predictions, "transcript": transcript}
