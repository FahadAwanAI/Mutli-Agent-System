"""
Moderator Agent - Guides the debate between prediction agents.
Uses Gemini or OpenAI depending on available API keys.
"""

import os
import google.generativeai as genai
from openai import OpenAI
from src.prompts import MODERATOR_SYSTEM_PROMPT


class ModeratorAgent:
    """Moderator that guides the debate between agents."""
    
    def __init__(self):
        self._client = None
        self._gemini_model = None
        self._provider = None
        self._detect_provider()

    def _detect_provider(self):
        """Detect which LLM provider to use for moderation."""
        # Check OpenAI first
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and len(openai_key) > 20 and not openai_key.startswith("your_"):
            self._provider = "openai"
            self._client = OpenAI(api_key=openai_key)
            return
        
        # Check Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and len(gemini_key) > 20 and not gemini_key.startswith("your_"):
            self._provider = "gemini"
            genai.configure(api_key=gemini_key)
            self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            return
        
        self._provider = None

    def has_valid_config(self) -> bool:
        if os.getenv("TEST_MODE", "false").lower() == "true":
            return True
        return self._provider is not None

    def provide_direction(self, event_title: str, predictions_summary: str, transcript: str) -> str:
        """Analyzes the debate and provides direction for the next turn."""
        
        # Mock mode
        if os.getenv("TEST_MODE", "false").lower() == "true":
            return "Agent C, defend your skepticism. Agent A, challenge the timeline assumptions."
        
        prompt = f"""EVENT: {event_title}

INITIAL PREDICTIONS:
{predictions_summary}

DEBATE TRANSCRIPT SO FAR:
{transcript}

Analyze the factual disagreements. Specify which agent should speak next and what claim they should address or defend."""

        try:
            if self._provider == "openai":
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": MODERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            
            elif self._provider == "gemini":
                full_prompt = f"{MODERATOR_SYSTEM_PROMPT}\n\n{prompt}"
                response = self._gemini_model.generate_content(full_prompt)
                return response.text
            
            else:
                return "No moderator configured."
                
        except Exception as e:
            return f"Moderation failed: {str(e)[:100]}"
