import asyncio
import json
import os
import time
from typing import Any, Optional

import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI
from pydantic import ValidationError

from src.agents.base_agent import BaseAgent
from src.agents.tools import OPENAI_TOOLS, OPENAI_TOOL_CHOICE, GEMINI_TOOLS, CLAUDE_TOOLS
from src.exceptions import LLMCallError, LLMResponseParseError, LLMValidationError
from src.logger import get_logger
from src.models import EventMetadata, PredictionOutput
from src.prompts import (
    CHATGPT_ARCHETYPE,
    GEMINI_ARCHETYPE,
    GROK_ARCHETYPE,
    PREDICTION_PROMPT,
    SYSTEM_PROMPT_PREFIX,
)
from src.research import ResponseScorer

logger = get_logger(__name__)

# Retry configuration
_MAX_RETRIES: int = 3
_RETRY_BASE_WAIT: float = 5.0   # seconds
_SELF_CORRECT_PROMPT: str = (
    "Your previous response could not be parsed. "
    "You MUST call the `submit_prediction` function with valid arguments. "
    "Do not respond with plain text."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_json(content: str) -> str:
    """Strip markdown/code fences that some models wrap around JSON."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return content.strip()


def detect_best_gemini_model(api_key: str) -> str:
    """Auto-detect the best available Gemini model for the given key."""
    if not api_key or len(api_key) < 20:
        return "gemini-2.0-flash"
    try:
        genai.configure(api_key=api_key)
        available = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for preferred in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"]:
            if preferred in available:
                return preferred
        return available[0] if available else "gemini-2.0-flash"
    except Exception:
        return "gemini-2.0-flash"


def detect_best_claude_model(api_key: str) -> str:
    """Detect the best available Claude model."""
    # Claude models are versioned; use the latest stable
    return "claude-3-5-sonnet-20241022"


def _validate_prediction(agent_name: str, data: dict[str, Any]) -> PredictionOutput:
    """
    Validate a parsed dict against the PredictionOutput Pydantic schema.

    Raises:
        LLMValidationError: if validation fails.
    """
    try:
        prediction = PredictionOutput(**data)
    except ValidationError as exc:
        logger.error(
            "LLM output failed Pydantic validation.",
            extra={"agent": agent_name, "errors": exc.errors(), "data": data},
        )
        raise LLMValidationError(
            agent_name=agent_name,
            parsed_data=data,
            validation_errors=str(exc),
        ) from exc

    logger.info(
        "Prediction validated successfully.",
        extra={
            "agent": agent_name,
            "prediction": prediction.prediction.value,
            "probability": prediction.probability,
        },
    )
    return prediction


def _apply_response_scoring(agent_name: str, prediction: PredictionOutput) -> PredictionOutput:
    """
    Apply response scoring to detect hallucination and bias.

    Updates the prediction with:
      - response_score: overall confidence (0.0-1.0)
      - hallucination_risk: risk of hallucination (0.0-1.0)
      - bias_risk: risk of bias (0.0-1.0)
      - confidence_level: HIGH/MEDIUM/LOW/UNRELIABLE

    Args:
        agent_name: Name of the agent for logging
        prediction: The prediction to score

    Returns:
        Updated prediction with scoring fields
    """
    # Score the rationale (main response text)
    score_result = ResponseScorer.score_response(prediction.rationale, prediction.probability)

    prediction.response_score = score_result["overall_score"]
    prediction.hallucination_risk = score_result["hallucination_risk"]
    prediction.bias_risk = score_result["bias_risk"]
    prediction.confidence_level = score_result["recommendation"]

    logger.info(
        "Response scoring applied.",
        extra={
            "agent": agent_name,
            "response_score": prediction.response_score,
            "hallucination_risk": prediction.hallucination_risk,
            "bias_risk": prediction.bias_risk,
            "confidence_level": prediction.confidence_level,
        },
    )

    return prediction


def _extract_tool_call_args_openai(agent_name: str, response: Any) -> dict[str, Any]:
    """
    Extract function call arguments from an OpenAI/xAI tool-use response.

    Raises:
        LLMResponseParseError: if no tool call is present or args are not valid JSON.
    """
    message = response.choices[0].message

    if not message.tool_calls:
        raw = message.content or ""
        logger.warning(
            "LLM did not call the tool — attempting JSON fallback.",
            extra={"agent": agent_name, "raw_snippet": raw[:200]},
        )
        # Fallback: try to parse the content as raw JSON
        try:
            return json.loads(clean_json(raw))
        except json.JSONDecodeError as exc:
            raise LLMResponseParseError(
                agent_name=agent_name, raw_response=raw, original=exc
            ) from exc

    raw_args = message.tool_calls[0].function.arguments
    logger.info(
        "Agent called tool via Function Calling.",
        extra={"agent": agent_name, "tool": message.tool_calls[0].function.name},
    )

    try:
        return json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise LLMResponseParseError(
            agent_name=agent_name, raw_response=raw_args, original=exc
        ) from exc


def _extract_tool_call_args_claude(agent_name: str, response: Any) -> dict[str, Any]:
    """
    Extract function call arguments from a Claude tool-use response.

    Claude returns tool calls in content blocks with type='tool_use'.
    The input is already a dict, not a JSON string.

    Raises:
        LLMResponseParseError: if no tool call is present.
    """
    # Find the tool_use block in the response
    tool_use_block = None
    for block in response.content:
        if block.type == "tool_use":
            tool_use_block = block
            break

    if not tool_use_block:
        raw = "".join(b.text for b in response.content if hasattr(b, "text"))
        logger.warning(
            "Claude did not call the tool — attempting JSON fallback.",
            extra={"agent": agent_name, "raw_snippet": raw[:200]},
        )
        try:
            return json.loads(clean_json(raw))
        except json.JSONDecodeError as exc:
            raise LLMResponseParseError(
                agent_name=agent_name, raw_response=raw, original=exc
            ) from exc

    logger.info(
        "Claude called tool via Function Calling.",
        extra={"agent": agent_name, "tool": tool_use_block.name},
    )

    # Claude's input is already a dict
    return tool_use_block.input


def _extract_tool_call_args_gemini(agent_name: str, response: Any) -> dict[str, Any]:
    """
    Extract function call arguments from a Gemini functionCall response.

    Gemini returns function calls in the response with args as a dict.

    Raises:
        LLMResponseParseError: if no function call is present.
    """
    # Check for function calls in the response
    if not response.candidates or not response.candidates[0].content.parts:
        raw = response.text if hasattr(response, "text") else ""
        logger.warning(
            "Gemini did not call the function — attempting JSON fallback.",
            extra={"agent": agent_name, "raw_snippet": raw[:200]},
        )
        try:
            return json.loads(clean_json(raw))
        except json.JSONDecodeError as exc:
            raise LLMResponseParseError(
                agent_name=agent_name, raw_response=raw, original=exc
            ) from exc

    # Look for function call in parts
    for part in response.candidates[0].content.parts:
        if hasattr(part, "function_call"):
            func_call = part.function_call
            logger.info(
                "Gemini called function via Function Calling.",
                extra={"agent": agent_name, "function": func_call.name},
            )
            # Gemini's args are already a dict-like object
            return dict(func_call.args)

    # No function call found, try text fallback
    raw = response.text if hasattr(response, "text") else ""
    logger.warning(
        "Gemini response had no function call — attempting JSON fallback.",
        extra={"agent": agent_name, "raw_snippet": raw[:200]},
    )
    try:
        return json.loads(clean_json(raw))
    except json.JSONDecodeError as exc:
        raise LLMResponseParseError(
            agent_name=agent_name, raw_response=raw, original=exc
        ) from exc


# ── Async retry wrapper ───────────────────────────────────────────────────────

async def _call_with_retry(
    agent_name: str,
    coro_factory,
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """
    Execute an async coroutine with exponential back-off retry.

    Retries on rate-limit (429) and transient server errors (5xx).
    Raises LLMCallError after all retries are exhausted.

    Args:
        agent_name:    Name of the calling agent (for logging).
        coro_factory:  Zero-argument callable that returns a new coroutine each call.
        max_retries:   Maximum number of attempts.
    """
    wait = _RETRY_BASE_WAIT
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "LLM API call attempt.",
                extra={"agent": agent_name, "attempt": attempt, "max": max_retries},
            )
            return await coro_factory()

        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
            is_server_error = "500" in err_str or "502" in err_str or "503" in err_str

            if (is_rate_limit or is_server_error) and attempt < max_retries:
                logger.warning(
                    "Transient API error — retrying with back-off.",
                    extra={
                        "agent": agent_name,
                        "attempt": attempt,
                        "wait_seconds": wait,
                        "error": err_str[:120],
                    },
                )
                await asyncio.sleep(wait)
                wait *= 2  # exponential back-off
            else:
                logger.error(
                    "API call failed — no more retries.",
                    extra={"agent": agent_name, "attempt": attempt, "error": err_str},
                    exc_info=True,
                )
                break

    raise LLMCallError(
        agent_name=agent_name,
        reason=str(last_exc),
        original=last_exc,
    )


# ── ChatGPT Agent ─────────────────────────────────────────────────────────────

class ChatGPTAgent(BaseAgent):
    """
    Precision-focused agent using OpenAI gpt-4o with native Function Calling.

    The LLM is forced to call `submit_prediction` via tool_choice,
    guaranteeing structured output instead of free-form text.
    """

    def __init__(self) -> None:
        self._openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self._use_openai: bool = bool(self._openai_key and self._openai_key.startswith("sk-"))

        if self._use_openai:
            self._async_client = AsyncOpenAI(api_key=self._openai_key)
            model_name = "gpt-4o"
            logger.info("ChatGPT configured with OpenAI API (native Function Calling).", extra={"model": model_name})
        else:
            self._async_client = None
            model_name = "gpt-4o"
            logger.warning("ChatGPT: OPENAI_API_KEY not configured.", extra={"hint": "Set OPENAI_API_KEY in .env"})

        super().__init__("ChatGPT", model_name, "Precision-Oriented")

    def has_valid_config(self) -> bool:
        return bool(self._openai_key) and len(self._openai_key) > 20 and self._openai_key.startswith("sk-")

    async def _call_openai(self, messages: list[dict]) -> Any:
        """Call OpenAI with native Function Calling enforced."""
        return await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice=OPENAI_TOOL_CHOICE,
        )

    async def generate_prediction_async(self, event: EventMetadata) -> PredictionOutput:
        """
        Async prediction generation with native Function Calling + response scoring.

        Flow:
          1. Research the event topic (with source validation & context optimization)
          2. Call the LLM with tool_choice forced to `submit_prediction`
          3. Extract and validate the tool call arguments
          4. Apply response scoring (hallucination & bias detection)
          5. If validation fails, self-correct once by re-prompting
        """
        logger.info("ChatGPT starting prediction.", extra={"event_id": event.event_id})

        context = self.research(f"{event.title} official sources documentation")
        system = f"{SYSTEM_PROMPT_PREFIX}\n{CHATGPT_ARCHETYPE}"
        user_content = PREDICTION_PROMPT.format(
            title=event.title,
            description=event.description,
            rules=event.resolution_rules,
            date=event.resolution_date,
            context=context,
            event_id=event.event_id,
        )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        response = await _call_with_retry(
            "ChatGPT",
            lambda: self._call_openai(messages),
        )
        try:
            data = _extract_tool_call_args_openai("ChatGPT", response)
            prediction = _validate_prediction("ChatGPT", data)
            # Apply response scoring
            prediction = _apply_response_scoring("ChatGPT", prediction)
            return prediction
        except (LLMResponseParseError, LLMValidationError) as exc:
            # ── Self-correction attempt ──────────────────────────────────
            logger.warning(
                "Self-correction triggered — re-prompting ChatGPT.",
                extra={"agent": "ChatGPT", "reason": str(exc)},
            )
            messages.append({"role": "user", "content": _SELF_CORRECT_PROMPT})
            response = await _call_with_retry(
                "ChatGPT",
                lambda: self._call_openai(messages),
            )
            data = _extract_tool_call_args_openai("ChatGPT", response)
            prediction = _validate_prediction("ChatGPT", data)
            # Apply response scoring
            prediction = _apply_response_scoring("ChatGPT", prediction)
            return prediction

    def generate_prediction(self, event: EventMetadata) -> PredictionOutput:
        """Synchronous wrapper — runs the async method in the event loop."""
        return asyncio.run(self.generate_prediction_async(event))


# ── Grok Agent ────────────────────────────────────────────────────────────────

class GrokAgent(BaseAgent):
    """
    Early-signal focused agent using xAI Grok with native Function Calling.
    xAI's API is OpenAI-compatible, so we use AsyncOpenAI with a custom base_url.
    """

    def __init__(self) -> None:
        self._xai_key: Optional[str] = os.getenv("XAI_API_KEY")
        self._use_xai: bool = bool(self._xai_key and self._xai_key.startswith("xai-"))

        if self._use_xai:
            self._async_client = AsyncOpenAI(api_key=self._xai_key, base_url="https://api.x.ai/v1")
            model_name = "grok-2-latest"
            logger.info("Grok configured with xAI API (native Function Calling).", extra={"model": model_name})
        else:
            self._async_client = None
            model_name = "grok-2-latest"
            logger.warning("Grok: XAI_API_KEY not configured.", extra={"hint": "Set XAI_API_KEY in .env"})

        super().__init__("Grok", model_name, "Early-Signal Oriented")

    def has_valid_config(self) -> bool:
        return bool(self._xai_key) and len(self._xai_key) > 20 and self._xai_key.startswith("xai-")

    async def _call_xai(self, messages: list[dict]) -> Any:
        """Call xAI with native Function Calling enforced."""
        return await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice=OPENAI_TOOL_CHOICE,
        )

    async def generate_prediction_async(self, event: EventMetadata) -> PredictionOutput:
        """Async prediction generation with native Function Calling + response scoring."""
        logger.info("Grok starting prediction.", extra={"event_id": event.event_id})

        context = self.research(f"{event.title} rumors leaks social sentiment trends")
        system = f"{SYSTEM_PROMPT_PREFIX}\n{GROK_ARCHETYPE}"
        user_content = PREDICTION_PROMPT.format(
            title=event.title,
            description=event.description,
            rules=event.resolution_rules,
            date=event.resolution_date,
            context=context,
            event_id=event.event_id,
        )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        response = await _call_with_retry("Grok", lambda: self._call_xai(messages))
        try:
            data = _extract_tool_call_args_openai("Grok", response)
            prediction = _validate_prediction("Grok", data)
            # Apply response scoring
            prediction = _apply_response_scoring("Grok", prediction)
            return prediction
        except (LLMResponseParseError, LLMValidationError) as exc:
            logger.warning("Self-correction triggered — re-prompting Grok.", extra={"reason": str(exc)})
            messages.append({"role": "user", "content": _SELF_CORRECT_PROMPT})
            response = await _call_with_retry("Grok", lambda: self._call_xai(messages))
            data = _extract_tool_call_args_openai("Grok", response)
            prediction = _validate_prediction("Grok", data)
            # Apply response scoring
            prediction = _apply_response_scoring("Grok", prediction)
            return prediction

    def generate_prediction(self, event: EventMetadata) -> PredictionOutput:
        """Synchronous wrapper."""
        return asyncio.run(self.generate_prediction_async(event))


# ── Gemini Agent ──────────────────────────────────────────────────────────────

class GeminiAgent(BaseAgent):
    """
    Constraint-focused agent using Google Gemini with native Function Calling.
    Uses Gemini's functionDeclarations API for structured tool use.
    """

    def __init__(self) -> None:
        self._api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self._gemini_model: Optional[Any] = None
        model_name = detect_best_gemini_model(self._api_key) if self._api_key else "gemini-2.0-flash"
        super().__init__("Gemini", model_name, "Constraint-Oriented")
        if self.has_valid_config():
            logger.info("Gemini configured with native Function Calling.", extra={"model": model_name})

    def has_valid_config(self) -> bool:
        key = self._api_key
        return bool(key) and len(key) > 20 and not key.startswith("your_")

    @property
    def _model(self) -> Any:
        if not self._gemini_model:
            genai.configure(api_key=self._api_key)
            self._gemini_model = genai.GenerativeModel(
                self.model_name,
                tools=[{"function_declarations": GEMINI_TOOLS}],
            )
        return self._gemini_model

    async def generate_prediction_async(self, event: EventMetadata) -> PredictionOutput:
        """Async prediction generation with native Gemini Function Calling + response scoring."""
        logger.info("Gemini starting prediction.", extra={"event_id": event.event_id})

        context = self.research(f"{event.title} historical constraints feasibility")
        system = f"{SYSTEM_PROMPT_PREFIX}\n{GEMINI_ARCHETYPE}"
        user_content = PREDICTION_PROMPT.format(
            title=event.title,
            description=event.description,
            rules=event.resolution_rules,
            date=event.resolution_date,
            context=context,
            event_id=event.event_id,
        )

        try:
            loop = asyncio.get_event_loop()
            response = await _call_with_retry(
                "Gemini",
                lambda: loop.run_in_executor(
                    None,
                    lambda: self._model.generate_content(
                        [
                            {"role": "user", "parts": [system]},
                            {"role": "user", "parts": [user_content]},
                        ]
                    ),
                ),
            )
        except LLMCallError:
            raise
        except Exception as exc:
            raise LLMCallError(agent_name="Gemini", reason=str(exc), original=exc) from exc

        try:
            data = _extract_tool_call_args_gemini("Gemini", response)
            prediction = _validate_prediction("Gemini", data)
            # Apply response scoring
            prediction = _apply_response_scoring("Gemini", prediction)
            return prediction
        except (LLMResponseParseError, LLMValidationError) as exc:
            # Self-correction: re-prompt once
            logger.warning("Gemini returned invalid response — attempting self-correction.")
            correction_content = [
                {"role": "user", "parts": [system]},
                {"role": "user", "parts": [user_content]},
                {"role": "user", "parts": [_SELF_CORRECT_PROMPT]},
            ]
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._model.generate_content(correction_content),
                )
                data = _extract_tool_call_args_gemini("Gemini", response)
                prediction = _validate_prediction("Gemini", data)
                # Apply response scoring
                prediction = _apply_response_scoring("Gemini", prediction)
                return prediction
            except Exception as exc2:
                raise LLMValidationError(
                    agent_name="Gemini",
                    parsed_data={},
                    validation_errors=str(exc2),
                ) from exc2

    def generate_prediction(self, event: EventMetadata) -> PredictionOutput:
        """Synchronous wrapper."""
        return asyncio.run(self.generate_prediction_async(event))


# ── Claude Agent ──────────────────────────────────────────────────────────────

class ClaudeAgent(BaseAgent):
    """
    Reasoning-focused agent using Anthropic Claude with native Function Calling.
    Uses Claude's tool_use blocks with input_schema for structured tool invocation.
    """

    def __init__(self) -> None:
        self._api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self._client: Optional[anthropic.Anthropic] = None
        model_name = detect_best_claude_model(self._api_key) if self._api_key else "claude-3-5-sonnet-20241022"
        super().__init__("Claude", model_name, "Reasoning-Oriented")
        if self.has_valid_config():
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info("Claude configured with native Function Calling.", extra={"model": model_name})

    def has_valid_config(self) -> bool:
        key = self._api_key
        return bool(key) and len(key) > 20 and not key.startswith("your_")

    async def _call_claude(self, messages: list[dict]) -> Any:
        """Call Claude with native Function Calling."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                tools=CLAUDE_TOOLS,
                messages=messages,
            ),
        )

    async def generate_prediction_async(self, event: EventMetadata) -> PredictionOutput:
        """Async prediction generation with native Claude Function Calling + response scoring."""
        logger.info("Claude starting prediction.", extra={"event_id": event.event_id})

        context = self.research(f"{event.title} reasoning analysis deep dive")
        system = f"{SYSTEM_PROMPT_PREFIX}\n{CHATGPT_ARCHETYPE}"  # Use ChatGPT archetype for consistency
        user_content = PREDICTION_PROMPT.format(
            title=event.title,
            description=event.description,
            rules=event.resolution_rules,
            date=event.resolution_date,
            context=context,
            event_id=event.event_id,
        )
        messages: list[dict] = [
            {"role": "user", "content": user_content},
        ]

        response = await _call_with_retry(
            "Claude",
            lambda: self._call_claude(messages),
        )
        try:
            data = _extract_tool_call_args_claude("Claude", response)
            prediction = _validate_prediction("Claude", data)
            # Apply response scoring
            prediction = _apply_response_scoring("Claude", prediction)
            return prediction
        except (LLMResponseParseError, LLMValidationError) as exc:
            logger.warning("Self-correction triggered — re-prompting Claude.", extra={"reason": str(exc)})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": _SELF_CORRECT_PROMPT})
            response = await _call_with_retry(
                "Claude",
                lambda: self._call_claude(messages),
            )
            data = _extract_tool_call_args_claude("Claude", response)
            prediction = _validate_prediction("Claude", data)
            # Apply response scoring
            prediction = _apply_response_scoring("Claude", prediction)
            return prediction

    def generate_prediction(self, event: EventMetadata) -> PredictionOutput:
        """Synchronous wrapper."""
        return asyncio.run(self.generate_prediction_async(event))
