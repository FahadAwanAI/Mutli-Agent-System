import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from pydantic import ValidationError

from src.agents.specialized_agents import (
    ChatGPTAgent,
    GeminiAgent,
    GrokAgent,
    _extract_tool_call_args_openai,
    _extract_tool_call_args_claude,
    _extract_tool_call_args_gemini,
    _validate_prediction,
    clean_json,
    _call_with_retry,
)
from src.agents.tools import OPENAI_TOOLS, OPENAI_TOOL_CHOICE
from src.exceptions import LLMCallError, LLMResponseParseError, LLMValidationError
from src.models import EventMetadata, PredictionOutcome, PredictionOutput


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_event() -> EventMetadata:
    return EventMetadata(
        event_id="test-001",
        title="Will GPT-5 be released before Q3 2025?",
        description="Prediction market for GPT-5 release timeline.",
        resolution_rules="Resolves YES if OpenAI officially releases GPT-5 before July 1 2025.",
        resolution_date="2025-07-01T00:00:00Z",
    )


@pytest.fixture
def valid_prediction_data() -> dict:
    return {
        "event_id": "test-001",
        "prediction": "YES",
        "probability": 0.72,
        "key_facts": [
            {"claim": "OpenAI has been training GPT-5 since late 2024.", "source": "https://openai.com"},
            {"claim": "Multiple leaks suggest a mid-2025 release window.", "source": "https://example.com"},
        ],
        "rationale": "Strong evidence points to a Q2 2025 release based on training timelines.",
    }


@pytest.fixture
def valid_prediction_json(valid_prediction_data) -> str:
    return json.dumps(valid_prediction_data)


def _make_tool_call_response(args_dict: dict) -> MagicMock:
    """Build a mock OpenAI response that contains a function tool call."""
    tool_call = MagicMock()
    tool_call.function.name = "submit_prediction"
    tool_call.function.arguments = json.dumps(args_dict)

    message = MagicMock()
    message.tool_calls = [tool_call]
    message.content = None

    return MagicMock(choices=[MagicMock(message=message)])


def _make_no_tool_call_response(content: str) -> MagicMock:
    """Build a mock OpenAI response with NO tool call (plain text fallback)."""
    message = MagicMock()
    message.tool_calls = None
    message.content = content
    return MagicMock(choices=[MagicMock(message=message)])


def _make_claude_tool_call_response(args_dict: dict) -> MagicMock:
    """Build a mock Claude response that contains a tool_use block."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "submit_prediction"
    tool_use_block.input = args_dict  # Claude's input is already a dict

    response = MagicMock()
    response.content = [tool_use_block]
    return response


def _make_gemini_tool_call_response(args_dict: dict) -> MagicMock:
    """Build a mock Gemini response that contains a functionCall."""
    func_call = MagicMock()
    func_call.name = "submit_prediction"
    func_call.args = args_dict  # Gemini's args are dict-like

    part = MagicMock()
    part.function_call = func_call

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    response.text = ""
    return response


# ── clean_json ────────────────────────────────────────────────────────────────

class TestCleanJson:
    def test_strips_json_code_fence(self):
        assert clean_json("```json\n{\"k\": 1}\n```") == '{"k": 1}'

    def test_strips_plain_code_fence(self):
        assert clean_json("```\n{\"k\": 1}\n```") == '{"k": 1}'

    def test_passthrough_plain_json(self):
        raw = '{"k": 1}'
        assert clean_json(raw) == raw

    def test_strips_whitespace(self):
        assert clean_json('  {"k": 1}  ') == '{"k": 1}'


# ── _extract_tool_call_args ───────────────────────────────────────────────────

class TestExtractToolCallArgs:
    def test_extracts_from_tool_call(self, valid_prediction_data):
        response = _make_tool_call_response(valid_prediction_data)
        result = _extract_tool_call_args_openai("TestAgent", response)
        assert result["prediction"] == "YES"

    def test_falls_back_to_json_content_when_no_tool_call(self, valid_prediction_json):
        response = _make_no_tool_call_response(valid_prediction_json)
        result = _extract_tool_call_args_openai("TestAgent", response)
        assert result["probability"] == 0.72

    def test_raises_parse_error_when_no_tool_call_and_bad_json(self):
        response = _make_no_tool_call_response("Sorry, I cannot help.")
        with pytest.raises(LLMResponseParseError) as exc_info:
            _extract_tool_call_args_openai("TestAgent", response)
        assert exc_info.value.details["agent"] == "TestAgent"


# ── _validate_prediction ──────────────────────────────────────────────────────

class TestValidatePrediction:
    def test_valid_data_returns_prediction(self, valid_prediction_data):
        result = _validate_prediction("TestAgent", valid_prediction_data)
        assert isinstance(result, PredictionOutput)
        assert result.prediction == PredictionOutcome.YES

    def test_missing_field_raises_validation_error(self):
        bad = {"event_id": "x", "probability": 0.5, "key_facts": [], "rationale": "r"}
        with pytest.raises(LLMValidationError) as exc_info:
            _validate_prediction("TestAgent", bad)
        assert exc_info.value.details["agent"] == "TestAgent"

    def test_probability_out_of_range_raises_validation_error(self, valid_prediction_data):
        valid_prediction_data["probability"] = 1.5
        with pytest.raises(LLMValidationError):
            _validate_prediction("TestAgent", valid_prediction_data)

    def test_invalid_enum_raises_validation_error(self, valid_prediction_data):
        valid_prediction_data["prediction"] = "MAYBE"
        with pytest.raises(LLMValidationError):
            _validate_prediction("TestAgent", valid_prediction_data)


# ── _call_with_retry ──────────────────────────────────────────────────────────

class TestCallWithRetry:
    def test_succeeds_on_first_attempt(self):
        async def run():
            call_count = 0

            async def coro():
                nonlocal call_count
                call_count += 1
                return "success"

            result = await _call_with_retry("Agent", coro, max_retries=3)
            assert result == "success"
            assert call_count == 1

        asyncio.run(run())

    def test_retries_on_rate_limit_then_succeeds(self):
        async def run():
            call_count = 0

            async def coro():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("429 rate limit exceeded")
                return "success"

            with patch("src.agents.specialized_agents.asyncio.sleep", new_callable=AsyncMock):
                result = await _call_with_retry("Agent", coro, max_retries=3)

            assert result == "success"
            assert call_count == 3

        asyncio.run(run())

    def test_raises_llm_call_error_after_all_retries_exhausted(self):
        async def run():
            async def coro():
                raise Exception("429 rate limit exceeded")

            with patch("src.agents.specialized_agents.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMCallError) as exc_info:
                    await _call_with_retry("Agent", coro, max_retries=2)

            assert "Agent" in exc_info.value.message

        asyncio.run(run())


# ── ChatGPTAgent ──────────────────────────────────────────────────────────────

class TestChatGPTAgent:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_has_valid_config_with_openai_key(self, mock_cls):
        mock_cls.return_value = MagicMock()
        agent = ChatGPTAgent()
        assert agent.has_valid_config() is True

    @patch.dict("os.environ", {}, clear=True)
    def test_has_valid_config_without_keys(self):
        agent = ChatGPTAgent()
        assert agent.has_valid_config() is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_uses_function_calling_tool_schema(self, mock_cls, sample_event, valid_prediction_data):
        """Verify the agent passes tools= and tool_choice= to the API call."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_tool_call_response(valid_prediction_data)
        )

        agent = ChatGPTAgent()
        agent._use_openai = True
        agent._async_client = mock_client

        with patch.object(agent, "research", return_value="mocked context"):
            result = agent.generate_prediction(sample_event)

        # Verify Function Calling params were passed
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == OPENAI_TOOLS
        assert call_kwargs["tool_choice"] == OPENAI_TOOL_CHOICE
        assert isinstance(result, PredictionOutput)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_happy_path_returns_prediction(self, mock_cls, sample_event, valid_prediction_data):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_tool_call_response(valid_prediction_data)
        )

        agent = ChatGPTAgent()
        agent._use_openai = True
        agent._async_client = mock_client

        with patch.object(agent, "research", return_value=""):
            result = agent.generate_prediction(sample_event)

        assert result.event_id == "test-001"
        assert result.prediction == PredictionOutcome.YES

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_api_failure_raises_llm_call_error(self, mock_cls, sample_event):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Connection timeout"))

        agent = ChatGPTAgent()
        agent._use_openai = True
        agent._async_client = mock_client

        with patch.object(agent, "research", return_value=""):
            with patch("src.agents.specialized_agents.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMCallError) as exc_info:
                    agent.generate_prediction(sample_event)

        assert "ChatGPT" in exc_info.value.message

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_self_correction_on_bad_first_response(self, mock_cls, sample_event, valid_prediction_data):
        """
        Edge case: first call returns no tool call (bad response),
        second call (self-correction) returns valid tool call.
        """
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        bad_response = _make_no_tool_call_response("I cannot answer that.")
        good_response = _make_tool_call_response(valid_prediction_data)

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[bad_response, good_response]
        )

        agent = ChatGPTAgent()
        agent._use_openai = True
        agent._async_client = mock_client

        with patch.object(agent, "research", return_value=""):
            result = agent.generate_prediction(sample_event)

        # Should have called the API twice (original + self-correction)
        assert mock_client.chat.completions.create.call_count == 2
        assert isinstance(result, PredictionOutput)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_edge_case_empty_tool_call_args_raises_validation_error(self, mock_cls, sample_event):
        """Edge case: tool call returns empty JSON object {}."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_tool_call_response({})
        )

        agent = ChatGPTAgent()
        agent._use_openai = True
        agent._async_client = mock_client

        with patch.object(agent, "research", return_value=""):
            # Empty dict fails Pydantic validation → self-correction triggered
            # Second call also returns empty → LLMValidationError raised
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    _make_tool_call_response({}),
                    _make_tool_call_response({}),
                ]
            )
            with pytest.raises(LLMValidationError):
                agent.generate_prediction(sample_event)


# ── GrokAgent ─────────────────────────────────────────────────────────────────

class TestGrokAgent:
    @patch.dict("os.environ", {"XAI_API_KEY": "xai-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_has_valid_config_with_xai_key(self, mock_cls):
        mock_cls.return_value = MagicMock()
        agent = GrokAgent()
        assert agent.has_valid_config() is True

    @patch.dict("os.environ", {}, clear=True)
    def test_has_valid_config_without_keys(self):
        agent = GrokAgent()
        assert agent.has_valid_config() is False

    @patch.dict("os.environ", {"XAI_API_KEY": "xai-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.AsyncOpenAI")
    def test_happy_path_returns_prediction(self, mock_cls, sample_event, valid_prediction_data):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_tool_call_response(valid_prediction_data)
        )

        agent = GrokAgent()
        agent._use_xai = True
        agent._async_client = mock_client

        with patch.object(agent, "research", return_value="mocked context"):
            result = agent.generate_prediction(sample_event)

        assert isinstance(result, PredictionOutput)
        assert result.probability == 0.72


# ── GeminiAgent ───────────────────────────────────────────────────────────────

class TestGeminiAgent:
    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key-for-testing-1234567890"})
    def test_has_valid_config_with_gemini_key(self):
        agent = GeminiAgent()
        assert agent.has_valid_config() is True

    @patch.dict("os.environ", {}, clear=True)
    def test_has_valid_config_without_key(self):
        agent = GeminiAgent()
        assert agent.has_valid_config() is False

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key-for-testing-1234567890"})
    def test_happy_path_returns_prediction(self, sample_event, valid_prediction_json):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=valid_prediction_json)

        agent = GeminiAgent()
        agent._gemini_model = mock_model

        with patch.object(agent, "research", return_value="mocked context"):
            result = agent.generate_prediction(sample_event)

        assert isinstance(result, PredictionOutput)
        assert result.probability == 0.72

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key-for-testing-1234567890"})
    def test_self_correction_on_bad_json(self, sample_event, valid_prediction_json):
        """
        Edge case: Gemini returns garbage first, then valid JSON on self-correction.
        """
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            MagicMock(text="Sorry, I cannot help with that."),  # bad first response
            MagicMock(text=valid_prediction_json),              # good self-correction
        ]

        agent = GeminiAgent()
        agent._gemini_model = mock_model

        with patch.object(agent, "research", return_value=""):
            result = agent.generate_prediction(sample_event)

        assert mock_model.generate_content.call_count == 2
        assert isinstance(result, PredictionOutput)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key-for-testing-1234567890"})
    def test_edge_case_empty_response_raises_parse_error(self, sample_event):
        """Edge case: Gemini returns empty string both times → LLMResponseParseError."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="")

        agent = GeminiAgent()
        agent._gemini_model = mock_model

        with patch.object(agent, "research", return_value=""):
            with pytest.raises((LLMResponseParseError, LLMValidationError, Exception)):
                agent.generate_prediction(sample_event)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key-for-testing-1234567890"})
    def test_validation_error_on_invalid_enum(self, sample_event):
        """Edge case: Gemini returns valid JSON but prediction='MAYBE' (invalid enum)."""
        bad_json = json.dumps({
            "event_id": "test-001",
            "prediction": "MAYBE",
            "probability": 0.5,
            "key_facts": [],
            "rationale": "Some rationale",
        })
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=bad_json)

        agent = GeminiAgent()
        agent._gemini_model = mock_model

        with patch.object(agent, "research", return_value=""):
            with pytest.raises(LLMValidationError):
                agent.generate_prediction(sample_event)



# ── ClaudeAgent ───────────────────────────────────────────────────────────────

class TestClaudeAgent:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.anthropic.Anthropic")
    def test_has_valid_config_with_anthropic_key(self, mock_cls):
        mock_cls.return_value = MagicMock()
        agent = ChatGPTAgent()  # Import ClaudeAgent when available
        # For now, we'll test the config validation logic
        from src.agents.specialized_agents import ClaudeAgent
        agent = ClaudeAgent()
        assert agent.has_valid_config() is True

    @patch.dict("os.environ", {}, clear=True)
    def test_has_valid_config_without_key(self):
        from src.agents.specialized_agents import ClaudeAgent
        agent = ClaudeAgent()
        assert agent.has_valid_config() is False

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.anthropic.Anthropic")
    def test_happy_path_returns_prediction(self, mock_cls, sample_event, valid_prediction_data):
        """Test Claude with native function calling."""
        from src.agents.specialized_agents import ClaudeAgent
        
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = MagicMock(
            return_value=_make_claude_tool_call_response(valid_prediction_data)
        )

        agent = ClaudeAgent()
        agent._client = mock_client

        with patch.object(agent, "research", return_value="mocked context"):
            result = agent.generate_prediction(sample_event)

        assert isinstance(result, PredictionOutput)
        assert result.probability == 0.72

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-fake-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.anthropic.Anthropic")
    def test_self_correction_on_bad_first_response(self, mock_cls, sample_event, valid_prediction_data):
        """Test Claude self-correction when first response is invalid."""
        from src.agents.specialized_agents import ClaudeAgent
        
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        bad_response = MagicMock()
        bad_response.content = [MagicMock(type="text", text="I cannot help with that.")]

        good_response = _make_claude_tool_call_response(valid_prediction_data)

        mock_client.messages.create = MagicMock(
            side_effect=[bad_response, good_response]
        )

        agent = ClaudeAgent()
        agent._client = mock_client

        with patch.object(agent, "research", return_value=""):
            result = agent.generate_prediction(sample_event)

        assert mock_client.messages.create.call_count == 2
        assert isinstance(result, PredictionOutput)


# ── Gemini Native Function Calling Tests ──────────────────────────────────────

class TestGeminiAgentNativeFC:
    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.genai.GenerativeModel")
    def test_happy_path_with_native_function_calling(self, mock_model_cls, sample_event, valid_prediction_data):
        """Test Gemini with native functionDeclarations."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.generate_content.return_value = _make_gemini_tool_call_response(valid_prediction_data)

        agent = GeminiAgent()
        agent._gemini_model = mock_model

        with patch.object(agent, "research", return_value="mocked context"):
            result = agent.generate_prediction(sample_event)

        assert isinstance(result, PredictionOutput)
        assert result.probability == 0.72

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key-for-testing-1234567890"})
    @patch("src.agents.specialized_agents.genai.GenerativeModel")
    def test_self_correction_on_bad_first_response(self, mock_model_cls, sample_event, valid_prediction_data):
        """Test Gemini self-correction with native function calling."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        bad_response = MagicMock()
        bad_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="I cannot help.")]))]
        bad_response.text = "I cannot help."

        good_response = _make_gemini_tool_call_response(valid_prediction_data)

        mock_model.generate_content.side_effect = [bad_response, good_response]

        agent = GeminiAgent()
        agent._gemini_model = mock_model

        with patch.object(agent, "research", return_value=""):
            result = agent.generate_prediction(sample_event)

        assert mock_model.generate_content.call_count == 2
        assert isinstance(result, PredictionOutput)


# ── Multi-LLM Tool Extraction Tests ───────────────────────────────────────────

class TestExtractToolCallArgsMultiLLM:
    def test_extract_from_claude_tool_use(self, valid_prediction_data):
        """Test Claude tool_use extraction."""
        from src.agents.specialized_agents import _extract_tool_call_args_claude
        response = _make_claude_tool_call_response(valid_prediction_data)
        result = _extract_tool_call_args_claude("Claude", response)
        assert result["prediction"] == "YES"
        assert result["probability"] == 0.72

    def test_extract_from_gemini_function_call(self, valid_prediction_data):
        """Test Gemini functionCall extraction."""
        from src.agents.specialized_agents import _extract_tool_call_args_gemini
        response = _make_gemini_tool_call_response(valid_prediction_data)
        result = _extract_tool_call_args_gemini("Gemini", response)
        assert result["prediction"] == "YES"
        assert result["probability"] == 0.72

    def test_claude_fallback_to_json_when_no_tool_use(self):
        """Test Claude fallback to JSON parsing when no tool_use block."""
        from src.agents.specialized_agents import _extract_tool_call_args_claude
        
        response = MagicMock()
        response.content = [MagicMock(type="text", text='{"prediction": "YES", "probability": 0.5}')]
        
        result = _extract_tool_call_args_claude("Claude", response)
        assert result["prediction"] == "YES"

    def test_gemini_fallback_to_json_when_no_function_call(self):
        """Test Gemini fallback to JSON parsing when no functionCall."""
        from src.agents.specialized_agents import _extract_tool_call_args_gemini
        
        response = MagicMock()
        response.candidates = [MagicMock(content=MagicMock(parts=[]))]
        response.text = '{"prediction": "NO", "probability": 0.3}'
        
        result = _extract_tool_call_args_gemini("Gemini", response)
        assert result["prediction"] == "NO"
