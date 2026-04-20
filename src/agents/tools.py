from typing import Any

# ── Shared tool definition (used by all three) ────────────────────────────────

TOOL_NAME = "submit_prediction"
TOOL_DESCRIPTION = (
    "Submit a structured YES/NO prediction for a Polymarket event. "
    "Call this function with your analysis. Do NOT respond with plain text."
)

# ── OpenAI / xAI Function Calling Schema ──────────────────────────────────────

OPENAI_PREDICT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The Polymarket event ID being predicted.",
                },
                "prediction": {
                    "type": "string",
                    "enum": ["YES", "NO"],
                    "description": "Your binary prediction for the event outcome.",
                },
                "probability": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Your confidence as a probability (0.0 = certain NO, 1.0 = certain YES).",
                },
                "key_facts": {
                    "type": "array",
                    "description": "3-5 key evidence claims supporting your prediction.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "A specific, verifiable factual claim.",
                            },
                            "source": {
                                "type": "string",
                                "description": "URL or source name for this claim.",
                            },
                        },
                        "required": ["claim", "source"],
                    },
                    "minItems": 1,
                },
                "rationale": {
                    "type": "string",
                    "description": "2-3 sentence explanation of your core reasoning.",
                },
            },
            "required": ["event_id", "prediction", "probability", "key_facts", "rationale"],
            "additionalProperties": False,
        },
    },
}

OPENAI_TOOLS: list[dict[str, Any]] = [OPENAI_PREDICT_TOOL]
OPENAI_TOOL_CHOICE: dict[str, Any] = {"type": "function", "function": {"name": TOOL_NAME}}

# ── Google Gemini Function Calling Schema ─────────────────────────────────────

GEMINI_PREDICT_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The Polymarket event ID being predicted.",
            },
            "prediction": {
                "type": "string",
                "enum": ["YES", "NO"],
                "description": "Your binary prediction for the event outcome.",
            },
            "probability": {
                "type": "number",
                "description": "Your confidence as a probability (0.0 = certain NO, 1.0 = certain YES).",
            },
            "key_facts": {
                "type": "array",
                "description": "3-5 key evidence claims supporting your prediction.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "A specific, verifiable factual claim.",
                        },
                        "source": {
                            "type": "string",
                            "description": "URL or source name for this claim.",
                        },
                    },
                    "required": ["claim", "source"],
                },
            },
            "rationale": {
                "type": "string",
                "description": "2-3 sentence explanation of your core reasoning.",
            },
        },
        "required": ["event_id", "prediction", "probability", "key_facts", "rationale"],
    },
}

GEMINI_TOOLS: list[dict[str, Any]] = [GEMINI_PREDICT_TOOL]

# ── Anthropic Claude Function Calling Schema ──────────────────────────────────

CLAUDE_PREDICT_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The Polymarket event ID being predicted.",
            },
            "prediction": {
                "type": "string",
                "enum": ["YES", "NO"],
                "description": "Your binary prediction for the event outcome.",
            },
            "probability": {
                "type": "number",
                "description": "Your confidence as a probability (0.0 = certain NO, 1.0 = certain YES).",
            },
            "key_facts": {
                "type": "array",
                "description": "3-5 key evidence claims supporting your prediction.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "A specific, verifiable factual claim.",
                        },
                        "source": {
                            "type": "string",
                            "description": "URL or source name for this claim.",
                        },
                    },
                    "required": ["claim", "source"],
                },
            },
            "rationale": {
                "type": "string",
                "description": "2-3 sentence explanation of your core reasoning.",
            },
        },
        "required": ["event_id", "prediction", "probability", "key_facts", "rationale"],
    },
}

CLAUDE_TOOLS: list[dict[str, Any]] = [CLAUDE_PREDICT_TOOL]
