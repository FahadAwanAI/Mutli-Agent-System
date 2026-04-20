import re
from typing import Any, Optional

from src.logger import get_logger

logger = get_logger(__name__)

# ── Hallucination Indicators ──────────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    r"(?i)(i (don't|do not) (know|have))",
    r"(?i)(i (can't|cannot) (verify|confirm))",
    r"(?i)(i (assume|guess|think))",
    r"(?i)(according to my training)",
    r"(?i)(as of my knowledge cutoff)",
]


class ContextChunker:
    """Chunks and summarizes research context to reduce token waste."""

    @staticmethod
    def chunk_content(content: str, max_chunk_size: int = 500) -> list[str]:
        """
        Split content into chunks of max_chunk_size characters.

        Args:
            content: Raw content to chunk
            max_chunk_size: Maximum characters per chunk

        Returns:
            List of content chunks
        """
        if len(content) <= max_chunk_size:
            return [content]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def extract_key_sentences(content: str, num_sentences: int = 3) -> str:
        """
        Extract the most important sentences from content.

        Uses simple heuristics:
          - Sentences with numbers/dates are important
          - Sentences with action verbs are important
          - First and last sentences are important

        Args:
            content: Raw content
            num_sentences: Number of key sentences to extract

        Returns:
            Concatenated key sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) <= num_sentences:
            return content

        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0

            # Position bonus
            if i == 0 or i == len(sentences) - 1:
                score += 2

            # Number/date bonus
            if re.search(r'\d+', sentence):
                score += 1

            # Action verb bonus
            if re.search(r'\b(released|announced|launched|confirmed|denied|stated)\b', sentence, re.I):
                score += 1

            # Length bonus (not too short, not too long)
            if 20 <= len(sentence) <= 200:
                score += 0.5

            scored_sentences.append((score, sentence))

        # Sort by score and take top N
        top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:num_sentences]
        # Re-order by original position
        top_sentences.sort(key=lambda x: sentences.index(x[1]))

        return " ".join([s[1] for s in top_sentences])

    @staticmethod
    def optimize_context(results: list[dict[str, Any]], max_tokens: int = 2000) -> str:
        """
        Optimize research results to fit within token budget.

        Strategy:
          1. Extract key sentences from each source
          2. Chunk and prioritize by content quality
          3. Stop when token budget is reached
          
        Note: Source validation is delegated to the LLM. We provide all sources
        with their content, and the LLM decides which sources to trust based on
        content analysis. This allows dynamic validation that adapts to context.

        Args:
            results: List of Tavily search results
            max_tokens: Maximum tokens to use (~4 chars per token)

        Returns:
            Optimized context string with all sources (LLM will filter)
        """
        max_chars = max_tokens * 4  # Rough estimate: 4 chars per token
        optimized_context = ""
        used_chars = 0

        # Build context with all sources - let LLM decide credibility
        for result in results:
            if used_chars >= max_chars:
                break

            url = result.get("url", "")
            content = result.get("content", "")

            # Extract key sentences instead of full content
            key_content = ContextChunker.extract_key_sentences(content, num_sentences=2)

            # Format source entry
            entry = f"Source: {url}\nContent: {key_content}\n\n"

            if used_chars + len(entry) <= max_chars:
                optimized_context += entry
                used_chars += len(entry)
                logger.info(
                    "Added source to context",
                    extra={"url": url, "chars": len(entry)},
                )
            else:
                logger.info(
                    "Skipped source (token budget exceeded)",
                    extra={"url": url, "remaining_chars": max_chars - used_chars},
                )

        if not optimized_context:
            return "No research sources found."

        # Add instruction for LLM to validate sources
        instruction = (
            "\n[IMPORTANT: Evaluate the credibility of each source based on content quality, "
            "specificity, and consistency. Prioritize sources with concrete facts, official statements, "
            "and verifiable information. Discard sources that are vague, speculative, or contradictory.]\n"
        )

        return optimized_context + instruction


class ResponseScorer:
    """Scores LLM responses for bias, hallucination, and confidence."""

    @staticmethod
    def detect_hallucination_indicators(response: str) -> dict[str, Any]:
        """
        Detect indicators of hallucination in the response.

        Returns:
            {
                "has_indicators": bool,
                "indicators": [str],
                "hallucination_score": float (0.0-1.0)
            }
        """
        indicators = []
        score = 0.0

        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, response):
                indicators.append(pattern)
                score += 0.2

        return {
            "has_indicators": len(indicators) > 0,
            "indicators": indicators,
            "hallucination_score": min(1.0, score),
        }

    @staticmethod
    def detect_bias_indicators(response: str) -> dict[str, Any]:
        """
        Detect indicators of bias in the response.

        Returns:
            {
                "has_bias_indicators": bool,
                "indicators": [str],
                "bias_score": float (0.0-1.0)
            }
        """
        indicators = []
        score = 0.0

        # Extreme language
        extreme_words = [
            r"\b(definitely|certainly|absolutely|obviously|clearly)\b",
            r"\b(never|always|impossible|guaranteed)\b",
        ]
        for pattern in extreme_words:
            if re.search(pattern, response, re.I):
                indicators.append(f"Extreme language: {pattern}")
                score += 0.15

        # Emotional language
        emotional_words = [
            r"\b(amazing|terrible|horrible|wonderful|disgusting)\b",
        ]
        for pattern in emotional_words:
            if re.search(pattern, response, re.I):
                indicators.append(f"Emotional language: {pattern}")
                score += 0.1

        # One-sided arguments
        if response.count("however") == 0 and response.count("but") == 0:
            indicators.append("No counterarguments presented")
            score += 0.1

        return {
            "has_bias_indicators": len(indicators) > 0,
            "indicators": indicators,
            "bias_score": min(1.0, score),
        }

    @staticmethod
    def score_response(response: str, probability: float) -> dict[str, Any]:
        """
        Comprehensive scoring of LLM response.

        Returns:
            {
                "overall_score": float (0.0-1.0),
                "confidence": float (0.0-1.0),
                "hallucination_risk": float (0.0-1.0),
                "bias_risk": float (0.0-1.0),
                "recommendation": str,
                "details": {...}
            }
        """
        hallucination = ResponseScorer.detect_hallucination_indicators(response)
        bias = ResponseScorer.detect_bias_indicators(response)

        # Confidence based on probability
        confidence = probability if probability >= 0.5 else 1.0 - probability

        # Overall score
        hallucination_risk = hallucination["hallucination_score"]
        bias_risk = bias["bias_score"]
        overall_score = confidence * (1.0 - hallucination_risk * 0.3 - bias_risk * 0.2)

        # Recommendation
        if overall_score >= 0.8:
            recommendation = "HIGH_CONFIDENCE"
        elif overall_score >= 0.6:
            recommendation = "MEDIUM_CONFIDENCE"
        elif overall_score >= 0.4:
            recommendation = "LOW_CONFIDENCE"
        else:
            recommendation = "UNRELIABLE"

        return {
            "overall_score": max(0.0, min(1.0, overall_score)),
            "confidence": confidence,
            "hallucination_risk": hallucination_risk,
            "bias_risk": bias_risk,
            "recommendation": recommendation,
            "details": {
                "hallucination": hallucination,
                "bias": bias,
            },
        }
