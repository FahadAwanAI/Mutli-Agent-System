import pytest

from src.research.source_validator import (
    ContextChunker,
    ResponseScorer,
)


# ── ContextChunker Tests ──────────────────────────────────────────────────────

class TestContextChunker:
    def test_chunk_content_short(self):
        """Test chunking of short content."""
        content = "This is a short sentence."
        chunks = ContextChunker.chunk_content(content, max_chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_chunk_content_long(self):
        """Test chunking of long content."""
        content = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = ContextChunker.chunk_content(content, max_chunk_size=50)
        assert len(chunks) > 1
        # Each chunk should be <= max_chunk_size
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_extract_key_sentences_short(self):
        """Test extraction from short content."""
        content = "First sentence. Second sentence."
        result = ContextChunker.extract_key_sentences(content, num_sentences=2)
        assert "First sentence" in result
        assert "Second sentence" in result

    def test_extract_key_sentences_with_numbers(self):
        """Test that sentences with numbers are prioritized."""
        content = (
            "The company was founded in 1998. "
            "It has many employees. "
            "Revenue reached $1 billion in 2024. "
            "The CEO is John Doe."
        )
        result = ContextChunker.extract_key_sentences(content, num_sentences=2)
        # Should include sentences with numbers
        assert "1998" in result or "2024" in result or "$1 billion" in result

    def test_optimize_context_single_source(self):
        """Test context optimization with single source."""
        results = [
            {
                "url": "https://bbc.com/article",
                "content": "The company announced a major breakthrough in AI research today.",
            }
        ]
        context = ContextChunker.optimize_context(results, max_tokens=500)
        assert "bbc.com" in context
        assert "breakthrough" in context

    def test_optimize_context_multiple_sources(self):
        """Test context optimization with multiple sources."""
        results = [
            {
                "url": "https://bbc.com/article1",
                "content": "The company announced a major breakthrough in AI research today.",
            },
            {
                "url": "https://reuters.com/article2",
                "content": "Industry experts praised the announcement as a game-changer.",
            },
            {
                "url": "https://random-blog.com/article3",
                "content": "Allegedly, the company is planning something secret.",
            },
        ]
        context = ContextChunker.optimize_context(results, max_tokens=500)
        # Should include sources - LLM will decide credibility
        assert "bbc.com" in context or "reuters.com" in context or "random-blog.com" in context

    def test_optimize_context_token_budget(self):
        """Test that context respects token budget."""
        results = [
            {
                "url": f"https://bbc.com/article{i}",
                "content": "This is a very long article. " * 100,
            }
            for i in range(10)
        ]
        context = ContextChunker.optimize_context(results, max_tokens=500)
        # Rough estimate: 4 chars per token
        assert len(context) <= 500 * 4 * 1.2  # Allow 20% buffer

    def test_optimize_context_includes_llm_instruction(self):
        """Test that context includes instruction for LLM to validate sources."""
        results = [
            {
                "url": "https://bbc.com/article",
                "content": "The company announced a major breakthrough in AI research today.",
            }
        ]
        context = ContextChunker.optimize_context(results, max_tokens=500)
        # Should include instruction for LLM to evaluate source credibility
        assert "credibility" in context.lower() or "evaluate" in context.lower()


# ── ResponseScorer Tests ──────────────────────────────────────────────────────

class TestResponseScorer:
    def test_detect_hallucination_indicators_none(self):
        """Test detection when no hallucination indicators present."""
        response = "Based on the evidence, the prediction is YES with 75% confidence."
        result = ResponseScorer.detect_hallucination_indicators(response)
        assert result["has_indicators"] is False
        assert result["hallucination_score"] == 0.0

    def test_detect_hallucination_indicators_present(self):
        """Test detection of hallucination indicators."""
        response = "I don't know if this is true, but I assume the answer is YES."
        result = ResponseScorer.detect_hallucination_indicators(response)
        assert result["has_indicators"] is True
        assert result["hallucination_score"] > 0.0

    def test_detect_bias_indicators_none(self):
        """Test detection when no bias indicators present."""
        response = "The evidence suggests YES (60%) but there are counterarguments (40%)."
        result = ResponseScorer.detect_bias_indicators(response)
        assert result["has_bias_indicators"] is False
        assert result["bias_score"] == 0.0

    def test_detect_bias_indicators_extreme_language(self):
        """Test detection of extreme language."""
        response = "This is definitely YES. It's absolutely certain and impossible to be wrong."
        result = ResponseScorer.detect_bias_indicators(response)
        assert result["has_bias_indicators"] is True
        assert result["bias_score"] > 0.0

    def test_detect_bias_indicators_emotional_language(self):
        """Test detection of emotional language."""
        response = "This amazing breakthrough is wonderful and will be terrible for competitors."
        result = ResponseScorer.detect_bias_indicators(response)
        assert result["has_bias_indicators"] is True
        assert result["bias_score"] > 0.0

    def test_detect_bias_indicators_one_sided(self):
        """Test detection of one-sided arguments."""
        response = "The evidence clearly shows YES. All indicators point to YES."
        result = ResponseScorer.detect_bias_indicators(response)
        assert result["has_bias_indicators"] is True
        # Should detect extreme language or lack of counterarguments
        assert len(result["indicators"]) > 0

    def test_score_response_high_confidence(self):
        """Test scoring of high-confidence response."""
        response = "Based on the evidence, the prediction is YES with 85% confidence."
        result = ResponseScorer.score_response(response, probability=0.85)
        assert result["overall_score"] >= 0.7
        assert result["recommendation"] == "HIGH_CONFIDENCE"

    def test_score_response_low_confidence(self):
        """Test scoring of low-confidence response."""
        response = "I don't know, but I guess the answer is NO. This is definitely wrong."
        result = ResponseScorer.score_response(response, probability=0.3)
        assert result["overall_score"] < 0.6
        assert result["recommendation"] in ["LOW_CONFIDENCE", "UNRELIABLE"]

    def test_score_response_medium_confidence(self):
        """Test scoring of medium-confidence response."""
        response = "The evidence suggests YES (60%) but there are counterarguments (40%)."
        result = ResponseScorer.score_response(response, probability=0.6)
        assert 0.4 <= result["overall_score"] <= 0.8
        assert result["recommendation"] in ["MEDIUM_CONFIDENCE", "HIGH_CONFIDENCE"]

    def test_score_response_structure(self):
        """Test that score response has all required fields."""
        response = "The prediction is YES."
        result = ResponseScorer.score_response(response, probability=0.7)
        assert "overall_score" in result
        assert "confidence" in result
        assert "hallucination_risk" in result
        assert "bias_risk" in result
        assert "recommendation" in result
        assert "details" in result
        assert 0.0 <= result["overall_score"] <= 1.0
        assert 0.0 <= result["hallucination_risk"] <= 1.0
        assert 0.0 <= result["bias_risk"] <= 1.0
