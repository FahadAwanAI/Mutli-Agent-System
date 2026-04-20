import pytest
import json
import tempfile
import os

from src.memory import (
    EmbeddingProvider,
    VectorMemory,
    ContextWindow,
    QuantizationType,
    QuantizationAnalyzer,
    ModelSpecs,
    QuantizationRecommender,
)


# ── EmbeddingProvider Tests ───────────────────────────────────────────────────

class TestEmbeddingProvider:
    def test_init_with_local_provider(self):
        """Test initialization with local provider."""
        provider = EmbeddingProvider(provider="local")
        assert provider.provider == "local"

    def test_embed_text_local(self):
        """Test embedding text with local provider."""
        provider = EmbeddingProvider(provider="local")
        embedding = provider.embed_text("This is a test sentence.")
        assert embedding is not None
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_text_empty(self):
        """Test embedding empty text."""
        provider = EmbeddingProvider(provider="local")
        embedding = provider.embed_text("")
        assert embedding is None

    def test_embed_batch(self):
        """Test batch embedding."""
        provider = EmbeddingProvider(provider="local")
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = provider.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(e is not None for e in embeddings)


# ── VectorMemory Tests ────────────────────────────────────────────────────────

class TestVectorMemory:
    def test_add_memory(self):
        """Test adding memory."""
        memory = VectorMemory(embedding_provider="local")
        result = memory.add_memory(
            event_id="test-001",
            prediction="YES",
            probability=0.75,
            rationale="Based on evidence, the prediction is YES.",
        )
        assert result is True
        assert len(memory.memories) == 1

    def test_add_multiple_memories(self):
        """Test adding multiple memories."""
        memory = VectorMemory(embedding_provider="local")
        for i in range(5):
            result = memory.add_memory(
                event_id=f"test-{i:03d}",
                prediction="YES" if i % 2 == 0 else "NO",
                probability=0.5 + (i * 0.1),
                rationale=f"Rationale for prediction {i}.",
            )
            assert result is True

        assert len(memory.memories) == 5

    def test_search_similar(self):
        """Test similarity search."""
        memory = VectorMemory(embedding_provider="local")

        # Add memories
        memory.add_memory(
            event_id="test-001",
            prediction="YES",
            probability=0.75,
            rationale="The company announced a major breakthrough in AI.",
        )
        memory.add_memory(
            event_id="test-002",
            prediction="NO",
            probability=0.3,
            rationale="The market is declining due to economic factors.",
        )

        # Search for similar
        results = memory.search_similar("AI breakthrough announcement", top_k=1)
        assert len(results) > 0
        assert results[0]["event_id"] == "test-001"

    def test_get_context_for_event(self):
        """Test retrieving context for specific event."""
        memory = VectorMemory(embedding_provider="local")
        memory.add_memory(
            event_id="test-001",
            prediction="YES",
            probability=0.75,
            rationale="Test rationale.",
        )

        context = memory.get_context_for_event("test-001")
        assert context is not None
        assert context["event_id"] == "test-001"
        assert context["prediction"] == "YES"

    def test_get_all_memories(self):
        """Test retrieving all memories."""
        memory = VectorMemory(embedding_provider="local")
        for i in range(3):
            memory.add_memory(
                event_id=f"test-{i}",
                prediction="YES",
                probability=0.5,
                rationale="Test",
            )

        all_memories = memory.get_all_memories()
        assert len(all_memories) == 3

    def test_clear_memories(self):
        """Test clearing memories."""
        memory = VectorMemory(embedding_provider="local")
        memory.add_memory(
            event_id="test-001",
            prediction="YES",
            probability=0.75,
            rationale="Test",
        )
        assert len(memory.memories) == 1

        memory.clear()
        assert len(memory.memories) == 0

    def test_save_and_load_memories(self):
        """Test saving and loading memories."""
        memory1 = VectorMemory(embedding_provider="local")
        memory1.add_memory(
            event_id="test-001",
            prediction="YES",
            probability=0.75,
            rationale="Test rationale 1.",
        )
        memory1.add_memory(
            event_id="test-002",
            prediction="NO",
            probability=0.3,
            rationale="Test rationale 2.",
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            result = memory1.save_to_file(temp_file)
            assert result is True
            assert os.path.exists(temp_file)

            # Load into new memory
            memory2 = VectorMemory(embedding_provider="local")
            result = memory2.load_from_file(temp_file)
            assert result is True
            assert len(memory2.memories) == 2
            assert memory2.memories[0]["event_id"] == "test-001"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


# ── ContextWindow Tests ───────────────────────────────────────────────────────

class TestContextWindow:
    def test_init(self):
        """Test context window initialization."""
        window = ContextWindow(max_tokens=4096)
        assert window.max_tokens == 4096
        assert window.current_tokens == 0

    def test_add_message(self):
        """Test adding message to context."""
        window = ContextWindow(max_tokens=1000)
        result = window.add_message("user", "This is a test message.")
        assert result is True
        assert len(window.messages) == 1

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        window = ContextWindow(max_tokens=1000)
        for i in range(5):
            result = window.add_message("user", f"Message {i}.")
            assert result is True

        assert len(window.messages) == 5

    def test_context_window_full(self):
        """Test context window overflow."""
        window = ContextWindow(max_tokens=100)
        # Add messages until full
        for i in range(20):
            result = window.add_message("user", "x" * 100)
            if not result:
                break

        # Should have failed at some point
        assert len(window.messages) < 20

    def test_get_token_usage(self):
        """Test token usage statistics."""
        window = ContextWindow(max_tokens=1000)
        window.add_message("user", "Test message.")
        usage = window.get_token_usage()

        assert "used" in usage
        assert "available" in usage
        assert "total" in usage
        assert "percent_used" in usage
        assert usage["total"] == 1000

    def test_clear_context(self):
        """Test clearing context window."""
        window = ContextWindow(max_tokens=1000)
        window.add_message("user", "Test message.")
        assert len(window.messages) == 1

        window.clear()
        assert len(window.messages) == 0
        assert window.current_tokens == 0

    def test_prune_oldest(self):
        """Test pruning oldest messages."""
        window = ContextWindow(max_tokens=10000)
        for i in range(5):
            window.add_message("user", f"Message {i}.")

        assert len(window.messages) == 5
        window.prune_oldest(num_messages=2)
        assert len(window.messages) == 3


# ── QuantizationAnalyzer Tests ────────────────────────────────────────────────

class TestQuantizationAnalyzer:
    def test_estimate_memory_fp32(self):
        """Test memory estimation for fp32."""
        result = QuantizationAnalyzer.estimate_memory(
            1_000_000_000, QuantizationType.FULL_PRECISION
        )
        assert result["total_gb"] == pytest.approx(4.0, rel=0.1)

    def test_estimate_memory_int4(self):
        """Test memory estimation for int4."""
        result = QuantizationAnalyzer.estimate_memory(
            1_000_000_000, QuantizationType.INT4
        )
        assert result["total_gb"] == pytest.approx(0.5, rel=0.1)

    def test_compare_quantizations(self):
        """Test comparing quantization options."""
        comparisons = QuantizationAnalyzer.compare_quantizations(
            1_000_000_000,
            [
                QuantizationType.FULL_PRECISION,
                QuantizationType.INT8,
                QuantizationType.INT4,
            ],
        )
        assert len(comparisons) == 3
        # Should be sorted by recommendation score
        assert comparisons[0]["recommendation_score"] >= comparisons[1]["recommendation_score"]

    def test_recommend_quantization_balanced(self):
        """Test quantization recommendation (balanced)."""
        result = QuantizationAnalyzer.recommend_quantization(
            70_000_000_000, priority="balanced"
        )
        assert "recommended_quantization" in result
        assert result["priority"] == "balanced"

    def test_recommend_quantization_memory(self):
        """Test quantization recommendation (memory priority)."""
        result = QuantizationAnalyzer.recommend_quantization(
            70_000_000_000, priority="memory"
        )
        assert "recommended_quantization" in result
        assert result["priority"] == "memory"

    def test_recommend_quantization_speed(self):
        """Test quantization recommendation (speed priority)."""
        result = QuantizationAnalyzer.recommend_quantization(
            70_000_000_000, priority="speed"
        )
        assert "recommended_quantization" in result
        assert result["priority"] == "speed"


# ── ModelSpecs Tests ──────────────────────────────────────────────────────────

class TestModelSpecs:
    def test_get_specs_gpt4(self):
        """Test getting specs for GPT-4."""
        specs = ModelSpecs.get_specs("gpt-4o")
        assert specs is not None
        assert "num_parameters" in specs
        assert "context_window" in specs

    def test_get_specs_unknown_model(self):
        """Test getting specs for unknown model."""
        specs = ModelSpecs.get_specs("unknown-model")
        assert specs is None

    def test_get_quantization_recommendation_api_model(self):
        """Test quantization recommendation for API model."""
        result = ModelSpecs.get_quantization_recommendation("gpt-4o")
        assert result["quantization_support"] is False

    def test_get_quantization_recommendation_local_model(self):
        """Test quantization recommendation for local model."""
        result = ModelSpecs.get_quantization_recommendation("mistral-7b")
        assert result["quantization_support"] is True
        assert "recommendation" in result


# ── QuantizationRecommender Tests ─────────────────────────────────────────────

class TestQuantizationRecommender:
    def test_for_production(self):
        """Test production recommendation."""
        result = QuantizationRecommender.for_production()
        assert result["scenario"] == "production"
        assert "recommendations" in result

    def test_for_edge_deployment(self):
        """Test edge deployment recommendation."""
        result = QuantizationRecommender.for_edge_deployment()
        assert result["scenario"] == "edge_deployment"
        assert result["priority"] == "memory"

    def test_for_high_speed(self):
        """Test high-speed recommendation."""
        result = QuantizationRecommender.for_high_speed()
        assert result["scenario"] == "high_speed"
        assert result["priority"] == "speed"
