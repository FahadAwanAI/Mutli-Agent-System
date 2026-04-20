import json
import os
from typing import Optional, List, Dict, Any
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProvider:
    """Handles text-to-embedding conversion."""

    def __init__(self, provider: str = "openai"):
        """
        Initialize embedding provider.

        Args:
            provider: "openai" (default) or "local"
        """
        self.provider = provider
        self.api_key = os.getenv("OPENAI_API_KEY")

        if provider == "openai" and not self.api_key:
            logger.warning(
                "OpenAI API key not configured. Embeddings will be unavailable.",
                extra={"provider": provider},
            )

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Convert text to embedding vector.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536-dim for OpenAI) or None if failed
        """
        if not text or len(text.strip()) == 0:
            return None

        if self.provider == "openai":
            return self._embed_openai(text)
        elif self.provider == "local":
            return self._embed_local(text)
        else:
            logger.error(f"Unknown embedding provider: {self.provider}")
            return None

    def _embed_openai(self, text: str) -> Optional[List[float]]:
        """Embed using OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8191],  # OpenAI limit
            )
            embedding = response.data[0].embedding
            logger.debug(
                "Text embedded successfully",
                extra={"provider": "openai", "dim": len(embedding)},
            )
            return embedding
        except Exception as exc:
            logger.error(
                "Failed to embed text with OpenAI",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return None

    def _embed_local(self, text: str) -> Optional[List[float]]:
        """
        Embed using local model (lightweight fallback).
        Uses a simple hash-based approach for demo purposes.
        In production, use sentence-transformers or similar.
        """
        try:
            # For demo: create a simple deterministic embedding
            # In production, use: from sentence_transformers import SentenceTransformer
            import hashlib

            # Create a 384-dim vector from text hash
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()

            # Convert hash to float vector
            embedding = [
                float(byte) / 255.0 for byte in hash_bytes
            ] + [0.0] * (384 - len(hash_bytes))

            logger.debug(
                "Text embedded locally",
                extra={"dim": len(embedding)},
            )
            return embedding[:384]
        except Exception as exc:
            logger.error(
                "Failed to embed text locally",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts."""
        return [self.embed_text(text) for text in texts]


class VectorMemory:
    """In-memory vector store for predictions and context."""

    def __init__(self, embedding_provider: str = "openai"):
        """
        Initialize vector memory.

        Args:
            embedding_provider: "openai" or "local"
        """
        self.embedder = EmbeddingProvider(embedding_provider)
        self.memories: List[Dict[str, Any]] = []
        self.embeddings: List[Optional[List[float]]] = []

    def add_memory(
        self,
        event_id: str,
        prediction: str,
        probability: float,
        rationale: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a prediction to memory.

        Args:
            event_id: Event ID
            prediction: YES or NO
            probability: Confidence (0.0-1.0)
            rationale: Reasoning
            metadata: Additional metadata

        Returns:
            True if added successfully
        """
        try:
            # Create embedding from rationale
            embedding = self.embedder.embed_text(rationale)
            if embedding is None:
                logger.warning(
                    "Failed to create embedding for memory",
                    extra={"event_id": event_id},
                )
                return False

            memory = {
                "event_id": event_id,
                "prediction": prediction,
                "probability": probability,
                "rationale": rationale,
                "metadata": metadata or {},
            }

            self.memories.append(memory)
            self.embeddings.append(embedding)

            logger.info(
                "Memory added",
                extra={"event_id": event_id, "total_memories": len(self.memories)},
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to add memory",
                extra={"event_id": event_id, "error": str(exc)},
                exc_info=True,
            )
            return False

    def search_similar(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar memories with similarity scores
        """
        if not self.memories:
            logger.info("No memories to search")
            return []

        # Embed query
        query_embedding = self.embedder.embed_text(query)
        if query_embedding is None:
            logger.warning("Failed to embed query")
            return []

        # Calculate similarities
        similarities = []
        for i, memory_embedding in enumerate(self.embeddings):
            if memory_embedding is None:
                continue

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, memory_embedding)
            if similarity >= threshold:
                similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for idx, similarity in similarities[:top_k]:
            memory = self.memories[idx].copy()
            memory["similarity_score"] = similarity
            results.append(memory)

        logger.info(
            "Similarity search completed",
            extra={"query_len": len(query), "results": len(results)},
        )
        return results

    def get_context_for_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get memory for a specific event."""
        for memory in self.memories:
            if memory["event_id"] == event_id:
                return memory
        return None

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories."""
        return self.memories.copy()

    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.embeddings.clear()
        logger.info("Memory cleared")

    def save_to_file(self, filepath: str) -> bool:
        """Save memories to JSON file (embeddings not saved)."""
        try:
            data = {
                "memories": self.memories,
                "count": len(self.memories),
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Memories saved", extra={"filepath": filepath})
            return True
        except Exception as exc:
            logger.error(
                "Failed to save memories",
                extra={"filepath": filepath, "error": str(exc)},
                exc_info=True,
            )
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load memories from JSON file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.memories = data.get("memories", [])
            # Re-embed all memories
            self.embeddings = self.embedder.embed_batch(
                [m["rationale"] for m in self.memories]
            )

            logger.info(
                "Memories loaded",
                extra={"filepath": filepath, "count": len(self.memories)},
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to load memories",
                extra={"filepath": filepath, "error": str(exc)},
                exc_info=True,
            )
            return False

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            arr1 = np.array(vec1)
            arr2 = np.array(vec2)

            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0


class ContextWindow:
    """Manages LLM context window efficiently."""

    def __init__(self, max_tokens: int = 4096):
        """
        Initialize context window.

        Args:
            max_tokens: Maximum tokens in context window
        """
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> bool:
        """
        Add message to context window.

        Args:
            role: "system", "user", or "assistant"
            content: Message content

        Returns:
            True if added, False if would exceed token limit
        """
        # Rough estimate: 4 chars = 1 token
        estimated_tokens = len(content) // 4

        if self.current_tokens + estimated_tokens > self.max_tokens:
            logger.warning(
                "Context window full",
                extra={
                    "current": self.current_tokens,
                    "needed": estimated_tokens,
                    "max": self.max_tokens,
                },
            )
            return False

        self.messages.append({"role": role, "content": content})
        self.current_tokens += estimated_tokens

        logger.debug(
            "Message added to context",
            extra={"role": role, "tokens": estimated_tokens, "total": self.current_tokens},
        )
        return True

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in context window."""
        return self.messages.copy()

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "used": self.current_tokens,
            "available": self.max_tokens - self.current_tokens,
            "total": self.max_tokens,
            "percent_used": (self.current_tokens / self.max_tokens) * 100,
        }

    def clear(self) -> None:
        """Clear context window."""
        self.messages.clear()
        self.current_tokens = 0
        logger.info("Context window cleared")

    def prune_oldest(self, num_messages: int = 1) -> None:
        """Remove oldest messages to free up space."""
        if len(self.messages) > num_messages:
            removed = self.messages[:num_messages]
            self.messages = self.messages[num_messages:]

            # Recalculate tokens
            self.current_tokens = sum(len(m["content"]) // 4 for m in self.messages)

            logger.info(
                "Context pruned",
                extra={"removed": len(removed), "remaining": len(self.messages)},
            )
