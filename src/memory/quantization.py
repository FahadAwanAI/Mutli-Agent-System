from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.logger import get_logger

logger = get_logger(__name__)


class QuantizationType(str, Enum):
    """Quantization types."""
    FULL_PRECISION = "fp32"  # 32-bit float (baseline)
    HALF_PRECISION = "fp16"  # 16-bit float
    BFLOAT16 = "bfloat16"    # Brain float 16
    INT8 = "int8"            # 8-bit integer
    INT4 = "int4"            # 4-bit integer (TurboQuant, GPTQ)
    NBIT = "nbit"            # N-bit quantization


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quantization_type: QuantizationType
    bits: int
    group_size: Optional[int] = None  # For GPTQ
    desc_act: bool = False  # For GPTQ
    use_cache: bool = True


class QuantizationAnalyzer:
    """Analyzes quantization options and recommendations."""

    # Memory footprint per parameter (in bytes)
    MEMORY_PER_PARAM = {
        QuantizationType.FULL_PRECISION: 4.0,  # 32-bit
        QuantizationType.HALF_PRECISION: 2.0,  # 16-bit
        QuantizationType.BFLOAT16: 2.0,        # 16-bit
        QuantizationType.INT8: 1.0,            # 8-bit
        QuantizationType.INT4: 0.5,            # 4-bit
    }

    # Speed multiplier (relative to fp32)
    SPEED_MULTIPLIER = {
        QuantizationType.FULL_PRECISION: 1.0,
        QuantizationType.HALF_PRECISION: 1.5,
        QuantizationType.BFLOAT16: 1.5,
        QuantizationType.INT8: 2.0,
        QuantizationType.INT4: 2.5,
    }

    # Quality loss (0.0 = no loss, 1.0 = complete loss)
    QUALITY_LOSS = {
        QuantizationType.FULL_PRECISION: 0.0,
        QuantizationType.HALF_PRECISION: 0.01,
        QuantizationType.BFLOAT16: 0.02,
        QuantizationType.INT8: 0.05,
        QuantizationType.INT4: 0.10,
    }

    @staticmethod
    def estimate_memory(
        num_parameters: int,
        quantization_type: QuantizationType,
    ) -> Dict[str, Any]:
        """
        Estimate memory footprint for a model.

        Args:
            num_parameters: Number of model parameters
            quantization_type: Type of quantization

        Returns:
            Memory estimation dict
        """
        bytes_per_param = QuantizationAnalyzer.MEMORY_PER_PARAM.get(
            quantization_type, 4.0
        )
        total_bytes = num_parameters * bytes_per_param
        total_gb = total_bytes / (1024 ** 3)

        return {
            "num_parameters": num_parameters,
            "quantization_type": quantization_type.value,
            "bytes_per_param": bytes_per_param,
            "total_bytes": total_bytes,
            "total_gb": total_gb,
            "total_mb": total_bytes / (1024 ** 2),
        }

    @staticmethod
    def compare_quantizations(
        num_parameters: int,
        quantization_types: Optional[list] = None,
    ) -> list[Dict[str, Any]]:
        """
        Compare different quantization options.

        Args:
            num_parameters: Number of model parameters
            quantization_types: List of quantization types to compare

        Returns:
            List of comparison dicts
        """
        if quantization_types is None:
            quantization_types = list(QuantizationType)

        comparisons = []
        baseline_memory = QuantizationAnalyzer.estimate_memory(
            num_parameters, QuantizationType.FULL_PRECISION
        )

        for q_type in quantization_types:
            memory = QuantizationAnalyzer.estimate_memory(num_parameters, q_type)
            speed_mult = QuantizationAnalyzer.SPEED_MULTIPLIER.get(q_type, 1.0)
            quality_loss = QuantizationAnalyzer.QUALITY_LOSS.get(q_type, 0.0)

            comparison = {
                "quantization_type": q_type.value,
                "memory_gb": memory["total_gb"],
                "memory_reduction": (
                    (baseline_memory["total_gb"] - memory["total_gb"])
                    / baseline_memory["total_gb"]
                    * 100
                ),
                "speed_multiplier": speed_mult,
                "quality_loss_percent": quality_loss * 100,
                "recommendation_score": (
                    (1 - quality_loss) * speed_mult
                ),  # Higher is better
            }
            comparisons.append(comparison)

        # Sort by recommendation score
        comparisons.sort(key=lambda x: x["recommendation_score"], reverse=True)
        return comparisons

    @staticmethod
    def recommend_quantization(
        num_parameters: int,
        target_memory_gb: Optional[float] = None,
        priority: str = "balanced",  # "speed", "memory", "quality", "balanced"
    ) -> Dict[str, Any]:
        """
        Recommend quantization based on constraints.

        Args:
            num_parameters: Number of model parameters
            target_memory_gb: Target memory footprint (optional)
            priority: Optimization priority

        Returns:
            Recommendation dict
        """
        comparisons = QuantizationAnalyzer.compare_quantizations(num_parameters)

        # Filter by target memory if specified
        if target_memory_gb:
            comparisons = [
                c for c in comparisons if c["memory_gb"] <= target_memory_gb
            ]

        if not comparisons:
            logger.warning(
                "No quantization option meets target memory",
                extra={"target_gb": target_memory_gb},
            )
            return {}

        # Select based on priority
        if priority == "speed":
            best = max(comparisons, key=lambda x: x["speed_multiplier"])
        elif priority == "memory":
            best = min(comparisons, key=lambda x: x["memory_gb"])
        elif priority == "quality":
            best = min(comparisons, key=lambda x: x["quality_loss_percent"])
        else:  # balanced
            best = comparisons[0]  # Already sorted by recommendation_score

        return {
            "recommended_quantization": best["quantization_type"],
            "memory_gb": best["memory_gb"],
            "memory_reduction_percent": best["memory_reduction"],
            "speed_multiplier": best["speed_multiplier"],
            "quality_loss_percent": best["quality_loss_percent"],
            "priority": priority,
        }


class ModelSpecs:
    """Specifications for common LLMs."""

    SPECS = {
        "gpt-4o": {
            "num_parameters": 175_000_000_000,  # ~175B (estimated)
            "context_window": 128_000,
            "quantization_support": False,  # Server-side only
        },
        "gpt-4-turbo": {
            "num_parameters": 175_000_000_000,
            "context_window": 128_000,
            "quantization_support": False,
        },
        "gpt-3.5-turbo": {
            "num_parameters": 20_000_000_000,  # ~20B (estimated)
            "context_window": 16_000,
            "quantization_support": False,
        },
        "claude-3-opus": {
            "num_parameters": 200_000_000_000,  # ~200B (estimated)
            "context_window": 200_000,
            "quantization_support": False,
        },
        "claude-3-sonnet": {
            "num_parameters": 70_000_000_000,  # ~70B (estimated)
            "context_window": 200_000,
            "quantization_support": False,
        },
        "gemini-2.0-flash": {
            "num_parameters": 100_000_000_000,  # ~100B (estimated)
            "context_window": 1_000_000,
            "quantization_support": False,
        },
        "llama-2-70b": {
            "num_parameters": 70_000_000_000,
            "context_window": 4_096,
            "quantization_support": True,
        },
        "llama-2-13b": {
            "num_parameters": 13_000_000_000,
            "context_window": 4_096,
            "quantization_support": True,
        },
        "mistral-7b": {
            "num_parameters": 7_000_000_000,
            "context_window": 32_000,
            "quantization_support": True,
        },
    }

    @staticmethod
    def get_specs(model_name: str) -> Optional[Dict[str, Any]]:
        """Get specifications for a model."""
        return ModelSpecs.SPECS.get(model_name)

    @staticmethod
    def get_quantization_recommendation(
        model_name: str,
        target_memory_gb: Optional[float] = None,
        priority: str = "balanced",
    ) -> Dict[str, Any]:
        """Get quantization recommendation for a model."""
        specs = ModelSpecs.get_specs(model_name)
        if not specs:
            logger.warning(f"Unknown model: {model_name}")
            return {}

        if not specs.get("quantization_support"):
            logger.info(
                f"Model {model_name} does not support quantization (server-side only)"
            )
            return {
                "model": model_name,
                "quantization_support": False,
                "note": "Quantization handled server-side by provider",
            }

        return {
            "model": model_name,
            "quantization_support": True,
            "recommendation": QuantizationAnalyzer.recommend_quantization(
                specs["num_parameters"],
                target_memory_gb,
                priority,
            ),
        }


class QuantizationRecommender:
    """Provides quantization recommendations for different scenarios."""

    @staticmethod
    def for_production() -> Dict[str, Any]:
        """Recommendation for production deployment."""
        return {
            "scenario": "production",
            "priority": "balanced",
            "recommendations": {
                "local_models": {
                    "llama-2-70b": QuantizationAnalyzer.recommend_quantization(
                        70_000_000_000, target_memory_gb=16, priority="balanced"
                    ),
                    "mistral-7b": QuantizationAnalyzer.recommend_quantization(
                        7_000_000_000, target_memory_gb=4, priority="balanced"
                    ),
                },
                "api_models": {
                    "gpt-4o": "Use OpenAI API (quantization handled server-side)",
                    "claude-3-opus": "Use Anthropic API (quantization handled server-side)",
                    "gemini-2.0-flash": "Use Google API (quantization handled server-side)",
                },
            },
        }

    @staticmethod
    def for_edge_deployment() -> Dict[str, Any]:
        """Recommendation for edge deployment (limited resources)."""
        return {
            "scenario": "edge_deployment",
            "priority": "memory",
            "recommendations": {
                "quantization_type": "int4",
                "models": {
                    "mistral-7b": QuantizationAnalyzer.recommend_quantization(
                        7_000_000_000, target_memory_gb=2, priority="memory"
                    ),
                    "llama-2-13b": QuantizationAnalyzer.recommend_quantization(
                        13_000_000_000, target_memory_gb=4, priority="memory"
                    ),
                },
            },
        }

    @staticmethod
    def for_high_speed() -> Dict[str, Any]:
        """Recommendation for high-speed inference."""
        return {
            "scenario": "high_speed",
            "priority": "speed",
            "recommendations": {
                "quantization_type": "int8",
                "models": {
                    "mistral-7b": QuantizationAnalyzer.recommend_quantization(
                        7_000_000_000, priority="speed"
                    ),
                    "llama-2-70b": QuantizationAnalyzer.recommend_quantization(
                        70_000_000_000, priority="speed"
                    ),
                },
            },
        }
