"""
Hardware-aware context management and KV-cache VRAM budgeting.

Computes exact KV-cache memory footprints across context lengths (4K, 8K, 16K, 32K)
and bounds dialogue contexts to prevent CUDA Out-Of-Memory (OOM) failures on local GPUs
such as the NVIDIA RTX 2000 Ada (8 GB VRAM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from genesis.core.logging import get_logger

log = get_logger("context_manager")


@dataclass(frozen=True)
class ModelArchitecture:
    """Architectural specifications for KV-cache estimation."""
    layers: int
    kv_heads: int
    head_dim: int = 128
    bytes_per_elem: int = 2  # FP16 KV-cache
    weights_mb: float = 4700.0  # Typical Q4_K_M size

    @property
    def bytes_per_token(self) -> int:
        """KV cache bytes per token = 2 (K & V) * layers * kv_heads * head_dim * bytes_per_elem."""
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_elem


# Known local model configurations
KNOWN_ARCHITECTURES: dict[str, ModelArchitecture] = {
    # Qwen 2.5 7B (28 layers, 4 KV heads with GQA, head dim 128)
    "qwen2.5:7b": ModelArchitecture(layers=28, kv_heads=4, head_dim=128, weights_mb=4700.0),
    "qwen2.5-coder:7b": ModelArchitecture(layers=28, kv_heads=4, head_dim=128, weights_mb=4700.0),
    # Qwen 2.5 1.5B (28 layers, 2 KV heads, head dim 128)
    "qwen2.5-coder:1.5b-base": ModelArchitecture(layers=28, kv_heads=2, head_dim=128, weights_mb=986.0),
    "qwen2.5:1.5b": ModelArchitecture(layers=28, kv_heads=2, head_dim=128, weights_mb=986.0),
    # Llama 3 / 3.1 8B (32 layers, 8 KV heads with GQA, head dim 128)
    "llama3:8b": ModelArchitecture(layers=32, kv_heads=8, head_dim=128, weights_mb=4900.0),
    "llama3.1:8b": ModelArchitecture(layers=32, kv_heads=8, head_dim=128, weights_mb=4900.0),
    # Default 7B fallback
    "default": ModelArchitecture(layers=28, kv_heads=4, head_dim=128, weights_mb=4700.0),
}


@dataclass
class ContextBudget:
    """Detailed VRAM and token budget for a given model and context size."""
    model_name: str
    context_length: int
    kv_cache_mb: float
    model_weights_mb: float
    cuda_overhead_mb: float
    total_vram_mb: float
    available_vram_mb: float
    is_safe: bool
    headroom_mb: float
    recommended_max_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "context_length": self.context_length,
            "kv_cache_mb": round(self.kv_cache_mb, 2),
            "model_weights_mb": round(self.model_weights_mb, 2),
            "cuda_overhead_mb": round(self.cuda_overhead_mb, 2),
            "total_vram_mb": round(self.total_vram_mb, 2),
            "available_vram_mb": round(self.available_vram_mb, 2),
            "is_safe": self.is_safe,
            "headroom_mb": round(self.headroom_mb, 2),
            "recommended_max_tokens": self.recommended_max_tokens,
            "metadata": self.metadata,
        }


class ContextManager:
    """
    Manages context length, token counts, and KV-cache VRAM budgeting.
    """

    def __init__(
        self,
        default_model: str = "qwen2.5:7b",
        total_gpu_vram_mb: float = 8192.0,  # RTX 2000 Ada (8 GB)
        safety_headroom_mb: float = 1024.0,  # 1 GB reserve for desktop/OS/CUDA buffers
    ) -> None:
        self.default_model = default_model
        self.total_gpu_vram_mb = total_gpu_vram_mb
        self.safety_headroom_mb = safety_headroom_mb

    def get_architecture(self, model_name: str | None = None) -> ModelArchitecture:
        """Retrieve model architectural parameters for KV cache computation."""
        name = (model_name or self.default_model).lower()
        for key, arch in KNOWN_ARCHITECTURES.items():
            if key in name or name in key:
                return arch
        return KNOWN_ARCHITECTURES["default"]

    def calculate_kv_cache_vram(self, context_tokens: int, model_name: str | None = None) -> float:
        """
        Calculate the exact memory consumed by the FP16 KV-cache in megabytes (MB).
        """
        arch = self.get_architecture(model_name)
        total_bytes = context_tokens * arch.bytes_per_token
        return total_bytes / (1024 * 1024)

    def get_budget(
        self,
        context_tokens: int = 4096,
        model_name: str | None = None,
        available_vram_mb: float | None = None,
    ) -> ContextBudget:
        """
        Compute VRAM budget, headroom, safety status, and recommended max tokens.
        """
        target_model = model_name or self.default_model
        arch = self.get_architecture(target_model)
        available_vram = available_vram_mb or self.total_gpu_vram_mb

        kv_cache_mb = self.calculate_kv_cache_vram(context_tokens, target_model)
        weights_mb = arch.weights_mb
        # CUDA context + scratch space baseline (~500 MB)
        cuda_overhead_mb = 512.0
        total_vram_mb = weights_mb + kv_cache_mb + cuda_overhead_mb
        headroom_mb = available_vram - total_vram_mb
        is_safe = headroom_mb >= self.safety_headroom_mb

        # Calculate recommended max tokens before breaching safety headroom
        usable_vram_for_kv = max(0.0, available_vram - weights_mb - cuda_overhead_mb - self.safety_headroom_mb)
        bytes_per_token = arch.bytes_per_token
        max_safe_tokens = int((usable_vram_for_kv * 1024 * 1024) / bytes_per_token)
        # Clamp to reasonable discrete increments (e.g. 2048 to 32768)
        recommended_max = max(1024, min(32768, (max_safe_tokens // 512) * 512))

        return ContextBudget(
            model_name=target_model,
            context_length=context_tokens,
            kv_cache_mb=kv_cache_mb,
            model_weights_mb=weights_mb,
            cuda_overhead_mb=cuda_overhead_mb,
            total_vram_mb=total_vram_mb,
            available_vram_mb=available_vram,
            is_safe=is_safe,
            headroom_mb=headroom_mb,
            recommended_max_tokens=recommended_max,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Fast token estimator calibrated for Qwen/Llama tokenizers.
        Average ~3.8 chars per token with word boundary awareness.
        """
        if not text:
            return 0
        char_count = len(text)
        word_count = len(text.split())
        # Blended heuristic: 0.6 * (chars / 3.8) + 0.4 * (words * 1.3)
        tokens = int(0.6 * (char_count / 3.8) + 0.4 * (word_count * 1.3))
        return max(1, tokens)

    def estimate_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate the token count for a list of chat message dicts."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            total += self.estimate_tokens(content) + self.estimate_tokens(role) + 4  # ChatML wrapper tokens
        return total

    def truncate_text(self, text: str, max_tokens: int, suffix: str = "\n... [truncated]") -> str:
        """Truncate text to fit within max_tokens cleanly."""
        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Approximate ratio
        ratio = max_tokens / max(1, current_tokens)
        target_chars = int(len(text) * ratio * 0.95)
        truncated = text[:target_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + suffix

    def compact_dialogue(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        preserve_system: bool = True,
    ) -> list[dict[str, str]]:
        """
        Compacts dialogue history to guarantee it fits within max_tokens:
        1. Always preserves system prompt if preserve_system=True.
        2. Always preserves the most recent user turn.
        3. Progressively retains the most recent message pairs from the end.
        """
        if not messages:
            return []

        total_tokens = self.estimate_messages_tokens(messages)
        if total_tokens <= max_tokens:
            return list(messages)

        system_msg: dict[str, str] | None = None
        other_messages: list[dict[str, str]] = []

        for msg in messages:
            if preserve_system and msg.get("role") == "system" and system_msg is None:
                system_msg = msg
            else:
                other_messages.append(msg)

        budget = max_tokens
        if system_msg:
            budget -= self.estimate_tokens(system_msg.get("content", "")) + 4

        # Always keep the last turn
        if not other_messages:
            return [system_msg] if system_msg else []

        retained: list[dict[str, str]] = [other_messages[-1]]
        current_used = self.estimate_tokens(other_messages[-1].get("content", "")) + 4

        # Traverse backwards from the second-to-last message
        for msg in reversed(other_messages[:-1]):
            msg_tokens = self.estimate_tokens(msg.get("content", "")) + 4
            if current_used + msg_tokens <= budget:
                retained.append(msg)
                current_used += msg_tokens
            else:
                break

        # Re-order forward
        retained.reverse()

        result: list[dict[str, str]] = []
        if system_msg:
            result.append(system_msg)
        result.extend(retained)

        log.info(
            "Compacted dialogue: %d messages -> %d messages (tokens ~%d)",
            len(messages),
            len(result),
            self.estimate_messages_tokens(result),
        )
        return result
