"""
Unit tests for ContextManager and InferenceProfiler (Phase 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator
import pytest

from genesis.core.database import initialize_database
from genesis.models import GenerationRequest, GenerationResponse, ModelInfo, ModelProvider
from genesis.models.context import ContextManager, KNOWN_ARCHITECTURES
from genesis.models.profiler import InferenceProfiler


class DummyStreamingProvider(ModelProvider):
    """Stub provider yielding chunks with simulated latency for deterministic profiling."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(content="Dummy response", model=request.model)

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        tokens = ["Quick", "Sort", " runs", " in", " O(n", " log", " n)", " average", " time."]
        for t in tokens:
            yield t

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def is_available(self) -> bool:
        return True

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        return None

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


def test_context_manager_architecture_resolution():
    cm = ContextManager()
    arch_7b = cm.get_architecture("qwen2.5:7b")
    assert arch_7b.layers == 28
    assert arch_7b.kv_heads == 4
    assert arch_7b.bytes_per_token == 57344

    arch_15b = cm.get_architecture("qwen2.5-coder:1.5b-base")
    assert arch_15b.kv_heads == 2
    assert arch_15b.bytes_per_token == 28672

    arch_default = cm.get_architecture("unknown_custom_model")
    assert arch_default == KNOWN_ARCHITECTURES["default"]


def test_calculate_kv_cache_vram():
    cm = ContextManager()
    # At 4096 tokens on 7B: 4096 * 57344 / 1048576 = 224 MB
    mb_4k = cm.calculate_kv_cache_vram(4096, "qwen2.5:7b")
    assert 220.0 < mb_4k < 230.0

    # At 8192 tokens: ~448 MB
    mb_8k = cm.calculate_kv_cache_vram(8192, "qwen2.5:7b")
    assert 440.0 < mb_8k < 460.0

    # At 32768 tokens: ~1792 MB
    mb_32k = cm.calculate_kv_cache_vram(32768, "qwen2.5:7b")
    assert 1780.0 < mb_32k < 1810.0


def test_context_budget_safety():
    cm = ContextManager(total_gpu_vram_mb=8192.0, safety_headroom_mb=1024.0)
    # 4K context on 7B (weights 4700 + kv ~224 + cuda ~512 = ~5436 MB; headroom ~2756 MB)
    budget_4k = cm.get_budget(4096, "qwen2.5:7b")
    assert budget_4k.is_safe is True
    assert budget_4k.headroom_mb > 2000.0
    assert budget_4k.recommended_max_tokens >= 8192

    # High context with small available VRAM -> should flag unsafe
    budget_tight = cm.get_budget(32768, "qwen2.5:7b", available_vram_mb=6000.0)
    assert budget_tight.is_safe is False
    assert budget_tight.headroom_mb < 1024.0


def test_token_estimation_and_truncation():
    cm = ContextManager()
    text = "The quick brown fox jumps over the lazy dog."
    tokens = cm.estimate_tokens(text)
    assert 8 <= tokens <= 15

    long_text = "word " * 1000
    truncated = cm.truncate_text(long_text, max_tokens=50)
    assert cm.estimate_tokens(truncated) <= 60
    assert truncated.endswith("[truncated]")


def test_compact_dialogue_preserves_system_and_last_user():
    cm = ContextManager()
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Turn 1 question: " + "alpha " * 100},
        {"role": "assistant", "content": "Turn 1 answer: " + "beta " * 100},
        {"role": "user", "content": "Turn 2 question: " + "gamma " * 100},
        {"role": "assistant", "content": "Turn 2 answer: " + "delta " * 100},
        {"role": "user", "content": "Turn 3 latest question."},
    ]

    compacted = cm.compact_dialogue(messages, max_tokens=150)
    assert len(compacted) < len(messages)
    # Must preserve system
    assert compacted[0]["role"] == "system"
    assert "helpful coding assistant" in compacted[0]["content"]
    # Must preserve latest turn
    assert compacted[-1]["role"] == "user"
    assert "Turn 3 latest question." in compacted[-1]["content"]


@pytest.mark.asyncio
async def test_inference_profiler_streaming(tmp_path: Path):
    db_file = tmp_path / "test_profiler.db"
    initialize_database(db_file)

    provider = DummyStreamingProvider()
    profiler = InferenceProfiler(provider=provider, db_path=db_file)

    result = await profiler.profile(
        model_id="qwen2.5:7b",
        prompt="Explain QuickSort.",
        max_tokens=64,
    )

    assert result.status == "completed"
    assert result.model_id == "qwen2.5:7b"
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.ttft_ms >= 0.0
    assert result.tokens_per_second >= 0.0
    assert result.kv_cache_vram_mb > 0.0

    # Query back from SQLite
    history = profiler.list_profiles(limit=5)
    assert len(history) == 1
    assert history[0].id == result.id
    assert history[0].model_id == "qwen2.5:7b"
