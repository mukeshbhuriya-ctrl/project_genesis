"""
Unit and integration tests for AdaptiveKnowledgeSynthesizer (Phase 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator
import pytest

from genesis.core.database import initialize_database
from genesis.memory import MemoryStore
from genesis.models import GenerationRequest, GenerationResponse, ModelInfo, ModelProvider
from genesis.research.synthesis import AdaptiveKnowledgeSynthesizer, Citation


class MockSynthesizingProvider(ModelProvider):
    """Stub provider generating a markdown brief with citations."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        content = (
            "# Executive Summary\n"
            "Project Genesis uses local quantization to run within 8 GB VRAM [1].\n\n"
            "## Key Synthesized Insights\n"
            "LoRA fine-tuning freezes the base weights and adapts low-rank matrices [1]. "
            "Evaluation benchmarks verify no catastrophic forgetting occurs [2].\n\n"
            "## Implications & Next Actions\n"
            "All capabilities operate strictly local without telemetry.\n\n"
            "## Reference Sources\n"
            "[1] Memory: architecture\n"
            "[2] Memory: evaluation"
        )
        return GenerationResponse(content=content, model=request.model)

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        yield ""

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def is_available(self) -> bool:
        return True

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        return None

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.05] * 768 for _ in texts]


def test_citation_verification_scoring():
    synthesizer = AdaptiveKnowledgeSynthesizer(model_id="stub")
    citations = [
        Citation(index=1, source_id="s1", source_type="doc", title="T1", snippet="C1"),
        Citation(index=2, source_id="s2", source_type="doc", title="T2", snippet="C2"),
    ]

    # Perfect grounding
    text_perfect = "Fact A is backed by [1], and Fact B is proven by [2]."
    score, valid, invalid = synthesizer.verify_citations(text_perfect, citations)
    assert score == 1.0
    assert valid == [1, 2]
    assert invalid == []

    # Hallucinated citation [9]
    text_partial = "Fact A is [1], but Fact C is [9]."
    score, valid, invalid = synthesizer.verify_citations(text_partial, citations)
    assert score == 0.5
    assert valid == [1]
    assert invalid == [9]


@pytest.mark.asyncio
async def test_adaptive_synthesis_flow(tmp_path: Path):
    db_file = tmp_path / "test_synthesis.db"
    initialize_database(db_file)

    # Seed initial memories
    mem_store = MemoryStore(db_path=str(db_file))
    mem_store.create(
        category="architecture",
        content="Project Genesis runs on Windows 11 with 8 GB VRAM RTX 2000 Ada GPU.",
        source="doc1",
    )
    mem_store.create(
        category="evaluation",
        content="Evaluation benchmarks detect regressions using seeded deterministic test cases.",
        source="doc2",
    )

    mock_provider = MockSynthesizingProvider()
    synthesizer = AdaptiveKnowledgeSynthesizer(
        provider=mock_provider,
        memory_store=mem_store,
        db_path=db_file,
        model_id="mock-qwen",
    )

    brief = await synthesizer.synthesize(
        topic="Project Genesis Local Execution",
        query="VRAM",
        max_sources=3,
        save_to_memory=True,
    )

    assert brief.status == "completed"
    assert brief.sources_count > 0
    assert len(brief.citations) > 0
    assert brief.confidence_score > 0.0
    assert "# Executive Summary" in brief.brief_markdown
    assert brief.completed_at is not None

    # Verify saved in SQLite research_synthesis
    history = synthesizer.list_syntheses(limit=5)
    assert len(history) == 1
    assert history[0].id == brief.id
    assert history[0].topic == "Project Genesis Local Execution"

    # Verify consolidated memory was saved under category 'research_brief'
    brief_mems = mem_store.search(category="research_brief")
    assert len(brief_mems) == 1
    assert "Research Brief on" in brief_mems[0].content
