"""
Adaptive Knowledge Synthesis tool for research briefs and memory integration.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any

from genesis.memory import MemoryStore
from genesis.models.ollama_backend import OllamaProvider
from genesis.research.synthesis import AdaptiveKnowledgeSynthesizer
from genesis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolPermission, ToolResult


class KnowledgeSynthesizerTool(BaseTool):
    """
    Synthesizes multi-source research insights and citations from persistent memory
    and indexed documents, producing a structured and verified research brief.
    """

    def __init__(
        self,
        db_path: str | Path = "data/genesis.db",
        provider: OllamaProvider | None = None,
        memory_store: MemoryStore | None = None,
        model_id: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.provider = provider
        self.memory_store = memory_store
        self.model_id = model_id

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="synthesize_knowledge",
            description=(
                "Synthesize multi-source research insights and citations from persistent memory "
                "and indexed documents for a specific topic, producing a structured research brief."
            ),
            permission=ToolPermission.MUTATING,
            parameters=[
                ToolParameter(
                    name="topic",
                    type="string",
                    description="Topic or research question to investigate and synthesize.",
                    required=True,
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="Optional focused search query to retrieve relevant passages.",
                    required=False,
                    default=None,
                ),
                ToolParameter(
                    name="max_sources",
                    type="integer",
                    description="Maximum number of memory and document sources to cite (default: 5).",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="save_to_memory",
                    type="boolean",
                    description="Whether to automatically store the synthesized brief in persistent memory (default: True).",
                    required=False,
                    default=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.execute_async(**kwargs))).result()
        return asyncio.run(self.execute_async(**kwargs))

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        topic = kwargs.get("topic")
        if not topic or not topic.strip():
            return ToolResult(success=False, error="Parameter 'topic' is required.")

        query = kwargs.get("query")
        max_sources = int(kwargs.get("max_sources", 5))
        save_to_memory = bool(kwargs.get("save_to_memory", True))

        try:
            synthesizer = AdaptiveKnowledgeSynthesizer(
                provider=self.provider,
                memory_store=self.memory_store,
                db_path=self.db_path,
                model_id=self.model_id or "qwen2.5:7b",
            )
            brief = await synthesizer.synthesize(
                topic=topic.strip(),
                query=query,
                max_sources=max_sources,
                save_to_memory=save_to_memory,
            )

            success = brief.status == "completed"
            return ToolResult(
                success=success,
                output={
                    "id": brief.id,
                    "topic": brief.topic,
                    "status": brief.status,
                    "confidence_score": brief.confidence_score,
                    "citations": brief.citations,
                    "brief_markdown": brief.brief_markdown,
                },
                error=brief.error if not success else None,
                metadata=brief.to_dict(),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Knowledge synthesis failed: {exc}",
            )
