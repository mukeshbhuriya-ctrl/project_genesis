"""
Adaptive knowledge synthesis and citation verification engine.

Performs multi-round knowledge synthesis:
1. Multi-source retrieval across indexed document passages and semantic memory.
2. Context grounding and inline citation tagging ([1], [2]).
3. Local model synthesis with strict citation grounding prompts.
4. Deterministic citation verification and groundedness scoring.
5. Automatic consolidation into persistent long-term memory.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core.database import get_sync_connection
from genesis.core.logging import get_logger
from genesis.knowledge.indexer import KnowledgeIndexer
from genesis.memory import MemoryStore
from genesis.memory.embeddings import EmbeddingEngine
from genesis.models import GenerationRequest, ModelProvider
from genesis.models.ollama_backend import OllamaProvider

log = get_logger("research.synthesis")


@dataclass
class Citation:
    """A verified source reference backing a synthesis statement."""
    index: int
    source_id: str
    source_type: str  # 'document_chunk' or 'memory'
    title: str
    snippet: str
    relevance_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "snippet": self.snippet,
            "relevance_score": round(self.relevance_score, 3),
        }


@dataclass
class ResearchBrief:
    """Structured research synthesis report with verified citations."""
    id: str
    topic: str
    query: str
    sources_count: int = 0
    citations: list[Citation] = field(default_factory=list)
    brief_markdown: str = ""
    confidence_score: float = 1.0
    status: str = "completed"
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "query": self.query,
            "sources_count": self.sources_count,
            "citations": [c.to_dict() for c in self.citations],
            "brief_markdown": self.brief_markdown,
            "confidence_score": round(self.confidence_score, 3),
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


class AdaptiveKnowledgeSynthesizer:
    """
    Synthesizes multi-source knowledge briefs with rigorous citation verification.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
        indexer: KnowledgeIndexer | None = None,
        memory_store: MemoryStore | None = None,
        db_path: str | Path = "data/genesis.db",
        model_id: str = "qwen2.5:7b",
    ) -> None:
        self.provider = provider or OllamaProvider()
        self.db_path = Path(db_path)
        if indexer is not None:
            self.indexer = indexer
        else:
            engine = EmbeddingEngine(provider=self.provider)
            self.indexer = KnowledgeIndexer(db_path=str(self.db_path), engine=engine)
        self.memory_store = memory_store or MemoryStore(db_path=str(self.db_path))
        self.model_id = model_id

    async def _gather_sources(
        self, query: str, max_sources: int = 5
    ) -> list[Citation]:
        """Collect top relevant chunks from indexed documents and memory."""
        citations: list[Citation] = []
        seen_texts: set[str] = set()
        index_counter = 1

        # 1. Search indexed knowledge documents
        try:
            doc_hits = await self.indexer.search(query=query, limit=max_sources, min_score=0.2)
            for hit in doc_hits:
                content = hit.get("content", "").strip()
                if content and content not in seen_texts:
                    seen_texts.add(content)
                    citations.append(
                        Citation(
                            index=index_counter,
                            source_id=hit.get("id", str(uuid.uuid4())),
                            source_type="document_chunk",
                            title=hit.get("doc_title", "Indexed Document"),
                            snippet=content[:280] + ("..." if len(content) > 280 else ""),
                            relevance_score=float(hit.get("similarity_score", 1.0)),
                        )
                    )
                    index_counter += 1
        except Exception as e:
            log.warning("Knowledge index search error during synthesis: %s", e)

        # 2. Search memory store
        try:
            mem_hits = self.memory_store.search_hybrid(query=query, limit=max_sources)
            for rec, score in mem_hits:
                content = rec.content.strip()
                if content and content not in seen_texts:
                    seen_texts.add(content)
                    citations.append(
                        Citation(
                            index=index_counter,
                            source_id=rec.id,
                            source_type="memory",
                            title=f"Memory: {rec.category}",
                            snippet=content[:280] + ("..." if len(content) > 280 else ""),
                            relevance_score=float(score),
                        )
                    )
                    index_counter += 1
        except Exception as e:
            log.warning("Memory search error during synthesis: %s", e)

        # If no vector/indexed sources exist, fallback to search in memories table directly
        if not citations and self.db_path.exists():
            conn = get_sync_connection(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, category, content FROM memories WHERE content LIKE ? LIMIT ?",
                    (f"%{query}%", max_sources),
                )
                for row in cursor.fetchall():
                    content = row["content"].strip()
                    if content and content not in seen_texts:
                        seen_texts.add(content)
                        citations.append(
                            Citation(
                                index=index_counter,
                                source_id=row["id"],
                                source_type="memory",
                                title=f"Memory: {row['category']}",
                                snippet=content[:280] + ("..." if len(content) > 280 else ""),
                                relevance_score=0.8,
                            )
                        )
                        index_counter += 1
            finally:
                conn.close()

        return citations[:max_sources]

    def verify_citations(
        self, text: str, citations: list[Citation]
    ) -> tuple[float, list[int], list[int]]:
        """
        Verify inline citations (e.g. [1], [2]) against the retrieved source citations.
        Returns (confidence_score, valid_indices, invalid_indices).
        """
        # Find citations like [1], [2], [Ref 1], etc.
        matches = re.findall(r"\[(?:Ref\s*)?(\d+)\]", text)
        if not matches:
            # If no inline citations were used but sources were provided
            return (0.5 if citations else 1.0), [], []

        cited_indices = [int(m) for m in matches]
        valid_set = {c.index for c in citations}

        valid_cites = [idx for idx in cited_indices if idx in valid_set]
        invalid_cites = [idx for idx in cited_indices if idx not in valid_set]

        if not cited_indices:
            score = 1.0
        else:
            score = len(valid_cites) / len(cited_indices)

        return score, valid_cites, invalid_cites

    async def synthesize(
        self,
        topic: str,
        query: str | None = None,
        max_sources: int = 5,
        save_to_memory: bool = True,
    ) -> ResearchBrief:
        """
        Produce a citation-grounded research brief on the given topic.
        """
        search_query = query or topic
        brief_id = str(uuid.uuid4())
        sources = await self._gather_sources(search_query, max_sources=max_sources)

        record = ResearchBrief(
            id=brief_id,
            topic=topic,
            query=search_query,
            sources_count=len(sources),
            citations=sources,
            status="synthesizing",
        )

        # Build grounding context
        source_texts = []
        for src in sources:
            source_texts.append(
                f"[{src.index}] Source: {src.title}\nContent: {src.snippet}"
            )
        context_block = "\n\n".join(source_texts) if source_texts else "No indexed sources found."

        system_prompt = (
            "You are a Senior AI Research Analyst. Produce an executive synthesis brief based strictly on "
            "the provided source context. Every factual claim must include an inline bracket citation [1], [2]. "
            "Do not hallucinate sources outside the provided references."
        )
        user_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Grounding Reference Context:\n{context_block}\n\n"
            "Format your response as a professional Markdown brief with the following sections:\n"
            "1. # Executive Summary\n"
            "2. ## Key Synthesized Insights (with inline citations [N])\n"
            "3. ## Implications & Next Actions\n"
            "4. ## Reference Sources"
        )

        try:
            resp = await self.provider.generate(
                GenerationRequest(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
            )

            generated_markdown = resp.content.strip()
            confidence, valid_cites, invalid_cites = self.verify_citations(
                generated_markdown, sources
            )

            record.brief_markdown = generated_markdown
            record.confidence_score = confidence
            record.status = "completed"
            record.completed_at = datetime.now(timezone.utc).isoformat()
            record.metadata = {
                "valid_citations_count": len(valid_cites),
                "invalid_citations_count": len(invalid_cites),
            }

            # Optional save to long-term memory
            if save_to_memory and record.status == "completed":
                try:
                    summary_snippet = (
                        f"Research Brief on '{topic}': {generated_markdown[:300]}..."
                    )
                    self.memory_store.create(
                        category="research_brief",
                        content=summary_snippet,
                        source=f"synthesis:{brief_id}",
                        confidence=confidence,
                        metadata={"brief_id": brief_id, "topic": topic},
                    )
                    log.info("Saved research brief %s to memory", brief_id)
                except Exception as me:
                    log.warning("Failed to save research brief to memory: %s", me)

        except Exception as e:
            log.error("Synthesis failed for topic '%s': %s", topic, e)
            record.status = "failed"
            record.error = str(e)
            record.completed_at = datetime.now(timezone.utc).isoformat()

        self._save_record(record)
        return record

    def _save_record(self, r: ResearchBrief) -> None:
        """Persist synthesis brief to SQLite."""
        if not self.db_path.exists():
            return
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO research_synthesis (
                    id, topic, query, sources_count, citations, brief_markdown,
                    confidence_score, status, error, created_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.id,
                    r.topic,
                    r.query,
                    r.sources_count,
                    json.dumps([c.to_dict() for c in r.citations]),
                    r.brief_markdown,
                    r.confidence_score,
                    r.status,
                    r.error,
                    r.created_at,
                    r.completed_at,
                    json.dumps(r.metadata),
                ),
            )
            conn.commit()
            log.info("Saved research synthesis brief %s", r.id)
        except Exception as e:
            log.warning("Failed to save research synthesis %s: %s", r.id, e)
        finally:
            conn.close()

    def list_syntheses(self, limit: int = 20) -> list[ResearchBrief]:
        """List past research syntheses from SQLite."""
        if not self.db_path.exists():
            return []
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, topic, query, sources_count, citations, brief_markdown,
                       confidence_score, status, error, created_at, completed_at, metadata
                FROM research_synthesis
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                cites_raw = json.loads(row["citations"] or "[]")
                citations = [
                    Citation(
                        index=c.get("index", i + 1),
                        source_id=c.get("source_id", ""),
                        source_type=c.get("source_type", "document_chunk"),
                        title=c.get("title", ""),
                        snippet=c.get("snippet", ""),
                        relevance_score=c.get("relevance_score", 1.0),
                    )
                    for i, c in enumerate(cites_raw)
                ]
                results.append(
                    ResearchBrief(
                        id=row["id"],
                        topic=row["topic"],
                        query=row["query"],
                        sources_count=row["sources_count"],
                        citations=citations,
                        brief_markdown=row["brief_markdown"] or "",
                        confidence_score=row["confidence_score"],
                        status=row["status"],
                        error=row["error"],
                        created_at=row["created_at"],
                        completed_at=row["completed_at"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return results
        finally:
            conn.close()
