"""
Dynamic task router and multi-model dispatch.

Directs tasks dynamically across locally installed models based on task intent
(coding, reasoning, fast_validation, embedding, general) and device VRAM headroom.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.device import DeviceOrchestrator
from genesis.core.logging import get_logger
from genesis.models.ollama_backend import OllamaProvider

log = get_logger("model_router")

TASK_CATEGORIES = ("coding", "reasoning", "fast_validation", "embedding", "general")


@dataclass
class ModelRoute:
    """Routing specification for a task category."""
    id: str
    task_category: str
    preferred_model: str
    fallback_model: str
    min_vram_mb: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_category": self.task_category,
            "preferred_model": self.preferred_model,
            "fallback_model": self.fallback_model,
            "min_vram_mb": self.min_vram_mb,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class RouteDecision:
    """The result of routing a task to an optimal model."""
    task_category: str
    selected_model: str
    is_fallback: bool
    reason: str
    confidence: float = 1.0
    estimated_vram_mb: float = 0.0
    available_headroom_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_category": self.task_category,
            "selected_model": self.selected_model,
            "is_fallback": self.is_fallback,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "estimated_vram_mb": round(self.estimated_vram_mb, 1),
            "available_headroom_mb": round(self.available_headroom_mb, 1),
            "metadata": self.metadata,
        }


# Pattern matchers for task categorization
_CODING_PATTERNS = [
    re.compile(r"\b(def|class|import|return|def\s+\w+|async\s+def)\b"),
    re.compile(r"\b(python|javascript|typescript|bash|sql|rust|c\+\+|html|css)\b", re.IGNORECASE),
    re.compile(r"\b(code|coding|script|function|algorithm|refactor|debug|bug|traceback|syntax|exception)\b", re.IGNORECASE),
    re.compile(r"\b(pytest|unittest|test\s+harness|assert|mock)\b", re.IGNORECASE),
    re.compile(r"```[a-z]*\n.*?\n```", re.DOTALL),
]

_FAST_VALIDATION_PATTERNS = [
    re.compile(r"\b(validate|valid|check\s+if|verify\s+syntax|schema\s+check)\b", re.IGNORECASE),
    re.compile(r"\b(is\s+valid\s+json|parse\s+json|lint|syntax\s+ok|boolean\s+check)\b", re.IGNORECASE),
    re.compile(r"\b(fast\s+check|shallow\s+eval|quick\s+verification)\b", re.IGNORECASE),
]

_REASONING_PATTERNS = [
    re.compile(r"\b(why|explain\s+step\s+by\s+step|derive|deduce|infer|rationale)\b", re.IGNORECASE),
    re.compile(r"\b(hypothesis|trade-offs|architectural\s+decision|root\s+cause\s+analysis)\b", re.IGNORECASE),
    re.compile(r"\b(multi-step\s+plan|decompose|mathematical\s+proof|evaluate\s+implications)\b", re.IGNORECASE),
    re.compile(r"\b(compare\s+and\s+contrast|deliberate|formal\s+logic)\b", re.IGNORECASE),
]

_EMBEDDING_PATTERNS = [
    re.compile(r"\b(embed|embedding|vectorize|cosine\s+similarity|dense\s+vector)\b", re.IGNORECASE),
    re.compile(r"\b(semantic\s+index|passage\s+encoding)\b", re.IGNORECASE),
]


class ModelRouter:
    """
    Intelligent router selecting optimal local models for task workloads.

    Takes task prompt or explicit category, evaluates VRAM and device availability,
    and returns a RouteDecision.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        provider: OllamaProvider | None = None,
        topology_mgr: DeviceOrchestrator | None = None,
    ):
        settings = get_settings()
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_db_path_from_url(settings.database_url)
        self.provider = provider or OllamaProvider()
        self.topology_mgr = topology_mgr or DeviceOrchestrator(db_path=self.db_path)
        self.default_model = getattr(settings, "default_model", None) or "qwen2.5:7b"

    def classify_task(self, prompt: str) -> tuple[str, float]:
        """
        Classifies task prompt into a category and confidence score.

        Returns (category, confidence).
        """
        prompt_clean = prompt.strip()
        if not prompt_clean:
            return "general", 0.5

        # Check embedding first if explicitly requested
        for pat in _EMBEDDING_PATTERNS:
            if pat.search(prompt_clean):
                return "embedding", 0.95

        # Count pattern hits
        coding_score = sum(1.5 if pat.search(prompt_clean) else 0.0 for pat in _CODING_PATTERNS)
        fast_val_score = sum(1.2 if pat.search(prompt_clean) else 0.0 for pat in _FAST_VALIDATION_PATTERNS)
        reasoning_score = sum(1.2 if pat.search(prompt_clean) else 0.0 for pat in _REASONING_PATTERNS)

        scores = {
            "coding": coding_score,
            "fast_validation": fast_val_score,
            "reasoning": reasoning_score,
        }

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score <= 0.0:
            return "general", 0.70

        # Normalize confidence between 0.65 and 0.98
        confidence = min(0.98, 0.65 + (best_score * 0.1))
        return best_category, confidence

    def get_route(self, task_category: str) -> Optional[ModelRoute]:
        """Retrieve the configured route for a given task category from SQLite."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT id, task_category, preferred_model, fallback_model, min_vram_mb, created_at, metadata "
                "FROM model_routes WHERE task_category = ?",
                (task_category,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            meta = {}
            if row["metadata"]:
                try:
                    meta = json.loads(row["metadata"])
                except Exception:
                    meta = {}
            return ModelRoute(
                id=row["id"],
                task_category=row["task_category"],
                preferred_model=row["preferred_model"],
                fallback_model=row["fallback_model"],
                min_vram_mb=float(row["min_vram_mb"] or 0.0),
                created_at=row["created_at"],
                metadata=meta,
            )
        finally:
            conn.close()

    def list_routes(self) -> list[ModelRoute]:
        """List all configured model routes."""
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT id, task_category, preferred_model, fallback_model, min_vram_mb, created_at, metadata "
                "FROM model_routes ORDER BY task_category ASC"
            )
            routes = []
            for row in cursor.fetchall():
                meta = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except Exception:
                        meta = {}
                routes.append(
                    ModelRoute(
                        id=row["id"],
                        task_category=row["task_category"],
                        preferred_model=row["preferred_model"],
                        fallback_model=row["fallback_model"],
                        min_vram_mb=float(row["min_vram_mb"] or 0.0),
                        created_at=row["created_at"],
                        metadata=meta,
                    )
                )
            return routes
        finally:
            conn.close()

    def set_route(
        self,
        task_category: str,
        preferred_model: str,
        fallback_model: str,
        min_vram_mb: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ModelRoute:
        """Create or update a model route."""
        if task_category not in TASK_CATEGORIES:
            raise ValueError(f"Invalid task_category '{task_category}'. Must be one of {TASK_CATEGORIES}")

        route_id = f"route-{task_category}"
        meta = metadata or {}
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO model_routes (id, task_category, preferred_model, fallback_model, min_vram_mb, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_category) DO UPDATE SET
                    preferred_model=excluded.preferred_model,
                    fallback_model=excluded.fallback_model,
                    min_vram_mb=excluded.min_vram_mb,
                    metadata=excluded.metadata
                """,
                (route_id, task_category, preferred_model, fallback_model, min_vram_mb, json.dumps(meta)),
            )
            conn.commit()
            return ModelRoute(
                id=route_id,
                task_category=task_category,
                preferred_model=preferred_model,
                fallback_model=fallback_model,
                min_vram_mb=min_vram_mb,
                metadata=meta,
            )
        finally:
            conn.close()

    async def route(
        self,
        task_prompt: str = "",
        task_category: Optional[str] = None,
        require_vram_mb: float = 0.0,
        available_models: Optional[list[str]] = None,
    ) -> RouteDecision:
        """
        Dynamically dispatches task to optimal model based on intent, VRAM, and local availability.
        """
        # 1. Determine task category
        confidence = 1.0
        if task_category and task_category in TASK_CATEGORIES:
            category = task_category
        else:
            category, confidence = self.classify_task(task_prompt)

        # 2. Get route definition
        route_def = self.get_route(category)
        if not route_def:
            preferred = self.default_model
            fallback = self.default_model
            min_vram = 0.0
        else:
            preferred = route_def.preferred_model
            fallback = route_def.fallback_model
            min_vram = max(route_def.min_vram_mb, require_vram_mb)

        # 3. Discover installed models if not provided
        installed: set[str] = set()
        if available_models is not None:
            installed = {m.lower() for m in available_models}
        else:
            try:
                models = await self.provider.list_models()
                installed = {m.id.lower() for m in models} | {m.name.lower() for m in models}
            except Exception as e:
                log.warning("Could not list local models for routing: %s", e)
                # Fallback to trusting preferred model
                installed = {preferred.lower(), fallback.lower(), self.default_model.lower()}

        # 4. Check hardware VRAM headroom
        headroom_mb = 8192.0  # Default reasonable assumption
        try:
            topology = self.topology_mgr.discover_topology()
            gpu_devices = [d for d in topology.devices if d.device_type == "cuda"]
            if gpu_devices:
                headroom_mb = sum(d.vram_free_mb for d in gpu_devices)
            else:
                headroom_mb = topology.total_ram_gb * 1024.0 * 0.5
        except Exception as e:
            log.warning("Hardware topology check failed during routing: %s", e)

        # 5. Check if preferred model is viable
        # Match either exact name or model name prefix
        def is_installed(m_name: str) -> bool:
            m_low = m_name.lower()
            return any(m_low == inst or inst.startswith(m_low.split(":")[0]) for inst in installed)

        preferred_installed = is_installed(preferred)
        vram_ok = headroom_mb >= min_vram if min_vram > 0 else True

        if preferred_installed and vram_ok:
            return RouteDecision(
                task_category=category,
                selected_model=preferred,
                is_fallback=False,
                reason=f"Selected preferred model '{preferred}' for '{category}' workload with {headroom_mb:.0f} MB headroom.",
                confidence=confidence,
                estimated_vram_mb=min_vram,
                available_headroom_mb=headroom_mb,
                metadata={"route_id": route_def.id if route_def else None},
            )

        # 6. Fallback resolution
        fallback_installed = is_installed(fallback)
        reason_parts = []
        if not preferred_installed:
            reason_parts.append(f"Preferred model '{preferred}' not installed")
        if not vram_ok:
            reason_parts.append(f"VRAM headroom ({headroom_mb:.0f} MB) < required ({min_vram:.0f} MB)")

        if fallback_installed:
            selected = fallback
            reason = f"{', '.join(reason_parts)}. Switched to fallback model '{fallback}'."
        else:
            selected = self.default_model
            reason = f"{', '.join(reason_parts)}, and fallback '{fallback}' missing. Switched to system default '{self.default_model}'."

        return RouteDecision(
            task_category=category,
            selected_model=selected,
            is_fallback=True,
            reason=reason,
            confidence=confidence * 0.85,
            estimated_vram_mb=min_vram * 0.5,
            available_headroom_mb=headroom_mb,
            metadata={"route_id": route_def.id if route_def else None, "original_preferred": preferred},
        )
