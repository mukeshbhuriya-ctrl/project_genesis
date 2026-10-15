"""
Inference profiling engine for local models.

Measures hardware performance metrics on the NVIDIA RTX 2000 Ada GPU:
- Time To First Token (TTFT in milliseconds)
- Inter-token latency (milliseconds per token)
- Generation throughput (tokens per second)
- Total latency
- Active GPU VRAM footprint & KV-cache allocation
- Persistence in SQLite research_profiles table
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core.database import get_sync_connection
from genesis.core.hardware import detect_gpu
from genesis.core.logging import get_logger
from genesis.models import GenerationRequest, ModelProvider
from genesis.models.context import ContextManager
from genesis.models.ollama_backend import OllamaProvider

log = get_logger("inference_profiler")


@dataclass
class InferenceProfileResult:
    """Detailed benchmark metrics for an inference execution run."""
    id: str
    model_id: str
    backend: str = "ollama"
    context_length: int = 4096
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    inter_token_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    gpu_vram_allocated_mb: float = 0.0
    kv_cache_vram_mb: float = 0.0
    status: str = "completed"
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "backend": self.backend,
            "context_length": self.context_length,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": round(self.ttft_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "inter_token_latency_ms": round(self.inter_token_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "gpu_vram_allocated_mb": round(self.gpu_vram_allocated_mb, 2),
            "kv_cache_vram_mb": round(self.kv_cache_vram_mb, 2),
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class InferenceProfiler:
    """Profiles local model execution throughput, latency, and VRAM utilization."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        context_manager: ContextManager | None = None,
        db_path: str | Path = "data/genesis.db",
    ) -> None:
        self.provider = provider or OllamaProvider()
        self.context_manager = context_manager or ContextManager()
        self.db_path = Path(db_path)

    def _get_gpu_vram_used(self) -> float:
        """Query current NVIDIA GPU VRAM used in MB."""
        try:
            gpus = detect_gpu()
            if gpus:
                # Return used MB
                return float(max(0, gpus[0].vram_total_mb - gpus[0].vram_free_mb))
        except Exception as e:
            log.debug("GPU VRAM detection skipped: %s", e)
        return 0.0

    async def profile(
        self,
        model_id: str,
        prompt: str = "Explain the QuickSort algorithm and its average time complexity in 3 short sentences.",
        max_tokens: int = 128,
        temperature: float = 0.2,
    ) -> InferenceProfileResult:
        """
        Execute a streaming generation run and measure TTFT, inter-token latency,
        throughput (tokens/sec), and VRAM utilization.
        """
        profile_id = str(uuid.uuid4())
        prompt_tokens = self.context_manager.estimate_tokens(prompt)
        initial_vram = self._get_gpu_vram_used()

        request = GenerationRequest(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        start_time = time.perf_counter()
        first_token_time: float | None = None
        token_timestamps: list[float] = []
        chunks: list[str] = []

        try:
            async for chunk in self.provider.generate_stream(request):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_timestamps.append(now)
                chunks.append(chunk)

            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000.0

            generated_text = "".join(chunks)
            completion_tokens = self.context_manager.estimate_tokens(generated_text)
            if completion_tokens == 0 and len(chunks) > 0:
                completion_tokens = len(chunks)

            # TTFT
            if first_token_time is not None:
                ttft_ms = (first_token_time - start_time) * 1000.0
            else:
                ttft_ms = total_latency_ms

            # Inter-token latency
            if len(token_timestamps) > 1:
                intervals = [
                    (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
                    for i in range(1, len(token_timestamps))
                ]
                inter_token_latency_ms = sum(intervals) / len(intervals)
            else:
                inter_token_latency_ms = 0.0

            # Tokens per second
            gen_duration_sec = end_time - (first_token_time or start_time)
            if gen_duration_sec > 0.001 and completion_tokens > 0:
                tokens_per_second = completion_tokens / gen_duration_sec
            else:
                tokens_per_second = 0.0

            # VRAM & KV Cache
            active_vram = self._get_gpu_vram_used()
            vram_used = max(active_vram, initial_vram)
            total_context_tokens = prompt_tokens + completion_tokens
            kv_cache_vram_mb = self.context_manager.calculate_kv_cache_vram(
                context_tokens=total_context_tokens,
                model_name=model_id,
            )

            result = InferenceProfileResult(
                id=profile_id,
                model_id=model_id,
                backend="ollama",
                context_length=total_context_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                tokens_per_second=tokens_per_second,
                inter_token_latency_ms=inter_token_latency_ms,
                total_latency_ms=total_latency_ms,
                gpu_vram_allocated_mb=vram_used,
                kv_cache_vram_mb=kv_cache_vram_mb,
                status="completed",
                metadata={"prompt_preview": prompt[:80], "chunks_received": len(chunks)},
            )

        except Exception as e:
            log.error("Profiling failed for model %s: %s", model_id, e)
            end_time = time.perf_counter()
            result = InferenceProfileResult(
                id=profile_id,
                model_id=model_id,
                backend="ollama",
                status="failed",
                error=str(e),
                total_latency_ms=(end_time - start_time) * 1000.0,
            )

        self._save_profile(result)
        return result

    def _save_profile(self, r: InferenceProfileResult) -> None:
        """Persist profile record to SQLite database."""
        if not self.db_path.exists():
            return
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO research_profiles (
                    id, model_id, backend, context_length, prompt_tokens,
                    completion_tokens, ttft_ms, tokens_per_second,
                    inter_token_latency_ms, total_latency_ms,
                    gpu_vram_allocated_mb, kv_cache_vram_mb, status, error,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.id,
                    r.model_id,
                    r.backend,
                    r.context_length,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.ttft_ms,
                    r.tokens_per_second,
                    r.inter_token_latency_ms,
                    r.total_latency_ms,
                    r.gpu_vram_allocated_mb,
                    r.kv_cache_vram_mb,
                    r.status,
                    r.error,
                    r.created_at,
                    json.dumps(r.metadata),
                ),
            )
            conn.commit()
            log.info("Saved inference profile %s for %s", r.id, r.model_id)
        except Exception as e:
            log.warning("Failed to persist profile %s: %s", r.id, e)
        finally:
            conn.close()

    def list_profiles(self, limit: int = 20) -> list[InferenceProfileResult]:
        """Retrieve recent inference profiles from SQLite."""
        if not self.db_path.exists():
            return []
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, model_id, backend, context_length, prompt_tokens,
                       completion_tokens, ttft_ms, tokens_per_second,
                       inter_token_latency_ms, total_latency_ms,
                       gpu_vram_allocated_mb, kv_cache_vram_mb, status, error,
                       created_at, metadata
                FROM research_profiles
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                results.append(
                    InferenceProfileResult(
                        id=row["id"],
                        model_id=row["model_id"],
                        backend=row["backend"],
                        context_length=row["context_length"],
                        prompt_tokens=row["prompt_tokens"],
                        completion_tokens=row["completion_tokens"],
                        ttft_ms=row["ttft_ms"],
                        tokens_per_second=row["tokens_per_second"],
                        inter_token_latency_ms=row["inter_token_latency_ms"],
                        total_latency_ms=row["total_latency_ms"],
                        gpu_vram_allocated_mb=row["gpu_vram_allocated_mb"],
                        kv_cache_vram_mb=row["kv_cache_vram_mb"],
                        status=row["status"],
                        error=row["error"],
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return results
        finally:
            conn.close()
