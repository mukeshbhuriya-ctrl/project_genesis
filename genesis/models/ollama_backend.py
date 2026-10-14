"""
Ollama model backend.

Communicates with a local Ollama server via its HTTP API.
Ollama manages model downloads, GPU offloading, and inference.
"""

from __future__ import annotations

from typing import AsyncIterator, Any

import httpx

from genesis.core.logging import get_logger
from genesis.models import (
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ModelProvider,
)

log = get_logger("models.ollama")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaProvider(ModelProvider):
    """Model provider backed by a local Ollama server."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List models available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

            models = []
            for m in data.get("models", []):
                details = m.get("details", {})
                models.append(ModelInfo(
                    id=m.get("name", ""),
                    name=m.get("name", ""),
                    backend="ollama",
                    size_bytes=m.get("size", 0),
                    parameter_count=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    context_length=0,  # Ollama doesn't expose this in list
                    family=details.get("family", ""),
                    format=details.get("format", ""),
                    modified_at=m.get("modified_at", ""),
                ))
            return models
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.warning(f"Failed to list Ollama models: {e}")
            return []

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get detailed info about a specific Ollama model."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model_id},
                )
                resp.raise_for_status()
                data = resp.json()

            details = data.get("details", {})
            params = data.get("model_info", {})

            # Try to extract context length from model_info
            ctx_len = 0
            for key, value in params.items():
                if "context_length" in key.lower():
                    ctx_len = int(value) if isinstance(value, (int, float)) else 0
                    break

            return ModelInfo(
                id=model_id,
                name=model_id,
                backend="ollama",
                size_bytes=0,
                parameter_count=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""),
                context_length=ctx_len,
                family=details.get("family", ""),
                format=details.get("format", ""),
            )
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.warning(f"Failed to get model info for {model_id}: {e}")
            return None

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a complete (non-streaming) response."""
        payload = self._build_payload(request, stream=False)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        return GenerationResponse(
            content=message.get("content", ""),
            model=data.get("model", request.model),
            role=message.get("role", "assistant"),
            finish_reason=data.get("done_reason", ""),
            token_count_prompt=data.get("prompt_eval_count", 0),
            token_count_completion=data.get("eval_count", 0),
            tool_calls=message.get("tool_calls", []),
            metadata={
                "total_duration_ns": data.get("total_duration", 0),
                "eval_duration_ns": data.get("eval_duration", 0),
            },
        )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        payload = self._build_payload(request, stream=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done", False):
                            return
                    except (ValueError, KeyError):
                        continue

    def _build_payload(
        self, request: GenerationRequest, stream: bool
    ) -> dict[str, Any]:
        """Build the Ollama API request payload."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": stream,
            "options": {},
        }

        if request.temperature is not None:
            payload["options"]["temperature"] = request.temperature

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        if request.stop:
            payload["options"]["stop"] = request.stop

        if request.tools:
            payload["tools"] = request.tools

        return payload
