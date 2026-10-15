"""
Model provider abstraction layer.

Defines the interface that all model backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Any


@dataclass
class GenerationRequest:
    """A request to generate a model response."""
    messages: list[dict[str, str]]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    stop: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """A response from a model generation."""
    content: str = ""
    model: str = ""
    role: str = "assistant"
    finish_reason: str = ""
    token_count_prompt: int = 0
    token_count_completion: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Metadata about an available model."""
    id: str
    name: str
    backend: str
    size_bytes: int = 0
    parameter_count: str = ""
    quantization: str = ""
    context_length: int = 0
    family: str = ""
    format: str = ""
    modified_at: str = ""


class ModelProvider(ABC):
    """Abstract interface for model inference backends."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a complete response."""
        ...

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming response, yielding text chunks."""
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models on this backend."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the backend is reachable and ready."""
        ...

    @abstractmethod
    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info about a specific model."""
        ...

    @abstractmethod
    async def embed(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        """Generate vector embeddings for one or more text inputs."""
        ...


from genesis.models.context import ContextBudget, ContextManager, ModelArchitecture
from genesis.models.profiler import InferenceProfileResult, InferenceProfiler
from genesis.models.router import ModelRoute, ModelRouter, RouteDecision, TASK_CATEGORIES
from genesis.models.vision import VisionInspector, VisionRequest, VisionResponse

__all__ = [
    "GenerationRequest",
    "GenerationResponse",
    "ModelInfo",
    "ModelProvider",
    "ContextBudget",
    "ContextManager",
    "ModelArchitecture",
    "InferenceProfileResult",
    "InferenceProfiler",
    "ModelRoute",
    "ModelRouter",
    "RouteDecision",
    "TASK_CATEGORIES",
    "VisionInspector",
    "VisionRequest",
    "VisionResponse",
]


