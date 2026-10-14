"""Models API endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from genesis.models.ollama_backend import OllamaProvider

router = APIRouter()


@router.get("/models")
async def list_models() -> dict:
    """List available models."""
    provider = OllamaProvider()
    if not await provider.is_available():
        return {"available": False, "models": [], "error": "Ollama is not running"}

    models = await provider.list_models()
    return {
        "available": True,
        "count": len(models),
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "backend": m.backend,
                "size_bytes": m.size_bytes,
                "parameter_count": m.parameter_count,
                "quantization": m.quantization,
                "family": m.family,
            }
            for m in models
        ],
    }


@router.get("/models/{model_id:path}")
async def get_model(model_id: str) -> dict:
    """Get detailed info about a specific model."""
    provider = OllamaProvider()
    info = await provider.get_model_info(model_id)
    if not info:
        return {"found": False, "error": f"Model '{model_id}' not found"}
    return {
        "found": True,
        "id": info.id,
        "name": info.name,
        "backend": info.backend,
        "parameter_count": info.parameter_count,
        "quantization": info.quantization,
        "context_length": info.context_length,
        "family": info.family,
        "format": info.format,
    }
