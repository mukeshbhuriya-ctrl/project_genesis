"""
REST API routes for Hardware-Targeted Micro-Quantization (Phase 14).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.models.quantization import (
    MicroQuantizer,
    QuantizationConfig,
    QuantizationStrategy,
)

router = APIRouter(prefix="/quantization", tags=["Hardware-Targeted Micro-Quantization"])


class ProfileQuantizationRequest(BaseModel):
    model_name: str = Field(default="qwen2.5:7b", description="Model architecture identifier")
    bits: int = Field(default=4, description="Quantization bit-width (2, 3, 4, 8)")
    strategy: QuantizationStrategy = Field(default="awq_outlier_salient", description="Quantization algorithm")
    block_size: int = Field(default=64, description="Granular block size for scaling factors (32, 64, 128)")
    outlier_channel_ratio: float = Field(default=0.01, ge=0.0, le=0.10, description="Fraction of salient activation outlier channels preserved in FP16")


@router.get("/strategies")
async def list_quantization_strategies():
    """List supported quantization algorithms, bit-widths, and hardware characteristics."""
    return {
        "strategies": [
            {
                "id": "awq_outlier_salient",
                "name": "Activation-Aware Weight Quantization (AWQ)",
                "description": "Protects top salient weight channels in FP16 while quantizing the remaining 99% in INT4/INT3/INT2 blocks.",
                "supported_bits": [2, 3, 4, 8],
                "recommended_for": "NVIDIA Ada Lovelace Tensor Cores with high-activation variance",
            },
            {
                "id": "asymmetric",
                "name": "Asymmetric Affine Quantization",
                "description": "Non-zero-centered quantization with scale S and zero-point Z for skewed weight distributions.",
                "supported_bits": [2, 3, 4, 8],
                "recommended_for": "Asymmetric ReLU/GELU projection layers",
            },
            {
                "id": "symmetric",
                "name": "Symmetric Uniform Block Quantization",
                "description": "Zero-centered linear quantization with per-block scaling factors, maximizing hardware throughput.",
                "supported_bits": [2, 3, 4, 8],
                "recommended_for": "Fast low-latency inference on general FP16 tensor pipelines",
            },
            {
                "id": "gptq_hessian",
                "name": "Second-Order Hessian-Guided Quantization",
                "description": "Hessian-based inverse curvature compensation for minimal loss degradation.",
                "supported_bits": [3, 4, 8],
                "recommended_for": "Extreme 3-bit / 4-bit compression runs",
            },
        ],
        "supported_block_sizes": [32, 64, 128],
    }


@router.post("/profile")
async def profile_quantization(req: ProfileQuantizationRequest):
    """Run mathematical and empirical micro-quantization benchmark and store profile."""
    quantizer = MicroQuantizer()
    try:
        cfg = QuantizationConfig(
            model_name=req.model_name,
            bits=req.bits,
            strategy=req.strategy,
            block_size=req.block_size,
            outlier_channel_ratio=req.outlier_channel_ratio,
        )
        res = quantizer.profile_quantization(config=cfg)
        return res.to_dict()
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Quantization profiling failed: {e}")


@router.get("/profiles")
async def list_quantization_profiles(limit: int = Query(default=20, ge=1, le=100)):
    """List historical quantization profiles."""
    quantizer = MicroQuantizer()
    profiles = quantizer.list_profiles(limit=limit)
    return [p.to_dict() for p in profiles]


@router.get("/profiles/{profile_id}")
async def get_quantization_profile(profile_id: str):
    """Retrieve details for a specific quantization profile."""
    quantizer = MicroQuantizer()
    profile = quantizer.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Quantization profile '{profile_id}' not found")
    return profile.to_dict()
