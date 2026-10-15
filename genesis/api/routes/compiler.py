"""
REST API routes for Real-Time Self-Compilation and Native Extension Kernels (Phase 11).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from genesis.core.compiler import NativeExtensionManager

router = APIRouter(prefix="/compiler", tags=["Self-Compilation & Native Kernels"])


# ── Request / Response Schemas ──

class CompileKernelRequest(BaseModel):
    name: str = Field(..., description="Unique kernel identifier (e.g. vector_dot_product, fast_quantize_int8)")
    language: str = Field(default="c", description="Target language: c, rust, cpp")
    source_code: str = Field(..., description="C or Rust source code implementing exportable functions")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Custom metadata")


class BenchmarkKernelRequest(BaseModel):
    iterations: int = Field(default=500, ge=10, le=10000, description="Benchmark loop iterations")


class InvokeKernelRequest(BaseModel):
    arguments: list[Any] = Field(default_factory=list, description="Positional arguments for kernel function")
    keyword_arguments: Optional[dict[str, Any]] = Field(default_factory=dict, description="Keyword arguments")


# ── Routes ──

@router.get("/compilers")
async def get_available_compilers():
    """Discover available host C/C++/Rust compilers."""
    mgr = NativeExtensionManager()
    compilers = mgr.detect_compilers()
    return {
        "compilers": compilers,
        "has_native_compiler": len(compilers) > 0,
        "supported_languages": ["c", "cpp", "rust"],
    }


@router.post("/compile")
async def compile_extension(req: CompileKernelRequest):
    """
    Synthesize and compile a native shared library extension.
    If no compiler is installed on the host, registers an optimized Python fallback.
    """
    mgr = NativeExtensionManager()
    result = mgr.compile_c_extension(
        name=req.name,
        source_code=req.source_code,
    )
    return result.to_dict()


@router.get("/extensions")
async def list_extensions():
    """List all registered and compiled native extensions."""
    mgr = NativeExtensionManager()
    extensions = mgr.list_extensions()
    return [ext.to_dict() for ext in extensions]


@router.post("/benchmark/{name}")
async def benchmark_extension(name: str, req: BenchmarkKernelRequest):
    """Benchmark performance and speedup factor of a compiled extension."""
    mgr = NativeExtensionManager()
    try:
        metrics = mgr.benchmark_kernel(name, iterations=req.iterations)
        return metrics
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e}")


@router.post("/invoke/{name}")
async def invoke_extension(name: str, req: InvokeKernelRequest):
    """Invoke a compiled native extension or fallback kernel."""
    mgr = NativeExtensionManager()
    try:
        res = mgr.invoke(name, *req.arguments, **(req.keyword_arguments or {}))
        return {
            "kernel": name,
            "result": res,
            "status": "success",
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invocation failed: {e}")
