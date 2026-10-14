# ADR-002: Local Model Runtime and Storage Allocation

## Status
Accepted

## Date
2026-10-14

## Context
Project Genesis requires a robust, local-first inference engine that:
1. Runs entirely on the host machine without external API dependencies.
2. Leverages the NVIDIA RTX 2000 Ada Generation Laptop GPU (8 GB VRAM) for accelerated inference.
3. Automatically offloads layers to GPU while fitting comfortably within available VRAM.
4. Manages model binaries cleanly without saturating the primary C: system partition (212 GB free vs. 364 GB free on NVMe D:).

## Decision
1. **Inference Runtime:** Adopted Ollama v0.40.4 as the core local model provider backend. Ollama packages llama.cpp, automatic CUDA 12/13 compute graph compilation, and Flash Attention.
2. **Dedicated Storage Location:** Configured `OLLAMA_MODELS` to point permanently to `D:\project_genesis\data\models\ollama`. All model weights, layer blobs, and manifests are isolated to the secondary high-speed NVMe storage partition.
3. **Default General Model:** Selected `qwen2.5:7b` (Q4_K_M quantization, 4.7 GB on disk). It offloads 100% of its 29 layers to the RTX 2000 Ada GPU, utilizing ~4.6 GB of VRAM, leaving ~3.4 GB VRAM for KV-cache and system graphics.
4. **Default Embedding Model:** Selected `nomic-embed-text` (274 MB) for local vector embeddings and dense semantic memory retrieval.
5. **Client Integration:** Implemented async HTTP integration via `genesis.models.ollama_backend.OllamaProvider` using `httpx`. Configured extended timeout tolerances (300s) to absorb initial cold-start model weight allocation and CUDA graph warming.

## Consequences
- **Positive:** Full offline inference capability, verified 100% GPU offloading, robust layer quantization, and reliable conversation persistence.
- **Negative:** Cold start on initial weight load requires ~15-20 seconds before tokens begin streaming.
- **Mitigation:** Ollama keeps the model resident in VRAM for 5 minutes (`OLLAMA_KEEP_ALIVE=5m`) by default, ensuring sub-second response times for subsequent queries in an active session.
