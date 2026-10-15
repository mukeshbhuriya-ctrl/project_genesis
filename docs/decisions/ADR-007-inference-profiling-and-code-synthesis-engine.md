# ADR-007: Hardware-Aware Context Management, Inference Profiling, Reflective Code Synthesis, and Adaptive Knowledge Synthesis

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis operates as a local-first, self-improving AI research and model development platform running on a dedicated Windows workstation equipped with an NVIDIA RTX 2000 Ada GPU (8 GB VRAM). Following the successful completion of Phase 5 (multi-format dataset management, VRAM-bounded LoRA adaptation, and automated baseline rollback), Phase 6 focuses on research expansion and core autonomous capability enhancement.

Four distinct architectural requirements emerge in Phase 6:
1. **Context Window & KV-Cache VRAM Allocation**: On an 8 GB VRAM GPU, running 7B quantized models (~4.7 GB base weights) with extended context windows (4K, 8K, 16K, 32K) consumes substantial memory for Key-Value (KV) attention caches. Unbounded context growth risks CUDA Out-Of-Memory (OOM) fatal crashes that terminate local inference processes.
2. **Inference Latency & Throughput Profiling**: Optimization requires empirical measurement of hardware performance, specifically Time-To-First-Token (TTFT), inter-token latency (ITL), and generation throughput (tokens/second) across varying context lengths.
3. **Reflective Code Synthesis & Sandbox Self-Repair**: Enabling autonomous code generation requires executing untrusted, generated code safely in an isolated subprocess environment with strict timeouts, capturing traceback diagnostics on failure, and feeding structured error feedback back into the local model for multi-round self-repair. Once verified, code samples must be fed back into Phase 5 training datasets, completing a genuine self-improving closed loop.
4. **Adaptive Multi-Source Knowledge Synthesis**: Generating factual research briefs from local vector memory and indexed documents requires deterministic citation verification (`[1]`, `[2]`), groundedness confidence scoring, and automatic consolidation into persistent long-term memory.

## Decisions

### 1. Database Schema Version 6 Migrations
Schema Version 6 introduces three relational tables in SQLite (`genesis/core/database.py`):
- `research_profiles`: Tracks model profiling runs (model_id, backend, context_length, prompt_tokens, completion_tokens, ttft_ms, tokens_per_second, inter_token_latency_ms, total_latency_ms, gpu_vram_allocated_mb, kv_cache_vram_mb, status, metadata).
- `code_experiments`: Tracks code synthesis runs, objective, language, initial and final code, unit test harnesses, execution tracebacks, status (`passed`, `failed`, `repairing`), iteration counts, and execution durations.
- `research_synthesis`: Tracks multi-round research briefs, grounding source counts, verified citation sets, markdown briefs, and groundedness confidence scores.

### 2. Context Window & KV-Cache VRAM Budgeting Engine (`genesis/models/context.py`)
- Computes exact FP16 KV-cache memory consumption:
  $$\text{KV Bytes per Token} = 2 \times L \times H_{kv} \times D_{head} \times 2 \text{ bytes}$$
  For Qwen2.5 7B ($L=28, H_{kv}=4, D_{head}=128$), KV cache is $57,344 \text{ bytes} \approx 56 \text{ KB per token}$ (224 MB at 4K context, 448 MB at 8K, 896 MB at 16K, and 1,792 MB at 32K).
- Enforces dynamic VRAM headroom: Evaluates base model weights, KV cache, and 512 MB CUDA baseline buffer against total VRAM (8,192 MB) and ensures a 1,024 MB safety reserve.
- Context truncation & dialogue compaction: `compact_dialogue` compresses chat histories to fit token budgets while strictly preserving the system prompt and the latest user turn.

### 3. Inference Profiling Engine (`genesis/models/profiler.py`)
- Streams generation chunks from `ModelProvider`, capturing timestamps for $t_0$, first chunk ($t_1$ for TTFT), and subsequent chunks $t_2, \dots, t_n$.
- Measures TTFT in milliseconds, inter-token intervals $(t_n - t_1) / (N - 1)$, and overall throughput in tokens/second.
- Reads active GPU VRAM from `genesis.core.hardware.detect_gpu()` and computes exact KV-cache allocation.
- Persists profiling runs into SQLite `research_profiles`.

### 4. Code Synthesis & Subprocess Sandbox with Self-Repair (`genesis/research/coding.py`)
- `CodeSandbox`: Executes generated code and unit test harnesses in an isolated temporary directory via `subprocess.run([sys.executable, ...])` with configurable timeout (default: 8.0s). Captures stdout, stderr, and categorizes failure modes (`assertion_failure`, `syntax_error`, `timeout`, `runtime_error`).
- `ReflectiveCodeSynthesizer`: Decomposes coding goals, prompts the local model for implementation and assert statements, executes tests in the sandbox, and iteratively repairs code up to `max_iterations` on failure.
- **Closed-Loop Self-Improvement**: When verified code passes all unit tests, it can optionally be appended to Phase 5 training datasets (`data/datasets/<id>.jsonl`) as verified instruction pairs for LoRA fine-tuning.

### 5. Adaptive Knowledge Synthesis with Citation Verification (`genesis/research/synthesis.py`)
- `AdaptiveKnowledgeSynthesizer`: Gathers source passages from both `KnowledgeIndexer` (dense vector embeddings) and `MemoryStore` (hybrid vector/lexical retrieval).
- Prompts local model with strict citation grounding constraints (`[1]`, `[2]`).
- Deterministic verification: Scans output text for citation indices, matches them against retrieved source candidates, and calculates groundedness confidence score $\frac{\text{valid citations}}{\text{total citations}}$.
- Memory Consolidation: Automatically stores verified research briefs in the persistent memory store under category `research_brief`.

### 6. REST API & Web Dashboard Integration
- Exposes endpoints under `/api/research`:
  - `GET /api/research/context-budget`
  - `POST /api/research/profile` and `GET /api/research/profiles`
  - `POST /api/research/coding/synthesize` and `GET /api/research/coding/experiments`
  - `POST /api/research/synthesize` and `GET /api/research/syntheses`
- Interactive Web UI: Dedicated **"Research & Capabilities"** tab in `genesis/web/index.html` featuring interactive KV-cache budget calculators, live inference latency test runner, code synthesis and sandbox tester with traceback view, knowledge synthesis generator, and historical logs.

## Consequences

### Positive
- **OOM Prevention**: Exact KV-cache tracking prevents out-of-VRAM allocation on the 8 GB RTX 2000 Ada GPU during multi-turn reasoning conversations.
- **Empirical Hardware Metrics**: Real-time TTFT and throughput profiling eliminates guesswork regarding model latency and quantization performance.
- **Robust Code Generation**: Sandboxed execution and reflective repair guarantee that only code passing executable unit tests is accepted.
- **Closed-Loop Evolution**: Verified code experiments directly feed into Phase 5 training datasets, creating an autonomous cycle of data generation, verification, fine-tuning, and regression benchmarking.
- **Zero Hallucination Grounding**: Adaptive synthesis verifies that claims are backed by indexed passages, scoring groundedness and consolidating verified knowledge into long-term memory.

### Negative / Trade-offs
- Subprocess execution adds a ~50-200ms latency overhead per test run.
- Iterative self-repair consumes local model inference cycles on failure, bounded by `max_iterations`.
