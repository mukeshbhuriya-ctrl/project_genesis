# PROJECT STATE

> Last updated: 2026-10-15 by Senior AI/ML Engineer (Phase 2 completion & Phase 3 activation)

## Current Phase

**Phase 3 Complete (Verified) → Phase 4: Evaluation and Feedback (Active)**

## Implemented

### Foundation & Infrastructure (Phase 0 & 1)
- [x] Repository initialized with Git
- [x] Master prompt committed (`MASTER_PROMPT_PROJECT_GENESIS.md`)
- [x] `.gitignore` configured for Python/AI project (data/, models, caches ignored)
- [x] `.env.example` and active `.env` configuration with `qwen2.5:7b` default model
- [x] `AGENTS.md` — agent/developer instructions and protocols
- [x] `README.md` — project overview and quick start
- [x] `PROJECT_STATE.md` — this file
- [x] `ROADMAP.md` — milestone tracker
- [x] `CHANGELOG.md` — change log
- [x] `docs/architecture.md` — modular architecture specification
- [x] `docs/decisions/` — ADR-001 (Tech selection), ADR-002 (Ollama/Storage isolation), ADR-003 (Vector memory & indexing), ADR-004 (Tool registry & orchestration)
- [x] Python package scaffolding (`genesis/`)
- [x] Core configuration module (`genesis/core/__init__.py`)
- [x] Logging module (`genesis/core/logging.py`)
- [x] Hardware discovery module (`genesis/core/hardware.py`)
- [x] Model provider abstraction (`genesis/models/__init__.py`)
- [x] Ollama backend integration with embedding support (`genesis/models/ollama_backend.py`)
- [x] Ollama v0.40.4 installed, persistent service running with storage on `D:\project_genesis\data\models\ollama`
- [x] Models pulled & verified: `qwen2.5:7b` (4.7 GB), `qwen2.5-coder:7b` (4.7 GB), `qwen2.5-coder:1.5b-base` (986 MB), `nomic-embed-text` (274 MB)
- [x] CLI entry point (`genesis/cli.py`) with init, hardware, status, and serve commands
- [x] Database initialization (`genesis/core/database.py`) with Schema Version 3 migrations

### Memory & Knowledge Subsystem (Phase 2)
- [x] Persistent structured memory store (`genesis/memory/__init__.py`) with CRUD, categories, confidence, metadata, versioning
- [x] Dense vector embeddings (`genesis/memory/embeddings.py`) with binary IEEE 754 float32 BLOB serialization and cosine similarity
- [x] Hybrid memory search fusing dense vector similarity ($\alpha=0.6$) with lexical keyword hits and confidence weighting
- [x] Memory consolidation & deduplication (`genesis/memory/consolidation.py`) with similarity suppression ($\ge 0.90$) and version bumping
- [x] Knowledge document indexer (`genesis/knowledge/indexer.py`) with paragraph/window chunking, SHA256 checksum change detection, document tracking, and semantic passage search
- [x] Atomic SQLite backup & disaster recovery (`genesis/core/backup.py`) with online backup API, integrity verification (`PRAGMA integrity_check`), restore, and backup rotation
- [x] REST API endpoints: `/api/memories` (hybrid/semantic/lexical search, consolidation), `/api/knowledge/` (index-file, documents, search), `/api/backups/` (create, list)
- [x] Interactive Web Dashboard UI (`genesis/web/`): single-page app (Chat, Memory inspector, Knowledge document search, Hardware diagnostics, Backup management) mounted at `/` and `/static/`

### Tools & Task Orchestration Subsystem (Phase 3)
- [x] Strongly-typed tool registry (`genesis/tools/base.py`, `genesis/tools/registry.py`) with parameter validation and Ollama/OpenAI schema export
- [x] Tiered permission model (`genesis/tools/permissions.py`) with workspace path confinement, directory traversal blocking, and system path protection
- [x] Safe built-in local tools (`genesis/tools/builtin/`): `read_file`, `write_file`, `list_directory`, `calculator`, `python_eval`
- [x] Tool execution engine (`genesis/tools/executor.py`) with timeout protection, error isolation, and execution latency metrics
- [x] Bounded task planner (`genesis/tasks/planner.py`) decomposing goals into discrete steps with local model (`qwen2.5:7b`), max step bounds, and fallback plan protection
- [x] Task executor (`genesis/tasks/executor.py`) with step status progression, error recovery, and audit log persistence in SQLite (`tasks` and `task_steps` tables)
- [x] Task verifier (`genesis/tasks/verifier.py`) with file existence, non-empty byte count, substring content, and numeric tolerance validation
- [x] REST API endpoints (`genesis/api/routes/`): `/api/tools`, `/api/tools/execute`, `/api/tasks`, `/api/tasks/plan`, `/api/tasks/run`, `/api/tasks/{task_id}`, `/api/tasks/{task_id}/verify`
- [x] Interactive Web Dashboard Tools & Tasks tab (`genesis/web/`): live tool registry browser, permission badges, and real-time task execution monitor
- [x] Comprehensive test suite in `tests/`: 88 tests covering all subsystems (100% passing)

## Verified

- [x] **88/88 unit and integration tests pass (100% passing)** across all modules
- [x] **Local model responds** — `qwen2.5:7b` verified via Ollama API and Python `OllamaProvider`
- [x] **GPU acceleration verified** via `nvidia-smi` (all 29 layers offloaded, ~4.6 GB VRAM utilized on RTX 2000 Ada)
- [x] **Dense vector embeddings verified** — `nomic-embed-text` produces 768-dimensional embeddings via Ollama `/api/embed`
- [x] **Hybrid retrieval verified** — vector cosine ranking combined with lexical scoring and confidence weighting
- [x] **Knowledge document indexing verified** — chunking, SHA256 deduplication, cascade deletions, passage search
- [x] **Memory consolidation verified** — semantic duplicates identified and suppressed
- [x] **Online atomic backup & restore verified** — SQLite integrity verified across snapshot and restore operations
- [x] **Bounded tool execution verified** — AI decomposes goals with local model (`qwen2.5:7b`), executes local tools, and produces verified file artifacts on disk
- [x] **Audit trace logging verified** — every step's status, tool name, arguments, output, and duration persisted in SQLite
- [x] **Filesystem sandboxing verified** — path escapes, traversal attacks, and external drive roots rejected with SecurityError
- [x] **Conversation persists across restart** — verified end-to-end via `test_real_model_chat_and_restart_survival`
- [x] **Web dashboard responsive** — `/` and `/static/` serve interactive UI; API endpoints operational

## Known System Configuration Notes

1. Ollama v0.40.4 running locally on port 11434 with `OLLAMA_MODELS=D:\project_genesis\data\models\ollama`.
2. Windows environment variables `OLLAMA_MODELS` and `Path` updated.
3. System has ~360 GB free NVMe storage on D: and 8 GB dedicated GPU VRAM.
4. Active background daemon serving models locally.

## Hardware Baseline

| Component | Value |
|-----------|-------|
| OS | Windows 11 Pro (Build 26200) |
| CPU | Intel i9-13900H (14C/20T) |
| RAM | 32 GB |
| GPU | NVIDIA RTX 2000 Ada (8 GB VRAM) |
| Storage | 1 TB NVMe, ~360 GB free on D: |
| Python | 3.14.6 in `.venv` |
| Node.js | 24.17.0 |
| Git | 2.54.0 |

## Next Steps (Phase 4: Evaluation and Feedback)

1. **Fixed Baseline Benchmarks (`genesis/evaluation/benchmarks.py`):** Define standard reasoning, coding, and tool invocation benchmark tasks with deterministic evaluation rubrics.
2. **Task Outcome Recording (`genesis/evaluation/models.py`):** Track benchmark pass rates, latencies, token consumption, and regression metrics over time in SQLite.
3. **Failure Analysis Engine (`genesis/evaluation/analysis.py`):** Automatic error taxonomy categorization (syntax error, tool misuse, timeout, hallucination, missing parameter).
4. **Regression Test Runner (`genesis/evaluation/regression.py`):** Automated benchmark verification before and after prompt updates, model swaps, or fine-tuning runs.
5. **Comparison Reports (`genesis/evaluation/reports.py`):** Generate markdown and visual evaluation comparisons between baseline and candidate configurations.
