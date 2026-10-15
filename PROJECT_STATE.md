# PROJECT STATE

> Last updated: 2026-10-15 by Senior AI/ML Engineer (Phase 2 completion & Phase 3 activation)

## Current Phase

**Phase 2 Complete (Verified) → Phase 3: Tools & Task Orchestration (Active)**

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
- [x] `docs/decisions/` — ADR-001 (Tech selection), ADR-002 (Ollama/Storage isolation), ADR-003 (Vector memory & indexing)
- [x] Python package scaffolding (`genesis/`)
- [x] Core configuration module (`genesis/core/__init__.py`)
- [x] Logging module (`genesis/core/logging.py`)
- [x] Hardware discovery module (`genesis/core/hardware.py`)
- [x] Model provider abstraction (`genesis/models/__init__.py`)
- [x] Ollama backend integration with embedding support (`genesis/models/ollama_backend.py`)
- [x] Ollama v0.40.4 installed, persistent service running with storage on `D:\project_genesis\data\models\ollama`
- [x] Models pulled & verified: `qwen2.5:7b` (4.7 GB), `qwen2.5-coder:7b` (4.7 GB), `qwen2.5-coder:1.5b-base` (986 MB), `nomic-embed-text` (274 MB)
- [x] CLI entry point (`genesis/cli.py`) with init, hardware, status, and serve commands
- [x] Database initialization (`genesis/core/database.py`) with Schema Version 2 migrations

### Memory & Knowledge Subsystem (Phase 2)
- [x] Persistent structured memory store (`genesis/memory/__init__.py`) with CRUD, categories, confidence, metadata, versioning
- [x] Dense vector embeddings (`genesis/memory/embeddings.py`) with binary IEEE 754 float32 BLOB serialization and cosine similarity
- [x] Hybrid memory search fusing dense vector similarity ($\alpha=0.6$) with lexical keyword hits and confidence weighting
- [x] Memory consolidation & deduplication (`genesis/memory/consolidation.py`) with similarity suppression ($\ge 0.90$) and version bumping
- [x] Knowledge document indexer (`genesis/knowledge/indexer.py`) with paragraph/window chunking, SHA256 checksum change detection, document tracking, and semantic passage search
- [x] Atomic SQLite backup & disaster recovery (`genesis/core/backup.py`) with online backup API, integrity verification (`PRAGMA integrity_check`), restore, and backup rotation
- [x] REST API endpoints (`genesis/api/routes/`): `/api/memories` (hybrid/semantic/lexical search, consolidation), `/api/knowledge/` (index-file, documents, search), `/api/backups/` (create, list)
- [x] Interactive Web Dashboard UI (`genesis/web/`): single-page app (Chat, Memory inspector, Knowledge document search, Hardware diagnostics, Backup management) mounted at `/` and `/static/`
- [x] Comprehensive test suite in `tests/`: 58 tests covering all subsystems (100% passing)

## Verified

- [x] **58/58 unit and integration tests pass (100% passing)** across all modules
- [x] **Local model responds** — `qwen2.5:7b` verified via Ollama API and Python `OllamaProvider`
- [x] **GPU acceleration verified** via `nvidia-smi` (all 29 layers offloaded, ~4.6 GB VRAM utilized on RTX 2000 Ada)
- [x] **Dense vector embeddings verified** — `nomic-embed-text` produces 768-dimensional embeddings via Ollama `/api/embed`
- [x] **Hybrid retrieval verified** — vector cosine ranking combined with lexical scoring and confidence weighting
- [x] **Knowledge document indexing verified** — chunking, SHA256 deduplication, cascade deletions, passage search
- [x] **Memory consolidation verified** — semantic duplicates identified and suppressed
- [x] **Online atomic backup & restore verified** — SQLite integrity verified across snapshot and restore operations
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

## Next Steps (Phase 3: Tools and Task Orchestration)

1. **Typed Tool Registry (`genesis/tools/registry.py`):** Define JSON schema / Pydantic schemas, tool definitions, typed input/output validation, and metadata.
2. **Permission Model (`genesis/tools/permissions.py`):** Establish execution security levels (`READ_ONLY`, `MUTATING`, `DANGEROUS`) with permission checks and sandboxing.
3. **Built-in Safe Tools (`genesis/tools/builtin/`):** Implement `read_file`, `write_file`, `list_directory`, `calculator`, `python_eval` (restricted evaluator).
4. **Sandboxed Executor (`genesis/tools/executor.py`):** Enforce strict timeout, execution limits, working directory confinement, and stderr/stdout capture.
5. **Task Orchestration (`genesis/tasks/`):** Implement `TaskPlanner`, `StepExecutor`, task states (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`), step-by-step trace logging in SQLite.
6. **Integration Verification:** Demonstrate an autonomous, bounded multi-step task execution using `qwen2.5:7b` / `qwen2.5-coder:7b` with real tools, verified outputs, and full audit trail.
