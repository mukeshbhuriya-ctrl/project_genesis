# CHANGELOG

All notable changes to Project Genesis will be documented in this file.

## [0.4.0] — 2026-10-15

### Added
- **Typed Tool Registry Subsystem**: Strongly-typed tool parameter schema validation and Ollama/OpenAI function calling specification generation (`genesis/tools/base.py`, `genesis/tools/registry.py`).
- **Tiered Security and Path Sandboxing**: Permission enforcement (`READ_ONLY`, `MUTATING`, `DANGEROUS`) with strict workspace path boundary verification, blocking traversal escapes, drive roots, and protected system files (`genesis/tools/permissions.py`).
- **Standard Built-in Local Tools**:
  - `read_file`: Line-range selective text file reading inside workspace (`genesis/tools/builtin/file_reader.py`).
  - `write_file`: Confined workspace file creation and atomic overwriting (`genesis/tools/builtin/file_writer.py`).
  - `list_directory`: Recursive and flat directory listing with size/type introspection (`genesis/tools/builtin/dir_lister.py`).
  - `calculator`: Pure AST mathematical evaluation supporting operators, math functions, and constants without `eval()` (`genesis/tools/builtin/calculator.py`).
  - `python_eval`: Subprocess-isolated Python code execution with bounded memory, timeout, and output capture (`genesis/tools/builtin/python_eval.py`).
- **Tool Execution Engine**: `ToolExecutor` with permission validation, execution timing, error trapping, and async timeout enforcement (`genesis/tools/executor.py`).
- **Bounded Task Orchestrator**: `TaskPlanner` with local LLM decomposition, step bounding, and fallback plan generation (`genesis/tasks/planner.py`).
- **Task Audit Log & State Persistence (Schema V3)**: `TaskExecutor` tracking step states (`pending`, `running`, `success`, `failed`, `skipped`) and full audit execution traces in SQLite (`tasks` and `task_steps` tables).
- **Artifact Verification**: `TaskVerifier` checking file existence, non-empty byte thresholds, required substring content, and numeric tolerances (`genesis/tasks/verifier.py`).
- **REST Endpoints**: `/api/tools`, `/api/tools/execute`, `/api/tasks`, `/api/tasks/plan`, `/api/tasks/run`, `/api/tasks/{task_id}`, `/api/tasks/{task_id}/verify`.
- **Interactive Web UI Tools & Tasks Tab**: Live registered tools visualizer, permission badges, and real-time execution trace monitor (`genesis/web/`).
- **Architecture Decision Record**: ADR-004 Tool Registry, Permission Boundaries, and Bounded Task Orchestration.
- **Test Suite**: 30 new tests across tools, tasks, API, and live E2E (`test_tools.py`, `test_tasks.py`, `test_api_tools_tasks.py`, `test_tool_orchestration_e2e.py`), bringing total passing test suite to 88 tests.

## [0.3.0] — 2026-10-15

### Added
- **Dense Vector Memory Subsystem**: Dense float32 binary BLOB vector storage in SQLite (`memory_embeddings` table) and pure-Python cosine similarity search (`genesis/memory/embeddings.py`).
- **Hybrid Memory Retrieval**: Combined semantic cosine vector scoring ($\alpha=0.6$) with lexical keyword matching and confidence weighting (`genesis/memory/__init__.py`).
- **Knowledge Document Indexer**: Ingestion pipeline (`genesis/knowledge/indexer.py`) supporting text/markdown/code chunking, SHA256 checksum change detection, document versioning, and semantic passage search.
- **Memory Consolidation**: Semantic deduplication ($\ge 0.90$ cosine similarity) and automatic version incrementing (`genesis/memory/consolidation.py`).
- **Atomic Backup and Disaster Recovery**: SQLite online atomic backup API (`genesis/core/backup.py`) with `PRAGMA integrity_check` verification, safe point-in-time restore, and backup rotation.
- **REST Endpoints**: `/api/memories` (hybrid/semantic/lexical search, consolidation), `/api/knowledge/index-file`, `/api/knowledge/documents`, `/api/knowledge/search`, `/api/backups`.
- **Interactive Single-Page Web Dashboard**: Modern dark-mode interface (`genesis/web/`) for Chat, Memory inspection, Document search, Hardware diagnostics, and Backups.
- **Architecture Decision Record**: ADR-003 Dense Vector Memory and Knowledge Indexing.
- **Subsystem Tests**: 18 new unit & integration tests (`test_memory_embeddings.py`, `test_knowledge_indexer.py`, `test_backup.py`), bringing total passing test suite to 58 tests.

## [0.2.0] — 2026-10-14

### Added
- Local Ollama v0.40.4 runtime installed and configured with persistent background daemon.
- Configured dedicated model storage on NVMe D: drive (`D:\project_genesis\data\models\ollama`).
- Downloaded and verified default reasoning model: `qwen2.5:7b` (4.7 GB) with 100% GPU offloading on RTX 2000 Ada (8 GB VRAM).
- Downloaded and verified embedding model: `nomic-embed-text` (274 MB).
- End-to-end integration tests (`tests/test_integration_e2e.py`) validating live GPU inference and conversation persistence across simulated application restarts.
- Chat API tests (`tests/test_api_chat.py`) covering conversation listing, message history, and error recovery.
- Full test suite passing (40/40 tests).

## [0.1.0] — 2026-10-14

### Added
- Initial project scaffolding and repository structure.
- Master engineering directive (`MASTER_PROMPT_PROJECT_GENESIS.md`).
- Continuity documents: `AGENTS.md`, `PROJECT_STATE.md`, `ROADMAP.md`, `CHANGELOG.md`.
- Architecture documentation (`docs/architecture.md`).
- ADR-001: Initial technology selection.
- Python package structure (`genesis/`) with:
  - Core configuration (`genesis/core/config.py`)
  - Structured logging (`genesis/core/logging.py`)
  - Hardware discovery (`genesis/core/hardware.py`)
  - Database initialization (`genesis/core/database.py`)
  - Model provider abstraction (`genesis/models/provider.py`)
  - Ollama backend (`genesis/models/ollama_backend.py`)
  - Memory subsystem foundation (`genesis/memory/`)
  - CLI entry point (`genesis/cli.py`)
- `pyproject.toml` with dependency management.
- Test suite foundation (`tests/`).
- `.gitignore`, `.env.example`, `README.md`.

### Infrastructure
- Git repository initialized.
- Windows 11, Python 3.14, RTX 2000 Ada (8GB VRAM) as baseline hardware.
