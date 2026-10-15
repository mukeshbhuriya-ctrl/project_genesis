# CHANGELOG

All notable changes to Project Genesis will be documented in this file.

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
