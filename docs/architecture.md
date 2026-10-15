# Architecture

## Overview

Project Genesis uses a modular, layered architecture designed to run in a single process on a Windows laptop while supporting future scaling to multi-process and multi-machine deployments.

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (Web UI)                 │
│              HTML/CSS/JS — Future Phase              │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼──────────────────────────────┐
│                  API Layer (FastAPI)                 │
│   Routes: /chat, /memory, /tools, /models, /health  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Orchestration Layer                     │
│   Task planning, execution, tool dispatch           │
└───┬──────┬──────┬──────┬──────┬──────┬─────────────┘
    │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────────┐
│Models││Memory││Know- ││Tools ││Train-││Evaluation│
│      ││      ││ledge ││      ││ing   ││          │
└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───────┘
   │       │       │       │       │       │
   ▼       ▼       ▼       ▼       ▼       ▼
┌─────────────────────────────────────────────────────┐
│              Data Layer (SQLite + Files)             │
│   genesis.db, vector store, model files, datasets   │
└─────────────────────────────────────────────────────┘
```

## Module Boundaries

### `genesis/core/` — Foundation
- **`config.py`** — Environment-based configuration with validation.
- **`logging.py`** — Structured logging (JSON + console).
- **`hardware.py`** — System capability detection (CPU, RAM, GPU, disk).
- **`database.py`** — SQLite connection management and schema migrations.
- **`exceptions.py`** — Shared exception hierarchy.

### `genesis/models/` — Model Inference
- **`provider.py`** — Abstract `ModelProvider` interface.
- **`ollama_backend.py`** — Ollama integration (primary).
- **`registry.py`** — Model registry (metadata, versions, status).
- Future: `llamacpp_backend.py`, `transformers_backend.py`, `vllm_backend.py`.

### `genesis/memory/` — Persistent Memory
- **`store.py`** — Memory CRUD with categories, metadata, versioning.
- **`search.py`** — Full-text and (future) semantic search.
- **`models.py`** — SQLAlchemy/dataclass models for memory records.

### `genesis/knowledge/` — Document Retrieval
- **`ingestion.py`** — File validation, content extraction, chunking.
- **`index.py`** — Search index management.
- **`retrieval.py`** — Query and relevance scoring.
- Future: embedding-based vector search.

### `genesis/tools/` — Tool Registry
- **`registry.py`** — Tool definitions, schemas, permissions.
- **`executor.py`** — Sandboxed execution with timeouts and limits.
- **`builtin/`** — Built-in tools (file reader, calculator, etc.).

### `genesis/tasks/` — Task Orchestration
- **`planner.py`** — Task decomposition and planning.
- **`executor.py`** — Step execution with state tracking.
- **`models.py`** — Task states (pending, running, completed, failed).

### `genesis/training/` — Fine-Tuning Subsystem (Phase 5)
- **`datasets.py`** — Multi-format dataset ingestion, normalization, deterministic splitting, and token statistics.
- **`preprocessing.py`** — SHA-256 deduplication, length filtering, AST syntax verification, and 8-gram benchmark contamination guard.
- **`experiments.py`** — LoRA hyperparameters, experiment tracking, loss metrics logging, and SQLite persistence.
- **`trainer.py`** — VRAM-bounded fine-tuning execution engine, step optimization loop, and callback handlers.
- **`export.py`** — Checkpoint selection, Ollama Modelfile synthesis, local model registration, and automated rollback.

### `genesis/evaluation/` — Benchmarking
- **`benchmarks.py`** — Benchmark definitions and runners.
- **`regression.py`** — Regression test suite.
- **`reports.py`** — Evaluation report generation.

### `genesis/api/` — REST API
- **`app.py`** — FastAPI application factory.
- **`routes/`** — Endpoint modules (chat, memory, models, health).

### `genesis/cli.py` — Command-Line Interface
- `init` — Initialize database and data directories.
- `serve` — Start the API server.
- `hardware` — Print hardware diagnostic report.
- `test` — Run evaluation benchmarks.

## Data Layout

```
data/
├── genesis.db          # SQLite database (conversations, memory, metadata)
├── models/
│   ├── registry.json   # Model registry metadata
│   └── weights/        # Downloaded model files (gitignored)
├── datasets/
│   ├── raw/            # Original datasets (gitignored)
│   └── processed/      # Transformed datasets (gitignored)
├── experiments/        # Experiment records and checkpoints
├── knowledge/
│   ├── documents/      # Ingested source documents
│   └── index/          # Search index files
├── backups/            # Database and config backups
└── logs/               # Application logs
```

## Key Design Decisions

1. **SQLite as the primary database** — No external database server required. Sufficient for single-user local operation. Migrateable to PostgreSQL if needed.
2. **Ollama as the initial model backend** — Simplifies model management on Windows. Handles GPU offloading, quantization, and model lifecycle.
3. **FastAPI for the API layer** — Async-native, good OpenAPI support, WebSocket support for streaming.
4. **Modular provider interfaces** — Every major subsystem (models, memory, knowledge, tools) has an abstract interface allowing backend replacement.

## Interfaces Between Modules

All inter-module communication uses typed Python function calls within a single process. No message queues, RPC, or microservices at this stage.

Future scaling path:
1. Extract model inference to a separate process (already isolated by Ollama).
2. Add async task queue for long-running operations.
3. Move to PostgreSQL if concurrent access is needed.
4. Distribute training to a separate machine via SSH/API.

## Constraints

- Must work without internet after initial setup.
- Must not require admin/root privileges for normal operation.
- Must not silently send data to external services.
- Must handle GPU absence gracefully (CPU fallback).
- Must preserve all data across application upgrades.
