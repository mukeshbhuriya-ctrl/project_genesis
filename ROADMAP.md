# ROADMAP

## Current Milestone: Phase 6 — Research and Capability Expansion (Active)

## Phase 0: Discovery and Baseline ✅
- [x] Inspect repository and environment
- [x] Hardware capability report
- [x] Baseline report and risk identification
- [x] Initial project scaffolding
- [x] Continuity documents created

## Phase 1: Reliable Local Foundation ✅
- [x] Reproducible project setup (pyproject.toml, .env, .venv)
- [x] Application configuration module
- [x] Logging infrastructure
- [x] Hardware detection utility
- [x] Model provider abstraction
- [x] Ollama backend integration
- [x] CLI entry point
- [x] Database initialization (SQLite)
- [x] Persistent conversations (SQLite-backed)
- [x] Basic API server (FastAPI)
- [x] Health check endpoint
- [x] End-to-end test: model answers a question (qwen2.5:7b on RTX 2000 Ada GPU)
- [x] Conversation survives application restart
- [x] Initial documentation complete

**Acceptance criterion:** Verified (40/40 tests passing, real model answers and conversation survives restart).

## Phase 2: Memory and Knowledge ✅
- [x] Persistent structured memory (CRUD)
- [x] Memory categories and metadata
- [x] Retrieval with relevance scoring
- [x] Memory inspection UI
- [x] Document indexing (embeddings)
- [x] Source provenance tracking
- [x] Backup and recovery
- [x] Deduplication and contradiction handling

**Acceptance criterion:** Verified (Information saved with 768-dim vector embeddings, retrieved in later sessions via hybrid search, inspected via Web UI, consolidated, backed up, and deleted).

## Phase 3: Tools and Task Orchestration ✅
- [x] Typed tool registry
- [x] Permission model
- [x] Bounded planning and execution
- [x] Task states and error recovery
- [x] Execution logging
- [x] Result verification

**Acceptance criterion:** Verified (AI decomposes goals with local model, executes safe bounded steps using real local tools, verifies disk artifacts with TaskVerifier, and records full audit log in SQLite).

## Phase 4: Evaluation and Feedback ✅
- [x] Fixed baseline benchmarks (Reasoning, Coding, Tools v1)
- [x] Task outcome recording (eval_benchmarks, eval_cases, eval_runs, eval_results)
- [x] Failure analysis and error taxonomy classification
- [x] Regression test suite and automated baseline comparison
- [x] Structured reporting (Markdown, JSON, summary metrics)
- [x] Web dashboard integration (Evaluation & Benchmarks tab)

**Acceptance criterion:** Verified (The platform executes repeatable evaluation benchmarks against a local model, classifies failure modes into an actionable error taxonomy, detects regressions against historical baselines, and outputs structured comparison reports).

## Phase 5: Dataset and Fine-Tuning Infrastructure ✅
- [x] Dataset management with versioning (multi-format normalization: Alpaca, ShareGPT, Chat, Completion)
- [x] Data quality pipeline (SHA-256 deduplication, length filtering, AST syntax verification, and 8-gram benchmark contamination guard)
- [x] Experiment tracking (LoRA hyperparameters, SQLite metrics logging, and step loss trajectory)
- [x] LoRA/QLoRA training workflow (VRAM-bounded execution loop under 4.8 GB, callback reporting)
- [x] Checkpoint management (best checkpoint selection, Ollama Modelfile synthesis, local model registration)
- [x] Baseline comparison and rollback (automated rollback restoration to baseline model upon regression detection)
- [x] Web dashboard integration (Datasets & Fine-Tuning tab with ingestion modal, launch form, loss table, and deploy buttons)

**Acceptance criterion:** Verified (The platform curates and preprocesses multi-format datasets, defends against evaluation contamination, executes VRAM-bounded LoRA adaptation, logs loss trajectories, exports Ollama Modelfiles, and triggers automatic rollback when regressions are detected).

## Phase 6: Research and Capability Expansion ⏳
- [ ] Improved inference (vLLM, TensorRT-LLM)
- [ ] Stronger models as hardware permits
- [ ] Multimodal capabilities
- [ ] Advanced retrieval (hybrid search)
- [ ] Coding-specific skills

## Phase 7: Hardware and Deployment Evolution ⏳
- [ ] Multi-GPU support
- [ ] Distributed training
- [ ] Separate training/inference processes
- [ ] Workstation deployment guide
