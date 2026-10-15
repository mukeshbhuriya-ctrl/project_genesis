# ROADMAP

## Current Milestone: Phase 8 Complete ✅ → Phase 9: Multi-Agent Specialization & Long-Horizon Evolution (Next)


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

## Phase 6: Research and Capability Expansion ✅
- [x] Hardware-aware context management and dynamic dialogue compaction (`genesis/models/context.py`)
- [x] KV-cache VRAM budgeting and 8 GB VRAM headroom enforcement
- [x] Empirical inference profiling measuring TTFT, throughput (tokens/sec), and inter-token latency (`genesis/models/profiler.py`)
- [x] Subprocess Python CodeSandbox with strict timeout limits and error categorization (`genesis/research/coding.py`)
- [x] Reflective code synthesizer with multi-round automated self-repair loops
- [x] Closed-loop integration exporting verified code experiments directly to Phase 5 fine-tuning datasets
- [x] Adaptive multi-source knowledge synthesis with deterministic citation verification (`genesis/research/synthesis.py`)
- [x] Groundedness confidence scoring and automatic memory consolidation under `research_brief` category
- [x] SQLite schema version 6 (`research_profiles`, `code_experiments`, `research_synthesis` tables)
- [x] REST API endpoints (`/api/research/context-budget`, `/api/research/profile`, `/api/research/coding/synthesize`, `/api/research/synthesize`)
- [x] Interactive Web Dashboard Research tab (`genesis/web/`): live budget calculator, inference profiler, code sandbox tester, and brief synthesizer
- [x] Unit and end-to-end integration test suites in `tests/` covering all Phase 6 subsystems

**Acceptance criterion:** Verified (The platform profiles inference latency and KV-cache VRAM consumption, autonomously synthesizes and verifies Python code in an isolated sandbox with self-repair, exports verified samples into Phase 5 datasets, and executes multi-source citation-verified knowledge synthesis consolidated into memory).

## Phase 7: Hardware and Deployment Evolution ✅
- [x] Multi-GPU & heterogeneous compute device topology orchestration (`genesis/core/device.py`)
- [x] Multi-device layer distribution planner (`single_gpu`, `pipeline_parallel`, `cpu_offload`) with VRAM headroom and KV-cache safety modeling
- [x] Decoupled persistent work job queue with atomic lease-locking, heartbeats, and automated orphan crash recovery (`genesis/core/queue.py`)
- [x] Standalone background worker daemon for decoupled fine-tuning, benchmarks, and synthesis (`genesis/tasks/worker.py`)
- [x] Autonomous tool integration: registered `synthesize_code` and `synthesize_knowledge` as first-class tools for bounded task orchestrator (`genesis/tools/builtin/`, `genesis/tasks/planner.py`)
- [x] REST API endpoints (`/api/deployment/topology`, `/api/deployment/layer-plan`, `/api/deployment/jobs`, `/api/deployment/jobs/{id}/cancel`) (`genesis/api/routes/deployment.py`)
- [x] CLI worker entry point `genesis worker` (`genesis/cli.py`)
- [x] Interactive Web Dashboard Deployment tab: device inventory, layer distribution calculator, and live job queue monitor (`genesis/web/`)
- [x] Workstation deployment and process separation guide (`docs/deployment.md`)
- [x] Architecture Decision Record ADR-008
- [x] 171/171 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform discovers multi-device compute topologies and computes optimal layer offload plans, provides a persistent crash-resilient queue with lease-locking and background worker daemon, and integrates code synthesis and knowledge synthesis as first-class tools in the bounded task orchestrator).

## Phase 8: Autonomous Meta-Improvement and Continuous Evolution ✅
- [x] Dynamic task routing and multi-model dispatch (`genesis/models/router.py`)
- [x] Hardware-aware VRAM headroom safety checks and fallback model resolution (`genesis/core/device.py`)
- [x] Automated benchmark expansion from failure edge-case mining (`genesis/evaluation/expansion.py`)
- [x] Closed-loop autonomous research cycles: evaluation -> failure diagnosis -> synthetic remediation dataset curation -> LoRA fine-tuning -> regression defense -> deployment or rollback (`genesis/research/meta.py`)
- [x] Meta REST API mounted at `/api/meta` (`genesis/api/routes/meta.py`)
- [x] Interactive Web Dashboard Meta-Evolution tab and modal (`genesis/web/`)
- [x] Database Schema Version 8 migrations (`genesis/core/database.py`)
- [x] Architecture Decision Record ADR-009
- [x] 195/195 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform autonomously routes tasks to optimal specialized models with VRAM safety checks, expands benchmark test suites from runtime failure traces, and executes closed-loop evolution cycles that fine-tune adapters and enforce automatic rollback defenses upon regression).

## Phase 9: Multi-Agent Specialization & Long-Horizon Evolution ⏳
- [ ] Role-specialized autonomous agent teams (Researcher, Architect, Implementer, Reviewer)
- [ ] Asynchronous multi-turn debate and consensus protocols
- [ ] Autonomous long-horizon research campaign orchestration
- [ ] Self-updating knowledge base and continuous automated benchmark discovery

