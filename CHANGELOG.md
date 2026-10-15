# CHANGELOG

All notable changes to Project Genesis will be documented in this file.

## [0.11.0] — 2026-10-15

### Added
- **Multi-Modal Vision Processing, Cross-Campaign Meta-Memory Knowledge Graph, and Autonomous Research Daemon (Phase 10)**:
  - **Multi-Modal Vision & Visual Artifact Inspector**: Local parsing and validation engine (`VisionRequest`, `VisionResponse`, `VisionInspector`) decoding raster headers (PNG IHDR, GIF, BMP, JPEG) and SVG viewports/text nodes without cloud dependencies. SQLite audit persistence into `visual_inspections` table (`genesis/models/vision.py`).
  - **Cross-Campaign Meta-Memory Knowledge Graph**: Relational and semantic graph engine (`KnowledgeNode`, `KnowledgeEdge`, `Subgraph`, `MetaMemoryGraph`) supporting BFS shortest-path queries (`find_path`), neighborhood traversal, depth-bounded subgraph extraction, packed 768-dim vector embedding semantic search (`semantic_search_nodes`), and campaign artifact graph ingestion (`sync_from_campaign`) (`genesis/memory/graph.py`).
  - **Autonomous Background Research Daemon**: Continuous closed-loop scientific research engine (`ResearchPaper`, `AutonomousResearchDaemon`) executing 5-stage cycles: hypothesis selection &rarr; dialectic consensus debate &rarr; sandboxed prototype verification (`CodeSandbox`) &rarr; formal IMRaD scientific paper synthesis &rarr; graph and memory persistence (`genesis/research/daemon.py`).
  - **Database Schema V10**: Added `knowledge_nodes`, `knowledge_edges`, `research_papers`, and `visual_inspections` tables in SQLite WAL mode with index coverage and migration logic (`genesis/core/database.py`).
  - **Phase 10 REST APIs**: Mounted `/api/graph/*` (nodes, edges, path, subgraph, export, sync), `/api/multimodal/*` (inspect, inspections), and `/api/research/daemon/*` (cycle, papers) in `genesis/api/app.py`.
  - **Interactive Web Dashboard Module**: Dedicated "Meta-Graph & Papers" navigation tab, Knowledge Graph topology explorer with node filtering, pathfinder, and subgraph extractor, Autonomous Publications manager with IMRaD reader, and Multi-Modal Visual Artifact Inspector (`genesis/web/`).
  - **Architecture Decision Record**: ADR-011 Multi-Modal Vision Inspection, Cross-Campaign Meta-Memory Knowledge Graph, and Autonomous Background Research Daemon.
  - **Comprehensive Test Suite**: 17 new unit and API tests across vision inspection, knowledge graph, daemon, and API routes, expanding repository baseline to 228 passing tests (100% pass rate).

## [0.10.0] — 2026-10-15

### Added
- **Multi-Agent Specialization, Dialectic Consensus Debates, and Autonomous Research Campaigns (Phase 9)**:
  - **Role-Specialized Cognitive Agent Ensembles**: Decomposition of the reasoning pipeline into specialized personas with distinct cognitive biases, task boundaries, and model dispatch requirements (`Researcher`, `Architect`, `Implementer`, `Reviewer`, `Synthesizer`). Each agent dynamically resolves its optimal local model via `ModelRouter` (`genesis/tasks/agents/`).
  - **Dialectic Consensus Debate Engine**: Multi-round structured debate protocol (`ConsensusEngine`, `DebateSession`, `DebateTurn`) coordinating iterative proposals, adversarial critiques, implementation rebuttals, consensus synthesis, and dissenting opinion extraction (`genesis/tasks/consensus.py`).
  - **Autonomous Long-Horizon Research Campaigns**: Phased research campaign orchestration (`CampaignManager`, `ResearchCampaign`, `ResearchMilestone`) guiding open-ended objectives across 4 milestones: hypothesis formulation &rarr; multi-agent consensus debate &rarr; code prototyping in `SafeCodeSandbox` &rarr; findings consolidation into structured memory (`genesis/research/campaigns.py`).
  - **Database Schema V9**: Added `agent_teams`, `debate_sessions`, `debate_turns`, and `research_campaigns` relational tables in SQLite WAL mode with index coverage and default `team-genesis-core` seed insertion (`genesis/core/database.py`).
  - **Multi-Agent REST API**: Mounted under `/api/agents`: `/api/agents/roles`, `/api/agents/teams`, `/api/agents/debate/run`, `/api/agents/debates`, `/api/agents/debates/{id}`, `/api/agents/campaigns`, `/api/agents/campaigns/{id}`, `/api/agents/campaigns/{id}/advance` (`genesis/api/routes/agents.py`).
  - **Interactive Web Dashboard Module**: Dedicated "Multi-Agent & Campaigns" tab (`#pane-agents`), active team roster grid, dialectic consensus arena with color-coded turn streams and verdict banners, long-horizon campaign roadmap board with interactive milestone steppers, and modals (`#debate-modal`, `#campaign-modal`) (`genesis/web/`).
  - **Architecture Decision Record**: ADR-010 Multi-Agent Specialization, Dialectic Consensus Debates, and Autonomous Long-Horizon Research Campaigns.
  - **Comprehensive Test Suite**: 16 new unit and integration tests across agents, consensus engine, campaign manager, and agent REST API, expanding repository baseline to 211 passing tests (100% pass rate).

## [0.9.0] — 2026-10-15

### Added
- **Autonomous Meta-Improvement and Continuous Evolution Subsystem (Phase 8)**:
  - **Dynamic Task Router & Multi-Model Dispatch**: Automatic intent classification across five task categories (`coding`, `reasoning`, `fast_validation`, `embedding`, `general`) with compiled regex heuristics. Evaluates live physical VRAM headroom using `DeviceOrchestrator` (`detect_gpu`) and dynamically dispatches workloads to specialized models (`qwen2.5-coder:7b`, `qwen2.5:7b`, `qwen2.5-coder:1.5b-base`, `nomic-embed-text`) with transparent fallback handling (`genesis/models/router.py`).
  - **Automated Benchmark Expansion from Edge-Case Mining**: Mining failed execution traces from `code_experiments` and `eval_results` tables, extracting syntax and runtime failure modes, and synthesizing deterministic `EvalCase` instances with automated name collision disambiguation (`genesis/evaluation/expansion.py`).
  - **Closed-Loop Autonomous Evolution Orchestrator**: 5-stage closed-loop pipeline (`MetaResearchOrchestrator` in `genesis/research/meta.py`): diagnostic evaluation against target benchmarks &rarr; synthetic remediation dataset curation &rarr; VRAM-bounded LoRA fine-tuning under 4.8 GB &rarr; regression defense comparison &rarr; deployment or automated rollback upon detected accuracy degradation.
  - **Meta REST API**: Endpoints mounted under `/api/meta`: `/api/meta/cycles/run`, `/api/meta/cycles`, `/api/meta/cycles/{id}`, `/api/meta/route`, `/api/meta/routes`, `/api/meta/routes/{category}`, `/api/meta/benchmark/expand` (`genesis/api/routes/meta.py`).
  - **Interactive Web Dashboard Module**: Dedicated "Meta-Evolution" tab (`#pane-meta`) and modal (`#meta-cycle-modal`) with live cycle history, real-time router playground with routing rationale feedback, model route configuration table, and benchmark expansion controls (`genesis/web/`).
  - **Database Schema V8**: Added `meta_cycles` and `model_routes` tables with indexes and default route seed insertions (`genesis/core/database.py`).
  - **Architecture Decision Record**: ADR-009 Autonomous Meta-Improvement Cycles, Dynamic Multi-Model Dispatch, and Automated Benchmark Expansion.
  - **Comprehensive Test Suite**: 24 new unit and API tests across router, benchmark expansion, meta cycles, and REST API, expanding the total repository baseline to 195 passing tests (100% pass rate).

## [0.8.0] — 2026-10-15

### Added
- **Hardware and Deployment Evolution Subsystem (Phase 7)**:
  - **Multi-Device Hardware Topology Orchestrator**: Dynamic hardware discovery for physical NVIDIA CUDA GPUs, host CPUs, and system RAM with active compute backend detection (`genesis/core/device.py`).
  - **Multi-Device Layer Distribution Planner**: Dynamic layer offload calculation supporting single GPU (`single_gpu`), pipeline-parallel multi-GPU (`pipeline_parallel`), and CPU layer offloading (`cpu_offload`) with strict 1,024 MB safety reserve and KV-cache modeling to prevent CUDA out-of-memory errors on high-parameter models ($\ge 14\text{B}$).
  - **Decoupled Work Job Queue**: Persistent SQLite WAL-mode queue supporting priority dispatch, atomic lease-based locking, worker heartbeats, progress tracking, retry limits, and automated orphan crash recovery (`genesis/core/queue.py`).
  - **Standalone Job Worker Daemon**: Background worker loop leasing jobs, emitting live heartbeats, and dispatching handlers for fine-tuning, benchmarks, code synthesis, and knowledge research (`genesis/tasks/worker.py`).
  - **Autonomous Tool Orchestration**: Built-in `CodeSynthesizerTool` (`synthesize_code`) and `KnowledgeSynthesizerTool` (`synthesize_knowledge`) registered in `ToolRegistry`, allowing `TaskPlanner` and `TaskExecutor` to deploy reflective code synthesis and research briefs as native bounded task steps (`genesis/tools/builtin/`).
  - **Deployment REST API**: Endpoints under `/api/deployment`: `/api/deployment/topology`, `/api/deployment/layer-plan`, `/api/deployment/jobs`, `/api/deployment/jobs/{id}`, `/api/deployment/jobs/{id}/cancel` (`genesis/api/routes/deployment.py`).
  - **CLI Worker Command**: `genesis worker` CLI command to start the decoupled background worker daemon (`genesis/cli.py`).
  - **Interactive Web Dashboard Module**: Dedicated "Deployment & Topologies" tab with real-time compute device inventory, interactive layer offload distribution calculator, and live worker job queue monitor (`genesis/web/`).
  - **Database Schema V7**: Introduced `device_topologies` and `work_jobs` tables with status, priority, and worker indexes (`genesis/core/database.py`).
  - **Workstation Deployment Guide**: Documented architecture topology, process separation, background service management, and layer offload planning in `docs/deployment.md`.
  - **Architecture Decision Record**: ADR-008 Multi-Device Hardware Topology Orchestration and Decoupled Background Worker Queue.
  - **Comprehensive Test Suite**: 21 new unit, API, and E2E integration tests across device topology, job queue, worker daemon, deployment API, and end-to-end task execution, expanding total repository baseline to 171 passing tests (100% pass rate).

## [0.7.0] — 2026-10-15


### Added
- **Research & Capability Expansion Subsystem (Phase 6)**:
  - **Hardware-Aware Context Management & KV-Cache Budgeting**: Exact FP16 attention memory modeling ($2 \times L \times H_{kv} \times D_{head} \times 2$ bytes), dynamic headroom validation against RTX 2000 Ada 8 GB VRAM, token estimation, clean boundary truncation, and `compact_dialogue` ensuring system prompt and latest user turn retention (`genesis/models/context.py`).
  - **Inference Latency & Throughput Profiler**: Empirical execution profiling measuring Time-To-First-Token (TTFT), inter-token latency (ITL), generation throughput (tokens/second), total latency, and live GPU VRAM / KV-cache allocation with SQLite persistence (`genesis/models/profiler.py`).
  - **Subprocess CodeSandbox**: Subprocess execution isolation in temporary workspaces with bounded timeouts (default: 8.0s), exit code tracking, stdout/stderr capture, and automated failure classification (`syntax_error`, `assertion_failure`, `timeout`, `runtime_error`) (`genesis/research/coding.py`).
  - **Reflective Code Synthesizer with Self-Repair**: Autonomous decomposition of coding objectives, generation of implementations and unit test harnesses, sandbox verification, and multi-round reflective self-repair using execution tracebacks.
  - **Autonomous Self-Improvement Loop**: Option to export verified code experiments directly to Phase 5 fine-tuning datasets (`data/datasets/<id>.jsonl`) as verified instruction pairs for subsequent LoRA adaptation.
  - **Adaptive Multi-Source Knowledge Synthesis**: Multi-source retrieval across indexed document passages and semantic memory, prompt grounding with citation tags (`[1]`, `[2]`), deterministic citation verification, groundedness confidence scoring, and automatic consolidation into persistent memory under category `research_brief` (`genesis/research/synthesis.py`).
  - **Database Schema V6**: Introduced `research_profiles`, `code_experiments`, and `research_synthesis` tables with indexes and migration tracking (`genesis/core/database.py`).
  - **REST API Endpoints**: `/api/research/context-budget`, `/api/research/profile`, `/api/research/profiles`, `/api/research/coding/synthesize`, `/api/research/coding/experiments`, `/api/research/synthesize`, `/api/research/syntheses` (`genesis/api/routes/research.py`).
  - **Interactive Web Dashboard Research Tab**: Live KV-cache calculator, one-click inference profiler, code synthesis runner with sandbox test and traceback visualizer, knowledge brief generator, and historical logs (`genesis/web/`).
  - **Architecture Decision Record**: ADR-007 Hardware-Aware Context Management, Inference Profiling, Reflective Code Synthesis, and Adaptive Knowledge Synthesis.
  - **Comprehensive Test Suite**: 19 new unit, API, and E2E integration tests (`test_context_profiler.py`, `test_research_coding.py`, `test_research_synthesis.py`, `test_api_research.py`, `test_research_e2e.py`), expanding total repository test baseline to 149 passing tests (100% pass rate).

## [0.6.0] — 2026-10-15

### Added
- **Dataset and Fine-Tuning Infrastructure (Phase 5)**:
  - **Multi-Format Dataset Management**: Full lifecycle ingestion, normalization, and stats tracking for Alpaca (`instruction`), OpenAI/ShareGPT (`chat`), prompt-completion (`completion`), and raw text (`raw`) schemas (`genesis/training/datasets.py`).
  - **Deterministic Dataset Splitting**: Seeded partitioning into train/validation/test splits with checksum tracking and JSONL disk persistence.
  - **Quality Preprocessing & Contamination Guard**: Five-stage filtering pipeline featuring exact SHA-256 deduplication, near-duplicate Jaccard shingling ($\ge 0.90$), min/max token length boundaries, AST Python code syntax validation (`ast.parse`), and 8-gram sequence matching against evaluation benchmark test cases (`genesis/training/preprocessing.py`).
  - **Experiment & Loss Trajectory Tracking**: LoRA/QLoRA hyperparameter configuration ($r=8, 16, 32$, alpha, dropout, learning rate with linear decay, batch size, gradient accumulation, 4-bit quantization) and step-by-step metrics logging in SQLite (`genesis/training/experiments.py`).
  - **Fine-Tuning Execution Engine**: `FineTuningTrainer` orchestrating dataset validation, step calculation, async training execution, simulated loss optimization, live callbacks, and periodic checkpoint emission bounded under 4.8 GB VRAM (`genesis/training/trainer.py`).
  - **Checkpoint Management & Rollback Defense**: Auto-detection of best checkpoints (lowest eval loss), Ollama `Modelfile` generation, candidate model registration via `ollama create`, and automated rollback to verified baseline models upon regression detection (`genesis/training/export.py`).
  - **REST API Endpoints**: `/api/training/datasets`, `/api/training/datasets/create`, `/api/training/datasets/{id}`, `/api/training/datasets/{id}/preprocess`, `/api/training/experiments`, `/api/training/experiments/create`, `/api/training/experiments/{id}`, `/api/training/experiments/{id}/start`, `/api/training/experiments/{id}/checkpoints`, `/api/training/checkpoints/{id}/deploy`, `/api/training/experiments/{id}/rollback` (`genesis/api/routes/training.py`).
  - **Interactive Web Dashboard Module**: Dedicated "Datasets & Fine-Tuning" tab with dataset ingestion modal, preprocessing reports, experiment configuration, step loss trajectory table, checkpoint deployment, and rollback controls (`genesis/web/`).
  - **Database Schema V5**: Added `training_datasets`, `training_experiments`, and `training_checkpoints` tables with foreign key constraints, cascading deletes, and optimized indexes (`genesis/core/database.py`).
  - **Architecture Decision Record**: ADR-006 Multi-Format Dataset Curation, Contamination Defense, VRAM-Bounded LoRA Fine-Tuning, and Rollback Orchestration.
  - **Comprehensive Test Suite**: 19 new unit, API, and E2E integration tests (`test_training_datasets.py`, `test_training_preprocessing.py`, `test_training_experiments.py`, `test_api_training.py`, `test_training_e2e.py`), bringing the total repository test suite to 130 passing tests (100% pass rate).

## [0.5.0] — 2026-10-15

### Added
- **Evaluation and Feedback Subsystem (Phase 4)**:
  - Standard Baseline Benchmark Suites: Seeded multi-domain suites for Reasoning (`bm-reasoning-v1`), Python Coding (`bm-coding-v1`), and Tool Invocation (`bm-tools-v1`).
  - Deterministic Evaluation Rubrics: Support for exact match, substring inclusion, regular expressions, numeric tolerance, and structured tool invocation checks (`genesis/evaluation/benchmarks.py`).
  - Failure Analysis & Error Taxonomy: Automated failure categorization into `SYNTAX_ERROR`, `TOOL_MISUSE`, `TIMEOUT`, `FORMAT_ERROR`, `INCORRECT_ANSWER`, and `HALLUCINATION` with frequency distributions and actionable engineering recommendations (`genesis/evaluation/analysis.py`).
  - Automated Regression Detection: `RegressionDetector` comparing candidate runs against baselines, detecting score drops, tracking regressed test cases and resolved cases, monitoring latency spikes (> 50%), and issuing verdicts (`PASSED`, `IMPROVED`, `REGRESSED`, `BASELINE_ESTABLISHED`) (`genesis/evaluation/regression.py`).
  - Evaluation Reporter: Automated Markdown and JSON report generator with executive summaries, case breakdown tables, failure distributions, and diff comparisons (`genesis/evaluation/reports.py`).
  - REST Endpoints: `/api/evaluation/benchmarks`, `/api/evaluation/benchmarks/{id}`, `/api/evaluation/run`, `/api/evaluation/runs`, `/api/evaluation/runs/{id}`, `/api/evaluation/runs/{id}/report`, `/api/evaluation/compare` (`genesis/api/routes/evaluation.py`).
  - Interactive Web Dashboard Integration: Dedicated "Evaluation & Benchmarks" tab in `genesis/web/` featuring benchmark execution, history tracking, detailed failure inspection, and side-by-side regression comparison views.
  - Database Schema V4: Added `eval_benchmarks`, `eval_cases`, `eval_runs`, and `eval_results` tables with automatic migrations and foreign key cascades (`genesis/core/database.py`).
  - Architecture Decision Record: ADR-005 Repeatable Benchmarking, Failure Taxonomies, and Automated Regression Defense.
  - Comprehensive Test Suite: 23 new unit, integration, REST API, and live model E2E tests (`test_evaluation.py`, `test_regression.py`, `test_api_evaluation.py`, `test_evaluation_e2e.py`).

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
