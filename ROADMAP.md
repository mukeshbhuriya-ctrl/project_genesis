# ROADMAP

## Current Milestone: Phase 14 Complete ✅ → Phase 15: Cross-Modal Synthetic World Models & Continuous Latent Dream Exploration (Next)


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

## Phase 9: Multi-Agent Specialization & Long-Horizon Evolution ✅
- [x] Role-specialized autonomous agent ensembles (Researcher, Architect, Implementer, Reviewer, Synthesizer) with cognitive personas (`genesis/tasks/agents/`)
- [x] Multi-agent dialectic consensus debate protocol with adversarial critique, implementation defense, and consensus synthesis (`genesis/tasks/consensus.py`)
- [x] Autonomous long-horizon research campaigns with 4-stage workflows (hypothesis, consensus, experiment, memory consolidation) (`genesis/research/campaigns.py`)
- [x] SQLite Schema Version 9 relational persistence (`agent_teams`, `debate_sessions`, `debate_turns`, `research_campaigns`) with pre-seeded `team-genesis-core` (`genesis/core/database.py`)
- [x] Multi-Agent & Campaign REST API mounted at `/api/agents` (`genesis/api/routes/agents.py`)
- [x] Interactive Web Dashboard UI: Multi-Agent & Campaigns tab (`#pane-agents`), active team roster, dialectic debate arena, and long-horizon campaign roadmap board (`genesis/web/`)
- [x] Architecture Decision Record ADR-010 (`docs/decisions/ADR-010-multi-agent-specialization-consensus-debates-and-long-horizon-campaigns.md`)
- [x] 211/211 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform coordinates specialized cognitive agent ensembles, runs multi-round dialectic consensus debates synthesizing agreement and logging dissenting views, and advances multi-stage research campaigns from hypothesis to sandboxed execution and memory consolidation).

## Phase 10: Multi-Modal Integration & Autonomous Self-Expansion ✅
- [x] Multi-modal sensory processing and visual artifact inspection for diagrams, UI layouts, and loss curves (`genesis/models/vision.py`)
- [x] Cross-campaign meta-memory knowledge graph with BFS pathfinding, subgraph extraction, and vector similarity search (`genesis/memory/graph.py`)
- [x] Autonomous background research daemon with closed-loop hypothesis-to-IMRaD paper synthesis (`genesis/research/daemon.py`)
- [x] SQLite Schema Version 10 migrations (`knowledge_nodes`, `knowledge_edges`, `research_papers`, `visual_inspections`) (`genesis/core/database.py`)
- [x] Phase 10 REST APIs mounted at `/api/graph`, `/api/multimodal`, `/api/research/daemon` (`genesis/api/app.py`)
- [x] Interactive Web Dashboard UI: Meta-Graph & Papers tab, Knowledge Graph topology explorer, publications manager, and multimodal artifact inspector (`genesis/web/`)
- [x] Architecture Decision Record ADR-011 (`docs/decisions/ADR-011-multimodal-inspection-knowledge-graphs-and-autonomous-research-daemon.md`)
- [x] 228/228 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform autonomously parses and validates visual artifacts, weaves cross-campaign concepts, debates, and experiments into a searchable meta-memory knowledge graph, and continuously runs closed-loop scientific research cycles publishing peer-structured IMRaD papers).

## Phase 11: Real-Time Self-Compilation & Distributed Fleet Synchronization ✅
- [x] Host compiler discovery (`clang`, `gcc`, `cl`, `rustc`) and dynamic C/Rust extension synthesis (`genesis/core/compiler.py`)
- [x] Zero-overhead pure Python fallback kernel registration for systems without native toolchains
- [x] Built-in canonical hot-path tensor acceleration kernels: `vector_dot_product`, `fast_quantize_int8`, `matrix_vector_mul`
- [x] Automated benchmark profiling and SQLite speedup factor tracking
- [x] Distributed peer-to-peer fleet node discovery, hardware detection, and heartbeat tracking (`genesis/deployment/fleet.py`)
- [x] Incremental delta bundle export with SHA-256 integrity checksums
- [x] Timestamp-based conflict-resolving delta bundle import with strict foreign key defense
- [x] Immutable replication audit logging in `fleet_sync_logs`
- [x] SQLite Schema Version 11 migrations (`compiled_extensions`, `fleet_nodes`, `fleet_sync_logs`) (`genesis/core/database.py`)
- [x] Phase 11 REST API routes mounted under `/api/compiler` and `/api/fleet` (`genesis/api/app.py`)
- [x] Interactive Web Dashboard UI: Fleet & Compilers tab (`#pane-fleet`), toolchain status cards, kernel benchmark triggers, cluster topology grid, and delta import/export modals (`genesis/web/`)
- [x] Architecture Decision Record ADR-012 (`docs/decisions/ADR-012-native-extension-self-compilation-and-distributed-fleet-synchronization.md`)
- [x] 241/241 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform dynamically detects host compilers, compiles hot-path tensor routines with transparent Python fallbacks and benchmark profiling, coordinates distributed fleet nodes with heartbeat tracking, and performs air-gapped or network delta bundle synchronization with timestamp conflict resolution).

## Phase 12: Autonomous Kernel Synthesis & Cross-Fleet Model Merging ✅
- [x] Automated LLM-driven C/SIMD kernel optimization, numerical tolerance validation, and reflective repair loop (`genesis/research/kernel_synthesis.py`)
- [x] Cross-workstation parameter merging (Linear, SLERP, TIES, DARE) for distributed LoRA adapters and model weights (`genesis/training/merging.py`)
- [x] Active interactive human-in-the-loop steering controller for operator directives (`genesis/research/steering.py`)
- [x] SQLite Schema Version 12 migrations (`kernel_experiments`, `model_merges`, `steering_events`) (`genesis/core/database.py`)
- [x] Phase 12 REST API routes mounted under `/api/kernel-synthesis`, `/api/merging`, and `/api/steering` (`genesis/api/app.py`)
- [x] Interactive Web Dashboard UI: Autonomous Kernel Synthesis runner, C code modal, Model Merging controls with density slider, and Operator Steering Console (`genesis/web/`)
- [x] Architecture Decision Record ADR-013 (`docs/decisions/ADR-013-autonomous-kernel-synthesis-and-cross-fleet-model-merging.md`)
- [x] 254/254 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform autonomously synthesizes, validates against ground-truth math, and reflectively repairs C/SIMD tensor acceleration kernels, performs cross-fleet parameter merging via SLERP, TIES, DARE, and Linear methods, and exposes live human-in-the-loop steering controls for operator oversight).

## Phase 13: Continual Online Learning & Neuro-Symbolic Verification Engine ✅
- [x] Streaming prioritized experience replay buffer with domain-balanced rehearsal sampling (`genesis/training/continual.py`)
- [x] Elastic Weight Consolidation (EWC) parameter penalty estimation using diagonal Fisher Information matrices ($F_i$) to prevent catastrophic forgetting
- [x] Dynamic virtual soft prompt prefix generation and task prefix vector interpolation (`genesis/training/continual.py`)
- [x] Neuro-symbolic formal verification engine (`genesis/research/symbolic.py`):
  - AST static analysis: termination invariant checking, recursion base-case verification, safety/purity enforcement, division-by-zero detection
  - Continuous mathematical bounds verification: non-negativity, range constraint checking, and monotonicity verification
  - Propositional logic resolution refutation theorem proving
- [x] SQLite Schema Version 13 migrations (`replay_buffers`, `ewc_checkpoints`, `symbolic_verifications`) (`genesis/core/database.py`)
- [x] Phase 13 REST API routes mounted under `/api/continual` and `/api/symbolic` (`genesis/api/app.py`)
- [x] Interactive Web Dashboard UI: Continual Learning & Experience Replay panel in Training tab, EWC Checkpoints table, Neuro-Symbolic Verification console in Research tab, and interactive modals (`genesis/web/`)
- [x] Architecture Decision Record ADR-014 (`docs/decisions/ADR-014-continual-online-learning-and-neuro-symbolic-verification.md`)
- [x] 266/266 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform mitigates catastrophic forgetting via streaming prioritized experience replay and EWC quadratic Fisher penalty regularization, provides parameter-efficient virtual prompt prefix tuning, and deterministically verifies code termination invariants, continuous mathematical bounds, and propositional logic hypotheses).

## Phase 14: Autonomous Recursive Self-Architecting & Hardware-Targeted Micro-Quantization ✅
- [x] Autonomous neural architecture search (NAS) & block pruning (attention head pruning, intermediate dimension scaling) for target VRAM profiles (`genesis/models/pruning.py`)
- [x] Hardware-targeted micro-quantization (2-bit/3-bit/4-bit mixed-precision AWQ with salient outlier channel protection) optimized for RTX 2000 Ada Tensor Cores (`genesis/models/quantization.py`)
- [x] Self-speculative decoding engine with draft model generation and parallel speculative verification for ultra-low latency local inference (`genesis/models/speculative.py`)
- [x] SQLite Schema Version 14 migrations (`pruning_experiments`, `quantization_profiles`, `speculative_runs`) (`genesis/core/database.py`)
- [x] Phase 14 REST API routes mounted under `/api/pruning`, `/api/quantization`, and `/api/speculative` (`genesis/api/app.py`)
- [x] Interactive Web Dashboard UI: Dedicated "NAS & Micro-Quant" navigation tab (`#pane-architect`), structural pruning experiment table, micro-quantization benchmarking card, self-speculative decoding session manager, and 3 interactive modal dialogs (`genesis/web/`)
- [x] Architecture Decision Record ADR-015 (`docs/decisions/ADR-015-autonomous-recursive-self-architecting-and-micro-quantization.md`)
- [x] 286/286 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform dynamically prunes transformer attention heads and SwiGLU dimensions to fit within strict VRAM budgets, quantizes model tensors to 2/3/4/8 bits with outlier channel preservation achieving >0.95 cosine similarity, and accelerates autoregressive inference via self-speculative draft proposal and parallel rejection sampling).

## Phase 15: Cross-Modal Synthetic World Models & Continuous Latent Dream Exploration ✅
- [x] Multimodal environment transition modeling: auto-regressive state-space prediction of code/system states from multi-step execution traces (`genesis/world_model/transition.py`)
- [x] Latent imagination and dream exploration: generative rollout of synthetic counterfactual scenarios with heuristic value estimation $V(s)$ and replay buffer export (`genesis/world_model/dream.py`)
- [x] Cross-modal representation alignment: unified embedding projections bridging symbolic AST graphs, visual inspection tokens, and textual reasoning traces (`genesis/world_model/alignment.py`)
- [x] Dynamic execution sandbox checkpointing with state-forking, SHA-256 state hashing, and rollback for tree-search exploration (`genesis/world_model/mcts.py`)
- [x] SQLite Schema Version 15 relational persistence (`world_transitions`, `dream_trajectories`, `mcts_search_runs`, `modal_alignments`) (`genesis/core/database.py`)
- [x] World Model & Latent Dreams REST API mounted at `/api/world` (`genesis/api/routes/world_model.py`)
- [x] Interactive Web Dashboard UI: World Model & Dreams tab (`#pane-world`), State Transitions inspector, Latent Dream Trajectories viewer, MCTS Tree Search monitor, and 3 interactive modals (`genesis/web/`)
- [x] Architecture Decision Record ADR-016 (`docs/decisions/ADR-016-cross-modal-synthetic-world-models-and-latent-dream-exploration.md`)
- [x] 311/311 comprehensive tests passing (100% pass rate)

**Acceptance criterion:** Verified (The platform accurately models environment state transitions and safety invariants before execution, simulates multi-turn counterfactual problem-solving dreams with branch pruning and replay buffer export, projects heterogeneous modalities into a shared normalized embedding space with semantic cosine retrieval, and executes speculative Monte Carlo Tree Search with atomic state rollbacks).

## Phase 16: Autonomous Metacognitive Compiler & Decentralized Swarm Consensus (Next) ⏳
- [ ] Metacognitive introspection & self-monitoring: continuous calibration of epistemic confidence, attention allocation, and cognitive load profiling
- [ ] Decentralized swarm consensus & multi-agent dialectics: Byzantine fault-tolerant consensus across peer model instances with dynamic leader election
- [ ] Self-hosting evolutionary compiler optimization: autonomous JIT compilation of internal Genesis routing algorithms and data structures
- [ ] Continuous multi-objective Pareto optimization: evolutionary search balancing latency, VRAM footprint, throughput, and accuracy across heterogeneous local hardware profiles



