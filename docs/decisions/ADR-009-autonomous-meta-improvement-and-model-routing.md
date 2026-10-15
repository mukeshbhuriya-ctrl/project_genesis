# ADR-009: Autonomous Meta-Improvement Cycles, Dynamic Multi-Model Dispatch, and Automated Benchmark Expansion

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis is a local-first, self-improving AI research and model development platform operating autonomously on Windows workstations equipped with local hardware accelerators (NVIDIA RTX 2000 Ada, 8 GB VRAM). Following Phase 7 (heterogeneous hardware topology orchestration, layer distribution planning, decoupled background worker queue, and native tool integration), Phase 8 implements the ultimate core capability of Project Genesis: **autonomous meta-improvement and closed-loop model evolution**.

Prior to Phase 8, the system possessed individual subsystems for evaluation (`BenchmarkRunner`, `RegressionDetector`), dataset curation (`DatasetManager`, `DataPreprocessor`), fine-tuning (`FineTuningTrainer`, `ExperimentTracker`), and inference (`ModelProvider`). However, these operations required manual coordination. To realize a truly self-improving AI platform, three critical capabilities were required:
1. **Dynamic Task Routing & Multi-Model Dispatch**: Different task domains require specialized model architectures. Coding synthesis requires `qwen2.5-coder:7b`; deep multi-step reasoning benefits from `qwen2.5:7b`; fast syntax validation can run on lightweight `qwen2.5-coder:1.5b-base`; semantic memory indexing utilizes `nomic-embed-text`. Dispatches must dynamically assess intent, consult configured route priorities, and verify available physical VRAM headroom using hardware diagnostics before dispatching workloads, falling back to lighter models if headroom is exhausted.
2. **Automated Benchmark Expansion from Failure Mining**: Fixed benchmarks suffer from ceiling effects and fail to reflect novel edge cases encountered during live agent execution. The platform must inspect failed code sandbox executions and evaluation cases, diagnose root failure modes, synthesize deterministic test cases, and automatically expand benchmark suites.
3. **Closed-Loop Autonomous Evolution Orchestration**: The platform must autonomously coordinate the entire end-to-end development cycle without human intervention: run diagnostic evaluations &rarr; isolate performance gaps &rarr; curate synthetic remediation datasets &rarr; execute VRAM-bounded LoRA fine-tuning &rarr; evaluate candidate checkpoints against baseline benchmarks &rarr; enforce automated regression defense &rarr; promote passing models to the local runtime or roll back regressions.

## Decisions

### 1. Database Schema Version 8 Migrations (`genesis/core/database.py`)
Schema Version 8 introduces two relational tables operating in SQLite WAL mode:
- `meta_cycles`: Tracks autonomous research and evolution cycles (`id`, `name`, `benchmark_id`, `model_id`, `status`, `baseline_score`, `candidate_score`, `verdict`, `dataset_id`, `experiment_id`, `checkpoint_id`, `stages`, `created_at`, `completed_at`, `metadata`). Indexes: `idx_meta_cycles_status`, `idx_meta_cycles_benchmark`.
- `model_routes`: Configures intent-driven routing policies (`task_category`, `primary_model`, `fallback_model`, `min_vram_gb`, `created_at`, `updated_at`, `metadata`). Pre-seeded with production defaults:
  - `coding`: `qwen2.5-coder:7b` (fallback: `qwen2.5:7b`, min VRAM: 5.0 GB)
  - `reasoning`: `qwen2.5:7b` (fallback: `qwen2.5-coder:7b`, min VRAM: 5.0 GB)
  - `fast_validation`: `qwen2.5-coder:1.5b-base` (fallback: `qwen2.5:7b`, min VRAM: 1.8 GB)
  - `embedding`: `nomic-embed-text` (fallback: `nomic-embed-text`, min VRAM: 0.8 GB)
  - `general`: `qwen2.5:7b` (fallback: `qwen2.5-coder:7b`, min VRAM: 4.5 GB)

### 2. Dynamic Task Router & Multi-Model Dispatch (`genesis/models/router.py`)
- Analyzes prompts using compiled regular expression pattern matchers and token heuristics across five task categories (`coding`, `reasoning`, `fast_validation`, `embedding`, `general`).
- Queries active routes from SQLite and inspects local hardware state via `DeviceOrchestrator` (`detect_gpu`).
- Evaluates real-time free VRAM against `min_vram_gb`. If VRAM headroom is constrained or the primary model is uninstalled in the local Ollama daemon, transparently selects the fallback model with an explicit audit rationale.
- Exposes complete CRUD operations (`get_routes`, `get_route`, `set_route`, `route_task`).

### 3. Automated Benchmark Expansion (`genesis/evaluation/expansion.py`)
- **Failure Edge-Case Mining**: Scans `code_experiments` and `eval_results` tables for failed traces, extracting syntax errors, runtime exceptions, assertion mismatches, and timeout conditions.
- **Deterministic Case Synthesis**: Generates targeted `EvalCase` instances with exact prompt instructions, expected outputs, and grading types (`contains`, `exact_match`, `regex`).
- **Benchmark Enrichment**: Updates benchmark suites in SQLite (`expand_benchmark` and `create_expanded_suite`), preventing duplicate test names via automatic disambiguation suffixes.

### 4. Closed-Loop Autonomous Evolution Orchestrator (`genesis/research/meta.py`)
Executes an end-to-end 5-stage pipeline encapsulated in `MetaResearchOrchestrator`:
- **Stage 1 (Diagnosis)**: Executes baseline evaluation on the target benchmark suite, identifying failed cases and isolating performance bottlenecks.
- **Stage 2 (Remediation Curation)**: Curates a synthetic training dataset targeting identified failure modes, processes samples through `DataPreprocessor` (deduplication, syntax verification, contamination checks), and registers the dataset.
- **Stage 3 (VRAM-Bounded LoRA Fine-Tuning)**: Initializes an experiment via `ExperimentTracker` and launches parameter adaptation with 4-bit quantization (QLoRA) and gradient accumulation, guaranteeing execution within 8 GB VRAM.
- **Stage 4 (Regression Defense Verification)**: Evaluates the candidate checkpoint against the baseline benchmark suite using `RegressionDetector` to compare accuracy scores.
- **Stage 5 (Decision & Action)**:
  - **Promotion**: If candidate accuracy exceeds baseline and regression is within tolerance (`<= regression_threshold`), candidate is deployed to the local model registry.
  - **Rollback**: If candidate accuracy degrades beyond tolerance, the orchestrator triggers an automatic rollback defense, marking the cycle as `rollback` and preventing degraded weights from entering deployment.

### 5. Meta REST API (`genesis/api/routes/meta.py`)
- `POST /api/meta/cycles/run`: Launches closed-loop evolution cycles with configurable training steps and regression thresholds.
- `GET /api/meta/cycles`: Lists historical evolution cycles with status and score deltas.
- `GET /api/meta/cycles/{id}`: Retrieves comprehensive telemetry and stage execution logs for a specific cycle.
- `POST /api/meta/route`: Dispatches a workload to the optimal local model given task prompt and category.
- `GET /api/meta/routes`: Lists configured model routing policies.
- `PUT /api/meta/routes/{category}`: Updates routing parameters for a category.
- `POST /api/meta/benchmark/expand`: Triggers automated benchmark expansion from mined failure edge cases.

### 6. Web Dashboard Evolution & Dispatch UI (`genesis/web/`)
- Added dedicated **"Meta-Evolution"** tab (`#pane-meta`) to the web interface.
- Features real-time evolution cycle history table with verdict badges, an interactive model routing playground with instant routing explanations, a route policy configuration table, benchmark expansion triggers with live synthesized case display, and a cycle launch modal (`#meta-cycle-modal`).

## Consequences

### Positive
- Fully autonomous self-improvement: Project Genesis can diagnose weaknesses, generate training data, tune adapters, and verify improvements without developer intervention.
- Strict regression defense prevents model degradation from automated fine-tuning runs.
- Workloads are automatically routed to optimal domain-specialized models, maximizing coding and reasoning performance while honoring workstation VRAM boundaries.
- Continuous evaluation: Benchmark suites expand dynamically as new failure modes are encountered, preventing static benchmark saturation.

### Negative
- Full closed-loop cycles require non-trivial compute time for LoRA training and dual benchmark evaluation passes.
- Expanded benchmark suites increase subsequent evaluation durations.

## Compliance
- Strict local-first architecture: Zero external API calls or network egress.
- Hardware constraints: All training and inference steps remain strictly bounded within 8 GB VRAM.
