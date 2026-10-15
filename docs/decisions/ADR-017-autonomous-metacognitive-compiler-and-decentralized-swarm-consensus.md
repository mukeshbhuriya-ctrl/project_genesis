# ADR-017: Autonomous Metacognitive Compiler, Decentralized Swarm Consensus, and Multi-Objective Pareto Frontier Optimization

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis operates locally on dedicated workstation hardware (Intel Core i9-13900H, 32 GB RAM, NVIDIA RTX 2000 Ada with 8,188 MB VRAM) under Windows 11 with zero external cloud dependencies.
As autonomous recursive capability expands through world models, mental rollouts, and multi-agent systems (Phases 10–15), four foundational challenges emerge for self-sustaining local AI research:

1. **Uncalibrated Model Confidence & Misallocated Reasoning**: Autoregressive models frequently express overconfidence on deceptive queries or underconfidence on straightforward tasks. Without epistemic uncertainty estimation and expected calibration error (ECE) tracking, the system cannot dynamically allocate reasoning compute (e.g. fast path vs. reflective repair vs. swarm escalation).
2. **Centralized Agent Vulnerability & Byzantine Failure**: As multiple local agent instances debate architectural refactorings and code generation, individual nodes can suffer from hallucinations, reasoning loops, or adversarial prompt injections. A decentralized Byzantine Fault Tolerant (BFT) consensus protocol with cryptographic proof-of-reasoning hash chaining is necessary to guarantee quorum-based decision validity without human intervention.
3. **Execution Overhead in Algorithmic Hot-Paths**: Core mathematical routines—such as dense vector cosine similarity in memory retrieval, UCB1 calculations in MCTS tree exploration, and non-dominated sorting in multi-objective optimization—are repeatedly executed in pure Python, creating execution bottlenecks. The platform requires a self-hosting JIT compiler that profiles, compiles to native C extensions with numerical equivalence guarantees ($< 1\text{e-}5$ error), and dynamically dispatches optimized kernels.
4. **Hardware-Constrained Multi-Objective Trade-offs**: Workstation deployment requires simultaneously optimizing conflicting objectives: minimizing latency, strictly bounding VRAM below the RTX 2000 Ada 8 GB ceiling, maximizing generation throughput (tokens/sec), and maintaining high task accuracy. Scalar loss functions fail to discover the non-dominated Pareto frontier across heterogeneous model architectures and quantization settings.

Phase 16 addresses these challenges directly through:
- **Metacognitive Epistemic Introspection**: Self-evaluating token perplexity, lexical complexity, expected calibration error (ECE), and dynamic reasoning budget allocation.
- **Decentralized Swarm Consensus & Proof-of-Reasoning**: Byzantine fault-tolerant multi-node voting with reputation updates, leader election, and SHA-256 cryptographic chain auditing.
- **Self-Hosting Evolutionary Native Compiler**: Automated hot-path compilation of internal routines with speedup benchmarking and numerical invariance verification.
- **Multi-Objective Continuous Pareto Optimizer**: Fast non-dominated sorting and crowding distance assignment (NSGA-II) for 4D hardware runtime configuration exploration.

---

## Decisions

### 1. Database Schema Version 16 Migrations (`genesis/core/database.py`)
Schema Version 16 introduces six relational tables in SQLite WAL mode:
- `metacognitive_evaluations`: Persists introspection assessments, confidence scores, perplexity, lexical complexity, routing decisions, and latency (`id`, `prompt`, `candidate_output`, `confidence`, `uncertainty`, `perplexity`, `lexical_complexity`, `decision`, `reasoning_budget_json`, `latency_ms`, `created_at`, `metadata`). Indexes: `idx_metacog_decision`, `idx_metacog_conf`.
- `swarm_nodes`: Registers peer swarm reasoning nodes with role specializations, reputations, and active status (`id`, `name`, `model_name`, `role`, `reputation`, `status`, `registered_at`, `last_active_at`, `metadata`). Index: `idx_swarm_nodes_status`.
- `swarm_consensus_sessions`: Stores consensus rounds, leader node, proposals, votes, quorum ratio, verdict, and proof-of-reasoning SHA-256 hash chains (`id`, `task_prompt`, `leader_id`, `rounds`, `quorum_ratio`, `confidence`, `verdict`, `proof_hash`, `session_data_json`, `status`, `created_at`, `metadata`). Indexes: `idx_swarm_sessions_status`, `idx_swarm_sessions_verdict`.
- `self_compiled_targets`: Tracks compiled C/SIMD hot-path targets, compiler toolchain, speedup factors, numerical errors, and library checksums (`name`, `module_path`, `c_source_path`, `binary_path`, `status`, `compiler_used`, `speedup_factor`, `max_numerical_error`, `checksum`, `compiled_at`, `metadata`). Index: `idx_self_compiled_status`.
- `pareto_experiments`: Records multi-objective search runs, objective dimensions, population size, generations, and frontier sizes (`id`, `name`, `population_size`, `generations`, `frontier_size`, `status`, `best_compromise_json`, `created_at`, `metadata`). Index: `idx_pareto_experiments_status`.
- `pareto_frontiers`: Stores non-dominated configuration points (rank 0) mapping model configurations to latency, VRAM, throughput, and accuracy metrics (`id`, `experiment_id`, `config_json`, `latency_ms`, `vram_mb`, `throughput_tps`, `accuracy_score`, `crowding_distance`, `rank`, `created_at`, `metadata`). Indexes: `idx_pareto_frontiers_exp`, `idx_pareto_frontiers_rank`.

### 2. Metacognitive Epistemic Introspector (`genesis/models/metacognition.py`)
- `MetacognitiveDecision`: Enum routing decisions across `fast_path` (high confidence $\ge 0.85$, low perplexity $\le 2.0$), `reflective_repair` (moderate confidence $\in [0.60, 0.85)$, moderate uncertainty), and `swarm_escalation` (low confidence $< 0.60$, high uncertainty $\ge 0.40$).
- `ReasoningBudget`: Dynamic computational allocation specifying `max_iterations`, `temperature`, `top_p`, and `search_beam_width`.
- Epistemic Metrics:
  - Token Perplexity: $PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p_i\right)$ computed from log probabilities or estimated from lexical and domain entropy.
  - Lexical Complexity: Ratio of unique normalized tokens to total tokens scaled by average word length.
  - Epistemic Confidence $C \in [0.0, 1.0]$: Synthesized from token likelihoods, historical domain performance, and structural certainty.
  - Expected Calibration Error (ECE): Computes sample-weighted calibration discrepancy across confidence bins:
    $$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
- Persistence: All introspections are logged to `metacognitive_evaluations` with aggregate calibration summaries accessible via `get_calibration_summary()`.

### 3. Decentralized Swarm Reasoning Engine & BFT Proof-of-Reasoning (`genesis/tasks/swarm.py`)
- Node Roles & Swarm Architecture:
  - Nodes register with distinct models and roles (`reasoner`, `critic`, `synthesizer`, `verifier`, `architect`).
  - Dynamic leader election weighted by current reputation score ($w_i \in [0.1, 5.0]$).
- Byzantine Fault Detection:
  - Proposals and votes are scrutinized for malicious or destructive instructions (e.g. `rm -rf`, `format`, `drop table`, `os.system`).
  - Byzantine attempts flag the node, set votes to `0.0`, and penalize reputation by $0.4\times$.
- Proof-of-Reasoning Cryptographic Hash Chain:
  - Each consensus round generates a deterministic SHA-256 block hash incorporating round index, previous proof hash, voter IDs, confidence scores, and proposal checksums:
    $$H_r = \text{SHA256}(H_{r-1} \parallel r \parallel \text{voter\_payloads})$$
  - Guarantees auditability and tamper-resistance of collaborative multi-agent decisions.
- Quorum & Approval Threshold:
  - Quorum achieved when non-faulty participating weight exceeds $\frac{2}{3}$ of total active reputation weight ($> 66.7\%$).

### 4. Self-Hosting Evolutionary Compiler Optimization Engine (`genesis/core/self_compiler.py`)
- Hot-Path Targets Profiled:
  1. `cosine_similarity`: Dense vector batch cosine distances in retrieval memory.
  2. `mcts_ucb1`: UCB1 node score evaluations across thousands of speculative tree branches.
  3. `pareto_dominance`: Pairwise multi-objective dominance comparisons in NSGA-II sorting.
- Compilation & Equivalence Verification Pipeline:
  - Emits ANSI C code optimized for SIMD vectorization.
  - Compiles via `NativeExtensionManager` using available local C toolchains (MSVC `cl.exe` / Clang / GCC).
  - Validates numerical invariance against pure Python reference implementations: maximum absolute error must remain strictly below $1\text{e-}5$.
  - Measures execution speedup over pure Python reference (typically $2.0\times$ to $15.0\times$).
- Dynamic Zero-Overhead Ctypes Dispatch:
  - C functions are dynamically loaded via `ctypes.CDLL`.
  - Seamless fallback to pure Python references if native toolchains are unavailable or compilation fails.

### 5. Multi-Objective Continuous Pareto Frontier Optimizer (`genesis/research/pareto.py`)
- Runtime Configuration Dimensions:
  - `model_name` (e.g. `qwen2.5:7b`, `qwen2.5-coder:7b`, `qwen2.5-coder:1.5b-base`, `llama3:8b`).
  - `quantization_bits` (2, 3, 4, 8, 16 bits).
  - `context_window` ($2048$ to $32768$ tokens).
  - `batch_size` (1 to 16).
  - `speculative_lookahead_k` (0 to 8).
- Four Objectives Evaluated:
  1. $\min \text{Latency}$ (ms)
  2. $\min \text{VRAM}$ (MB) — strictly bounded by the RTX 2000 Ada 8,188 MB ceiling
  3. $\max \text{Throughput}$ (tokens/second)
  4. $\max \text{Accuracy}$ score $[0.0, 1.0]$
- NSGA-II Implementation:
  - Fast non-dominated sorting partitions candidates into Pareto ranks $F_0, F_1, \dots$.
  - Crowding distance assignment ensures uniform population diversity across the trade-off surface. Boundary points are assigned distance $10^6$ (avoiding JSON `inf` serialization breakages).
  - Simulated crossover and polynomial mutation explore feasible hardware operating configurations.

### 6. Phase 16 REST API Routes (`genesis/api/routes/metaswarm.py`)
Mounted under `/api/` in `genesis/api/app.py`:
- Metacognition:
  - `POST /api/metacognition/introspect`
  - `GET /api/metacognition/evaluations`
  - `GET /api/metacognition/summary`
- Swarm Consensus:
  - `GET /api/swarm/nodes`
  - `POST /api/swarm/nodes`
  - `POST /api/swarm/consensus`
  - `GET /api/swarm/sessions`
  - `GET /api/swarm/sessions/{session_id}`
- Self-Hosting Compiler:
  - `GET /api/self-compiler/targets`
  - `POST /api/self-compiler/compile`
  - `GET /api/self-compiler/targets/{target_name}`
- Pareto Optimization:
  - `POST /api/pareto/optimize`
  - `GET /api/pareto/experiments`
  - `GET /api/pareto/experiments/{experiment_id}/frontiers`

### 7. Interactive Web Dashboard UI (`genesis/web/`)
- Added dedicated navigation tab `🐝 Metacognition & Swarms` (`#tab-metaswarm-btn`, `#pane-metaswarm`).
- Four comprehensive monitoring cards:
  1. **Metacognitive Epistemic Calibration & Cognitive Load**: Live stats grid (total evaluations, average confidence, ECE calibration error, perplexity) and historical introspection table.
  2. **Decentralized Swarm Reasoning Arena**: Peer node reputation badges and BFT consensus session audit table with proof-of-reasoning SHA-256 tags.
  3. **Self-Hosting Compiler Targets**: Table of profiled hot-path native SIMD kernels with compiler, verified status, speedup factor, numerical error, and checksum. Includes "Recompile All Targets" trigger.
  4. **Multi-Objective Pareto Frontier Explorer**: Visual experiment cards displaying population, generations, non-dominated frontier sizes, and interactive point chips (`Latency | VRAM | Throughput`).
- Three interactive modal dialogs (`#introspect-modal`, `#swarm-modal`, `#pareto-modal`) wired in `app.js` with validated syntax and responsive styling in `index.css`.

---

## Consequences

### Positive
- **Guaranteed Epistemic Routing**: Low-confidence or high-perplexity prompts automatically escalate to reflective self-repair loops or multi-agent swarms, avoiding hallucinated single-shot responses.
- **Resilient Multi-Agent Reasoning**: Swarm decisions are protected against single-node failures, Byzantine inputs, and reasoning divergence through quorum requirements ($> 2/3$) and cryptographic hash chaining.
- **Measurable Computation Speedups**: Hot-path mathematical loops compile to native extensions with verified zero-degradation numerical error ($< 1\text{e-}5$), reducing latency across memory retrieval and MCTS rollouts.
- **Deterministic VRAM Safety**: Pareto optimization explicitly respects the 8,188 MB workstation hardware limit, preventing out-of-memory GPU driver crashes during complex multi-model workloads.
- **Zero Cloud Leakage**: All components execute strictly locally on workstation CPU and RTX 2000 Ada GPU in full accordance with Project Genesis privacy and autonomy guarantees.

### Negative / Trade-offs
- Native compilation depends on locally installed C compilers (e.g. MSVC or Clang); pure Python fallbacks are maintained when native toolchains are absent.
- Full multi-round swarm consensus increases wall-clock response time for escalated queries; the metacognitive router mitigates this by maintaining fast single-model paths for high-confidence tasks.

---

## Verification
- Comprehensive test coverage across 5 new test suites:
  - `tests/test_metacognition.py` (5 tests passing)
  - `tests/test_swarm.py` (6 tests passing)
  - `tests/test_self_compiler.py` (4 tests passing)
  - `tests/test_pareto.py` (5 tests passing)
  - `tests/test_api_metaswarm.py` (5 tests passing)
- Full Project Genesis regression test suite: **336 tests passed with 100% pass rate** in `pytest`.
- Web frontend JavaScript syntax verified cleanly with Node.js parser (`node -c genesis/web/app.js`).
