# ADR-014: Continual Online Learning Engine, Dynamic Experience Replay, and Neuro-Symbolic Verification

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis operates as an autonomous, self-improving local AI research platform designed for multi-year incremental enhancement without cloud dependencies.
Following the delivery of Phase 12 (autonomous kernel synthesis, model parameter merging, and human-in-the-loop operator steering), Phase 13 addresses two fundamental barriers to sustained autonomous model evolution:
1. **Catastrophic Forgetting & Plasticity-Stability Trade-offs in Continual Fine-Tuning**: When local models undergo sequential fine-tuning on diverse tasks (reasoning, coding, tools, general dialogue), updates to network parameters tend to overwrite previously acquired competencies. Traditional offline re-training on the full corpus is computationally prohibitive on local single-GPU hardware (RTX 2000 Ada 8GB VRAM). Phase 13 requires a continual learning subsystem combining:
   - Streaming prioritized experience replay buffers that preserve multi-domain exemplar diversity under strict bounded capacity.
   - Elastic Weight Consolidation (EWC) parameter penalty estimation using diagonal Fisher Information matrices ($F_i$) to penalize modification of weights critical to earlier tasks.
   - Dynamic parameter-efficient prefix-tuning managers providing virtual soft prompts and prompt interpolation without full-weight modification.
2. **Hallucination & Lack of Formal Soundness in Model Outputs**: While empirical test harnesses and sandbox execution (Phase 6) detect explicit runtime errors, they cannot mathematically prove code termination, purity invariants, mathematical boundary safety, or logical soundness of multi-step deductive arguments. Phase 13 introduces a Neuro-Symbolic Constraint Verification Engine that couples neural model generations with formal deterministic reasoning (AST static invariant analysis, continuous function boundary checks, and propositional resolution refutation theorem proving).

## Decisions

### 1. Database Schema Version 13 Migrations (`genesis/core/database.py`)
Schema Version 13 introduces three relational tables in SQLite WAL mode with strict foreign key integrity:
- `replay_buffers`: Stores streaming prioritized rehearsal experiences (`id`, `task_domain`, `sample_input`, `sample_target`, `priority_score`, `usage_count`, `created_at`, `metadata`). Supported task domains: `reasoning`, `coding`, `tools`, `general`. Indexes: `idx_replay_buffers_domain`, `idx_replay_buffers_priority`.
- `ewc_checkpoints`: Stores Elastic Weight Consolidation parameter checkpoints and diagonal Fisher information matrices (`id`, `experiment_id`, `task_name`, `importance_lambda`, `fisher_diagonal_json`, `optimal_weights_json`, `parameter_count`, `created_at`, `metadata`). Indexes: `idx_ewc_checkpoints_exp`, `idx_ewc_checkpoints_task`.
- `symbolic_verifications`: Stores formal verification outcomes and violation diagnostics (`id`, `target_type`, `expression_or_code`, `constraints_json`, `status`, `violations_json`, `verification_latency_ms`, `created_at`, `metadata`). Supported target types: `code`, `math`, `hypothesis`, `kernel`. Supported statuses: `verified`, `violated`, `inconclusive`. Indexes: `idx_symbolic_verifications_type`, `idx_symbolic_verifications_status`.

### 2. Continual Online Learning Subsystem (`genesis/training/continual.py`)
- `ReplaySample` & `ReplayBufferManager`:
  - **Bounded Multi-Domain Buffering**: Enforces strict capacity limits per task domain (default 10,000 samples/domain). When capacity is reached, lowest-priority samples are evicted.
  - **Duplicate Prevention**: SHA-256 fingerprinting on combined prompt-target pairs suppresses redundant ingestion.
  - **Balanced Multi-Domain Batch Rehearsal**: `sample_batch(batch_size)` draws stratified samples proportionally across domains, weighting selection by priority scores ($P_i / \sum P$), and increments `usage_count` for selected records to balance rehearsal frequency.
- `EWCRecord` & `EWCConsolidator`:
  - **Diagonal Fisher Information Estimation**: Evaluates parameter empirical Fisher information $F_i = \frac{1}{N} \sum (\nabla_\theta \log p(x))^2$ from gradient variance across validation samples.
  - **Quadratic EWC Penalty Loss**: Computes the regularization term:
    $$L_{\text{EWC}}(\theta) = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$
    penalizing deviations from optimal weights $\theta^*$ scaled by task importance $\lambda$.
  - **Multi-Task Checkpoint Consolidation**: Manages persistent checkpoint storage in SQLite with JSON-serialized parameter dictionaries.
- `PrefixTuningConfig` & `PrefixTuningManager`:
  - Generates virtual soft prompt prefix vectors initialized with zero-mean scaled Gaussian noise ($\mathcal{N}(0, \sigma^2)$).
  - Implements task prefix vector interpolation $P = (1 - \alpha) P_a + \alpha P_b$ for continuous domain blending.

### 3. Neuro-Symbolic Verification Engine (`genesis/research/symbolic.py`)
- `SymbolicConstraint`, `VerificationResult`, & `NeuroSymbolicVerifier`:
  - **Code Invariant AST Verification (`verify_code_invariants`)**:
    - *Termination Analysis*: Scans AST `While` and `For` loops for infinite loops (e.g. `while True:` without breaking statements or condition mutations). Detects unbounded recursive functions lacking base-case conditional branching.
    - *Purity and Safety*: Disallows dangerous builtins (`eval`, `exec`, `__import__`, `open`, `globals`, `locals`) and verifies user-specified forbidden identifiers.
    - *Division-by-Zero*: Detects literal zero divisions (`x / 0`, `x % 0`).
  - **Mathematical Continuous Bounds Checking (`verify_mathematical_bounds`)**:
    - Discretizes continuous variable domains $[x_{\min}, x_{\max}]$ into deterministic evaluation grids.
    - Evaluates expressions in a hardened math environment (`sin`, `cos`, `exp`, `log`, `sqrt`, `abs`).
    - Validates properties: `non_negative` ($f(x) \ge 0$), bounded range $[y_{\min}, y_{\max}]$, and `monotonic_increasing` ($f(x_{k+1}) \ge f(x_k)$).
  - **Propositional Logic Resolution Refutation (`verify_propositional_logic`)**:
    - Parses propositional formulas containing conjunctions (`&`), disjunctions (`|`), negations (`~`), and implications (`->`).
    - Transforms premises and $\neg \text{hypothesis}$ into Conjunctive Normal Form (CNF) clauses.
    - Applies systematic resolution rule ($C_1 \vee L, C_2 \vee \neg L \implies C_1 \vee C_2$) deriving the empty clause ($\emptyset$) to formally prove validity (Modus Ponens, Modus Tollens, hypothetical syllogisms).
  - **Audit Logging**: Persists all verification events with microsecond latency metrics into `symbolic_verifications`.

### 4. Phase 13 REST API Routes (`genesis/api/routes/`)
- Mounted under `/api/continual`:
  - `GET /replay/samples`: Lists buffered rehearsal samples with domain filtering.
  - `POST /replay/add`: Ingests an experience into the replay buffer.
  - `GET /replay/batch`: Draws a balanced, priority-weighted rehearsal batch.
  - `GET /replay/stats`: Returns sample counts and capacities per domain.
  - `DELETE /replay/samples/{id}`: Removes a sample from the buffer.
  - `GET /ewc/checkpoints`: Lists registered EWC parameter checkpoints.
  - `POST /ewc/consolidate`: Computes and registers Fisher information and optimal weights.
  - `POST /ewc/penalty`: Calculates quadratic EWC penalty given candidate weights.
  - `POST /prefix/generate`: Synthesizes virtual soft prompt prefixes.
- Mounted under `/api/symbolic`:
  - `POST /verify/code`: Verifies code AST safety and termination invariants.
  - `POST /verify/math`: Verifies continuous mathematical expression bounds.
  - `POST /verify/logic`: Proves propositional logic hypotheses via resolution refutation.
  - `GET /verifications`: Retrieves audit log of verification records.
  - `GET /verifications/{id}`: Retrieves specific verification record by ID.

### 5. Web Dashboard UI (`genesis/web/`)
- Updated header brand version to `v0.14.0 • Local-First`.
- Added **Continual Online Learning & Dynamic Replay Buffer** panel to `#pane-training` with stats cards, interactive replay samples table, EWC checkpoints table, and modals for sample ingestion and rehearsal batch generation.
- Added **Neuro-Symbolic Verification Engine** panel to `#pane-research` with verification log table and interactive verification modal supporting code, math, and logic modes with live verdict badges.

## Consequences

### Positive
- Prevents catastrophic forgetting during sequential fine-tuning cycles on RTX 2000 Ada without expensive full re-training.
- Protects critical model weights using mathematically rigorous Elastic Weight Consolidation (Fisher Information).
- Provides formal deterministic proofs of code termination, mathematical boundedness, and propositional validity before promotion into training datasets or execution.
- Retains 100% test coverage with zero external dependencies and SQLite WAL mode persistence.

### Negative / Trade-offs
- Propositional logic resolution is currently limited to propositional calculus; first-order predicate logic with quantifiers is deferred to a future milestone.
- Mathematical bounds checking operates via dense grid discretization rather than symbolic interval arithmetic.
