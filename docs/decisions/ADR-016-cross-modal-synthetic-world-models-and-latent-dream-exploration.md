# ADR-016: Cross-Modal Synthetic World Models, Continuous Latent Dream Exploration, and Speculative MCTS with State-Forking Rollback

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis operates locally on dedicated workstation hardware (Intel Core i9-13900H, 32 GB RAM, NVIDIA RTX 2000 Ada with 8 GB VRAM) under Windows 11 with zero external cloud dependencies.
As model capabilities and autonomy expand through recursive self-architecting (Phase 14) and continual online learning (Phase 13), long-horizon engineering tasks, code refactorings, and multi-step research campaigns face critical operational risks:
1. **Physical Execution Risk & State Corruption**: Executing unverified shell commands, code mutations, or filesystem edits directly against the workspace can lead to unrecoverable syntax breakages, directory escapes, or system regressions. Agents require predictive world models to simulate the outcomes of candidate actions prior to physical execution.
2. **Inference Latency in Multi-Turn Counterfactual Simulation**: Exploring hypothetical problem-solving branches by running full autoregressive GPU model inference at every step exhausts local compute bandwidth and slows planning loops. Generative mental rollouts (latent dreams) evaluate multi-step trajectories in memory with heuristic value functions $V(s)$ and branch pruning.
3. **Representation Divergence Across Modalities**: Software engineering tasks operate simultaneously across natural language requirements, symbolic abstract syntax trees (ASTs), and visual user interfaces. Without a common embedding projection, cross-modal grounding fails.
4. **State-Space Combinatorial Explosion**: Complex refactoring tasks require structured exploration over combinatorial mutation spaces. Unstructured exploration easily diverges without tree search heuristics and rollback guarantees.

Phase 15 addresses these requirements by introducing:
- **Multimodal Environment Transition Modeling**: Predictive simulation of filesystem deltas, AST mutations, stdout/stderr streams, and safety invariants before execution.
- **Continuous Latent Dream Exploration**: Generative mental rollouts in memory with value estimation, uncertainty quantification, branch pruning, and continual learning replay buffer export.
- **Cross-Modal Representation Alignment**: Deterministic multi-hash latent vector projections bridging symbolic ASTs, textual specifications, and visual layout primitives into a unified $\mathbb{R}^{64}$ space.
- **Sandboxed State-Forking & Speculative MCTS Engine**: Lightweight workspace snapshotting, UCB1 tree search exploration, rollout simulation, and atomic rollback defense upon invariant violation.

---

## Decisions

### 1. Database Schema Version 15 Migrations (`genesis/core/database.py`)
Schema Version 15 introduces four relational tables in SQLite WAL mode:
- `world_transitions`: Records transition predictions, actual execution states, error scores, and invariant verifications (`id`, `session_id`, `action_type`, `predicted_state_json`, `actual_state_json`, `transition_error_score`, `safety_predicted`, `safety_actual`, `created_at`, `metadata`). Supported action types: `file_write`, `file_delete`, `ast_mutation`, `command_exec`, `tool_call`. Indexes: `idx_world_transitions_session`, `idx_world_transitions_action`.
- `dream_trajectories`: Persists latent mental rollouts, value estimates, and branch pruning metrics (`id`, `prompt`, `steps_count`, `total_value_score`, `pruned_branches`, `status`, `trajectory_json`, `created_at`, `metadata`). Supported statuses: `imagining`, `completed`, `pruned`, `failed`. Indexes: `idx_dream_trajectories_status`, `idx_dream_trajectories_value`.
- `mcts_search_runs`: Stores speculative Monte Carlo Tree Search explorations, node counts, and rollback statistics (`id`, `root_task`, `iterations`, `nodes_explored`, `best_path_json`, `best_value`, `fork_rollbacks`, `status`, `created_at`, `metadata`). Supported statuses: `searching`, `completed`, `aborted`, `failed`. Index: `idx_mcts_search_status`.
- `modal_alignments`: Stores cross-modal projection alignments across text, AST, and visual representations (`id`, `text_prompt`, `ast_symbol`, `visual_token`, `similarity_score`, `projection_dim`, `created_at`, `metadata`). Index: `idx_modal_alignments_score`.

### 2. Multimodal Environment Transition Predictor (`genesis/world_model/transition.py`)
- `EnvironmentState`: Captures workspace state snapshots including files (`rel_path -> content`), AST symbols (`rel_path -> list[str]`), stdout, stderr, exit code, and invariant status flags.
- `Action`: Candidate action specification with `action_type`, `target`, `payload`, and `params`.
- `extract_ast_symbols`: Safely parses Python source code via the standard `ast` module, extracting declared function signatures with arguments, class definitions, and import statements without code execution. Flags syntax errors gracefully.
- `check_safety`: Deterministically scans candidate actions against security invariants:
  - Traversal attacks (`..`, absolute drive roots) penalize safety score by $0.5$.
  - Dangerous system destruction patterns (`rm -rf`, `del /s /q`, `format`, fork bombs) penalize safety score by $0.8$.
  - AST syntax errors penalize safety score by $0.3$.
- `predict_transition`: Simulates state transformations:
  - `file_write`: Updates file map, parses code into AST symbols, updates delta lists.
  - `file_delete`: Removes target from file map and AST symbols.
  - `ast_mutation`: Simulates targeted function/class injection.
  - `command_exec`: Predicts exit codes and terminal stream behaviors.
- `calculate_transition_error`: Measures normalized structural discrepancy $E \in [0.0, 1.0]$ between predicted state and actual execution state across exit codes, file contents, AST symbols, and invariants. Perfect predictions evaluate to $0.000$.

### 3. Latent Dream & Continuous Imagination Rollout Engine (`genesis/world_model/dream.py`)
- `DreamStep`: Represents a mental step featuring `hypothesis_action`, `imagined_outcome`, value estimate $V(s) \in [-1.0, 1.0]$, uncertainty metric $U \in [0.0, 1.0]$, and pruning flag.
- `evaluate_step_value`: Heuristic value function evaluating goal proximity, invariant preservation, and absence of hallucinatory or unsafe patterns.
- `simulate_dream`: Generatively unfolds multi-step hypothetical problem-solving rollouts in memory without disk or GPU overhead. When a branch value drops below threshold $\tau$ (default $0.20$), the branch is pruned and an alternative counterfactual recovery step is simulated.
- `export_to_replay_buffer`: Seamlessly bridges Phase 15 with Phase 13 continual learning by extracting high-reward dream steps ($V \ge 0.45$) and registering them as prioritized training samples in `replay_buffers`.

### 4. Cross-Modal Representation Alignment Subsystem (`genesis/world_model/alignment.py`)
- `ModalityEmbedding`: Represents normalized latent vectors in $\mathbb{R}^d$ ($d=64$).
- Token Extraction:
  - `extract_text_tokens`: Extracts clean alphanumeric keywords, filtering stopwords.
  - `extract_ast_tokens`: Extracts structural AST nodes, decomposing compound function/class/variable identifiers into constituent sub-tokens.
  - `extract_visual_tokens`: Extracts UI component tags, property keys, and layout primitives.
- `project_tokens_to_latent`: Projects token sets into $\mathbb{R}^{64}$ using dual-hash orthogonal basis projections with semantic bridging. Suppresses generic Python AST boilerplate (`load`, `store`, `arguments`, `call`) and applies $3.0\times$ priority weighting to semantic identifiers. Vectors are $\ell_2$-normalized.
- `align_modalities`: Projects text, AST, and visual representations concurrently, computing pairwise cosine similarities $\text{sim}(\text{text}, \text{ast})$, $\text{sim}(\text{ast}, \text{vis})$, $\text{sim}(\text{text}, \text{vis})$ and overall alignment score.
- `cross_modal_retrieve`: Ranks heterogeneous candidate artifacts across modalities against a target query based on cosine proximity.

### 5. Sandboxed State-Forking & Speculative MCTS Engine (`genesis/world_model/mcts.py`)
- `StateSnapshot`: Lightweight in-memory snapshot of workspace files and variables with SHA256 checksums and deep copy state forking.
- `MCTSNode`: Represents an exploration node with visit count $N(s)$, cumulative reward $Q(s)$, and UCB1 selection metric:
  $$\text{UCB1} = \frac{Q(s, a)}{N(s, a)} + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}, \quad c = 1.414$$
  Unvisited nodes are guaranteed infinite priority ($\infty$).
- Search Phases:
  1. **Selection**: Traverses the existing tree using UCB1 until an expandable leaf is reached.
  2. **Expansion**: Generates speculative mutations (unit test synthesis, refactoring, defensive guardrail injection, scaffolding cleanup) and forks snapshots.
  3. **Simulation**: Evaluates candidate mutations for syntax validity and quality. Invariant breakages trigger immediate rollback ($Q=0, \text{is\_terminal}=\text{True}$).
  4. **Backpropagation**: Propagates simulation rewards back to the root, updating visit counts and cumulative rewards.
- Best path is selected by traversing the trajectory of highest mean values.

### 6. Phase 15 REST API Routes (`genesis/api/routes/world_model.py`)
- Mounted under `/api/world/` in `genesis/api/app.py`:
  - `POST /api/world/predict-transition`
  - `POST /api/world/record-transition`
  - `GET /api/world/transitions`
  - `GET /api/world/transitions/{record_id}`
  - `POST /api/world/dream-rollout`
  - `GET /api/world/trajectories`
  - `GET /api/world/trajectories/{trajectory_id}`
  - `POST /api/world/align-modalities`
  - `GET /api/world/alignments`
  - `POST /api/world/cross-modal-retrieve`
  - `POST /api/world/mcts-search`
  - `GET /api/world/mcts-runs`
  - `GET /api/world/mcts-runs/{run_id}`

### 7. Interactive Web Dashboard UI (`genesis/web/`)
- Added dedicated navigation tab `🌌 World & Dreams` (`#tab-world-btn`, `#pane-world`).
- Three interactive monitoring sections:
  1. Multimodal Environment Transition Predictor with safety badge and audit table.
  2. Latent Dream Explorer with multi-turn mental rollout step cards and value badges.
  3. Speculative MCTS Tree Explorer displaying best discovered trajectories and defended rollbacks.
- Three interactive modal forms with real-time API integrations in `app.js` and glassmorphic styling in `index.css`.

---

## Consequences

### Positive
- **Zero-Risk Safe Exploration**: Agents can simulate proposed filesystem deltas and code refactorings before applying changes to disk, preventing syntax corruption or security escapes.
- **Compute-Efficient Planning**: Latent dream rollouts evaluate multi-step reasoning trajectories in memory without requiring full autoregressive GPU inference passes at every hypothetical turn.
- **Continual Learning Bridge**: High-reward mental rollouts automatically feed into Phase 13 prioritized replay buffers, enabling the platform to learn from imagined problem-solving traces.
- **Cross-Modal Coherence**: Unified latent projection space ensures text specifications, code ASTs, and visual layouts are aligned and retrievable across modalities.
- **Speculative Tree Search**: Complex multi-step software engineering tasks can be explored via MCTS with deterministic rollback defenses upon invariant violations.

### Neutral / Trade-offs
- In-memory state snapshots track text-based workspace files; large binary datasets or multi-gigabyte models remain indexed via SQLite metadata pointers to maintain lightweight tree exploration.
- Heuristic value estimation provides fast bounds for mental rollouts; physical verification benchmarks remain authoritative for final deployment decisions.
