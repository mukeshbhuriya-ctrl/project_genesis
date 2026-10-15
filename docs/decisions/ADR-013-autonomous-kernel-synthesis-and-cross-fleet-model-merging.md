# ADR-013: Autonomous LLM-Driven Kernel Synthesis, Cross-Fleet Model Parameter Merging, and Human-in-the-Loop Steering

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis operates as an autonomous, self-improving local AI research platform designed for multi-year incremental enhancement without cloud lock-in.
Following the successful delivery of Phase 11 (native extension self-compilation, compiler toolchain auto-discovery, and distributed peer-to-peer fleet synchronization), Phase 12 resolves three major research and engineering frontiers:
1. **Autonomous LLM-Driven Kernel Synthesis & Reflective C/SIMD Optimization**: While Phase 11 established compiler invocation and pure Python fallbacks, kernel implementations were largely static or seed-based. Phase 12 introduces an autonomous synthesis engine that generates, benchmarks, numerically validates (against reference mathematical formulations), and reflectively repairs low-level C tensor kernels (GELU, RMSNorm, SwiGLU, INT8 symmetric quantization, and SIMD vector dot product) with zero manual intervention.
2. **Cross-Fleet Model Parameter Merging**: Fine-tuning individual LoRA adapters or model weights creates task specialization, but combining multi-task capabilities across nodes without expensive full retraining requires algebraic parameter merging. Phase 12 implements four canonical merging algorithms: Weighted Linear Averaging, Spherical Linear Interpolation (SLERP), Task Arithmetic with Trimming and Sign Consensus (TIES-Merging), and Drop And REscale (DARE-Merging).
3. **Human-in-the-Loop Steering Controller**: While Genesis operates autonomously, human researchers require the ability to pause execution, critique hypotheses, inject domain guidance, or trigger emergency aborts without terminating runtime processes or invalidating working state.

## Decisions

### 1. Database Schema Version 12 Migrations (`genesis/core/database.py`)
Schema Version 12 introduces three relational tables in SQLite WAL mode with strict foreign key integrity:
- `kernel_experiments`: Stores autonomous kernel synthesis trials (`id`, `operation_type`, `status`, `iteration_count`, `speedup_factor`, `max_abs_error`, `numerical_check_passed`, `c_source`, `error_message`, `created_at`, `metadata`). Supported operation types: `gelu`, `rmsnorm`, `swiglu`, `quantize_int8`, `dot_product`. Supported statuses: `success`, `repair_succeeded`, `fallback_python`, `failed`. Indexes: `idx_kernel_experiments_op`, `idx_kernel_experiments_status`.
- `model_merges`: Stores parameter merging experiment recipes and outputs (`id`, `name`, `base_model`, `method`, `adapter_paths`, `weights`, `density`, `output_path`, `metrics`, `created_at`, `metadata`). Supported methods: `linear`, `slerp`, `ties`, `dare`. Indexes: `idx_model_merges_method`, `idx_model_merges_name`.
- `steering_events`: Stores human operator steering interventions (`id`, `action`, `target_subsystem`, `payload`, `status`, `created_at`, `applied_at`, `metadata`). Supported actions: `pause`, `resume`, `critique`, `override_hypothesis`, `abort`, `inject_guidance`. Supported statuses: `received`, `applied`. Indexes: `idx_steering_events_action`, `idx_steering_events_status`.

### 2. Autonomous LLM-Driven Kernel Synthesis Engine (`genesis/research/kernel_synthesis.py`)
- `KernelSynthesisExperiment`: Strongly typed dataclass tracking operation type, iterative attempts, C source code, numerical tolerance metrics, and speedup achievements.
- `KernelSynthesisEngine`:
  - **Canonical Mathematical References**: Reference implementations for `ref_gelu`, `ref_rmsnorm`, `ref_swiglu`, `ref_quantize_int8`, and `ref_dot_product`.
  - **Synthesis Seed Templates**: Optimized C templates incorporating platform-aware exports (`__declspec(dllexport)` on Windows), fast math approximations (tanh-based GELU, reciprocal square roots for RMSNorm), and SIMD loop structures.
  - **Ground-Truth Numerical Tolerance Validation**: Validates synthesized kernels against ground truth with strict tolerance thresholds (default `max_abs_error <= 1e-4`), preventing hallucinated numerical inaccuracies.
  - **Reflective Self-Repair Loop**: Automatically detects compiler syntax errors, missing includes, or numerical tolerance failures and applies multi-step reflective source corrections up to `max_iterations`.
  - **Graceful Compiler Degradation**: Fully integrated with `NativeExtensionManager`; when host compilers are absent, registers pure Python fallbacks (`status="fallback_python"`) with zero runtime crashes.

### 3. Cross-Fleet Model Parameter Merging Engine (`genesis/training/merging.py`)
- `ModelMergeRecipe`: Typed model representing merge hyperparameters (name, base model, method, density, weights, adapter paths).
- `ModelMergeManager`:
  - **Weighted Linear Averaging (`merge_linear`)**: Computes $\sum w_i \theta_i$ with normalized weights for parameter tensors.
  - **Spherical Linear Interpolation (`merge_slerp`)**: Interpolates between two parameter vectors along the hypersphere:
    $$\text{slerp}(v_1, v_2, t) = \frac{\sin((1-t)\Omega)}{\sin(\Omega)} v_1 + \frac{\sin(t\Omega)}{\sin(\Omega)} v_2$$
    Includes robust collinear fallback ($|\cos \Omega| > 0.9995$) to linear interpolation.
  - **TIES-Merging (`merge_ties`)**: Trims bottom $(1 - \text{density})$ magnitude parameters, computes sign consensus across models ($\text{sgn}(\sum \text{sgn}(\tau_i))$), filters conflicting signs, and computes disjoint mean averages.
  - **DARE-Merging (`merge_dare`)**: Randomly drops parameter updates with probability $(1 - \text{density})$, rescales remaining parameters by $1/\text{density}$, and computes linear averages.
  - **Disk Serialization & Registry**: Serializes merged dictionaries to `data/merges/` and persists metadata into SQLite.

### 4. Human-in-the-Loop Steering Subsystem (`genesis/research/steering.py`)
- `SteeringEvent`: Typed record representing operator directives.
- `SteeringController`:
  - Registers operator directives (`pause`, `resume`, `critique`, `override_hypothesis`, `abort`, `inject_guidance`).
  - Provides active directive polling for asynchronous workers and daemons.
  - State transitions (`received` $\to$ `applied`) with execution timestamps.

### 5. Phase 12 REST API Routes (`genesis/api/routes/`)
- `/api/kernel-synthesis/`:
  - `GET /templates`: Lists available kernel synthesis operations.
  - `POST /run`: Executes the iterative synthesis loop.
  - `GET /experiments`: Lists past kernel synthesis experiments.
- `/api/merging/`:
  - `GET /methods`: Lists supported merge algorithms (`linear`, `slerp`, `ties`, `dare`).
  - `POST /execute`: Executes parameter merging on target adapters.
  - `GET /merges`: Lists registered merge artifacts.
  - `GET /merges/{id}`: Retrieves specific merge details.
- `/api/steering/`:
  - `GET /directives`: Lists all steering events and active directives.
  - `POST /directives`: Dispatches an operator directive.
  - `POST /directives/{id}/apply`: Marks a directive as applied.

### 6. Interactive Web Dashboard Integration (`genesis/web/`)
- Upgraded version banner to `v0.13.0 • Local-First`.
- Added **Autonomous Kernel Synthesis** section to Fleet panel with live experiment logs and C code modal viewer.
- Added **Cross-Fleet Model Parameter Merging** section with recipe triggers, density slider, and merge history.
- Added **Human-in-the-Loop Operator Steering Console** to Research panel for real-time directive transmission and status monitoring.

## Consequences

### Positive
- High-performance tensor kernel synthesis is fully automated and self-repairing without requiring human C programming.
- Parameter merging allows zero-cost multi-task adaptation across fine-tuned local models.
- Human researchers possess real-time intervention controls over long-running autonomous research cycles.
- 100% test pass rate maintained (254 passing tests) with zero cloud dependencies.

### Negative / Trade-offs
- On Windows systems without a native C compiler installed, synthesized kernels fall back to verified pure Python execution; actual SIMD speedup requires MSVC, LLVM Clang, or GCC on `PATH`.
- SLERP algorithm is strictly pairwise; merging $\ge 3$ models requires sequential hierarchical folding or TIES/DARE.
