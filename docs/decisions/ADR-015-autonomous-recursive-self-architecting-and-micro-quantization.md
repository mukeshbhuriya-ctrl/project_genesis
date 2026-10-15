# ADR-015: Autonomous Recursive Self-Architecting, Hardware-Targeted Micro-Quantization, and Self-Speculative Decoding

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis is designed to operate locally on dedicated workstation hardware (Intel Core i9-13900H, 32 GB RAM, NVIDIA RTX 2000 Ada with 8 GB VRAM) under Windows 11 with zero external cloud dependencies.
As model capabilities expand through continual learning (Phase 13), two primary operational bottlenecks emerge on local edge hardware:
1. **Memory Capacity and KV-Cache Footprint Under Full 7B/8B Models**: Standard 7B/8B parameter models require ~4.2–5.2 GB in 4-bit representation, leaving limited VRAM for multi-turn conversational context, attention KV caches, and intermediate activation tensors during speculative planning or agent campaigns. Uncontrolled expansion of attention heads or intermediate feedforward dimensions risks CUDA out-of-memory (OOM) failures.
2. **Autoregressive Generation Latency on Edge Tensor Cores**: Standard token-by-token autoregressive generation on 7B models is inherently memory-bandwidth bound. Running multi-agent consensus debates (Phase 10) or reflective code repair loops (Phase 12) accumulates substantial inference latency.
3. **Accuracy Degradation in Ultra-Low-Bit Quantization**: Naive linear INT4/INT3 quantization uniformly clips weight distributions, destroying critical activation spikes ("salient outliers") that represent linguistic and reasoning features.

Phase 14 addresses these challenges by introducing:
- **Autonomous Neural Architecture Search (NAS) & Structural Block Pruning**: Sensitivity-driven pruning of attention heads and SwiGLU MLP intermediate projections to fit precise VRAM budgets ($< 5.5\text{ GB}$).
- **Hardware-Targeted Micro-Quantization Engine**: Activation-Aware Weight Quantization (AWQ) with salient outlier preservation in FP16 alongside 2-bit, 3-bit, 4-bit, and 8-bit blocked quantization for NVIDIA Ada Lovelace Tensor Cores.
- **Self-Speculative Decoding Engine**: Pairing low-overhead draft generation (1.5B) with parallel target verification (7B) via Leviathan-style rejection sampling to achieve 1.8x–2.6x inference speedup without loss of output fidelity.

## Decisions

### 1. Database Schema Version 14 Migrations (`genesis/core/database.py`)
Schema Version 14 introduces three relational tables in SQLite WAL mode with strict foreign key integrity:
- `pruning_experiments`: Tracks structural pruning blueprints, parameter reduction, and verification metrics (`id`, `model_name`, `target_vram_mb`, `pruning_strategy`, `heads_pruned`, `remaining_heads`, `mlp_expansion_ratio`, `original_parameters`, `pruned_parameters`, `parameter_reduction_pct`, `estimated_vram_mb`, `status`, `metrics_json`, `created_at`, `metadata`). Supported strategies: `structured_head`, `mlp_intermediate`, `taylor_sensitivity`, `magnitude_l1`. Supported statuses: `planned`, `pruned`, `benchmarking`, `verified`, `rejected`. Indexes: `idx_pruning_experiments_model`, `idx_pruning_experiments_status`.
- `quantization_profiles`: Stores mathematical and empirical quantization profiles and fidelity metrics (`id`, `model_name`, `bits`, `strategy`, `block_size`, `outlier_channel_ratio`, `original_size_mb`, `quantized_size_mb`, `compression_ratio`, `mse_loss`, `status`, `created_at`, `metadata`). Supported bits: 2, 3, 4, 8. Supported strategies: `symmetric`, `asymmetric`, `awq_outlier_salient`, `gptq_hessian`. Supported block sizes: 32, 64, 128. Supported statuses: `quantizing`, `completed`, `verified`, `failed`. Indexes: `idx_quant_profiles_model`, `idx_quant_profiles_bits`.
- `speculative_runs`: Stores speculative decoding sessions, token acceptance statistics, and speedup factors (`id`, `target_model`, `draft_model`, `lookahead_k`, `prompt`, `output_text`, `accepted_tokens`, `total_draft_tokens`, `acceptance_rate`, `speedup_factor`, `target_latency_ms`, `speculative_latency_ms`, `status`, `created_at`, `metadata`). Supported lookahead $K \in [1, 8]$. Supported statuses: `running`, `completed`, `fallback_autoregressive`, `failed`. Indexes: `idx_speculative_runs_models`, `idx_speculative_runs_status`.

### 2. Autonomous Structural Block Pruning & NAS Engine (`genesis/models/pruning.py`)
- `ModelArchitectureConfig` & `KNOWN_ARCH_CONFIGS`: Defines formal layer, head, dimension, and byte-per-parameter specifications for `qwen2.5:7b`, `qwen2.5-coder:7b`, `qwen2.5-coder:1.5b-base`, and `llama3:8b`.
- `calculate_parameter_count`: Provides closed-form analytical calculation of transformer parameter counts:
  $$\text{Params} = V \cdot d + L \left( 2 d (h_q + 2 h_{kv}) d_{\text{head}} + 3 d \cdot d_{\text{mlp}} + 2 d \right) + d$$
- `estimate_head_importance`: Computes attention head sensitivity ranking:
  - First-order Taylor approximation $|W \cdot \nabla_W L|$ factoring layer depth and normalized head position.
  - L1 weight magnitude proxy for rapid un-differentiated pruning passes.
- `create_pruning_plan`: Performs autonomous search over surviving head counts ($h \ge h_{kv}$) and SwiGLU MLP ratios ($m \in [0.6, 1.0]$) to find optimal configurations matching the requested target VRAM boundary. Generates per-layer head survival masks.
- `execute_pruning`: Commits and verifies the pruning blueprint against safety constraints, recording idempotent entries in `pruning_experiments`.

### 3. Hardware-Targeted Micro-Quantization Engine (`genesis/models/quantization.py`)
- **Symmetric Uniform Quantization**:
  $$S = \frac{\max(|w|)}{2^{b-1} - 1}, \quad q = \text{clamp}\left(\text{round}\left(\frac{w}{S}\right), -q_{\max}, q_{\max}\right), \quad \hat{w} = q \cdot S$$
- **Asymmetric Affine Quantization**:
  Guarantees real zero preservation by bounding $w_{\min} \le 0.0 \le w_{\max}$:
  $$S = \frac{w_{\max} - w_{\min}}{2^b - 1}, \quad Z = \text{clamp}\left(\text{round}\left(-\frac{w_{\min}}{S}\right), 0, 2^b - 1\right), \quad \hat{w} = (q - Z) \cdot S$$
- **Blocked Quantization**: Partitions tensor matrices into blocks of $N \in \{32, 64, 128\}$ elements, assigning independent scaling factors to minimize dynamic range clipping.
- **Activation-Aware Weight Quantization (AWQ)**:
  - Identifies top salient outlier channels based on activation magnitudes.
  - Retains salient columns ($~1\%$) strictly in FP16 precision.
  - Quantizes non-outlier channels with INT4/INT3/INT2 blocked scaling.
  - Evaluates mean squared error (MSE) loss and cosine similarity ($> 0.98$).

### 4. Self-Speculative Decoding Engine (`genesis/models/speculative.py`)
- `verify_tokens_rejection_sampling`: Implements Leviathan rejection sampling:
  For each draft token $x_i$ with draft distribution $q(x)$ and target verification $p(x)$:
  - Accepts token if $\frac{p(x_i)}{q(x_i)} \ge \gamma$ (threshold default $0.75$).
  - Upon rejection, terminates sequence acceptance and emits target model corrected token $x_i'$.
- `execute_speculative_run`:
  - Generates draft lookahead sequence of length $K \in [1, 8]$ using `qwen2.5-coder:1.5b-base`.
  - Concurrently evaluates acceptance with `qwen2.5:7b`.
  - Computes empirical acceleration factor:
    $$\text{Speedup} = \frac{1 + \alpha K}{1 + K \cdot \frac{C_{\text{draft}}}{C_{\text{target}}}}$$
  - Stores end-to-end telemetry into `speculative_runs`.

### 5. REST API Architecture (`genesis/api/routes/`)
- `/api/pruning`:
  - `GET /architectures`: Known architectural specifications.
  - `POST /plan`: Computes structural pruning blueprint and parameter reduction.
  - `POST /execute`: Executes and records pruning experiment.
  - `GET /experiments`: Lists historical pruning experiments.
  - `GET /experiments/{id}`: Detailed pruning experiment record.
- `/api/quantization`:
  - `GET /strategies`: Supported algorithms and hardware profiles.
  - `POST /profile`: Micro-quantization benchmark simulation and metrics.
  - `GET /profiles`: Lists past quantization profiles.
  - `GET /profiles/{id}`: Detailed quantization profile.
- `/api/speculative`:
  - `GET /configs`: Recommended model pairing configurations.
  - `POST /run`: Executes speculative decoding session with live telemetry.
  - `GET /runs`: Lists historical speculative sessions.
  - `GET /runs/{id}`: Detailed speculative run record.

### 6. Web Dashboard UI (`genesis/web/`)
- Added dedicated navigation tab `📐 NAS & Micro-Quant` (`#tab-architect-btn`, `#pane-architect`).
- Integrated 3 responsive data tables with live telemetry badges.
- Implemented modal dialogs for interactive pruning blueprint configuration (`#pruning-modal`), quantization profiling (`#quant-modal`), and speculative session execution (`#speculative-modal`).

## Consequences

### Positive
- Models can be dynamically pruned to fit comfortably within the 8 GB VRAM budget of the local RTX 2000 Ada GPU while preserving attention and SwiGLU expressiveness.
- AWQ salient outlier protection enables 4-bit and 3-bit compression with $> 0.95$ cosine fidelity, preventing quality collapse.
- Speculative decoding enables 1.8x–2.6x inference latency reduction for long-horizon agent reasoning and campaign synthesis.
- Complete SQLite auditability across all architectural modifications, quantization profiles, and inference benchmarks.

### Negative
- Extreme 2-bit quantization exhibits noticeable MSE increase ($> 0.001$) and should be restricted to non-critical auxiliary embeddings.
- Speculative decoding speedup degrades when the draft model acceptance rate falls below $50\%$, triggering autoregressive fallback.
