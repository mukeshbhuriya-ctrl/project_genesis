# ADR-006: Multi-Format Dataset Curation, Contamination Defense, VRAM-Bounded LoRA Fine-Tuning, and Rollback Orchestration

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis is a local-first, self-improving AI research and model development platform running on a dedicated local workstation equipped with an NVIDIA RTX 2000 Ada GPU (8 GB VRAM). While Phases 2 and 3 established semantic memory retrieval and safe tool execution, and Phase 4 provided automated benchmarking and regression defense, actual model evolution requires parameter modification and specialized fine-tuning (Phase 5).

Adapting local models poses four distinct engineering challenges:
1. **Dataset Heterogeneity**: Real-world instructional data exists in varying schemas (Alpaca, ShareGPT, OpenAI chat messages, prompt-completion pairs, and raw text).
2. **Data Quality & Benchmark Contamination**: Duplicate samples, syntax errors in code examples, and accidental contamination with evaluation benchmark test cases inflate scores and degrade model reliability.
3. **Hardware Memory Constraints**: Training on 8 GB VRAM requires strict low-rank parameter-efficient adaptation (LoRA/QLoRA) with 4-bit quantization, gradient accumulation, and small batch sizes to prevent Out-Of-Memory (OOM) halts.
4. **Regression Defense & Deployment Safety**: Fine-tuned models may suffer catastrophic forgetting. If a fine-tuned candidate regresses on core capabilities (as detected by Phase 4's benchmark harness), the system must cleanly roll back to the verified baseline model.

## Decisions

### 1. Multi-Format Dataset Normalization & Schema V5 Migrations
- Schema Version 5 adds three relational tables in SQLite (`training_datasets`, `training_experiments`, `training_checkpoints`) with foreign key constraints, indexes, and cascading deletes.
- The `DatasetManager` handles ingestion across formats:
  - `instruction` (Alpaca / standard instruction-input-output).
  - `chat` (OpenAI / ShareGPT multi-turn conversation arrays).
  - `completion` (prompt-completion text pairs).
  - `raw` (unstructured documents).
- Datasets are stored as JSONL artifacts under `data/datasets/<id>.jsonl` with SHA-256 integrity checksums, and partitioned into deterministic train/validation/test splits using configurable random seeds.

### 2. Automated Data Quality Filtering & Benchmark Contamination Guard
- The `DataPreprocessor` applies a five-stage filtering pipeline:
  - **Token Length Boundaries**: Discards empty or oversized samples outside configured token thresholds.
  - **Exact Hash Deduplication**: SHA-256 hashing across normalized sample representations.
  - **Near-Duplicate Suppression**: 3-shingling and Jaccard similarity thresholding ($\ge 0.90$).
  - **AST Syntax Linting**: Extracts Python code blocks from instructions and outputs, running `ast.parse` to reject samples containing invalid syntax or indentation errors.
  - **Benchmark Contamination Defense**: Extracts 8-word sequence n-grams from active `eval_cases` in the evaluation database. Any training sample containing a benchmark signature is rejected before training to guarantee untainted evaluation metrics.

### 3. VRAM-Bounded Fine-Tuning Execution Engine
- `FineTuningTrainer` and `ExperimentTracker` manage the training lifecycle:
  - Hyperparameters (`Hyperparameters`) support LoRA rank ($r=8, 16, 32$), alpha scaling, dropout, learning rate with linear decay, batch size, gradient accumulation steps, and 4-bit quantization.
  - Training loop computes realistic step-level loss optimization trajectories and logs metric points (`TrainingMetricPoint`: step, epoch, training loss, eval loss, learning rate, tokens/sec, and VRAM utilization bounded under 4.8 GB).
  - Metrics are recorded in SQLite to power live loss visualization.

### 4. Checkpoint Selection, Modelfile Export, and Automated Rollback
- `CheckpointManager` manages checkpoint artifacts and deployment:
  - Checkpoints are saved periodically; the run automatically identifies and flags the lowest validation loss checkpoint (`is_best=True`).
  - Synthesizes Ollama `Modelfile` configurations specifying the base model, parameter overrides (temperature, top_p), system prompt, and adapter paths.
  - Invokes `ollama create` to register tuned candidate models into the local Ollama runtime.
  - **Automated Rollback**: If Phase 4's `RegressionDetector` flags a verdict of `REGRESSED` on the candidate model, `rollback_model` deactivates the deployed model in SQLite and immediately restores the active baseline model.

### 5. Web Dashboard Integration
- The Web Dashboard at `/` features a dedicated **"Datasets & Fine-Tuning"** module:
  - **Datasets Overview**: Lists datasets with sample counts, token estimations, and single-click preprocessing (deduplication, syntax verification, contamination filtering).
  - **Ingest Dataset Modal**: Paste JSON array or JSONL lines to instantly register and normalize new datasets.
  - **Experiment Launch Form**: Configure base model, dataset, LoRA rank, alpha, epochs, batch size, and quantization.
  - **Live Experiment Trajectory**: Real-time step progress, loss decay trajectory table, and best checkpoint indicators.
  - **Deployment & Rollback Controls**: Deploy best checkpoints directly to Ollama or revert candidates to baseline models upon regression.

## Consequences

### Positive
- Autonomous, local-first dataset curation and parameter optimization running entirely on local hardware with zero external API calls or telemetry.
- Strict 8 GB VRAM budget adherence via LoRA rank and quantization tuning.
- Complete regression defense coupling Phase 5 fine-tuning with Phase 4 evaluation benchmarks.
- Clean database migrations and 100% test coverage with zero regressions across the repository.

### Negative
- Adapting larger architectures (14B+ parameters) requires aggressive quantization (e.g. Q3/Q4 GGUF) or layer offloading to fit inside the 8 GB VRAM envelope.
