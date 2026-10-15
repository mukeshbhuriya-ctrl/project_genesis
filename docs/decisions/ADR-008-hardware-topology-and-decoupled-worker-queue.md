# ADR-008: Multi-Device Hardware Topology Orchestration and Decoupled Background Worker Queue

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis is a local-first, self-improving AI research and model development platform designed to operate autonomously on workstations. Following the successful completion of Phase 6 (hardware-aware context management, inference latency profiling, reflective code synthesis in isolated sandboxes, and citation-verified knowledge synthesis), Phase 7 addresses hardware scaling and deployment evolution.

As Project Genesis advances into continuous model fine-tuning, automated benchmarking, and multi-round synthesis, three architectural challenges emerge:
1. **Heterogeneous Hardware Topology & High-Parameter Models**: Running models exceeding 8 GB VRAM (such as 14B, 32B, or 70B parameter models) or extended context windows (32K+) on single-GPU or multi-GPU workstations requires dynamic, hardware-aware layer offload calculation across available GPUs and host RAM. The platform must compute precise layer distribution plans (`single_gpu`, `pipeline_parallel`, and `cpu_offload`) that guarantee allocated weights and KV-cache never exceed physical VRAM headroom.
2. **Process Decoupling & Crash Recovery**: Long-running workloads (LoRA adaptation epochs, 50+ benchmark test cases, iterative code repair) must not run directly on the primary FastAPI web server event loop. Blocking the API event loop impairs system responsiveness, and an unexpected web server restart would silently abort in-flight training experiments. Workloads require an independent, persistent work queue with atomic lease locking, worker heartbeats, and automated orphan recovery.
3. **First-Class Autonomous Tool Integration**: The reflective code synthesizer and adaptive knowledge synthesizer from Phase 6 must be exposed as first-class callable tools within the bounded task orchestrator (`TaskPlanner` and `TaskExecutor`), allowing autonomous agents to plan and execute programming objectives and research briefs as discrete steps.

## Decisions

### 1. Database Schema Version 7 Migrations
Schema Version 7 introduces two relational tables in SQLite WAL mode (`genesis/core/database.py`):
- `device_topologies`: Persists hardware compute snapshots (id, hostname, devices_json, total_vram_mb, total_ram_gb, compute_backend, created_at, metadata).
- `work_jobs`: Implements a persistent work job queue supporting decoupled workers (id, job_type, payload, status, priority, worker_id, progress, max_retries, retries, created_at, leased_at, completed_at, heartbeat_at, error, result, metadata) with status, priority, and worker indexes.

### 2. Device Topology Orchestrator (`genesis/core/device.py`)
- Discovers physical CUDA GPUs, host CPUs, and system RAM dynamically using low-level hardware diagnostics (`detect_gpu`, `detect_cpu`, `detect_ram`).
- Computes optimal model layer distribution plans based on parameter size, quantization bit depth, target context length, and safety reserve (default: 1,024 MB):
  - **Single GPU (`single_gpu`)**: Full offload when model weights and KV-cache fit completely inside primary GPU usable VRAM.
  - **Pipeline Parallelism (`pipeline_parallel`)**: Proportional layer splitting across multiple physical CUDA devices when more than one GPU is present.
  - **CPU Offload Fallback (`cpu_offload`)**: Exact split placing maximum possible layers onto the GPU while offloading excess transformer layers into host system RAM.

### 3. Decoupled Work Job Queue (`genesis/core/queue.py`)
- Persistent SQLite-backed FIFO queue supporting priority ordering (`priority DESC, created_at ASC`).
- **Atomic Lease-Based Locking**: Workers lease pending jobs using atomic `UPDATE work_jobs SET status = 'leased', worker_id = ? WHERE id = ? AND status = 'queued'`.
- **Heartbeats & Progress Tracking**: Running workers emit periodic heartbeats updating `heartbeat_at` and a normalized `progress` float (0.0 to 1.0).
- **Automated Crash Recovery**: During every lease cycle, the queue scans for orphaned jobs (leased or running jobs whose heartbeats have expired beyond a configurable lease timeout). If retry attempts remain, orphaned jobs are reclaimed, incremented in `retries`, and reset to `queued`; otherwise, they are transitioned to `failed`.
- **Retry Handling**: Configurable retry limits (`max_retries`) with automatic requeuing on transient failures and permanent failure marking when exhausted.

### 4. Standalone Job Worker Daemon (`genesis/tasks/worker.py`)
- Independent asynchronous daemon process that continuously polls the work queue, leases available jobs, spawns background heartbeat tasks, and dispatches execution to registered subsystem handlers:
  - `custom`: Benchmark and diagnostic workloads.
  - `code_synthesis`: Reflective code synthesis and sandbox testing.
  - `research_synthesis`: Citation-verified knowledge synthesis.
  - `fine_tune`: Bounded LoRA adaptation runs via `FineTuningTrainer`.
  - `eval_benchmark`: Benchmark evaluation runs via `BenchmarkRunner`.
- Graceful shutdown on SIGINT / cancellation, ensuring current turns finish or leases are cleanly released.

### 5. Built-in Tool Integration for Autonomous Orchestration
- Created `CodeSynthesizerTool` (`synthesize_code`) and `KnowledgeSynthesizerTool` (`synthesize_knowledge`) in `genesis/tools/builtin/`.
- Exported and registered both tools in `create_default_registry()` in `genesis/tools/registry.py`.
- Enables `TaskPlanner` to decompose abstract user goals into bounded steps utilizing code synthesis with sandbox verification and knowledge brief generation.

### 6. Deployment REST API (`genesis/api/routes/deployment.py`)
- `GET /api/deployment/topology`: Discovers device topology and active compute backend with optional hardware rescan.
- `POST /api/deployment/layer-plan`: Calculates multi-device layer allocation and VRAM projections for target models.
- `POST /api/deployment/jobs`: Enqueues background work jobs with configurable priority and retry policies.
- `GET /api/deployment/jobs`: Lists jobs with status and job type filters.
- `GET /api/deployment/jobs/{job_id}`: Queries status, progress, and results of a specific work job.
- `POST /api/deployment/jobs/{job_id}/cancel`: Cancels queued or running jobs.

### 7. CLI Worker Command (`genesis/cli.py`)
- Exposes `genesis worker` CLI command with `--worker-id`, `--poll-interval`, and `--heartbeat-interval` options to run the worker daemon in detached or foreground console sessions.

### 8. Web Dashboard Deployment & Hardware UI (`genesis/web/`)
- Added dedicated **"Deployment & Topologies"** tab to the web dashboard.
- Features real-time compute device inventory, an interactive multi-device layer planner with dynamic offload strategy visualization, and a live background worker job queue monitor with progress indicators and job cancellation controls.

## Consequences

### Positive
- Heavy model fine-tuning and benchmark runs no longer block the FastAPI web server event loop.
- Workers can be run as separate OS background processes or across dedicated console sessions.
- In-flight jobs survive worker crashes and are automatically reclaimed and retried without data loss.
- High-parameter models (14B, 32B, 70B) are dynamically modeled with safe layer distribution plans, preventing CUDA OOM crashes.
- Autonomous task planner can deploy code synthesis and knowledge research as native tool steps.

### Negative
- Requires maintaining the `work_jobs` database table and lease timeouts.
- Background worker processes must be managed (started/stopped) either via CLI or task runner.

## Compliance
- All operations remain strictly local-first with zero external cloud or network dependencies.
- Hardware safety reserves (1,024 MB buffer) are strictly enforced across all distribution calculations.
