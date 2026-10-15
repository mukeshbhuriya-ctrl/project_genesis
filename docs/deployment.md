# Workstation Deployment & Process Separation Guide

## Overview

Project Genesis is designed for local workstations running Windows 11 with NVIDIA RTX GPUs. 
Phase 7 introduces **Process Separation** and **Heterogeneous Device Topology Orchestration**, allowing resource-heavy fine-tuning, benchmark evaluation suites, and multi-round code/knowledge synthesis to run in dedicated background worker processes without degrading the responsiveness of the primary FastAPI web server.

---

## 1. Architecture Topology

```
┌────────────────────────────────────────────────────────┐
│                   Web Browser / UI                     │
│               http://localhost:8000                    │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / REST
┌──────────────────────────▼─────────────────────────────┐
│             Genesis API Server (FastAPI)               │
│          genesis serve --host 127.0.0.1 --port 8000    │
└──────────────┬───────────────────────────┬─────────────┘
               │                           │
        Enqueue Jobs                Discover Topology
               │                           │
┌──────────────▼───────────────────────────▼─────────────┐
│          SQLite WAL Database (data/genesis.db)         │
│          Tables: work_jobs, device_topologies          │
└──────────────────────────┬─────────────────────────────┘
                           │
                 Lease / Heartbeat Locks
                           │
┌──────────────────────────▼─────────────────────────────┐
│         Decoupled Background Job Worker Daemon         │
│                     genesis worker                     │
│   Executes: Fine-Tuning, Benchmarks, Code Synthesis    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Running Services on a Workstation

### Service 1: Ollama Model Runtime (Persistent Background Daemon)
Ensure Ollama is running locally with local storage on NVMe:
```powershell
$env:OLLAMA_MODELS = "D:\project_genesis\data\models\ollama"
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

### Service 2: Genesis Primary Web & API Server
Launches the FastAPI application serving REST endpoints and the single-page dashboard:
```powershell
.venv\Scripts\python.exe -m genesis.cli serve --host 127.0.0.1 --port 8000
```
- Web Dashboard: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### Service 3: Background Job Worker Daemon
Launches an autonomous, decoupled worker leasing jobs from SQLite:
```powershell
.venv\Scripts\python.exe -m genesis.cli worker --poll-interval 2.0 --heartbeat-interval 10.0
```
Multiple workers can be started concurrently across separate terminal windows or as Windows scheduled background tasks. SQLite WAL mode and atomic lease locking guarantee that each job is leased to exactly one worker.

---

## 3. Hardware Topology & Layer Offload Planning

The `DeviceOrchestrator` (`genesis/core/device.py`) models hardware capabilities and calculates optimal layer offloading:

### Strategy 1: Single GPU (`single_gpu`)
- **Condition**: Base model weights + target KV-cache + 512 MB CUDA buffer + 1,024 MB safety reserve $\le$ physical GPU VRAM.
- **Example**: Qwen 2.5 7B (Q4_K_M ~4.7 GB) on RTX 2000 Ada (8 GB VRAM) at 4K context. All 28 layers run directly on the GPU.

### Strategy 2: Pipeline Parallel Multi-GPU (`pipeline_parallel`)
- **Condition**: Multiple CUDA GPUs detected.
- **Allocation**: Layers are distributed across GPUs proportional to their available free VRAM.

### Strategy 3: CPU Layer Offloading (`cpu_offload`)
- **Condition**: Model parameters exceed available GPU VRAM (e.g. 14B, 32B, or 70B models).
- **Allocation**: Allocates maximum feasible layers onto the GPU up to the safety reserve headroom; remaining transformer layers are offloaded to host CPU RAM.

---

## 4. Crash Recovery & Resilience

- **Heartbeats**: Active workers update `heartbeat_at` and `progress` every 10 seconds.
- **Orphan Recovery**: If a worker process is terminated (power outage, laptop lid close, SIGKILL), the next worker cycle detects that its heartbeat expired beyond the lease timeout (default: 300s). The job is automatically reclaimed, incremented in `retries`, and reset to `queued` status.
- **Retry Bounds**: Failed jobs retry up to `max_retries` (default: 3) before transitioning permanently to `failed`.

---

## 5. Deployment REST API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/deployment/topology` | `GET` | Return hardware devices, VRAM, and RAM |
| `/api/deployment/layer-plan` | `POST` | Calculate layer distribution plan for a target model |
| `/api/deployment/jobs` | `POST` | Enqueue a background work job |
| `/api/deployment/jobs` | `GET` | List queued, running, and completed jobs |
| `/api/deployment/jobs/{id}` | `GET` | Retrieve job status, progress, and results |
| `/api/deployment/jobs/{id}/cancel` | `POST` | Cancel a pending or active job |
