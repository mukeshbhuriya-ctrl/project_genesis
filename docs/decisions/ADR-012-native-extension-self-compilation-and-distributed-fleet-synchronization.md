# ADR-012: Real-Time Self-Compilation, Native Binary Extensions, and Distributed Multi-Workstation Fleet Synchronization

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis operates as an autonomous, self-improving local AI research platform designed for multi-year incremental enhancement without cloud lock-in.
Following the successful delivery of Phase 10 (multi-modal vision processing, meta-memory knowledge graph, and autonomous research daemon), Phase 11 tackles two foundational capability bottlenecks:
1. **Host-Level Computational Acceleration (Hot-Path Kernels)**: Pure Python implementations of tensor operations (dense vector dot products, SIMD float-to-int8 quantization, matrix-vector multiplication) incur significant interpreter overhead. Real-time self-compilation enables the system to dynamically synthesize optimized C/Rust shared libraries, detect local compilers (`clang`, `gcc`, `cl.exe`, `rustc`), build binaries, and load them via `ctypes`. Crucially, on systems where no native C/Rust compiler is present in `PATH`, the subsystem must gracefully degrade to verified, zero-overhead pure Python fallbacks with zero runtime failures.
2. **Distributed Multi-Workstation Fleet Synchronization**: Single-workstation setups are limited by physical VRAM and compute bounds. Distributed fleet synchronization enables peer workstations across local LANs (or air-gapped transport bundles) to discover one another, track heartbeats and hardware capabilities, export incremental delta bundles, and import knowledge nodes, edges, and research publications using deterministic timestamp-based conflict resolution.

## Decisions

### 1. Database Schema Version 11 Migrations (`genesis/core/database.py`)
Schema Version 11 introduces three relational tables in SQLite WAL mode:
- `compiled_extensions`: Stores dynamically synthesized and compiled kernel records (`id`, `name`, `language`, `source_code`, `binary_path`, `status`, `compiler_used`, `speedup_factor`, `checksum`, `created_at`, `metadata`). Supported statuses: `compiled`, `fallback_python`, `failed`. Indexes: `idx_compiled_extensions_name`, `idx_compiled_extensions_status`.
- `fleet_nodes`: Stores workstation membership in the distributed cluster (`id`, `hostname`, `address`, `port`, `gpu_device`, `vram_gb`, `status`, `last_heartbeat`, `sync_cursor`, `created_at`, `metadata`). Supported statuses: `online`, `unreachable`, `syncing`, `offline`. Indexes: `idx_fleet_nodes_status`, `idx_fleet_nodes_address`.
- `fleet_sync_logs`: Immutable audit journal for synchronization transactions (`id`, `node_id`, `entity_type`, `direction`, `items_synced`, `status`, `checksum`, `created_at`, `metadata`). Foreign key `node_id REFERENCES fleet_nodes(id) ON DELETE CASCADE`. Index: `idx_fleet_sync_logs_node`.

### 2. Real-Time Self-Compilation Engine (`genesis/core/compiler.py`)
- `CompilationResult`: Strongly typed record tracking compilation outcome, compiler utilized, binary filesystem path, speedup metrics, and execution diagnostics.
- `NativeExtensionManager`:
  - **Toolchain Discovery**: Detects locally installed compilers (`clang`, `gcc`, `cl`, `rustc`) using `shutil.which`.
  - **Dynamic Compilation**: Generates C/Rust sources, executes compiler processes with `-O3` / `/O2` optimization and shared library flags (`-shared -fPIC` or `/LD`), dynamic DLL loading via `ctypes.CDLL`, and signature binding.
  - **Resilient Fallback Mode**: When no compiler is detected or compilation errors occur, gracefully registers mathematically equivalent pure Python kernels with `status="fallback_python"`, preventing any downtime or exceptions.
  - **Built-in Canonical Kernels**: Implements `vector_dot_product`, `fast_quantize_int8`, and `matrix_vector_mul`.
  - **Benchmarking & Profiling**: Automated performance benchmarking (`benchmark_kernel`) measuring speedup factors against naive baselines and recording metrics into SQLite.

### 3. Distributed Fleet Synchronization Engine (`genesis/deployment/fleet.py`)
- `FleetNode` & `FleetSyncResult`: Typed models representing cluster nodes and replication events.
- `FleetManager`:
  - **Local Host Self-Registration**: Automatically registers host machine as `node-self-{hash}` with hardware detection via `DeviceOrchestrator` (`detect_topology`).
  - **Node Management & Heartbeats**: Node CRUD with heartbeat tracking (`update_heartbeat`).
  - **Incremental Delta Export (`export_delta_bundle`)**: Extracts newly created/updated `knowledge_nodes`, `knowledge_edges`, and `research_papers` since an ISO timestamp cursor, generates a deterministic SHA-256 bundle checksum.
  - **Conflict-Resolving Import (`import_delta_bundle`)**: Merges delta bundles using `excluded.updated_at >= existing.updated_at` timestamp-based conflict resolution. Enforces parent existence checks (e.g. verifying `campaign_id` and ensuring `source_node_id` exists in `fleet_nodes`) to ensure strict foreign key integrity under SQLite `PRAGMA foreign_keys=ON`.
  - **Peer-to-Peer Replication (`sync_with_peer`)**: Asynchronously pulls and applies delta bundles from remote nodes via HTTP.

### 4. Phase 11 REST API Routes (`genesis/api/routes/`)
- `/api/compiler/`:
  - `GET /compilers`: Detects local host compilers.
  - `POST /compile`: Compiles dynamic C/Rust extensions or registers Python fallbacks.
  - `GET /extensions`: Lists all registered extensions and active speedup factors.
  - `POST /benchmark/{name}`: Benchmarks execution latency and throughput.
  - `POST /invoke/{name}`: Invokes a registered extension or fallback kernel.
- `/api/fleet/`:
  - `GET /nodes` & `POST /nodes`: Cluster node discovery and manual registration.
  - `GET /nodes/{id}` & `DELETE /nodes/{id}`: Node retrieval and eviction.
  - `POST /nodes/{id}/ping`: Updates peer heartbeat.
  - `GET /export-delta`: Exports incremental delta bundle.
  - `POST /import-delta`: Merges an incoming delta bundle into local database.
  - `POST /sync/{id}`: Triggers network delta synchronization with peer.
  - `GET /logs`: Lists historical replication audit logs.

### 5. Interactive Web Dashboard Module (`genesis/web/`)
- Dedicated **Fleet & Compilers** navigation tab (`#pane-fleet`).
- Global status cards for host toolchains, compiled extensions, fleet nodes, and logged delta events.
- Toolchain inspection grid (`clang`, `gcc`, `cl`, `rustc`) with status pills.
- Native extension kernel registry table with one-click benchmark triggers.
- Multi-workstation fleet topology cards with ping and delta-sync buttons.
- Complete delta synchronization audit log table.
- Interactive modals for peer registration (`#fleet-register-modal`) and offline delta bundle import (`#fleet-import-modal`).

## Consequences

### Positive
- Accelerated hot-path tensor routines with zero required third-party package dependencies beyond Python standard library (`ctypes`, `subprocess`, `hashlib`).
- Guaranteed operational stability on all environments through transparent Python fallbacks.
- Multi-machine scalability enabling multi-workstation clusters to pool research findings, empirical papers, and knowledge graphs.
- Fully offline and air-gapped support via file-based delta bundle export/import.
- Repository test baseline expanded to 241 passing tests with 100% pass rate.

### Neutral / Trade-offs
- Pure Python fallback kernels execute at standard interpreter speeds (~1.0x) until native compilers are installed.
- Cross-workstation network sync assumes reachability over LAN or VPN on the configured port (default 8000).
