# ADR-011: Multi-Modal Vision Inspection, Cross-Campaign Meta-Memory Knowledge Graph, and Autonomous Background Research Daemon

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis operates as a self-improving, local-first AI research and model development platform on Windows workstations equipped with local hardware accelerators (NVIDIA RTX 2000 Ada, 8 GB VRAM). Following Phase 9 (multi-agent specialization, dialectic consensus debates, and autonomous research campaigns), Phase 10 advances the system into **multi-modal vision processing, a relational and semantic meta-memory knowledge graph, and an autonomous background research daemon**.

Prior to Phase 10, the platform was limited to textual representations and linear memory retrieval. Three critical capabilities were required:
1. **Multi-Modal Vision & Visual Artifact Inspection**: Real AI research produces and consumes visual artifacts—architectural component diagrams (SVG), dashboard layout renders, and training loss curves. Project Genesis required local visual parsing without external cloud dependencies, combining deterministic header/DOM/geometry parsing with vision models to audit generated visual assets.
2. **Cross-Campaign Meta-Memory Knowledge Graph**: Linear memory queries and episodic lists fail to represent the relational topology between concepts, hypotheses, consensus debate verdicts, code prototypes, and empirical findings across disparate research campaigns. A graph engine was needed to support BFS shortest-path queries, neighborhood traversals, subgraph extractions, and vector-similarity node matching.
3. **Autonomous Background Research Daemon**: Scientific inquiry requires an uninterrupted, closed-loop engine that operates continuously: selecting research topics, orchestrating multi-agent consensus debates, prototyping implementations in a sandboxed runtime, synthesizing formal peer-reviewed IMRaD publications (Introduction, Methods, Results, and Discussion), and integrating discoveries into the knowledge graph and long-term memory.

## Decisions

### 1. Database Schema Version 10 Migrations (`genesis/core/database.py`)
Schema Version 10 introduces four relational tables in SQLite WAL mode:
- `knowledge_nodes`: Stores graph nodes (`id`, `node_type`, `label`, `properties_json`, `embedding`, `created_at`, `updated_at`, `metadata`). Supported types: `concept`, `hypothesis`, `debate`, `experiment`, `campaign`, `metric`, `finding`, `paper`, `component`. Index: `idx_knodes_type`.
- `knowledge_edges`: Stores directed weighted semantic relationships (`id`, `source_id`, `target_id`, `relation_type`, `weight`, `created_at`, `metadata`). Relations include `relates_to`, `supports`, `refutes`, `derives_from`, `implements`, `evaluates`, `synthesizes`. Indexes: `idx_kedges_source`, `idx_kedges_target`, `idx_kedges_relation`.
- `research_papers`: Stores formal scientific publications (`id`, `title`, `abstract`, `content_markdown`, `campaign_id`, `cited_nodes_json`, `status`, `created_at`, `updated_at`, `metadata`). Indexes: `idx_papers_status`, `idx_papers_campaign`.
- `visual_inspections`: Audit trail of visual evaluations (`id`, `artifact_name`, `artifact_type`, `format`, `width`, `height`, `passed`, `elements_detected_json`, `missing_elements_json`, `analysis_report`, `metadata`, `created_at`). Indexes: `idx_visual_type`, `idx_visual_passed`.

### 2. Multi-Modal Vision Engine (`genesis/models/vision.py`)
- `VisionRequest` & `VisionResponse`: Strongly typed dataclasses specifying image payloads (base64, data URI, raw SVG, or binary), artifact categories, prompt guidelines, and expected structural elements.
- `VisionInspector`:
  - Deterministic binary header decoding for PNG (IHDR chunks), GIF (screen descriptors), BMP (BITMAPINFOHEADER), and JPEG (SOF0/SOF2 segments) without requiring bulky external C-extension libraries.
  - XML/SVG structural analysis verifying coordinate viewports and text element presence against expected specifications.
  - Vision model prompting via local Ollama backend for high-level semantic validation when supported.
  - Persists all visual validation runs into `visual_inspections` for regression defense.

### 3. Cross-Campaign Meta-Memory Knowledge Graph (`genesis/memory/graph.py`)
- `KnowledgeNode` & `KnowledgeEdge`: Typed dataclasses representing vertices and directed semantic edges with customizable property dictionaries and packed 768-dimensional float32 vector embeddings.
- `MetaMemoryGraph`:
  - Node and edge CRUD operations with SQLite transactions.
  - Neighborhood traversal with direction filters (`in`, `out`, `both`).
  - Graph pathfinding via Breadth-First Search (`find_path`) returning step sequences between arbitrary vertices.
  - Subgraph extraction (`get_subgraph`) up to depth $k$ around center vertices.
  - Semantic node search (`semantic_search_nodes`) using cosine similarity over packed vector embeddings.
  - Automated graph ingestion from research campaigns (`sync_from_campaign`), mapping campaign nodes, hypotheses, milestones, and debate consensus into a cohesive graph topology.

### 4. Autonomous Background Research Daemon (`genesis/research/daemon.py`)
- `ResearchPaper`: Structured model for scientific publications formatted in standard IMRaD structure.
- `AutonomousResearchDaemon`:
  - Coordinates closed-loop research cycles:
    1. **Hypothesis & Topic Formulation**: Selects or receives research problem.
    2. **Dialectic Multi-Agent Debate**: Convenes specialized ensemble for multi-round deliberation and consensus synthesis.
    3. **Empirical Sandbox Prototyping**: Synthesizes executable Python verification code and evaluates results in `CodeSandbox`.
    4. **Formal Paper Synthesis**: Compiles complete IMRaD markdown publication documenting background, methodology, empirical metrics, and conclusions.
    5. **Graph & Memory Weaving**: Links paper and concept nodes into the knowledge graph and records findings in `MemoryStore`.

### 5. Phase 10 REST API Routes (`genesis/api/routes/`)
- `/api/graph/`:
  - `GET /nodes` & `POST /nodes`: Node management and type filtering.
  - `GET /edges` & `POST /edges`: Edge management and relationship weighting.
  - `GET /neighbors/{id}`: Directional neighbor retrieval.
  - `GET /path`: BFS shortest path finding between nodes.
  - `GET /subgraph`: Localized subgraph extraction.
  - `POST /search`: Vector-semantic similarity node search.
  - `POST /sync/campaigns`: Cross-campaign knowledge synchronization.
  - `GET /export`: Complete graph topological export with metrics.
- `/api/multimodal/`:
  - `POST /inspect`: Inspects and validates visual artifacts.
  - `GET /inspections`: Retrieves historical visual audit log.
- `/api/research/daemon/`:
  - `POST /cycle`: Triggers an autonomous closed-loop research cycle.
  - `GET /papers` & `GET /papers/{id}`: Lists and reads published papers.

### 6. Interactive Web Dashboard UI (`genesis/web/`)
- Added dedicated **Meta-Graph & Papers** navigation tab.
- Knowledge Graph Explorer with node type filters, interactive node cards, pathfinding tools, and subgraph inspector.
- Autonomous Publications manager with IMRaD reader and cycle dispatch modal.
- Multi-Modal Visual Artifact Inspector with diagram payload submission and verification report drawer.

## Consequences

### Positive
- Fully closed-loop autonomous scientific research can now be initiated, debated, prototyped, published, and stored without human intervention.
- Cross-campaign learnings are permanently interconnected in a searchable knowledge graph.
- Generated diagrams and visual assets are verifiable locally without cloud APIs.
- Retains 100% test pass rate (228/228 tests passing).

### Negative / Trade-Offs
- Graph pathfinding scales with BFS memory; depth limits are enforced on subgraph extraction to keep traversal bounded.
- Raster image parsing is currently restricted to metadata, dimensions, and header validation in the absence of a dedicated local vision model (e.g. `llava` / `llama3.2-vision`).
