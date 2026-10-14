# ROADMAP

## Current Milestone: Phase 1 — Reliable Local Foundation

## Phase 0: Discovery and Baseline ✅
- [x] Inspect repository and environment
- [x] Hardware capability report
- [x] Baseline report and risk identification
- [x] Initial project scaffolding
- [x] Continuity documents created

## Phase 1: Reliable Local Foundation 🔄 (Active)
- [x] Reproducible project setup (pyproject.toml, .env, .venv)
- [x] Application configuration module
- [x] Logging infrastructure
- [x] Hardware detection utility
- [x] Model provider abstraction
- [x] Ollama backend integration
- [x] CLI entry point
- [x] Database initialization (SQLite)
- [ ] Persistent conversations (SQLite-backed)
- [ ] Basic API server (FastAPI)
- [ ] Health check endpoint
- [ ] End-to-end test: model answers a question
- [ ] Conversation survives application restart
- [ ] Initial documentation complete

**Acceptance criterion:** A real local model can answer a message and the conversation survives an application restart.

## Phase 2: Memory and Knowledge ⏳
- [ ] Persistent structured memory (CRUD)
- [ ] Memory categories and metadata
- [ ] Retrieval with relevance scoring
- [ ] Memory inspection UI
- [ ] Document indexing (embeddings)
- [ ] Source provenance tracking
- [ ] Backup and recovery
- [ ] Deduplication and contradiction handling

**Acceptance criterion:** Approved information can be saved, retrieved in a later session, inspected, and deleted.

## Phase 3: Tools and Task Orchestration ⏳
- [ ] Typed tool registry
- [ ] Permission model
- [ ] Bounded planning and execution
- [ ] Task states and error recovery
- [ ] Execution logging
- [ ] Result verification

**Acceptance criterion:** The AI can complete and verify a safe, bounded task using a real local tool.

## Phase 4: Evaluation and Feedback ⏳
- [ ] Fixed baseline benchmarks
- [ ] Task outcome recording
- [ ] Failure analysis
- [ ] Regression test suite
- [ ] Versioned skill candidates
- [ ] Comparison reports

## Phase 5: Dataset and Fine-Tuning Infrastructure ⏳
- [ ] Dataset management with versioning
- [ ] Data quality pipeline
- [ ] Experiment tracking
- [ ] LoRA/QLoRA training workflow
- [ ] Checkpoint management
- [ ] Baseline comparison and rollback

## Phase 6: Research and Capability Expansion ⏳
- [ ] Improved inference (vLLM, TensorRT-LLM)
- [ ] Stronger models as hardware permits
- [ ] Multimodal capabilities
- [ ] Advanced retrieval (hybrid search)
- [ ] Coding-specific skills

## Phase 7: Hardware and Deployment Evolution ⏳
- [ ] Multi-GPU support
- [ ] Distributed training
- [ ] Separate training/inference processes
- [ ] Workstation deployment guide
