# ADR-003: Dense Vector Memory, Hybrid Retrieval, and Document Indexing

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis requires long-term continuity through persistent structured memory and document retrieval. A pure lexical text search (`LIKE %query%`) cannot match semantically related concepts or handle synonyms and paraphrases. Conversely, pure vector search cannot reliably locate exact keyword tokens or identifiers. Furthermore, unconstrained memory accumulation leads to duplicate and contradictory assertions.

## Decision
1. **Embedding Model & Vector Dimension:** Integrated local `nomic-embed-text` via Ollama (`POST /api/embed`), generating 768-dimensional dense vector embeddings.
2. **Compact Binary Serialization:** Serialized vector embeddings into binary IEEE 754 float32 blobs (`struct.pack/unpack`), storing 768 dimensions in exactly 3,072 bytes per record in SQLite without requiring heavyweight C++ vector extensions.
3. **Database Schema v2:**
   - `memory_embeddings`: Linked to `memories` table for vector similarity search.
   - `knowledge_documents` and `knowledge_chunks`: Ingesting local files with SHA256 checksum tracking, paragraph chunking, and passage embeddings.
4. **Hybrid Retrieval Strategy:** Implemented fused scoring:
   $$\text{score} = \alpha \cdot S_{\text{vector}} + (1 - \alpha) \cdot S_{\text{lexical}} \times \text{confidence}$$
   balancing dense semantic proximity ($\alpha=0.6$) with lexical keyword hits and confidence weighting.
5. **Memory Consolidation:** Automatic deduplication suppresses near-identical memory candidates ($\ge 0.90$ similarity), incrementing record versions and updating timestamps rather than bloating the database.
6. **Disaster Recovery:** Implemented point-in-time atomic online backup snapshots via `sqlite3.Connection.backup()` with automated `PRAGMA integrity_check` verification.
7. **User Interface:** Developed a responsive single-page Web application (`genesis/web/`) using HTML5, Vanilla CSS design tokens, and client-side JavaScript, mounted at the root `/` endpoint of the FastAPI server.

## Consequences
- **Positive:** Phase 2 acceptance criterion fully verified. Memory CRUD, hybrid semantic ranking, document ingestion, and atomic backup snapshots operate completely offline with 100% test coverage (58/58 passing).
- **Negative:** Generating embeddings for large batches of documents introduces minor CPU/GPU compute load during initial indexing.
- **Mitigation:** Checksum-based deduplication skips re-indexing for unchanged documents.
