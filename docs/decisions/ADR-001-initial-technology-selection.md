# ADR-001: Initial Technology Selection

## Status
Accepted

## Date
2026-10-14

## Context
Project Genesis needs a technology stack that:
- Runs on Windows 11 natively
- Supports local AI model inference with GPU acceleration
- Is maintainable for years
- Minimizes external dependencies
- Can scale from a laptop to a workstation

## Decision

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12+ | ML ecosystem, type hints, async support |
| API Framework | FastAPI | Async, OpenAPI, WebSocket, lightweight |
| Database | SQLite (via aiosqlite) | Zero-config, file-based, sufficient for single-user |
| Model Backend | Ollama (primary) | Windows-native, manages GPU, model lifecycle |
| Package Manager | pip + pyproject.toml | Universal, reliable; uv as optional upgrade |
| Frontend | Vanilla HTML/CSS/JS | No build step, no framework churn |
| Testing | pytest | Standard, extensive ecosystem |
| Version Control | Git | Standard, with descriptive atomic commits |

## Alternatives Considered

- **PostgreSQL**: More powerful but requires a running server. Deferred until concurrent access is needed.
- **Django**: Full-featured but heavy for an API-centric application.
- **LangChain**: Provides abstractions but adds opaque complexity and rapid API churn. We build our own thin abstractions.
- **React/Vue frontend**: Adds build tooling complexity. Deferred to later phases.
- **Docker**: Useful for deployment but adds overhead. Kept optional.

## Consequences

- SQLite limits concurrent write throughput (acceptable for single-user).
- Ollama must be installed separately (documented in setup guide).
- No hot-reload frontend dev server initially.
- Python 3.12+ excludes some older environments (acceptable tradeoff for modern features).
