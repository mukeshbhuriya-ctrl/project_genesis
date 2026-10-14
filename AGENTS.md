# AGENTS.md — Developer & Agent Instructions

## Project Identity

**Project Genesis** is a local-first, self-improving AI research and model development platform.
It runs on Windows, uses locally available models, and is designed for years of incremental improvement.

## Mandatory Startup Protocol

Before making any substantive changes:

1. Read this file (`AGENTS.md`).
2. Read `PROJECT_STATE.md` for the current working state.
3. Read `ROADMAP.md` for priorities and milestones.
4. Read `docs/architecture.md` for subsystem boundaries.
5. Inspect `git log --oneline -10` for recent changes.
6. Run `python -m pytest tests/ -q` to verify the current test baseline.
7. Continue from the highest-value unfinished task listed in `PROJECT_STATE.md`.

## Session Close Protocol

Before ending a session:

1. Update `PROJECT_STATE.md` with what was accomplished and what remains.
2. Add any new architecture decisions to `docs/decisions/`.
3. Update `ROADMAP.md` if priorities changed.
4. Run tests and record results.
5. Write precise next steps for the next agent.
6. Commit with a descriptive message.

## Repository Conventions

- **Language:** Python 3.12+ for backend; vanilla HTML/CSS/JS for frontend.
- **Package management:** Use `uv` if available, otherwise `pip` with a virtual environment (`.venv/`).
- **Configuration:** Environment variables via `.env`, loaded by `genesis/config.py`.
- **Data directories:** All mutable data under `data/` (gitignored for large files). Structure defined in `docs/architecture.md`.
- **Testing:** `pytest` in `tests/`. Mocked tests must not be presented as proof of real integration.
- **Formatting:** Follow PEP 8. Use type hints.
- **Imports:** Standard library → third-party → local, separated by blank lines.
- **Documentation:** Every subsystem has a corresponding `docs/*.md` file.
- **Git:** Atomic, descriptive commits. Never force-push main.

## Security Rules

- No secrets in source code or git history.
- No external API calls unless explicitly configured and documented.
- Generated code runs in controlled environments with limited filesystem access.
- Destructive operations (DB drops, file deletions, model overwrites) require explicit authorization.
- Never silently download multi-GB artifacts.

## What NOT to Do

- Do not claim a feature works without running its tests.
- Do not fabricate performance metrics or progress bars.
- Do not replace working code without documenting why.
- Do not introduce unnecessary framework lock-in.
- Do not assume GPU availability without checking `nvidia-smi`.
- Do not confuse memory (retrieval) with training (weight modification).
