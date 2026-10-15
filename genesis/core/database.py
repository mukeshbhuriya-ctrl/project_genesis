"""
Database initialization and connection management.

Uses SQLite via aiosqlite for async operations.
Schema migrations are version-tracked.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from genesis.core.logging import get_logger

log = get_logger("database")

# ── Schema version history ──
SCHEMA_VERSION = 4

SCHEMA_V1 = """
-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    model_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- Messages within conversations
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    model_id TEXT,
    token_count INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);

-- Memory records
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT 'general',
    content TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    version INTEGER DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    backend TEXT NOT NULL,
    base_model TEXT,
    format TEXT,
    quantization TEXT,
    context_length INTEGER,
    license TEXT,
    file_path TEXT,
    file_size_bytes INTEGER,
    status TEXT DEFAULT 'available',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- Experiment records
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    objective TEXT,
    baseline_model TEXT,
    candidate_model TEXT,
    dataset_version TEXT,
    config TEXT DEFAULT '{}',
    results TEXT DEFAULT '{}',
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
"""

SCHEMA_V2 = """
-- Memory vector embeddings
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL DEFAULT 768,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Document knowledge sources
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- Chunked knowledge passages with embeddings
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_doc ON knowledge_chunks(document_id, chunk_index);
"""

SCHEMA_V3 = """
-- Tasks table for bounded orchestration
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'planning', 'running', 'completed', 'failed', 'cancelled')),
    plan TEXT DEFAULT '[]',
    current_step INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- Task steps audit log / trace
CREATE TABLE IF NOT EXISTS task_steps (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    tool_name TEXT,
    tool_arguments TEXT DEFAULT '{}',
    tool_result TEXT DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'success', 'failed', 'skipped')),
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_task_steps_task ON task_steps(task_id, step_number);
"""

SCHEMA_V4 = """
-- Evaluation benchmarks
CREATE TABLE IF NOT EXISTS eval_benchmarks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL CHECK (category IN ('reasoning', 'coding', 'tools', 'knowledge', 'memory', 'custom')),
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Evaluation benchmark test cases
CREATE TABLE IF NOT EXISTS eval_cases (
    id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL REFERENCES eval_benchmarks(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected_output TEXT,
    eval_type TEXT NOT NULL CHECK (eval_type IN ('exact_match', 'contains', 'regex', 'numeric_tolerance', 'tool_call', 'llm_rubric')),
    criteria TEXT DEFAULT '{}',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eval_cases_benchmark ON eval_cases(benchmark_id);

-- Evaluation runs
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL REFERENCES eval_benchmarks(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    total_cases INTEGER NOT NULL DEFAULT 0,
    passed_cases INTEGER NOT NULL DEFAULT 0,
    failed_cases INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_benchmark ON eval_runs(benchmark_id, started_at);

-- Evaluation results per case
CREATE TABLE IF NOT EXISTS eval_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    case_id TEXT NOT NULL REFERENCES eval_cases(id) ON DELETE CASCADE,
    passed INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    actual_output TEXT,
    error TEXT,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    tokens_generated INTEGER NOT NULL DEFAULT 0,
    failure_category TEXT NOT NULL DEFAULT 'none',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);
"""


def get_sync_connection(db_path: str | Path) -> sqlite3.Connection:
    """Create a synchronous SQLite connection with recommended pragmas."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def initialize_database(db_path: str | Path) -> None:
    """
    Create the database and apply all schema migrations.

    Safe to call multiple times — uses IF NOT EXISTS throughout.
    """
    conn = get_sync_connection(db_path)
    try:
        # Check current schema version
        try:
            cursor = conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            )
            row = cursor.fetchone()
            current_version = row[0] if row and row[0] else 0
        except sqlite3.OperationalError:
            current_version = 0

        if current_version < 1:
            log.info("Applying schema v1...")
            conn.executescript(SCHEMA_V1)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (1, 'Initial schema: conversations, messages, memories, models, experiments')"
            )
            conn.commit()
            log.info("Schema v1 applied successfully.")

        if current_version < 2:
            log.info("Applying schema v2...")
            conn.executescript(SCHEMA_V2)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (2, 'Schema v2: memory_embeddings, knowledge_documents, knowledge_chunks')"
            )
            conn.commit()
            log.info("Schema v2 applied successfully.")

        if current_version < 3:
            log.info("Applying schema v3...")
            conn.executescript(SCHEMA_V3)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (3, 'Schema v3: tasks and task_steps for bounded orchestration')"
            )
            conn.commit()
            log.info("Schema v3 applied successfully.")

        if current_version < 4:
            log.info("Applying schema v4...")
            conn.executescript(SCHEMA_V4)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (4, 'Schema v4: evaluation benchmarks, test cases, runs, and results')"
            )
            conn.commit()
            log.info("Schema v4 applied successfully.")
        else:
            log.info(f"Database schema is at version {current_version}, up to date.")

    finally:
        conn.close()



def get_db_path_from_url(db_url: str) -> Path:
    """Extract the file path from a sqlite:/// URL."""
    prefix = "sqlite:///"
    if db_url.startswith(prefix):
        return Path(db_url[len(prefix):])
    return Path(db_url)
