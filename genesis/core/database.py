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
SCHEMA_VERSION = 7

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

SCHEMA_V5 = """
-- Training datasets
CREATE TABLE IF NOT EXISTS training_datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    format TEXT NOT NULL CHECK (format IN ('instruction', 'chat', 'completion', 'raw')),
    file_path TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    checksum TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Training experiments / runs
CREATE TABLE IF NOT EXISTS training_experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_model TEXT NOT NULL,
    dataset_id TEXT NOT NULL REFERENCES training_datasets(id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'preparing', 'training', 'completed', 'failed', 'cancelled')),
    hyperparameters TEXT DEFAULT '{}',
    current_epoch INTEGER NOT NULL DEFAULT 0,
    total_epochs INTEGER NOT NULL DEFAULT 1,
    current_step INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    training_loss REAL,
    eval_loss REAL,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    metrics TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_training_experiments_dataset ON training_experiments(dataset_id);

-- Checkpoints and exported models
CREATE TABLE IF NOT EXISTS training_checkpoints (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES training_experiments(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    epoch REAL NOT NULL,
    training_loss REAL,
    eval_loss REAL,
    file_path TEXT NOT NULL,
    format TEXT NOT NULL DEFAULT 'adapter',
    is_best INTEGER NOT NULL DEFAULT 0,
    deployed_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_training_checkpoints_exp ON training_checkpoints(experiment_id, step);
"""

SCHEMA_V6 = """
-- Hardware & Inference Profiling records
CREATE TABLE IF NOT EXISTS research_profiles (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    backend TEXT NOT NULL DEFAULT 'ollama',
    context_length INTEGER NOT NULL DEFAULT 4096,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    ttft_ms REAL NOT NULL DEFAULT 0.0,
    tokens_per_second REAL NOT NULL DEFAULT 0.0,
    inter_token_latency_ms REAL NOT NULL DEFAULT 0.0,
    total_latency_ms REAL NOT NULL DEFAULT 0.0,
    gpu_vram_allocated_mb REAL NOT NULL DEFAULT 0.0,
    kv_cache_vram_mb REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_research_profiles_model ON research_profiles(model_id, created_at);

-- Code synthesis and self-repair experiments
CREATE TABLE IF NOT EXISTS code_experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    objective TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'python',
    initial_code TEXT,
    final_code TEXT,
    tests TEXT,
    test_results TEXT DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('pending', 'generating', 'verifying', 'repairing', 'passed', 'failed', 'error')),
    iterations INTEGER NOT NULL DEFAULT 0,
    max_iterations INTEGER NOT NULL DEFAULT 3,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_code_experiments_status ON code_experiments(status, created_at);

-- Multi-round adaptive knowledge synthesis briefs
CREATE TABLE IF NOT EXISTS research_synthesis (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    query TEXT NOT NULL,
    sources_count INTEGER NOT NULL DEFAULT 0,
    citations TEXT DEFAULT '[]',
    brief_markdown TEXT,
    confidence_score REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL CHECK (status IN ('pending', 'synthesizing', 'completed', 'failed')),
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_research_synthesis_topic ON research_synthesis(topic);
"""

SCHEMA_V7 = """
-- Hardware & Device Topology Snapshots
CREATE TABLE IF NOT EXISTS device_topologies (
    id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    devices_json TEXT NOT NULL DEFAULT '[]',
    total_vram_mb REAL NOT NULL DEFAULT 0.0,
    total_ram_gb REAL NOT NULL DEFAULT 0.0,
    compute_backend TEXT NOT NULL DEFAULT 'cuda',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- Decoupled Work Job Queue for Background Workers
CREATE TABLE IF NOT EXISTS work_jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL CHECK (job_type IN ('fine_tune', 'eval_benchmark', 'knowledge_index', 'code_synthesis', 'research_synthesis', 'custom')),
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('queued', 'leased', 'running', 'completed', 'failed', 'cancelled')),
    priority INTEGER NOT NULL DEFAULT 10,
    worker_id TEXT,
    progress REAL NOT NULL DEFAULT 0.0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    retries INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    leased_at TEXT,
    completed_at TEXT,
    heartbeat_at TEXT,
    error TEXT,
    result TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_work_jobs_status ON work_jobs(status, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_work_jobs_worker ON work_jobs(worker_id, status);
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

        if current_version < 5:
            log.info("Applying schema v5...")
            conn.executescript(SCHEMA_V5)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (5, 'Schema v5: training datasets, experiments, and checkpoints')"
            )
            conn.commit()
            log.info("Schema v5 applied successfully.")

        if current_version < 6:
            log.info("Applying schema v6...")
            conn.executescript(SCHEMA_V6)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (6, 'Schema v6: research profiles, code experiments, and research synthesis')"
            )
            conn.commit()
            log.info("Schema v6 applied successfully.")

        if current_version < 7:
            log.info("Applying schema v7...")
            conn.executescript(SCHEMA_V7)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (7, 'Schema v7: device topologies and decoupled work job queue')"
            )
            conn.commit()
            log.info("Schema v7 applied successfully.")
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
