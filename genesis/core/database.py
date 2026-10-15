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
SCHEMA_VERSION = 8


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

SCHEMA_V8 = """
-- Autonomous Meta-Improvement cycles
CREATE TABLE IF NOT EXISTS meta_cycles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    benchmark_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'diagnosing', 'curating', 'fine_tuning', 'validating', 'completed', 'regressed', 'failed')),
    baseline_score REAL NOT NULL DEFAULT 0.0,
    candidate_score REAL NOT NULL DEFAULT 0.0,
    verdict TEXT NOT NULL CHECK (verdict IN ('pending', 'promoted', 'rolled_back', 'failed', 'inconclusive')),
    dataset_id TEXT,
    experiment_id TEXT,
    checkpoint_id TEXT,
    stages_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meta_cycles_status ON meta_cycles(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_cycles_model ON meta_cycles(model_id);

-- Dynamic Multi-Model Routes
CREATE TABLE IF NOT EXISTS model_routes (
    id TEXT PRIMARY KEY,
    task_category TEXT NOT NULL UNIQUE CHECK (task_category IN ('coding', 'reasoning', 'fast_validation', 'embedding', 'general')),
    preferred_model TEXT NOT NULL,
    fallback_model TEXT NOT NULL,
    min_vram_mb REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_model_routes_category ON model_routes(task_category);

-- Default system routes
INSERT OR IGNORE INTO model_routes (id, task_category, preferred_model, fallback_model, min_vram_mb, metadata)
VALUES
    ('route-coding', 'coding', 'qwen2.5-coder:7b', 'qwen2.5:7b', 4000.0, '{"description": "Specialized coding and debugging model"}'),
    ('route-reasoning', 'reasoning', 'qwen2.5:7b', 'qwen2.5-coder:7b', 4000.0, '{"description": "Multi-step reasoning and analytical planning model"}'),
    ('route-fast-validation', 'fast_validation', 'qwen2.5-coder:1.5b-base', 'qwen2.5:7b', 1500.0, '{"description": "Ultra-low latency model for quick validation and classification"}'),
    ('route-embedding', 'embedding', 'nomic-embed-text:latest', 'nomic-embed-text:latest', 500.0, '{"description": "Dense vector embedding provider"}'),
    ('route-general', 'general', 'qwen2.5:7b', 'qwen2.5:7b', 4000.0, '{"description": "Default general-purpose conversational and instruction model"}');
"""

SCHEMA_V9 = """
-- Agent Teams / Ensembles
CREATE TABLE IF NOT EXISTS agent_teams (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    roles_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- Multi-Agent Consensus Debate Sessions
CREATE TABLE IF NOT EXISTS debate_sessions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    team_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'consensus_reached', 'consensus_failed', 'cancelled')),
    rounds INTEGER NOT NULL DEFAULT 1,
    consensus_verdict TEXT,
    dissenting_opinions TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_debate_sessions_status ON debate_sessions(status, created_at DESC);

-- Individual Debate Turns & Critiques
CREATE TABLE IF NOT EXISTS debate_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES debate_sessions(id) ON DELETE CASCADE,
    round_num INTEGER NOT NULL,
    agent_role TEXT NOT NULL,
    model_id TEXT NOT NULL,
    stance TEXT NOT NULL CHECK (stance IN ('propose', 'critique', 'rebut', 'concur', 'dissent')),
    content TEXT NOT NULL,
    critique_target_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_debate_turns_session ON debate_turns(session_id, round_num);

-- Long-Horizon Autonomous Research Campaigns
CREATE TABLE IF NOT EXISTS research_campaigns (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'paused', 'completed', 'failed')),
    current_milestone INTEGER NOT NULL DEFAULT 0,
    total_milestones INTEGER NOT NULL DEFAULT 1,
    hypotheses_json TEXT NOT NULL DEFAULT '[]',
    findings_json TEXT NOT NULL DEFAULT '[]',
    milestones_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON research_campaigns(status, created_at DESC);

-- Default system agent team seed
INSERT OR IGNORE INTO agent_teams (id, name, description, roles_json, metadata)
VALUES (
    'team-genesis-core',
    'Genesis Core Dialectic Ensemble',
    'Primary four-role dialectic team consisting of Researcher, Architect, Implementer, and Reviewer',
    '["researcher", "architect", "implementer", "reviewer"]',
    '{"system_seeded": true}'
);
"""

SCHEMA_V10 = """
-- Knowledge Graph Nodes (concepts, hypotheses, debates, experiments, papers, findings)
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL CHECK (node_type IN ('concept', 'hypothesis', 'debate', 'experiment', 'campaign', 'metric', 'finding', 'paper', 'component')),
    label TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_label ON knowledge_nodes(label);

-- Knowledge Graph Edges (semantic, causal, and developmental relationships)
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL CHECK (relation_type IN ('hypothesizes', 'debates', 'validates', 'contradicts', 'implements', 'refines', 'derived_from', 'measured_by', 'cites', 'relates_to')),
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source ON knowledge_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_target ON knowledge_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_relation ON knowledge_edges(relation_type);

-- Autonomous Research Papers / Synthesis Publications
CREATE TABLE IF NOT EXISTS research_papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    content_markdown TEXT NOT NULL,
    campaign_id TEXT REFERENCES research_campaigns(id) ON DELETE SET NULL,
    cited_nodes_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL CHECK (status IN ('draft', 'published', 'archived')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_research_papers_status ON research_papers(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_papers_campaign ON research_papers(campaign_id);

-- Multimodal Visual Artifact Inspections
CREATE TABLE IF NOT EXISTS visual_inspections (
    id TEXT PRIMARY KEY,
    artifact_name TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK (artifact_type IN ('diagram', 'ui_screenshot', 'chart', 'loss_curve', 'image')),
    file_path TEXT,
    prompt TEXT NOT NULL,
    inspection_result TEXT NOT NULL,
    passed INTEGER NOT NULL DEFAULT 1,
    detected_elements_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_visual_inspections_type ON visual_inspections(artifact_type, created_at DESC);
"""

SCHEMA_V11 = """
-- Native Compiled Binary Extensions (hot-path acceleration kernels)
CREATE TABLE IF NOT EXISTS compiled_extensions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    language TEXT NOT NULL CHECK (language IN ('c', 'rust', 'cpp')),
    source_code TEXT NOT NULL,
    binary_path TEXT,
    status TEXT NOT NULL CHECK (status IN ('compiled', 'fallback_python', 'failed')),
    compiler_used TEXT,
    speedup_factor REAL NOT NULL DEFAULT 1.0,
    checksum TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_compiled_extensions_name ON compiled_extensions(name);
CREATE INDEX IF NOT EXISTS idx_compiled_extensions_status ON compiled_extensions(status);

-- Distributed Fleet Peer Nodes
CREATE TABLE IF NOT EXISTS fleet_nodes (
    id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    address TEXT NOT NULL,
    port INTEGER NOT NULL DEFAULT 8000,
    gpu_device TEXT,
    vram_gb REAL DEFAULT 0.0,
    status TEXT NOT NULL CHECK (status IN ('online', 'unreachable', 'syncing', 'offline')) DEFAULT 'online',
    last_heartbeat TEXT NOT NULL DEFAULT (datetime('now')),
    sync_cursor TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_fleet_nodes_status ON fleet_nodes(status);
CREATE INDEX IF NOT EXISTS idx_fleet_nodes_address ON fleet_nodes(address, port);

-- Fleet Delta Synchronization Audit Logs
CREATE TABLE IF NOT EXISTS fleet_sync_logs (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES fleet_nodes(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('push', 'pull', 'bidirectional')),
    items_synced INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('success', 'failed', 'conflict_resolved')) DEFAULT 'success',
    checksum TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_fleet_sync_logs_node ON fleet_sync_logs(node_id, created_at DESC);
"""

SCHEMA_V12 = """
-- Autonomous LLM-Driven Kernel Synthesis Experiments
CREATE TABLE IF NOT EXISTS kernel_experiments (
    id TEXT PRIMARY KEY,
    kernel_name TEXT NOT NULL,
    operation_type TEXT NOT NULL CHECK (operation_type IN ('activation', 'quantization', 'normalization', 'reduction', 'gemm')),
    iteration INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL CHECK (status IN ('synthesizing', 'compiling', 'validating', 'success', 'failed')),
    c_source TEXT NOT NULL,
    speedup_factor REAL NOT NULL DEFAULT 1.0,
    max_abs_error REAL NOT NULL DEFAULT 0.0,
    passed_numerical_check INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_kernel_exp_name ON kernel_experiments(kernel_name);
CREATE INDEX IF NOT EXISTS idx_kernel_exp_status ON kernel_experiments(status);

-- Cross-Fleet Model Parameter Merging (Linear, SLERP, TIES, DARE)
CREATE TABLE IF NOT EXISTS model_merges (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    merge_method TEXT NOT NULL CHECK (merge_method IN ('linear', 'slerp', 'ties', 'dare')),
    target_model TEXT NOT NULL,
    source_adapters_json TEXT NOT NULL DEFAULT '[]',
    weights_json TEXT NOT NULL DEFAULT '[]',
    density REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL CHECK (status IN ('pending', 'merging', 'completed', 'failed')),
    output_path TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_model_merges_method ON model_merges(merge_method);
CREATE INDEX IF NOT EXISTS idx_model_merges_status ON model_merges(status);

-- Interactive Human-in-the-Loop Steering Events
CREATE TABLE IF NOT EXISTS steering_events (
    id TEXT PRIMARY KEY,
    target_type TEXT NOT NULL CHECK (target_type IN ('daemon_cycle', 'campaign', 'debate', 'kernel_synthesis')),
    target_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('pause', 'resume', 'critique', 'override_hypothesis', 'abort', 'inject_guidance')),
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('received', 'applied', 'rejected')) DEFAULT 'received',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_steering_target ON steering_events(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_steering_status ON steering_events(status);
"""

SCHEMA_V13 = """
-- Continual Learning Streaming Memory Replay Buffers
CREATE TABLE IF NOT EXISTS replay_buffers (
    id TEXT PRIMARY KEY,
    task_domain TEXT NOT NULL CHECK (task_domain IN ('reasoning', 'coding', 'tools', 'general')),
    sample_input TEXT NOT NULL,
    sample_target TEXT NOT NULL,
    priority_score REAL NOT NULL DEFAULT 1.0,
    usage_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_replay_buffers_domain ON replay_buffers(task_domain);
CREATE INDEX IF NOT EXISTS idx_replay_buffers_priority ON replay_buffers(priority_score DESC);

-- Elastic Weight Consolidation (EWC) Checkpoints & Fisher Diagonal
CREATE TABLE IF NOT EXISTS ewc_checkpoints (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES training_experiments(id) ON DELETE CASCADE,
    task_name TEXT NOT NULL,
    importance_lambda REAL NOT NULL DEFAULT 1000.0,
    fisher_diagonal_json TEXT NOT NULL DEFAULT '{}',
    optimal_weights_json TEXT NOT NULL DEFAULT '{}',
    parameter_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_ewc_checkpoints_exp ON ewc_checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ewc_checkpoints_task ON ewc_checkpoints(task_name);

-- Neuro-Symbolic Verification Records
CREATE TABLE IF NOT EXISTS symbolic_verifications (
    id TEXT PRIMARY KEY,
    target_type TEXT NOT NULL CHECK (target_type IN ('code', 'math', 'hypothesis', 'kernel')),
    expression_or_code TEXT NOT NULL,
    constraints_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL CHECK (status IN ('verified', 'violated', 'inconclusive')),
    violations_json TEXT NOT NULL DEFAULT '[]',
    verification_latency_ms REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_symbolic_target_type ON symbolic_verifications(target_type);
CREATE INDEX IF NOT EXISTS idx_symbolic_status ON symbolic_verifications(status);
"""

SCHEMA_V14 = """
-- Neural Architecture Search & Block Pruning Experiments
CREATE TABLE IF NOT EXISTS pruning_experiments (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    target_vram_mb REAL NOT NULL,
    pruning_strategy TEXT NOT NULL CHECK (pruning_strategy IN ('structured_head', 'mlp_intermediate', 'taylor_sensitivity', 'magnitude_l1')),
    heads_pruned INTEGER NOT NULL DEFAULT 0,
    remaining_heads INTEGER NOT NULL,
    mlp_expansion_ratio REAL NOT NULL DEFAULT 1.0,
    original_parameters INTEGER NOT NULL,
    pruned_parameters INTEGER NOT NULL,
    parameter_reduction_pct REAL NOT NULL,
    estimated_vram_mb REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('planned', 'pruned', 'benchmarking', 'verified', 'rejected')),
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_pruning_model ON pruning_experiments(model_name);
CREATE INDEX IF NOT EXISTS idx_pruning_status ON pruning_experiments(status);

-- Hardware-Targeted Micro-Quantization Profiles
CREATE TABLE IF NOT EXISTS quantization_profiles (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    bits INTEGER NOT NULL CHECK (bits IN (2, 3, 4, 8)),
    strategy TEXT NOT NULL CHECK (strategy IN ('symmetric', 'asymmetric', 'awq_outlier_salient', 'gptq_hessian')),
    block_size INTEGER NOT NULL DEFAULT 64 CHECK (block_size IN (32, 64, 128)),
    outlier_channel_ratio REAL NOT NULL DEFAULT 0.01,
    original_size_mb REAL NOT NULL,
    quantized_size_mb REAL NOT NULL,
    compression_ratio REAL NOT NULL,
    mse_loss REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL CHECK (status IN ('quantizing', 'completed', 'verified', 'failed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_quant_model ON quantization_profiles(model_name);
CREATE INDEX IF NOT EXISTS idx_quant_bits ON quantization_profiles(bits);

-- Self-Speculative Decoding Inference Runs
CREATE TABLE IF NOT EXISTS speculative_runs (
    id TEXT PRIMARY KEY,
    target_model TEXT NOT NULL,
    draft_model TEXT NOT NULL,
    lookahead_k INTEGER NOT NULL DEFAULT 4,
    prompt TEXT NOT NULL,
    output_text TEXT NOT NULL,
    accepted_tokens INTEGER NOT NULL DEFAULT 0,
    total_draft_tokens INTEGER NOT NULL DEFAULT 0,
    acceptance_rate REAL NOT NULL DEFAULT 0.0,
    speedup_factor REAL NOT NULL DEFAULT 1.0,
    target_latency_ms REAL NOT NULL DEFAULT 0.0,
    speculative_latency_ms REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'fallback_autoregressive', 'failed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_speculative_models ON speculative_runs(target_model, draft_model);
CREATE INDEX IF NOT EXISTS idx_speculative_status ON speculative_runs(status);
"""

SCHEMA_V15 = """
-- Multimodal Environment Transition Predictions & Error Verifications
CREATE TABLE IF NOT EXISTS world_transitions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    action_type TEXT NOT NULL CHECK (action_type IN ('file_write', 'file_delete', 'ast_mutation', 'command_exec', 'tool_call')),
    predicted_state_json TEXT NOT NULL DEFAULT '{}',
    actual_state_json TEXT DEFAULT '{}',
    transition_error_score REAL NOT NULL DEFAULT 0.0,
    safety_predicted INTEGER NOT NULL DEFAULT 1,
    safety_actual INTEGER DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_world_transitions_session ON world_transitions(session_id);
CREATE INDEX IF NOT EXISTS idx_world_transitions_action ON world_transitions(action_type);

-- Latent Dream Trajectories & Synthetic Mental Rollouts
CREATE TABLE IF NOT EXISTS dream_trajectories (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    steps_count INTEGER NOT NULL DEFAULT 1,
    total_value_score REAL NOT NULL DEFAULT 0.0,
    pruned_branches INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('imagining', 'completed', 'pruned', 'failed')),
    trajectory_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_dream_trajectories_status ON dream_trajectories(status);
CREATE INDEX IF NOT EXISTS idx_dream_trajectories_value ON dream_trajectories(total_value_score DESC);

-- Speculative Monte Carlo Tree Search (MCTS) Runs
CREATE TABLE IF NOT EXISTS mcts_search_runs (
    id TEXT PRIMARY KEY,
    root_task TEXT NOT NULL,
    iterations INTEGER NOT NULL DEFAULT 10,
    nodes_explored INTEGER NOT NULL DEFAULT 0,
    best_path_json TEXT NOT NULL DEFAULT '[]',
    best_value REAL NOT NULL DEFAULT 0.0,
    fork_rollbacks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('searching', 'completed', 'aborted', 'failed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mcts_search_status ON mcts_search_runs(status);

-- Cross-Modal Latent Alignments
CREATE TABLE IF NOT EXISTS modal_alignments (
    id TEXT PRIMARY KEY,
    text_prompt TEXT NOT NULL,
    ast_symbol TEXT NOT NULL,
    visual_token TEXT NOT NULL,
    similarity_score REAL NOT NULL DEFAULT 0.0,
    projection_dim INTEGER NOT NULL DEFAULT 64,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_modal_alignments_score ON modal_alignments(similarity_score DESC);
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

        if current_version < 8:
            log.info("Applying schema v8...")
            conn.executescript(SCHEMA_V8)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (8, 'Schema v8: autonomous meta-improvement cycles and dynamic model routes')"
            )
            conn.commit()
            log.info("Schema v8 applied successfully.")

        if current_version < 9:
            log.info("Applying schema v9...")
            conn.executescript(SCHEMA_V9)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (9, 'Schema v9: multi-agent teams, debate consensus sessions, and research campaigns')"
            )
            conn.commit()
            log.info("Schema v9 applied successfully.")

        if current_version < 10:
            log.info("Applying schema v10...")
            conn.executescript(SCHEMA_V10)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (10, 'Schema v10: knowledge graph nodes/edges, research papers, and visual inspections')"
            )
            conn.commit()
            log.info("Schema v10 applied successfully.")

        if current_version < 11:
            log.info("Applying schema v11...")
            conn.executescript(SCHEMA_V11)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (11, 'Schema v11: compiled extensions, fleet nodes, and fleet sync logs')"
            )
            conn.commit()
            log.info("Schema v11 applied successfully.")

        if current_version < 12:
            log.info("Applying schema v12...")
            conn.executescript(SCHEMA_V12)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (12, 'Schema v12: autonomous kernel experiments, cross-fleet model merges, and steering events')"
            )
            conn.commit()
            log.info("Schema v12 applied successfully.")

        if current_version < 13:
            log.info("Applying schema v13...")
            conn.executescript(SCHEMA_V13)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (13, 'Schema v13: continual replay buffers, EWC checkpoints, and neuro-symbolic verifications')"
            )
            conn.commit()
            log.info("Schema v13 applied successfully.")

        if current_version < 14:
            log.info("Applying schema v14...")
            conn.executescript(SCHEMA_V14)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (14, 'Schema v14: neural architecture pruning, micro-quantization profiles, and speculative decoding runs')"
            )
            conn.commit()
            log.info("Schema v14 applied successfully.")

        if current_version < 15:
            log.info("Applying schema v15...")
            conn.executescript(SCHEMA_V15)
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, description) "
                "VALUES (15, 'Schema v15: world transitions, dream trajectories, mcts search runs, and modal alignments')"
            )
            conn.commit()
            log.info("Schema v15 applied successfully.")
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
