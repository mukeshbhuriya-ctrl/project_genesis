# ADR-004: Tool Registry, Permission Boundaries, and Bounded Task Orchestration

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis is an autonomous, local-first AI research and model development platform. To progress from pure dialogue and passive memory into active model research and software experiments, the system requires the capability to invoke local tools and orchestrate multi-step execution tasks safely.

However, unconstrained AI tool invocation on a developer workstation presents major security and operational risks:
1. **Unbounded recursive loops**: A model might repeatedly execute commands without terminating or consuming unbounded compute resources.
2. **Filesystem escapes & data destruction**: Model-generated tool parameters could traverse directory trees, delete critical repository assets, or overwrite operating system files.
3. **Black-box execution without auditability**: Without structured, persistent step traces, debugging task failures and reproducing experiments becomes impossible.

## Decisions

### 1. Explicit Typed Tool Registry and Parameter Validation
- Every tool inherits from `BaseTool` with a strongly-typed `ToolDefinition` and `ToolParameter` list.
- Parameters define expected types (`string`, `integer`, `number`, `boolean`, `array`, `object`) and required fields.
- Schema export (`to_ollama_tool()`) formats definitions into standard OpenAI/Ollama function calling specifications.

### 2. Tiered Permission Model and Strict Filesystem Sandboxing
- Tool operations are categorized into three explicit safety tiers:
  - `READ_ONLY`: Safe query/inspection tools (`calculator`, `read_file`, `list_directory`).
  - `MUTATING`: Workspace state modification tools (`write_file`, `python_eval`).
  - `DANGEROUS`: Shell command execution, process control, destructive filesystem operations.
- The `PermissionManager` enforces workspace path confinement:
  - All file access is strictly validated against resolved workspace boundaries (`settings.project_root`).
  - Path traversal (`..`), drive root targets (`C:\`, `D:\`), and protected system paths (`Windows`, `System32`, `Program Files`, `.ssh`) are rejected with `SecurityError`.

### 3. Bounded Planning and Execution Loop
- `TaskPlanner` uses the local model (`qwen2.5:7b` / `qwen2.5-coder:7b`) with low temperature ($0.1$) and explicit step bounds (`max_steps = 10`) to decompose goals into JSON step arrays.
- Malformed model responses automatically fall back to deterministic or single-step execution rather than crashing.
- `ToolExecutor` wraps all tool execution in async timeouts (default: 30s) and isolates errors into structured `ToolResult` objects with millisecond execution timings.

### 4. SQLite Task Trace and Audit Logging (Schema V3)
- All tasks and individual steps are persisted in the primary SQLite database (`tasks` and `task_steps` tables).
- Each step records `tool_name`, `tool_arguments`, `tool_result`, `status` (`pending`, `running`, `success`, `failed`, `skipped`), `error`, `duration_ms`, and timestamps.
- If an individual critical step fails, the task execution halts safely and records the failure reason in SQLite without corrupting prior state.

### 5. Deterministic Result Verification
- `TaskVerifier` independently checks output artifacts (e.g. file existence, non-empty byte count, required substring content, numeric tolerance) before marking task outcomes as verified.

## Consequences

### Positive
- Full local tool execution without external cloud dependencies.
- Strict security isolation prevents rogue file writes or escapes outside the project directory.
- Full auditability: Every single tool invocation, parameter, output, and execution duration is permanently recorded in SQLite.
- Enables safe progression into automated benchmark runs and fine-tuning pipelines in later phases.

### Negative / Trade-offs
- File tools cannot manipulate files outside the project root without explicit configuration.
- Complex tool chains with dynamic loops require pre-planning or recursive task expansion.
