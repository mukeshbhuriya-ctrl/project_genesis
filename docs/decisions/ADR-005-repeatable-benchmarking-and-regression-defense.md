# ADR-005: Repeatable Benchmarking, Failure Taxonomies, and Automated Regression Defense

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis is a local-first, self-improving AI research and model development platform. As the system introduces prompt improvements, memory updates, quantization formats, and eventually local LoRA/QLoRA fine-tuning, it requires a deterministic mechanism to answer a fundamental engineering question:

*Did the latest model update or system change make the AI better, worse, or induce regressions?*

Without standard baseline benchmarks, automated error categorization, and historical baseline regression detection, model iteration quickly degenerates into subjective impression and unverified claims.

## Decisions

### 1. Deterministic Multi-Domain Benchmark Suites (Schema V4)
- Standard baseline benchmarks are seeded into SQLite tables (`eval_benchmarks`, `eval_cases`, `eval_runs`, `eval_results`):
  - **Reasoning Baseline v1 (`bm-reasoning-v1`)**: Multi-step mathematical reasoning, constraint satisfaction, and logical deduction.
  - **Python Coding Baseline v1 (`bm-coding-v1`)**: Algorithmic Python implementation, string manipulation, and data structure operations.
  - **Tool Invocation Baseline v1 (`bm-tools-v1`)**: Proper tool identification, parameter structuring, and calculation formatting.
- Deterministic grading rubrics support:
  - `EXACT_MATCH`: Normalized case-insensitive matching.
  - `CONTAINS`: Essential key token/phrase inclusion.
  - `REGEX`: Regular expression pattern matching for code syntax.
  - `NUMERIC_TOLERANCE`: Float distance evaluation within configurable epsilon bounds.
  - `TOOL_CALL`: Function naming and argument structure validation.

### 2. Actionable Failure Taxonomy and Diagnostic Engine
- Failures are automatically categorized by `FailureAnalyzer` into a discrete taxonomy:
  - `SYNTAX_ERROR`: Code contains invalid syntax or indentation errors.
  - `TOOL_MISUSE`: Erroneous parameter formats or invalid tool selection.
  - `TIMEOUT`: Model generation exceeded execution latency bounds.
  - `FORMAT_ERROR`: Output violates structured response schemas.
  - `INCORRECT_ANSWER`: Output was well-formed but factually inaccurate.
  - `HALLUCINATION`: Refusal boilerplates or fabricated entities.
- Each evaluation run computes aggregate failure distributions and produces contextual recommendations for prompt tuning, memory augmentation, or fine-tuning intervention.

### 3. Automated Regression Detection and Historical Baseline Comparison
- `RegressionDetector` compares candidate evaluation runs against historical baselines:
  - Detects score drops beyond tolerance thresholds.
  - Tracks specific test cases that regressed (passed in baseline, failed in candidate).
  - Tracks resolved test cases (failed in baseline, passed in candidate).
  - Detects latency degradation spikes (> 50% slower runtime).
  - Issues explicit verdicts: `PASSED`, `IMPROVED`, `REGRESSED`, `BASELINE_ESTABLISHED`.
- If no baseline run is explicitly supplied, the system automatically detects the most recent completed run of the same benchmark prior to the candidate run.

### 4. Structured Reporting and Web Dashboard Integration
- `EvaluationReporter` outputs both detailed Markdown documents and structured JSON payloads for runs and regression comparisons.
- The interactive Web Dashboard at `/` features a dedicated **"Evaluation & Benchmarks"** tab:
  - Displays registered baseline benchmark suites with category badges and case counts.
  - Triggers single-click benchmark execution against any registered local model.
  - Displays historical run tables with pass rates, case counts, and latency metrics.
  - Provides in-browser detailed run inspection cards and side-by-side regression comparison views.

## Consequences

### Positive
- Fully automated, local-first evaluation without external cloud dependencies or API keys.
- Eliminates subjective evaluation: performance deltas are quantified with exact percentages and case diffs.
- Clear error taxonomies guide developers and autonomous agents on the exact area needing improvement.
- Prepares the platform for Phase 5 (Autonomous Learning Loops) and Phase 6 (Local Model Fine-Tuning).

### Negative / Trade-offs
- Executing multi-case benchmark suites against 7B parameter models on local GPU requires ~10-25 seconds per suite.
- Exact and regex rubrics require well-specified expected outputs to avoid false negatives.
