"""
Data models and representations for the Genesis Evaluation Subsystem.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BenchmarkCategory(str, Enum):
    """Domain category for evaluation benchmarks."""
    REASONING = "reasoning"
    CODING = "coding"
    TOOLS = "tools"
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    CUSTOM = "custom"


class EvalType(str, Enum):
    """Method used to grade model responses against test cases."""
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    TOOL_CALL = "tool_call"


class FailureCategory(str, Enum):
    """Taxonomy of failure modes for error analysis."""
    NONE = "none"
    SYNTAX_ERROR = "syntax_error"
    TOOL_MISUSE = "tool_misuse"
    TIMEOUT = "timeout"
    HALLUCINATION = "hallucination"
    INCORRECT_ANSWER = "incorrect_answer"
    FORMAT_ERROR = "format_error"


class RunStatus(str, Enum):
    """Lifecycle state of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EvalCase:
    """A single deterministic test case within a benchmark suite."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_id: str = ""
    name: str = ""
    prompt: str = ""
    expected_output: str = ""
    eval_type: EvalType = EvalType.CONTAINS
    criteria: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "eval_type": self.eval_type.value,
            "criteria": self.criteria,
            "weight": self.weight,
        }


@dataclass
class Benchmark:
    """A suite of evaluation test cases for a specific capability domain."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = ""
    cases: list[EvalCase] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "case_count": len(self.cases),
            "cases": [c.to_dict() for c in self.cases],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class EvalResult:
    """Graded result of a single benchmark case evaluation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    case_id: str = ""
    passed: bool = False
    score: float = 0.0
    actual_output: str = ""
    error: str | None = None
    latency_ms: float = 0.0
    tokens_generated: int = 0
    failure_category: FailureCategory = FailureCategory.NONE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "passed": self.passed,
            "score": round(self.score, 4),
            "actual_output": self.actual_output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_generated": self.tokens_generated,
            "failure_category": self.failure_category.value,
            "created_at": self.created_at,
        }


@dataclass
class EvalRun:
    """Execution run of a benchmark suite against a target model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_id: str = ""
    model_id: str = ""
    status: RunStatus = RunStatus.PENDING
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    score: float = 0.0  # Percentage 0.0 to 100.0
    duration_ms: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "benchmark_id": self.benchmark_id,
            "model_id": self.model_id,
            "status": self.status.value,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "score": round(self.score, 2),
            "duration_ms": round(self.duration_ms, 2),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }
