"""
Domain-specific code synthesis, sandbox execution, and reflective self-repair engine.

Implements an automated closed-loop code generation workflow:
1. Synthesizes algorithmic Python implementation + automated test harness.
2. Executes test harness in an isolated, timeout-bounded subprocess sandbox.
3. On failure, captures error traceback and performs multi-round reflective self-repair.
4. On success, persists verified code experiment and optionally registers verified
   syntheses into Phase 5 training datasets for fine-tuning.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core.database import get_sync_connection
from genesis.core.logging import get_logger
from genesis.models import GenerationRequest, ModelProvider
from genesis.models.ollama_backend import OllamaProvider

log = get_logger("code_research")


@dataclass
class SandboxResult:
    """Outcome of running code and test suite in an isolated sandbox."""
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    traceback: str
    duration_ms: float
    error_type: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "traceback": self.traceback,
            "duration_ms": round(self.duration_ms, 2),
            "error_type": self.error_type,
        }


class CodeSandbox:
    """Executes Python code and tests inside an isolated, timeout-governed subprocess."""

    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str, tests: str = "") -> SandboxResult:
        """
        Execute combined code and test harness in a temporary workspace.
        """
        combined = f"{code}\n\n# --- Test Harness ---\n{tests}\n"
        start_time = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "sandbox_test.py"
            script_path.write_text(combined, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=tmp_dir,
                )
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                passed = proc.returncode == 0
                error_type = "none" if passed else self._categorize_error(proc.stderr)

                return SandboxResult(
                    passed=passed,
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    traceback=proc.stderr.strip(),
                    duration_ms=duration_ms,
                    error_type=error_type,
                )
            except subprocess.TimeoutExpired as e:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return SandboxResult(
                    passed=False,
                    exit_code=-1,
                    stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
                    stderr=f"TimeoutExpired: Execution exceeded {self.timeout_seconds}s limit",
                    traceback=f"Execution timed out after {self.timeout_seconds} seconds.",
                    duration_ms=duration_ms,
                    error_type="timeout",
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return SandboxResult(
                    passed=False,
                    exit_code=-2,
                    stdout="",
                    stderr=str(e),
                    traceback=str(e),
                    duration_ms=duration_ms,
                    error_type="system_error",
                )

    def run(self, code: str, tests: str = "") -> SandboxResult:
        """Alias for execute to provide interface consistency."""
        return self.execute(code, tests)

    def _categorize_error(self, stderr: str) -> str:
        """Categorize failure mode from stderr."""
        if not stderr:
            return "unknown"
        if "SyntaxError" in stderr or "IndentationError" in stderr:
            return "syntax_error"
        if "AssertionError" in stderr:
            return "assertion_failure"
        if "NameError" in stderr:
            return "name_error"
        if "TypeError" in stderr:
            return "type_error"
        if "IndexError" in stderr or "KeyError" in stderr:
            return "index_key_error"
        if "TimeoutExpired" in stderr:
            return "timeout"
        return "runtime_error"


@dataclass
class CodeExperimentRecord:
    """Complete record of a code synthesis and reflective self-repair run."""
    id: str
    name: str
    objective: str
    language: str = "python"
    initial_code: str = ""
    final_code: str = ""
    tests: str = ""
    test_results: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    iterations: int = 0
    max_iterations: int = 3
    duration_ms: float = 0.0
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "objective": self.objective,
            "language": self.language,
            "initial_code": self.initial_code,
            "final_code": self.final_code,
            "tests": self.tests,
            "test_results": self.test_results,
            "status": self.status,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


class ReflectiveCodeSynthesizer:
    """
    Synthesizes code and tests with local models and executes iterative self-repair.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
        sandbox: CodeSandbox | None = None,
        db_path: str | Path = "data/genesis.db",
        model_id: str = "qwen2.5-coder:7b",
    ) -> None:
        self.provider = provider or OllamaProvider()
        self.sandbox = sandbox or CodeSandbox()
        self.db_path = Path(db_path)
        self.model_id = model_id

    def extract_code_blocks(self, text: str) -> str:
        """Extract clean Python code from markdown code fences or raw text."""
        # Find all ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(m.strip() for m in matches)
        # Fallback to stripped text
        return text.strip()

    def split_code_and_tests(self, content: str) -> tuple[str, str]:
        """
        Separate main implementation code from unit tests if marked with comments.
        """
        markers = [
            "# --- Test Harness ---",
            "# --- Tests ---",
            "# Tests",
            "# Unit Tests",
            "if __name__ == '__main__':",
        ]
        for marker in markers:
            if marker in content:
                parts = content.split(marker, 1)
                code_part = parts[0].strip()
                test_part = marker + "\n" + parts[1].strip()
                return code_part, test_part

        # If no split marker found, inspect AST or return as combined code
        return content.strip(), ""

    async def synthesize_and_repair(
        self,
        objective: str,
        name: str | None = None,
        max_iterations: int = 3,
        export_dataset_id: str | None = None,
    ) -> CodeExperimentRecord:
        """
        Execute the full autonomous code generation and self-repair pipeline:
        1. Prompt model for initial code + test assertions.
        2. Execute in sandbox.
        3. If failed, construct reflective repair prompt with traceback and re-run.
        4. Repeat until green or max_iterations reached.
        """
        exp_id = str(uuid.uuid4())
        exp_name = name or f"code-synth-{exp_id[:8]}"
        start_time = time.perf_counter()

        record = CodeExperimentRecord(
            id=exp_id,
            name=exp_name,
            objective=objective,
            max_iterations=max_iterations,
            status="generating",
        )

        # ── Step 1: Initial Synthesis Prompt ──
        system_prompt = (
            "You are an expert Python software engineer. Implement clean, robust, and PEP-8 compliant code.\n"
            "Format your response with the implementation first, followed by '# --- Test Harness ---' "
            "and standalone assert statements that thoroughly verify correctness."
        )
        user_prompt = (
            f"Objective: {objective}\n\n"
            "Provide the complete Python code followed by test assertions under '# --- Test Harness ---'.\n"
            "Use only Python standard library."
        )

        log.info("Starting code synthesis for experiment %s: '%s'", exp_id, objective[:60])

        try:
            resp = await self.provider.generate(
                GenerationRequest(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1500,
                )
            )
            raw_content = self.extract_code_blocks(resp.content)
            code_part, test_part = self.split_code_and_tests(raw_content)

            # If no explicit test harness was generated, add simple syntax validation test
            if not test_part:
                test_part = "# --- Test Harness ---\nassert True, 'Basic execution verification'"

            record.initial_code = code_part
            record.final_code = code_part
            record.tests = test_part
            record.status = "verifying"

            # ── Step 2: Verification and Reflective Self-Repair Loop ──
            iteration = 0
            last_sandbox_result: SandboxResult | None = None

            while iteration < max_iterations:
                iteration += 1
                record.iterations = iteration
                log.info("Running sandbox iteration %d/%d for %s", iteration, max_iterations, exp_id)

                sandbox_res = self.sandbox.execute(record.final_code, record.tests)
                last_sandbox_result = sandbox_res
                record.test_results = sandbox_res.to_dict()

                if sandbox_res.passed:
                    log.info("Sandbox verification passed on iteration %d for %s", iteration, exp_id)
                    record.status = "passed"
                    break

                # Self-repair needed
                log.warning(
                    "Sandbox failed on iteration %d (%s): %s",
                    iteration,
                    sandbox_res.error_type,
                    sandbox_res.traceback[:120],
                )

                if iteration >= max_iterations:
                    record.status = "failed"
                    record.error = f"Max repair iterations reached. Last error: {sandbox_res.error_type}"
                    break

                record.status = "repairing"
                repair_prompt = (
                    f"The previous implementation failed during test execution with error type '{sandbox_res.error_type}'.\n\n"
                    f"Traceback / Error Output:\n```\n{sandbox_res.traceback}\n```\n\n"
                    f"Previous Code:\n```python\n{record.final_code}\n\n{record.tests}\n```\n\n"
                    "Carefully analyze why this failed, fix the bug, and provide the complete corrected code and test assertions.\n"
                    "Include '# --- Test Harness ---' between the implementation and the assertions."
                )

                repair_resp = await self.provider.generate(
                    GenerationRequest(
                        model=self.model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": repair_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=1500,
                    )
                )

                repaired_content = self.extract_code_blocks(repair_resp.content)
                new_code, new_tests = self.split_code_and_tests(repaired_content)
                if new_code:
                    record.final_code = new_code
                if new_tests:
                    record.tests = new_tests

            total_ms = (time.perf_counter() - start_time) * 1000.0
            record.duration_ms = total_ms
            record.completed_at = datetime.now(timezone.utc).isoformat()

            # ── Step 3: Self-Improvement Loop (Register into Phase 5 Dataset) ──
            if record.status == "passed" and export_dataset_id:
                self._export_to_training_dataset(export_dataset_id, record)

        except Exception as e:
            log.error("Code synthesis experiment failed: %s", e)
            record.status = "error"
            record.error = str(e)
            record.duration_ms = (time.perf_counter() - start_time) * 1000.0
            record.completed_at = datetime.now(timezone.utc).isoformat()

        self._save_record(record)
        return record

    def _export_to_training_dataset(self, dataset_id: str, rec: CodeExperimentRecord) -> None:
        """Append verified code generation pair to Phase 5 dataset for fine-tuning."""
        try:
            conn = get_sync_connection(self.db_path)
            cursor = conn.execute(
                "SELECT file_path FROM training_datasets WHERE id = ?", (dataset_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return

            dataset_file = Path(row["file_path"])
            if dataset_file.exists():
                entry = {
                    "instruction": f"Implement the following in Python: {rec.objective}",
                    "input": "",
                    "output": f"```python\n{rec.final_code}\n```",
                }
                with dataset_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                log.info("Exported verified code sample to training dataset %s", dataset_id)
        except Exception as e:
            log.warning("Failed to export verified code to dataset %s: %s", dataset_id, e)

    def _save_record(self, r: CodeExperimentRecord) -> None:
        """Persist experiment record into SQLite."""
        if not self.db_path.exists():
            return
        conn = get_sync_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO code_experiments (
                    id, name, objective, language, initial_code, final_code,
                    tests, test_results, status, iterations, max_iterations,
                    duration_ms, error, created_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.id,
                    r.name,
                    r.objective,
                    r.language,
                    r.initial_code,
                    r.final_code,
                    r.tests,
                    json.dumps(r.test_results),
                    r.status,
                    r.iterations,
                    r.max_iterations,
                    r.duration_ms,
                    r.error,
                    r.created_at,
                    r.completed_at,
                    json.dumps(r.metadata),
                ),
            )
            conn.commit()
            log.info("Saved code experiment %s with status %s", r.id, r.status)
        except Exception as e:
            log.warning("Failed to persist code experiment %s: %s", r.id, e)
        finally:
            conn.close()

    def list_experiments(self, limit: int = 20) -> list[CodeExperimentRecord]:
        """Query recent code synthesis experiments from SQLite."""
        if not self.db_path.exists():
            return []
        conn = get_sync_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, name, objective, language, initial_code, final_code,
                       tests, test_results, status, iterations, max_iterations,
                       duration_ms, error, created_at, completed_at, metadata
                FROM code_experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                results.append(
                    CodeExperimentRecord(
                        id=row["id"],
                        name=row["name"],
                        objective=row["objective"],
                        language=row["language"],
                        initial_code=row["initial_code"] or "",
                        final_code=row["final_code"] or "",
                        tests=row["tests"] or "",
                        test_results=json.loads(row["test_results"] or "{}"),
                        status=row["status"],
                        iterations=row["iterations"],
                        max_iterations=row["max_iterations"],
                        duration_ms=row["duration_ms"],
                        error=row["error"],
                        created_at=row["created_at"],
                        completed_at=row["completed_at"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return results
        finally:
            conn.close()
