"""
Unit tests for the Genesis Task Orchestration Subsystem (genesis/tasks/).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from genesis.core.database import initialize_database
from genesis.tasks.executor import TaskExecutor
from genesis.tasks.models import StepStatus, Task, TaskStatus, TaskStep
from genesis.tasks.planner import TaskPlanner
from genesis.tasks.verifier import TaskVerifier
from genesis.tools.permissions import PermissionManager
from genesis.tools.registry import create_default_registry


class TestTaskModels:
    """Test task and task step state representations."""

    def test_task_step_lifecycle(self):
        step = TaskStep(
            step_number=1,
            description="Calculate optimal batch size",
            tool_name="calculator",
            tool_arguments={"expression": "1024 / 32"},
        )
        assert step.status == StepStatus.PENDING
        assert step.tool_name == "calculator"
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["status"] == "pending"

    def test_task_lifecycle(self):
        task = Task(goal="Benchmark memory latency")
        assert task.status == TaskStatus.PENDING
        assert task.current_step == 0
        d = task.to_dict()
        assert d["goal"] == "Benchmark memory latency"
        assert d["status"] == "pending"


class TestTaskPlanner:
    """Test deterministic and structured plan generation."""

    def test_create_deterministic_plan(self):
        planner = TaskPlanner()
        steps = [
            {"description": "Calculate VRAM", "tool_name": "calculator", "tool_arguments": {"expression": "8 * 1024"}},
            {"description": "Write report", "tool_name": "write_file", "tool_arguments": {"path": "vram.txt", "content": "8192 MB"}},
        ]
        task = planner.create_deterministic_plan("Analyze VRAM", steps)
        assert len(task.steps) == 2
        assert task.total_steps == 2
        assert task.steps[0].tool_name == "calculator"
        assert task.steps[1].tool_name == "write_file"

    def test_extract_json_steps(self):
        planner = TaskPlanner()
        sample_markdown = (
            "Here is the execution plan for your request:\n"
            "```json\n"
            "[\n"
            '  {"step_number": 1, "description": "Calc", "tool_name": "calculator", "tool_arguments": {"expression": "2+2"}}\n'
            "]\n"
            "```\n"
            "Hope this helps!"
        )
        extracted = planner._extract_json_steps(sample_markdown)
        assert extracted is not None
        assert len(extracted) == 1
        assert extracted[0]["tool_name"] == "calculator"


class TestTaskExecutorAndPersistence:
    """Test task execution, SQLite state persistence, and audit logging."""

    @pytest.fixture
    def test_db_path(self, tmp_path: Path) -> Path:
        db_file = tmp_path / "genesis_task_test.db"
        initialize_database(db_file)
        return db_file

    def test_save_and_retrieve_task(self, test_db_path: Path):
        executor = TaskExecutor(db_path=test_db_path)
        task = Task(goal="Verify database speed")
        task.steps = [
            TaskStep(
                task_id=task.id,
                step_number=1,
                description="Math step",
                tool_name="calculator",
                tool_arguments={"expression": "100 * 5"},
            )
        ]
        task.total_steps = 1
        executor.save_task(task)

        retrieved = executor.get_task(task.id)
        assert retrieved is not None
        assert retrieved.goal == "Verify database speed"
        assert len(retrieved.steps) == 1
        assert retrieved.steps[0].tool_name == "calculator"

    def test_execute_multi_step_task_synchronously(self, test_db_path: Path, tmp_path: Path):
        pm = PermissionManager(workspace_roots=[tmp_path])
        reg = create_default_registry(permission_manager=pm)
        executor = TaskExecutor(registry=reg, db_path=test_db_path)
        planner = TaskPlanner(registry=reg)

        output_file = tmp_path / "summary.txt"

        steps = [
            {
                "description": "Calculate product",
                "tool_name": "calculator",
                "tool_arguments": {"expression": "128 * 8"},
            },
            {
                "description": "Write product to file",
                "tool_name": "write_file",
                "tool_arguments": {"path": str(output_file), "content": "Calculated value: 1024"},
            },
            {
                "description": "Verify by reading file",
                "tool_name": "read_file",
                "tool_arguments": {"path": str(output_file)},
            },
        ]

        task = planner.create_deterministic_plan("Run math and file workflow", steps)
        completed = executor.run_task(task)

        assert completed.status == TaskStatus.COMPLETED
        assert completed.current_step == 3
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == "Calculated value: 1024"

        # Check DB audit trace
        from_db = executor.get_task(task.id)
        assert from_db is not None
        assert from_db.status == TaskStatus.COMPLETED
        assert len(from_db.steps) == 3
        assert all(s.status == StepStatus.SUCCESS for s in from_db.steps)
        assert from_db.steps[0].tool_result["output"] == 1024

    def test_task_aborts_on_failed_step(self, test_db_path: Path, tmp_path: Path):
        pm = PermissionManager(workspace_roots=[tmp_path])
        reg = create_default_registry(permission_manager=pm)
        executor = TaskExecutor(registry=reg, db_path=test_db_path)
        planner = TaskPlanner(registry=reg)

        steps = [
            {
                "description": "Read missing file",
                "tool_name": "read_file",
                "tool_arguments": {"path": str(tmp_path / "does_not_exist.txt")},
            },
            {
                "description": "Step that should not run",
                "tool_name": "calculator",
                "tool_arguments": {"expression": "1+1"},
            },
        ]

        task = planner.create_deterministic_plan("Expect failure", steps)
        completed = executor.run_task(task)

        assert completed.status == TaskStatus.FAILED
        assert completed.current_step == 1
        assert "failed" in completed.error.lower()
        assert completed.steps[0].status == StepStatus.FAILED
        assert completed.steps[1].status == StepStatus.PENDING


class TestTaskVerifier:
    """Test result and artifact verification."""

    def test_verify_file_checks(self, tmp_path: Path):
        verifier = TaskVerifier(permission_manager=PermissionManager(workspace_roots=[tmp_path]))
        test_file = tmp_path / "benchmark.log"
        test_file.write_text("Throughput: 45.2 tokens/sec\nLatency: 22ms", encoding="utf-8")

        # Exists
        ok, msg = verifier.verify_file_exists(test_file, min_size_bytes=10)
        assert ok

        # Contains
        ok_c, msg_c = verifier.verify_file_contains(test_file, "45.2 tokens/sec")
        assert ok_c

        # Missing substring
        bad_c, _ = verifier.verify_file_contains(test_file, "Nonexistent String")
        assert not bad_c

    def test_verify_numeric_result(self):
        verifier = TaskVerifier()
        ok, _ = verifier.verify_numeric_result(3.14159, 3.14150, tolerance=1e-3)
        assert ok

        bad, _ = verifier.verify_numeric_result(100.0, 105.0, tolerance=1.0)
        assert not bad
