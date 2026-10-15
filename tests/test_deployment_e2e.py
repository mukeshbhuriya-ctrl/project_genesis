"""
End-to-End integration test for Phase 7 Hardware and Deployment Evolution.

Acceptance Criteria:
"The platform discovers multi-device compute topologies and computes optimal layer offload plans,
provides a persistent crash-resilient queue with lease-locking and background worker daemon,
and integrates code synthesis and knowledge synthesis as first-class tools in the bounded task orchestrator."
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator
import pytest

from genesis.core.database import initialize_database
from genesis.core.device import DeviceOrchestrator
from genesis.core.queue import JobQueue
from genesis.models import GenerationRequest, GenerationResponse, ModelInfo, ModelProvider
from genesis.tasks.executor import TaskExecutor
from genesis.tasks.planner import TaskPlanner
from genesis.tasks.worker import JobWorker
from genesis.tools.builtin.code_synthesizer import CodeSynthesizerTool
from genesis.tools.builtin.knowledge_synthesizer import KnowledgeSynthesizerTool
from genesis.tools.registry import create_default_registry


class MockPhase7Provider(ModelProvider):
    """Deterministic model provider for Phase 7 end-to-end integration test."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        system_role = request.messages[0]["content"] if request.messages else ""

        if "task planner" in system_role:
            # Plan utilizing the new synthesize_code tool
            plan_json = json.dumps([
                {
                    "step_number": 1,
                    "description": "Synthesize and verify merge sort algorithm",
                    "tool_name": "synthesize_code",
                    "tool_arguments": {"objective": "Implement merge sort in Python"},
                }
            ])
            return GenerationResponse(content=f"```json\n{plan_json}\n```", model=request.model)

        elif "expert Python software engineer" in system_role:
            code = (
                "def merge_sort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    mid = len(arr) // 2\n"
                "    left = merge_sort(arr[:mid])\n"
                "    right = merge_sort(arr[mid:])\n"
                "    res = []\n"
                "    i = j = 0\n"
                "    while i < len(left) and j < len(right):\n"
                "        if left[i] <= right[j]:\n"
                "            res.append(left[i])\n"
                "            i += 1\n"
                "        else:\n"
                "            res.append(right[j])\n"
                "            j += 1\n"
                "    res.extend(left[i:])\n"
                "    res.extend(right[j:])\n"
                "    return res\n\n"
                "# --- Test Harness ---\n"
                "assert merge_sort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]\n"
                "assert merge_sort([]) == []\n"
                "assert merge_sort([42]) == [42]\n"
            )
            return GenerationResponse(content=f"```python\n{code}```", model=request.model)

        elif "Senior AI Research Analyst" in system_role:
            brief = (
                "# Executive Summary\n"
                "Project Genesis features decoupled worker queues and device topologies [1].\n\n"
                "## Key Synthesized Insights\n"
                "Hardware topology ensures VRAM budgeting safety [1].\n\n"
                "## Reference Sources\n"
                "[1] Device Topology Subsystem"
            )
            return GenerationResponse(content=brief, model=request.model)

        return GenerationResponse(content="Default Phase 7 mock response", model=request.model)

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        yield "Phase 7 streaming test"

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def is_available(self) -> bool:
        return True

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        return None

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.05] * 768 for _ in texts]


@pytest.mark.asyncio
async def test_phase7_hardware_and_deployment_e2e(tmp_path: Path):
    """
    Complete end-to-end verification of Phase 7:
    1. Discover hardware topology and verify compute devices are recognized.
    2. Compute multi-device layer allocation plans for single-GPU and CPU offload setups.
    3. Submit background jobs to persistent JobQueue.
    4. Run JobWorker to process queued jobs with lease locking and heartbeats.
    5. Verify TaskPlanner decomposes complex goal using synthesize_code tool.
    6. Execute planned steps and verify results in SQLite task logs.
    """
    db_file = tmp_path / "genesis_phase7_e2e.db"
    initialize_database(db_file)
    mock_provider = MockPhase7Provider()

    # ── 1. Device Topology Discovery ──
    orchestrator = DeviceOrchestrator(db_path=db_file)
    topo = orchestrator.discover_topology()

    assert topo.id is not None
    assert len(topo.devices) >= 1
    assert topo.total_ram_gb > 0.0

    # ── 2. Layer Distribution Planning ──
    # Plan A: Standard 7B fits inside single GPU
    plan_7b = orchestrator.calculate_layer_distribution(
        model_name="qwen2.5:7b",
        total_layers=28,
        parameter_size_b=7.0,
        quant_bits=4,
        context_tokens=4096,
        safety_reserve_mb=1024.0,
    )
    assert plan_7b.total_layers == 28
    assert "strategy" in plan_7b.to_dict()

    # Plan B: Giant 70B model requires CPU offload
    plan_70b = orchestrator.calculate_layer_distribution(
        model_name="qwen2.5:70b",
        total_layers=80,
        parameter_size_b=70.0,
        quant_bits=8,
        context_tokens=8192,
        safety_reserve_mb=1024.0,
    )
    assert plan_70b.total_layers == 80
    assert plan_70b.is_full_offload is False

    # ── 3. Persistent Job Queue & Worker Execution ──
    queue = JobQueue(db_path=db_file)
    job = queue.submit(
        job_type="custom",
        payload={"action": "device_memory_calibration", "duration": 0.05},
        priority=20,
    )
    assert job.status == "queued"

    worker = JobWorker(worker_id="e2e-worker-01", queue=queue, db_path=db_file)
    processed = await worker.run_once()
    assert processed is True

    completed_job = queue.get_job(job.id)
    assert completed_job.status == "completed"
    assert completed_job.worker_id == "e2e-worker-01"
    assert completed_job.result["action"] == "device_memory_calibration"

    # ── 4. Tool Registry with Synthesis Tools ──
    reg = create_default_registry()
    assert "synthesize_code" in reg
    assert "synthesize_knowledge" in reg

    # Replace tools with mock-injected versions for deterministic E2E test
    code_tool = CodeSynthesizerTool(db_path=db_file, provider=mock_provider)
    reg.register(code_tool)
    kn_tool = KnowledgeSynthesizerTool(db_path=db_file, provider=mock_provider)
    reg.register(kn_tool)

    # ── 5. Bounded Task Planner Decomposes Goal with synthesize_code ──
    planner = TaskPlanner(registry=reg, model_provider=mock_provider, max_steps=5)
    task = await planner.plan_goal_async(goal="Implement merge sort in Python with test cases")

    assert task.status.value == "pending"
    assert len(task.steps) == 1
    assert task.steps[0].tool_name == "synthesize_code"
    assert "merge sort" in task.steps[0].tool_arguments.get("objective", "").lower()

    # ── 6. Task Executor Executes Plan ──
    from genesis.tasks.models import StepStatus, TaskStatus
    executor = TaskExecutor(registry=reg, db_path=db_file)
    exec_task = await executor.run_task_async(task)

    assert exec_task.status == TaskStatus.COMPLETED
    assert exec_task.steps[0].status == StepStatus.SUCCESS
    step_output = exec_task.steps[0].tool_result["output"]
    assert step_output["status"] == "passed"
    assert "def merge_sort" in step_output["final_code"]


