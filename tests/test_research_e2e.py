"""
End-to-End integration test for Phase 6 Research & Capability Expansion.

Acceptance Criterion:
"The platform profiles inference latency (TTFT, throughput) and KV-cache VRAM consumption,
autonomously synthesizes and verifies Python code in an isolated sandbox with self-repair,
and performs multi-source citation-verified knowledge synthesis consolidated into memory."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
import pytest

from genesis.core.database import initialize_database, get_sync_connection
from genesis.memory import MemoryStore
from genesis.models import GenerationRequest, GenerationResponse, ModelInfo, ModelProvider
from genesis.models.context import ContextManager
from genesis.models.profiler import InferenceProfiler
from genesis.research.coding import CodeSandbox, ReflectiveCodeSynthesizer
from genesis.research.synthesis import AdaptiveKnowledgeSynthesizer


class MockGenesisE2EProvider(ModelProvider):
    """Provides deterministic responses for end-to-end Phase 6 verification."""

    def __init__(self) -> None:
        self.code_gen_calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        system_role = request.messages[0]["content"] if request.messages else ""

        if "expert Python software engineer" in system_role:
            self.code_gen_calls += 1
            if self.code_gen_calls == 1:
                # Produce code with a deliberate bug to exercise self-repair
                code = (
                    "def binary_search(arr, target):\n"
                    "    # Intentionally buggy initial implementation\n"
                    "    return -1\n\n"
                    "# --- Test Harness ---\n"
                    "assert binary_search([1, 2, 3, 4, 5], 3) == 2\n"
                )
            else:
                # Fixed code
                code = (
                    "def binary_search(arr, target):\n"
                    "    left, right = 0, len(arr) - 1\n"
                    "    while left <= right:\n"
                    "        mid = (left + right) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        elif arr[mid] < target:\n"
                    "            left = mid + 1\n"
                    "        else:\n"
                    "            right = mid - 1\n"
                    "    return -1\n\n"
                    "# --- Test Harness ---\n"
                    "assert binary_search([1, 2, 3, 4, 5], 3) == 2\n"
                    "assert binary_search([1, 2, 3, 4, 5], 1) == 0\n"
                    "assert binary_search([1, 2, 3, 4, 5], 5) == 4\n"
                    "assert binary_search([1, 2, 3, 4, 5], 9) == -1\n"
                )
            return GenerationResponse(content=f"```python\n{code}```", model=request.model)

        elif "Senior AI Research Analyst" in system_role:
            brief = (
                "# Executive Summary\n"
                "Project Genesis integrates self-improving code synthesis and knowledge synthesis [1].\n\n"
                "## Key Synthesized Insights\n"
                "The sandbox prevents untrusted or faulty code from contaminating the system [1]. "
                "Citations verify alignment with long-term memory [2].\n\n"
                "## Implications & Next Actions\n"
                "Verified outputs are fed into the dataset pipeline for LoRA fine-tuning.\n\n"
                "## Reference Sources\n"
                "[1] Sandbox Subsystem\n"
                "[2] Persistent Memory"
            )
            return GenerationResponse(content=brief, model=request.model)

        return GenerationResponse(content="Default E2E response", model=request.model)

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        chunks = ["Binary", " Search", " executes", " in", " logarithmic", " time."]
        for c in chunks:
            yield c

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def is_available(self) -> bool:
        return True

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        return None

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.02] * 768 for _ in texts]


@pytest.mark.asyncio
async def test_phase6_full_research_expansion_lifecycle(tmp_path: Path):
    """
    Complete end-to-end verification of Phase 6:
    1. Verify context budgeting and KV-cache calculations across context windows.
    2. Run inference profiling measuring TTFT, throughput, and VRAM estimation.
    3. Execute autonomous code synthesis with sandbox failure detection and reflective self-repair.
    4. Export the verified implementation into a Phase 5 training dataset.
    5. Execute adaptive knowledge synthesis with citation verification.
    6. Verify consolidated memory creation and SQLite persistence across all tables.
    """
    db_file = tmp_path / "genesis_phase6_e2e.db"
    initialize_database(db_file)

    # ── 1. Context Budgeting ──
    cm = ContextManager(total_gpu_vram_mb=8192.0, safety_headroom_mb=1024.0)
    budget_4k = cm.get_budget(4096, "qwen2.5:7b")
    budget_16k = cm.get_budget(16384, "qwen2.5:7b")

    assert budget_4k.is_safe is True
    assert budget_4k.kv_cache_mb < budget_16k.kv_cache_mb
    assert budget_16k.kv_cache_mb > 800.0  # 16K KV cache is ~900 MB

    # ── 2. Inference Profiling ──
    provider = MockGenesisE2EProvider()
    profiler = InferenceProfiler(provider=provider, context_manager=cm, db_path=db_file)
    profile = await profiler.profile(
        model_id="qwen2.5:7b",
        prompt="Explain Binary Search.",
        max_tokens=64,
    )
    assert profile.status == "completed"
    assert profile.ttft_ms >= 0.0
    assert profile.tokens_per_second >= 0.0
    assert profile.kv_cache_vram_mb > 0.0

    # ── 3. Seed Phase 5 Dataset for Closed-Loop Export ──
    ds_file = tmp_path / "research_verified_dataset.jsonl"
    ds_file.write_text("", encoding="utf-8")
    ds_id = "ds-phase6-verified"

    conn = get_sync_connection(db_file)
    conn.execute(
        """
        INSERT INTO training_datasets (
            id, name, format, file_path, checksum, sample_count, total_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ds_id, "Verified Research Syntheses", "instruction", str(ds_file), "chk123", 0, 0),
    )
    conn.commit()
    conn.close()

    # ── 4. Code Synthesis & Self-Repair ──
    synthesizer = ReflectiveCodeSynthesizer(
        provider=provider,
        db_path=db_file,
        model_id="qwen2.5-coder:7b",
    )
    code_record = await synthesizer.synthesize_and_repair(
        objective="Implement binary search algorithm in Python",
        name="e2e-binary-search",
        max_iterations=3,
        export_dataset_id=ds_id,
    )
    assert code_record.status == "passed"
    assert code_record.iterations == 2  # Repaired on iteration 2
    assert "right = mid - 1" in code_record.final_code

    # Verify exported into Phase 5 dataset
    ds_content = ds_file.read_text(encoding="utf-8").strip()
    assert ds_content != ""
    ds_entry = json.loads(ds_content)
    assert "binary search" in ds_entry["instruction"].lower()
    assert "def binary_search" in ds_entry["output"]

    # ── 5. Adaptive Knowledge Synthesis & Memory Consolidation ──
    mem_store = MemoryStore(db_path=str(db_file))
    mem_store.create(
        category="sandbox",
        content="The Python CodeSandbox isolates execution inside a subprocess with a strict timeout.",
        source="e2e_doc",
    )
    mem_store.create(
        category="memory",
        content="Persistent memories are stored in SQLite and indexed with dense vector embeddings.",
        source="e2e_doc",
    )

    kn_synthesizer = AdaptiveKnowledgeSynthesizer(
        provider=provider,
        memory_store=mem_store,
        db_path=db_file,
        model_id="qwen2.5:7b",
    )
    brief = await kn_synthesizer.synthesize(
        topic="Project Genesis Subsystem Security",
        query="CodeSandbox",
        max_sources=2,
        save_to_memory=True,
    )
    assert brief.status == "completed"
    assert brief.confidence_score > 0.0
    assert len(brief.citations) > 0
    assert "# Executive Summary" in brief.brief_markdown

    # ── 6. Verification of SQLite Persistence ──
    # Check research_profiles
    profiles = profiler.list_profiles()
    assert len(profiles) >= 1
    assert profiles[0].model_id == "qwen2.5:7b"

    # Check code_experiments
    experiments = synthesizer.list_experiments()
    assert len(experiments) >= 1
    assert experiments[0].name == "e2e-binary-search"

    # Check research_synthesis
    syntheses = kn_synthesizer.list_syntheses()
    assert len(syntheses) >= 1
    assert syntheses[0].topic == "Project Genesis Subsystem Security"

    # Check consolidated memory created
    mems = mem_store.search(category="research_brief")
    assert len(mems) == 1
    assert "Research Brief on 'Project Genesis Subsystem Security'" in mems[0].content
