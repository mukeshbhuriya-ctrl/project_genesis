"""
Unit and integration tests for CodeSandbox and ReflectiveCodeSynthesizer (Phase 6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
import pytest

from genesis.core.database import initialize_database
from genesis.models import GenerationRequest, GenerationResponse, ModelInfo, ModelProvider
from genesis.research.coding import CodeSandbox, ReflectiveCodeSynthesizer


class MockRepairingProvider(ModelProvider):
    """
    Simulates a model that produces buggy code on initial generation,
    then repairs the bug on the reflection turn.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_count += 1
        prompt = request.messages[-1]["content"]

        if self.call_count == 1:
            # Buggy initial code: reverse string bug
            content = (
                "```python\n"
                "def reverse_words(s: str) -> str:\n"
                "    # Intentionally buggy: reverses entire string instead of words\n"
                "    return s[::-1]\n\n"
                "# --- Test Harness ---\n"
                "assert reverse_words('hello world') == 'world hello'\n"
                "```"
            )
        else:
            # Repaired code: splits and reverses words
            content = (
                "```python\n"
                "def reverse_words(s: str) -> str:\n"
                "    return ' '.join(s.split()[::-1])\n\n"
                "# --- Test Harness ---\n"
                "assert reverse_words('hello world') == 'world hello'\n"
                "assert reverse_words('the sky is blue') == 'blue is sky the'\n"
                "```"
            )

        return GenerationResponse(content=content, model=request.model)

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        yield ""

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def is_available(self) -> bool:
        return True

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        return None

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


def test_sandbox_passing_code():
    sb = CodeSandbox(timeout_seconds=5.0)
    code = "def multiply(a, b):\n    return a * b"
    tests = "assert multiply(3, 4) == 12\nassert multiply(-2, 5) == -10"

    res = sb.execute(code, tests)
    assert res.passed is True
    assert res.exit_code == 0
    assert res.error_type == "none"
    assert res.duration_ms > 0.0


def test_sandbox_assertion_failure():
    sb = CodeSandbox(timeout_seconds=5.0)
    code = "def add(a, b):\n    return a - b"
    tests = "assert add(2, 2) == 4"

    res = sb.execute(code, tests)
    assert res.passed is False
    assert res.exit_code != 0
    assert res.error_type == "assertion_failure"
    assert "AssertionError" in res.traceback


def test_sandbox_syntax_error():
    sb = CodeSandbox(timeout_seconds=5.0)
    code = "def broken_func(:\n    pass"

    res = sb.execute(code)
    assert res.passed is False
    assert res.error_type == "syntax_error"


def test_sandbox_timeout():
    sb = CodeSandbox(timeout_seconds=0.6)
    code = "import time\ntime.sleep(2.0)"

    res = sb.execute(code)
    assert res.passed is False
    assert res.error_type == "timeout"
    assert "timed out" in res.traceback.lower()


@pytest.mark.asyncio
async def test_reflective_self_repair_loop(tmp_path: Path):
    db_file = tmp_path / "test_research_coding.db"
    initialize_database(db_file)

    mock_provider = MockRepairingProvider()
    synthesizer = ReflectiveCodeSynthesizer(
        provider=mock_provider,
        db_path=db_file,
        model_id="mock-coder",
    )

    record = await synthesizer.synthesize_and_repair(
        objective="Reverse words in a sentence.",
        name="test-word-reversal",
        max_iterations=3,
    )

    assert record.status == "passed"
    assert record.iterations == 2  # Passed on iteration 2 (after 1 repair)
    assert mock_provider.call_count == 2
    assert "s.split()[::-1]" in record.final_code
    assert record.duration_ms > 0.0

    # Verify persisted in SQLite
    history = synthesizer.list_experiments(limit=5)
    assert len(history) == 1
    assert history[0].id == record.id
    assert history[0].status == "passed"
    assert history[0].iterations == 2


@pytest.mark.asyncio
async def test_code_synthesis_exports_to_phase5_dataset(tmp_path: Path):
    db_file = tmp_path / "test_export_dataset.db"
    initialize_database(db_file)

    # Create dummy dataset entry in training_datasets table
    dataset_file = tmp_path / "fine_tune_data.jsonl"
    dataset_file.write_text("", encoding="utf-8")

    from genesis.core.database import get_sync_connection
    conn = get_sync_connection(db_file)
    dataset_id = "ds-code-synth-001"
    conn.execute(
        """
        INSERT INTO training_datasets (
            id, name, format, file_path, checksum, sample_count, total_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset_id, "Code Synth Verifications", "instruction", str(dataset_file), "dummy", 0, 0),
    )
    conn.commit()
    conn.close()

    mock_provider = MockRepairingProvider()
    synthesizer = ReflectiveCodeSynthesizer(
        provider=mock_provider,
        db_path=db_file,
        model_id="mock-coder",
    )

    record = await synthesizer.synthesize_and_repair(
        objective="Reverse words in a sentence.",
        max_iterations=3,
        export_dataset_id=dataset_id,
    )

    assert record.status == "passed"

    # Verify sample was appended to the jsonl dataset file
    content = dataset_file.read_text(encoding="utf-8").strip()
    assert content != ""
    entry = json.loads(content)
    assert "Reverse words in a sentence" in entry["instruction"]
    assert "s.split()[::-1]" in entry["output"]
