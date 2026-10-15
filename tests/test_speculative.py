"""
Tests for Self-Speculative Decoding Engine (Phase 14).
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from genesis.core.database import initialize_database
from genesis.models import GenerationResponse
from genesis.models.speculative import (
    SpeculativeConfig,
    SpeculativeEngine,
    SpeculativeRunResult,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "spec_test.db"
    initialize_database(db_file)
    return db_file


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="The autonomous recursive self architecting engine achieves optimal parameter compression.",
            model="qwen2.5:7b",
            token_count_prompt=10,
            token_count_completion=12,
        )
    )
    return provider


def test_rejection_sampling_all_accepted():
    """Verify rejection sampling accepts entire draft sequence when target supports it."""
    draft_tokens = ["autonomous", "recursive", "optimization"]
    draft_probs = [0.8, 0.85, 0.9]
    target_probs = [0.8, 0.9, 0.95]

    accepted, count, correction = SpeculativeEngine.verify_tokens_rejection_sampling(
        draft_tokens, draft_probs, target_probs, acceptance_threshold=0.75
    )

    assert count == 3
    assert accepted == draft_tokens
    assert correction is None


def test_rejection_sampling_rejection_and_correction():
    """Verify rejection sampling halts at low probability ratio and yields correction token."""
    draft_tokens = ["high", "risk", "hallucination"]
    draft_probs = [0.9, 0.95, 0.9]
    target_probs = [0.9, 0.1, 0.1]  # Second token rejected: ratio 0.1 / 0.95 = ~0.105 < 0.75

    accepted, count, correction = SpeculativeEngine.verify_tokens_rejection_sampling(
        draft_tokens, draft_probs, target_probs, acceptance_threshold=0.75
    )

    assert count == 1
    assert accepted == ["high"]
    assert correction == "risk_corrected"


@pytest.mark.asyncio
async def test_execute_speculative_run(test_db: Path, mock_provider):
    """Verify end-to-end speculative decoding run with provider and persistence."""
    engine = SpeculativeEngine(db_path=test_db, provider=mock_provider)

    cfg = SpeculativeConfig(
        target_model="qwen2.5:7b",
        draft_model="qwen2.5-coder:1.5b-base",
        lookahead_k=4,
        acceptance_threshold=0.70,
    )

    res = await engine.execute_speculative_run(
        prompt="Explain autonomous neural network pruning.",
        config=cfg,
        run_id="spec-test-001",
    )

    assert isinstance(res, SpeculativeRunResult)
    assert res.id == "spec-test-001"
    assert res.target_model == "qwen2.5:7b"
    assert res.draft_model == "qwen2.5-coder:1.5b-base"
    assert res.total_draft_tokens > 0
    assert 0.0 <= res.acceptance_rate <= 1.0
    assert res.speedup_factor >= 1.0
    assert res.status in ("completed", "fallback_autoregressive")

    # Verify SQLite persistence
    runs = engine.list_runs()
    assert len(runs) == 1
    assert runs[0].id == "spec-test-001"
    assert runs[0].speedup_factor == res.speedup_factor

    # Fetch by ID
    fetched = engine.get_run("spec-test-001")
    assert fetched is not None
    assert fetched.id == "spec-test-001"
    assert fetched.prompt == "Explain autonomous neural network pruning."

    # Non-existent
    assert engine.get_run("spec-nonexistent") is None


def test_speculative_config_validation():
    """Verify validation of lookahead K bounds."""
    with pytest.raises(ValueError, match="lookahead_k must be between 1 and 8"):
        SpeculativeConfig(lookahead_k=0)

    with pytest.raises(ValueError, match="lookahead_k must be between 1 and 8"):
        SpeculativeConfig(lookahead_k=10)
