"""
Tests for Autonomous Recursive Structural Pruning & NAS Engine (Phase 14).
"""

import pytest
from pathlib import Path

from genesis.core.database import initialize_database
from genesis.models.pruning import (
    KNOWN_ARCH_CONFIGS,
    ModelArchitectureConfig,
    PruningPlan,
    PruningExperimentRecord,
    RecursiveModelArchitect,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "pruning_test.db"
    initialize_database(db_file)
    return db_file


def test_parameter_calculation_analytical():
    """Verify exact analytical parameter calculation across transformer blocks."""
    architect = RecursiveModelArchitect(db_path=":memory:")
    arch = KNOWN_ARCH_CONFIGS["qwen2.5:7b"]

    full_params = architect.calculate_parameter_count(arch, arch.attention_heads, 1.0)
    # Qwen 2.5 7B parameters should be approximately 7.6 billion
    assert 7.0e9 < full_params < 8.2e9

    # Pruned attention heads should strictly reduce parameter count
    pruned_params = architect.calculate_parameter_count(arch, remaining_heads=20, mlp_ratio=0.85)
    assert pruned_params < full_params
    assert (full_params - pruned_params) > 5.0e8


def test_head_importance_ranking():
    """Verify sensitivity ranking under Taylor approximation and L1 magnitude."""
    architect = RecursiveModelArchitect(db_path=":memory:")

    # Taylor sensitivity
    taylor_scores = architect.estimate_head_importance(
        layer_idx=14, num_heads=28, strategy="taylor_sensitivity", seed=123
    )
    assert len(taylor_scores) == 28
    # Sorted ascending by importance score
    for i in range(len(taylor_scores) - 1):
        assert taylor_scores[i][1] <= taylor_scores[i + 1][1]

    # Magnitude L1
    mag_scores = architect.estimate_head_importance(
        layer_idx=5, num_heads=28, strategy="magnitude_l1", seed=456
    )
    assert len(mag_scores) == 28
    assert all(score > 0 for _, score in mag_scores)


def test_create_pruning_plan_with_target_vram():
    """Verify autonomous parameter search respects target VRAM boundary."""
    architect = RecursiveModelArchitect(db_path=":memory:")

    # Target 3500 MB VRAM footprint (fits comfortably in RTX 2000 Ada 8GB with context KV cache)
    plan = architect.create_pruning_plan(
        model_name="qwen2.5:7b",
        target_vram_mb=3500.0,
        strategy="taylor_sensitivity",
    )

    assert isinstance(plan, PruningPlan)
    assert plan.model_name == "qwen2.5:7b"
    assert plan.heads_pruned > 0
    assert plan.remaining_heads >= 4  # Should not prune below KV heads (4)
    assert plan.pruned_parameters < plan.original_parameters
    assert plan.parameter_reduction_pct > 0.0
    assert plan.estimated_vram_mb <= plan.original_vram_mb
    assert plan.speedup_estimate > 1.0
    assert len(plan.layer_head_mask) == 28
    # Each layer should retain exactly 'remaining_heads' heads
    for layer_idx, surviving_heads in plan.layer_head_mask.items():
        assert len(surviving_heads) == plan.remaining_heads


def test_execute_and_persist_pruning(test_db: Path):
    """Verify execution recording and retrieval in SQLite pruning_experiments table."""
    architect = RecursiveModelArchitect(db_path=test_db)

    plan = architect.create_pruning_plan(
        model_name="qwen2.5-coder:1.5b-base",
        target_vram_mb=800.0,
        strategy="magnitude_l1",
    )

    record = architect.execute_pruning(plan=plan, experiment_id="exp-nas-001")
    assert isinstance(record, PruningExperimentRecord)
    assert record.id == "exp-nas-001"
    assert record.model_name == "qwen2.5-coder:1.5b-base"
    assert record.status in ("verified", "planned")

    # List experiments
    experiments = architect.list_experiments()
    assert len(experiments) == 1
    assert experiments[0].id == "exp-nas-001"
    assert experiments[0].remaining_heads == plan.remaining_heads

    # Get single experiment
    fetched = architect.get_experiment("exp-nas-001")
    assert fetched is not None
    assert fetched.id == "exp-nas-001"
    assert fetched.parameter_reduction_pct == record.parameter_reduction_pct

    # Non-existent
    assert architect.get_experiment("exp-nas-999") is None
