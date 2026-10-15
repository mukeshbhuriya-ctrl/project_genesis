"""
Unit tests for Cross-Fleet Model Parameter Merging (Phase 12).
"""

import math
import pytest
from pathlib import Path

from genesis.core.database import initialize_database
from genesis.training.merging import (
    ModelMergeManager,
    ModelMergeRecipe,
    merge_dare,
    merge_linear,
    merge_slerp,
    merge_ties,
)


def test_merge_linear_math():
    """Verify linear weighted averaging."""
    t1 = [1.0, 2.0, 3.0]
    t2 = [3.0, 4.0, 5.0]

    # Equal weights (0.5 each)
    res_eq = merge_linear([t1, t2])
    assert res_eq == [2.0, 3.0, 4.0]

    # Asymmetric weights (0.25 and 0.75)
    res_weighted = merge_linear([t1, t2], weights=[0.25, 0.75])
    # 0.25*1 + 0.75*3 = 2.5
    # 0.25*2 + 0.75*4 = 3.5
    # 0.25*3 + 0.75*5 = 4.5
    assert pytest.approx(res_weighted[0], 0.001) == 2.5
    assert pytest.approx(res_weighted[1], 0.001) == 3.5
    assert pytest.approx(res_weighted[2], 0.001) == 4.5


def test_merge_slerp_interpolation():
    """Verify spherical linear interpolation (SLERP)."""
    # Orthogonal unit vectors
    t1 = [1.0, 0.0]
    t2 = [0.0, 1.0]

    # At t=0.5, angle is 45 deg, components should be equal and norm preserved
    res_mid = merge_slerp(t1, t2, t=0.5)
    assert pytest.approx(res_mid[0], 0.001) == math.sqrt(0.5)
    assert pytest.approx(res_mid[1], 0.001) == math.sqrt(0.5)

    # At t=0, output equals t1
    res_0 = merge_slerp(t1, t2, t=0.0)
    assert pytest.approx(res_0[0], 0.001) == 1.0
    assert pytest.approx(res_0[1], 0.001) == 0.0

    # At t=1, output equals t2
    res_1 = merge_slerp(t1, t2, t=1.0)
    assert pytest.approx(res_1[0], 0.001) == 0.0
    assert pytest.approx(res_1[1], 0.001) == 1.0


def test_merge_ties_trimming_and_sign_consensus():
    """Verify TIES-Merging trimming and sign election."""
    # 3 vectors of length 4
    # t1: positive strong
    # t2: positive weak
    # t3: negative interference
    t1 = [10.0, 5.0, 1.0, 0.1]
    t2 = [8.0, 4.0, 0.5, 0.05]
    t3 = [-1.0, -0.5, -20.0, 0.01]

    # With density=0.5, top 2 elements kept per vector
    res = merge_ties([t1, t2, t3], density=0.5)
    assert len(res) == 4
    # In index 0: t1 (10) and t2 (8) positive kept, t3 trimmed to 0 -> consensus positive
    assert res[0] > 0.0
    # In index 2: t3 (-20) kept, t1 and t2 trimmed -> consensus negative
    assert res[2] < 0.0


def test_merge_dare_drop_and_rescale():
    """Verify DARE drop and rescale preserves expected scale."""
    t1 = [1.0] * 100
    t2 = [2.0] * 100

    # DARE with density 0.8
    res = merge_dare([t1, t2], density=0.8, seed=42)
    assert len(res) == 100
    mean_val = sum(res) / len(res)
    # Expected mean of 0.5*1.0 + 0.5*2.0 = 1.5
    assert pytest.approx(mean_val, 0.25) == 1.5


def test_model_merge_manager_crud_and_methods(tmp_path):
    """Verify ModelMergeManager executes linear, slerp, ties, and dare merges."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    mgr = ModelMergeManager(db_path=test_db)

    # 1. Execute Linear Merge
    recipe_linear = mgr.execute_merge(
        name="test-linear-merge",
        merge_method="linear",
        target_model="qwen2.5:7b",
        source_adapters=["adapter-rig-1", "adapter-rig-2"],
        weights=[0.6, 0.4],
    )
    assert isinstance(recipe_linear, ModelMergeRecipe)
    assert recipe_linear.status == "completed"
    assert recipe_linear.merge_method == "linear"
    assert Path(recipe_linear.output_path).is_file()

    # 2. Execute SLERP Merge
    recipe_slerp = mgr.execute_merge(
        name="test-slerp-merge",
        merge_method="slerp",
        target_model="qwen2.5-coder:7b",
        source_adapters=["adapter-alpha", "adapter-beta"],
        weights=[0.5, 0.5],
    )
    assert recipe_slerp.status == "completed"
    assert recipe_slerp.merge_method == "slerp"

    # 3. Execute TIES Merge
    recipe_ties = mgr.execute_merge(
        name="test-ties-merge",
        merge_method="ties",
        target_model="qwen2.5:7b",
        source_adapters=["adapter-1", "adapter-2", "adapter-3"],
        density=0.7,
    )
    assert recipe_ties.status == "completed"
    assert recipe_ties.merge_method == "ties"

    # 4. Execute DARE Merge
    recipe_dare = mgr.execute_merge(
        name="test-dare-merge",
        merge_method="dare",
        target_model="qwen2.5:7b",
        source_adapters=["adapter-1", "adapter-2"],
        density=0.8,
    )
    assert recipe_dare.status == "completed"
    assert recipe_dare.merge_method == "dare"

    # List merges
    merges = mgr.list_merges()
    assert len(merges) == 4

    # Get specific merge
    fetched = mgr.get_merge(recipe_linear.id)
    assert fetched is not None
    assert fetched.name == "test-linear-merge"

    # Delete merge
    deleted = mgr.delete_merge(recipe_linear.id)
    assert deleted is True
    assert mgr.get_merge(recipe_linear.id) is None
    assert not Path(recipe_linear.output_path).is_file()


def test_merge_validation_errors(tmp_path):
    """Verify validation error handling for unsupported methods or invalid inputs."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    mgr = ModelMergeManager(db_path=test_db)

    # Invalid merge method
    with pytest.raises(ValueError, match="Unsupported merge method"):
        mgr.execute_merge(
            name="bad-merge",
            merge_method="unknown_quantum_merge",
            target_model="qwen2.5:7b",
            source_adapters=["a1", "a2"],
        )

    # Insufficient source adapters for SLERP (needs at least 2)
    with pytest.raises(ValueError, match="requires at least 2"):
        mgr.execute_merge(
            name="insufficient-slerp",
            merge_method="slerp",
            target_model="qwen2.5:7b",
            source_adapters=["only_one"],
        )
