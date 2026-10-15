"""
Tests for Latent Dream & Imagination Rollout Subsystem (Phase 15).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database, get_sync_connection
from genesis.world_model.dream import LatentDreamEngine


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "dream_test.db"
    initialize_database(db_file)
    return db_file


def test_evaluate_step_value():
    """Verify heuristic value estimator rewards validation and penalizes hallucinations."""
    # Positive action/outcome
    val1, uncert1 = LatentDreamEngine.evaluate_step_value(
        "Validate input invariants with unit tests",
        "Verification passed: invariant held across all boundaries",
    )
    assert val1 >= 0.75
    assert uncert1 <= 0.30

    # Negative action/outcome
    val2, uncert2 = LatentDreamEngine.evaluate_step_value(
        "Blindly force overwrite production checkpoint",
        "SyntaxError and memory leak detected; execution diverged",
    )
    assert val2 <= 0.0
    assert uncert2 >= 0.40


def test_simulate_dream_basic(test_db: Path):
    """Verify generative dream rollout generates multi-turn trajectory and persists."""
    engine = LatentDreamEngine(db_path=test_db)

    traj = engine.simulate_dream(
        prompt="Design crash-resilient vector cache with LRU eviction",
        depth=3,
        trajectory_id="dream-001",
    )

    assert traj.id == "dream-001"
    assert len(traj.steps) >= 3
    assert traj.total_value_score > 0.40
    assert traj.status == "completed"

    # Query from DB
    fetched = engine.get_trajectory("dream-001")
    assert fetched is not None
    assert fetched.prompt == "Design crash-resilient vector cache with LRU eviction"
    assert len(fetched.steps) >= 3


def test_simulate_dream_with_pruning(test_db: Path):
    """Verify unpromising seed action triggers pruning and counterfactual recovery."""
    engine = LatentDreamEngine(db_path=test_db)

    # Supply an intentionally dangerous seed action
    bad_seeds = [
        "Blindly ignore errors and force overwrite state",
        "Validate invariant guardrails",
    ]

    traj = engine.simulate_dream(
        prompt="Execute critical parameter update",
        seed_steps=bad_seeds,
        depth=2,
        prune_threshold=0.30,
        trajectory_id="dream-prune-001",
    )

    assert traj.pruned_branches >= 1
    # Check that a recovery step was appended
    recovery_steps = [s for s in traj.steps if "counterfactual recovery" in s.hypothesis_action.lower()]
    assert len(recovery_steps) >= 1


def test_export_to_replay_buffer(test_db: Path):
    """Verify high-reward dream trajectories export to continual learning replay buffer."""
    engine = LatentDreamEngine(db_path=test_db)

    traj = engine.simulate_dream(
        prompt="Verify mathematical convergence of gradient updates",
        depth=4,
        trajectory_id="dream-replay-001",
    )

    exported = engine.export_to_replay_buffer(traj, domain="reasoning")
    assert exported > 0

    # Verify rows in replay_buffers table
    conn = get_sync_connection(test_db)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM replay_buffers WHERE task_domain = 'reasoning'")
        count = cursor.fetchone()[0]
        assert count == exported
    finally:
        conn.close()


def test_list_trajectories(test_db: Path):
    """Verify trajectory listing filters and limits."""
    engine = LatentDreamEngine(db_path=test_db)

    engine.simulate_dream("Task A", depth=2, trajectory_id="dream-a")
    engine.simulate_dream("Task B", depth=2, trajectory_id="dream-b")

    trajs = engine.list_trajectories(limit=10)
    assert len(trajs) >= 2
