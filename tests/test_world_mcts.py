"""
Tests for Sandboxed State-Forking & MCTS Rollback Subsystem (Phase 15).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.world_model.mcts import (
    MCTSNode,
    MCTSSearchEngine,
    StateSnapshot,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "mcts_test.db"
    initialize_database(db_file)
    return db_file


def test_state_snapshot_fork_and_checksum():
    """Verify state snapshot creates independent deep copies with stable checksums."""
    snap1 = StateSnapshot(files={"app.py": "print('hello')"})
    assert len(snap1.checksum) == 16

    snap2 = snap1.fork()
    assert snap2.checksum == snap1.checksum

    # Mutate fork
    snap2.files["app.py"] = "print('world')"
    snap2.checksum = snap2.compute_checksum()

    # Original remains untouched
    assert snap1.files["app.py"] == "print('hello')"
    assert snap2.checksum != snap1.checksum


def test_mcts_node_ucb1():
    """Verify UCB1 prioritizes unvisited nodes and balances exploitation vs exploration."""
    snap = StateSnapshot()
    node = MCTSNode(node_id="n1", parent_id=None, action="act1", state_snapshot=snap)

    # Unvisited node must return infinity
    assert node.ucb1(c=1.414, parent_visits=10) == float("inf")

    # Visited node
    node.visits = 4
    node.total_reward = 3.0  # Mean = 0.75
    score = node.ucb1(c=1.414, parent_visits=16)
    assert score > 0.75  # 0.75 + exploration bonus


def test_simulation_rollback_on_syntax_error():
    """Verify MCTS simulation detects code syntax violations and triggers rollback."""
    # Valid mutation
    valid_files = {"calc.py": "def add(a, b):\n    return a + b\n"}
    rew1, is_valid1 = MCTSSearchEngine.evaluate_simulation("refactor", valid_files)
    assert is_valid1 is True
    assert rew1 >= 0.50

    # Broken syntax mutation
    broken_files = {"calc.py": "def broken(:\n    pass\n"}
    rew2, is_valid2 = MCTSSearchEngine.evaluate_simulation("refactor", broken_files)
    assert is_valid2 is False
    assert rew2 == 0.0


def test_mcts_search_and_persistence(test_db: Path):
    """Verify end-to-end MCTS tree search produces best trajectory and persists in SQLite."""
    engine = MCTSSearchEngine(db_path=test_db, exploration_c=1.414)

    initial_files = {
        "core.py": "def compute(data):\n    return sum(data)\n",
    }

    result = engine.search(
        root_task="Harden compute function with unit testing and safety invariants",
        initial_files=initial_files,
        iterations=10,
        max_depth=3,
        run_id="mcts-run-001",
    )

    assert result.id == "mcts-run-001"
    assert result.nodes_explored >= 3
    assert len(result.best_path) >= 1
    assert result.best_value >= 0.0
    assert result.status in ("completed", "aborted")

    # Fetch from SQLite
    fetched = engine.get_run("mcts-run-001")
    assert fetched is not None
    assert fetched.root_task == "Harden compute function with unit testing and safety invariants"
    assert len(fetched.best_path) >= 1

    listed = engine.list_runs(limit=10)
    assert len(listed) >= 1
