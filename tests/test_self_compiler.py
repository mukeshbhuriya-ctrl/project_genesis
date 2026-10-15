"""
Tests for Self-Hosting Evolutionary Compiler Optimization Subsystem (Phase 16).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.core.self_compiler import (
    SelfHostingCompilerEngine,
    _py_cosine_similarity,
    _py_ucb1,
    _py_pareto_dominates,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "self_compiler_test.db"
    initialize_database(db_file)
    return db_file


def test_pure_python_reference_implementations():
    """Verify reference mathematical accuracy of pure Python hot paths."""
    # 1. Cosine similarity
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert _py_cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-5)

    v3 = [0.0, 1.0, 0.0]
    assert _py_cosine_similarity(v1, v3) == pytest.approx(0.0, abs=1e-5)

    # 2. UCB1 calculation
    ucb = _py_ucb1(q_val=0.6, parent_visits=100, node_visits=10, c_param=1.414)
    assert ucb > 0.6

    ucb_unvisited = _py_ucb1(q_val=0.0, parent_visits=10, node_visits=0)
    assert ucb_unvisited >= 1e8

    # 3. Pareto dominance
    # Minimize objectives: candidate [1.0, 2.0] dominates [1.5, 2.5]
    assert _py_pareto_dominates([1.0, 2.0], [1.5, 2.5]) is True
    # Incomparable trade-off
    assert _py_pareto_dominates([1.0, 3.0], [2.0, 2.0]) is False
    # Identical does not strictly dominate
    assert _py_pareto_dominates([1.0, 2.0], [1.0, 2.0]) is False


def test_compile_target_and_persistence(test_db: Path):
    """Verify compiling an individual target, numerical equivalence, and database persistence."""
    engine = SelfHostingCompilerEngine(db_path=test_db)

    target = engine.compile_target("cosine_similarity")
    assert target.target_name == "cosine_similarity"
    assert target.status in ("compiled_native", "fallback_python")
    assert target.numerical_error <= 1e-4
    assert target.speedup_factor >= 1.0

    # Retrieve from DB
    fetched = engine.get_target("cosine_similarity")
    assert fetched is not None
    assert fetched.id == target.id
    assert fetched.target_name == "cosine_similarity"


def test_compile_all_targets(test_db: Path):
    """Verify compiling all registered internal Genesis optimization targets."""
    engine = SelfHostingCompilerEngine(db_path=test_db)
    targets = engine.compile_all()

    assert len(targets) == 3
    names = {t.target_name for t in targets}
    assert "cosine_similarity" in names
    assert "mcts_ucb1" in names
    assert "pareto_dominance" in names

    listed = engine.list_targets()
    assert len(listed) >= 3


def test_fast_dispatch_functions():
    """Verify high-performance public dispatchers with fallback safety."""
    # Cosine dispatch
    sim = SelfHostingCompilerEngine.fast_cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert sim == pytest.approx(1.0, abs=1e-4)

    # UCB1 dispatch
    score = SelfHostingCompilerEngine.fast_ucb1(0.75, 50, 10, 1.414)
    assert score > 0.75

    # Pareto dispatch
    dom = SelfHostingCompilerEngine.fast_pareto_dominates([0.5, 1.0], [0.8, 1.2])
    assert dom is True
