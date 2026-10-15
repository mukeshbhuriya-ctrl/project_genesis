"""
Tests for Continuous Multi-Objective Pareto Optimization Subsystem (Phase 16).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.research.pareto import (
    ParetoOptimizer,
    ParetoPoint,
    RuntimeConfig,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "pareto_test.db"
    initialize_database(db_file)
    return db_file


def test_evaluate_config():
    """Verify objective evaluation calculations across VRAM, latency, throughput, and accuracy."""
    cfg_4bit = RuntimeConfig(quant_bits=4, context_window=2048, lookahead_k=4, batch_size=1, thread_count=8)
    lat4, vram4, tps4, acc4 = ParetoOptimizer.evaluate_config(cfg_4bit)

    assert vram4 < 5000.0  # Fits comfortably on 8 GB RTX 2000 Ada
    assert lat4 > 0.0
    assert tps4 > 0.0
    assert 0.90 <= acc4 <= 1.0

    # 16-bit config requires much more VRAM and drops throughput
    cfg_16bit = RuntimeConfig(quant_bits=16, context_window=2048, lookahead_k=1, batch_size=1, thread_count=8)
    lat16, vram16, tps16, acc16 = ParetoOptimizer.evaluate_config(cfg_16bit)

    assert vram16 > vram4
    assert acc16 >= acc4


def test_dominates():
    """Verify 4D dominance comparison logic."""
    cfg = RuntimeConfig()
    # Point 1: Better on all metrics
    p1 = ParetoPoint(config=cfg, latency_ms=50.0, vram_mb=3000.0, throughput_tps=80.0, accuracy_score=0.98)
    # Point 2: Worse on all metrics
    p2 = ParetoPoint(config=cfg, latency_ms=100.0, vram_mb=6000.0, throughput_tps=40.0, accuracy_score=0.90)

    assert ParetoOptimizer.dominates(p1, p2) is True
    assert ParetoOptimizer.dominates(p2, p1) is False

    # Trade-off: p3 is faster but less accurate
    p3 = ParetoPoint(config=cfg, latency_ms=30.0, vram_mb=3000.0, throughput_tps=90.0, accuracy_score=0.85)
    assert ParetoOptimizer.dominates(p1, p3) is False
    assert ParetoOptimizer.dominates(p3, p1) is False


def test_fast_non_dominated_sort():
    """Verify NSGA-II partitioning into ranked fronts."""
    cfg = RuntimeConfig()
    p1 = ParetoPoint(config=cfg, latency_ms=50.0, vram_mb=3000.0, throughput_tps=80.0, accuracy_score=0.98)
    p2 = ParetoPoint(config=cfg, latency_ms=100.0, vram_mb=6000.0, throughput_tps=40.0, accuracy_score=0.90)
    p3 = ParetoPoint(config=cfg, latency_ms=30.0, vram_mb=2500.0, throughput_tps=95.0, accuracy_score=0.88)

    fronts = ParetoOptimizer.fast_non_dominated_sort([p1, p2, p3])

    assert len(fronts) >= 2
    # Front 0 should contain non-dominated points p1 and p3
    front_0_ids = {p.id for p in fronts[0]}
    assert p1.id in front_0_ids
    assert p3.id in front_0_ids

    # Front 1 contains dominated point p2
    front_1_ids = {p.id for p in fronts[1]}
    assert p2.id in front_1_ids


def test_calculate_crowding_distance():
    """Verify crowding distance calculation assigns inf to boundaries."""
    cfg = RuntimeConfig()
    pts = [
        ParetoPoint(config=cfg, latency_ms=20.0, vram_mb=2000.0, throughput_tps=100.0, accuracy_score=0.80),
        ParetoPoint(config=cfg, latency_ms=50.0, vram_mb=3000.0, throughput_tps=80.0, accuracy_score=0.90),
        ParetoPoint(config=cfg, latency_ms=80.0, vram_mb=4000.0, throughput_tps=60.0, accuracy_score=0.98),
    ]

    ParetoOptimizer.calculate_crowding_distance(pts)
    assert any(p.crowding_distance >= 1_000_000.0 for p in pts)


def test_run_optimization_and_persistence(test_db: Path):
    """Verify end-to-end evolutionary search, Pareto front discovery, and SQLite persistence."""
    optimizer = ParetoOptimizer(db_path=test_db)

    experiment, front = optimizer.run_optimization(
        name="Test-RTX-Frontier",
        population_size=12,
        generations=3,
    )

    assert experiment.status == "completed"
    assert experiment.pareto_front_size == len(front)
    assert len(front) >= 1

    # Verify points on Pareto front are rank 0
    for pt in front:
        assert pt.rank == 0
        assert pt.latency_ms > 0
        assert pt.vram_mb > 0

    # Retrieve from DB
    experiments = optimizer.list_experiments(limit=5)
    assert len(experiments) >= 1
    assert experiments[0].id == experiment.id

    db_points = optimizer.get_frontiers(experiment.id)
    assert len(db_points) == len(front)
    assert db_points[0].config.quant_bits in (2, 3, 4, 8, 16)
