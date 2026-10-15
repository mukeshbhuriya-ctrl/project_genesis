"""
Tests for Metacognitive Introspection & Calibration Subsystem (Phase 16).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.models.metacognition import (
    MetacognitiveDecision,
    MetacognitiveIntrospector,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "metacog_test.db"
    initialize_database(db_file)
    return db_file


def test_compute_lexical_complexity():
    """Verify lexical complexity calculation distinguishes simple from dense technical text."""
    simple = "Hello world! This is a test."
    complex_text = "Synthesize Byzantine fault-tolerant consensus invariants with Merkle tree proofs and Pareto optimal scheduling."

    score_simple = MetacognitiveIntrospector.compute_lexical_complexity(simple)
    score_complex = MetacognitiveIntrospector.compute_lexical_complexity(complex_text)

    assert 0.0 <= score_simple <= 0.45
    assert score_complex > score_simple
    assert score_complex >= 0.50


def test_estimate_perplexity():
    """Verify perplexity calculation with log probs and entropy fallback."""
    # From log probs
    log_probs = [-0.1, -0.2, -0.15, -0.05]
    ppl = MetacognitiveIntrospector.estimate_perplexity("sample text", log_probs)
    assert 1.0 <= ppl <= 1.5

    # High negative log probs (high uncertainty)
    high_uncertainty_lps = [-2.5, -3.1, -2.8]
    ppl_high = MetacognitiveIntrospector.estimate_perplexity("sample text", high_uncertainty_lps)
    assert ppl_high > 10.0

    # Heuristic fallback
    ppl_fallback = MetacognitiveIntrospector.estimate_perplexity("The quick brown fox jumps over the lazy dog")
    assert ppl_fallback >= 1.0


def test_evaluate_confidence_hedging_and_certainty():
    """Verify hedging suppresses confidence while certainty terms boost confidence."""
    hedging_prompt = "Maybe we might perhaps face an error or fail, but I am not sure."
    confident_prompt = "We have definitely proven and verified that the mathematical invariant held."

    conf_hedged, uncert_hedged = MetacognitiveIntrospector.evaluate_confidence(hedging_prompt)
    conf_strong, uncert_strong = MetacognitiveIntrospector.evaluate_confidence(confident_prompt)

    assert conf_strong > conf_hedged
    assert uncert_hedged > uncert_strong
    assert conf_strong >= 0.70
    assert conf_hedged <= 0.60


def test_expected_calibration_error():
    """Verify calculation of Expected Calibration Error (ECE)."""
    # Perfectly calibrated predictions: conf 0.8 with 80% accuracy, conf 0.2 with 20% accuracy
    confs = [0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2]
    accs = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # bin 1: 4/5 = 0.8, bin 0: 1/5 = 0.2

    ece = MetacognitiveIntrospector.calculate_expected_calibration_error(confs, accs, n_bins=5)
    assert ece == pytest.approx(0.0, abs=0.05)

    # Completely uncalibrated: conf 0.99 but acc 0.0
    bad_confs = [0.95, 0.95, 0.95]
    bad_accs = [0.0, 0.0, 0.0]
    bad_ece = MetacognitiveIntrospector.calculate_expected_calibration_error(bad_confs, bad_accs, n_bins=5)
    assert bad_ece >= 0.90


def test_introspect_and_persistence(test_db: Path):
    """Verify end-to-end introspection, reasoning budget decision, and SQLite persistence."""
    introspector = MetacognitiveIntrospector(db_path=test_db)

    # 1. Fast path candidate
    eval_fast = introspector.introspect(
        prompt="Write a Python function to square a number.",
        candidate_output="def square(x):\n    return x * x\n",
        token_log_probs=[-0.05, -0.02, -0.01],
    )
    assert eval_fast.confidence_score >= 0.75
    assert eval_fast.decision in (MetacognitiveDecision.FAST_PATH, MetacognitiveDecision.REFLECTIVE_REPAIR)

    # 2. Swarm escalation candidate (contradictory, highly complex byzantine problem with hedging)
    eval_swarm = introspector.introspect(
        prompt="Synthesize Byzantine fault-tolerant distributed consensus with Merkle proof invariants, though deadlock is perhaps uncertain.",
        token_log_probs=[-3.5, -4.2, -3.8],
    )
    assert eval_swarm.decision == MetacognitiveDecision.SWARM_ESCALATION

    # 3. Verify retrieval and summary
    history = introspector.list_evaluations(limit=10)
    assert len(history) == 2
    assert history[0].id == eval_swarm.id
    assert history[1].id == eval_fast.id

    summary = introspector.get_calibration_summary()
    assert summary["total_evaluations"] == 2
    assert "swarm_escalation" in summary["decisions"]
