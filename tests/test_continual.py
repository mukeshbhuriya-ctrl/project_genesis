"""
Unit and integration tests for Continual Online Learning and EWC Regularization.
"""

import tempfile
from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.training.continual import (
    EWCConsolidator,
    PrefixTuningManager,
    ReplayBufferManager,
    ReplaySample,
)


@pytest.fixture
def test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


def test_replay_buffer_crud_and_capacity(test_db):
    mgr = ReplayBufferManager(test_db, capacity_per_domain=3)

    # Add samples in reasoning domain
    s1 = mgr.add_sample("If P then Q. P is true. What is Q?", "Q is true.", task_domain="reasoning", priority=1.5)
    assert s1.id.startswith("replay-")
    assert s1.task_domain == "reasoning"
    assert s1.priority_score == 1.5

    s2 = mgr.add_sample("All men are mortal. Socrates is a man.", "Socrates is mortal.", task_domain="reasoning", priority=2.0)
    s3 = mgr.add_sample("Prove sqrt(2) is irrational.", "Proof by contradiction...", task_domain="reasoning", priority=1.0)

    # Check listing
    samples = mgr.list_samples(task_domain="reasoning")
    assert len(samples) == 3
    # Ordered by priority DESC
    assert samples[0].id == s2.id

    # Capacity enforcement: adding 4th should prune lowest priority (s3 with priority 1.0)
    s4 = mgr.add_sample("Is 17 prime?", "Yes, 17 has only 1 and 17 as divisors.", task_domain="reasoning", priority=2.5)
    samples_after = mgr.list_samples(task_domain="reasoning")
    assert len(samples_after) == 3
    ids = [s.id for s in samples_after]
    assert s3.id not in ids
    assert s4.id in ids

    # Test duplicate input updates priority
    updated = mgr.add_sample("Is 17 prime?", "Yes, prime.", task_domain="reasoning", priority=5.0)
    assert updated.id == s4.id
    assert updated.priority_score == 5.0

    # Delete sample
    assert mgr.delete_sample(s1.id) is True
    assert mgr.get_sample(s1.id) is None


def test_replay_buffer_batch_sampling(test_db):
    mgr = ReplayBufferManager(test_db)

    # Populate multiple domains
    mgr.add_sample("def add(a, b):", "return a + b", task_domain="coding", priority=1.0)
    mgr.add_sample("def mul(a, b):", "return a * b", task_domain="coding", priority=2.0)
    mgr.add_sample("Find derivative of x^2", "2x", task_domain="reasoning", priority=1.5)
    mgr.add_sample("read_file('foo.txt')", "content", task_domain="tools", priority=1.2)
    mgr.add_sample("Hello Genesis", "Greetings", task_domain="general", priority=0.8)

    # Balanced batch sampling
    batch = mgr.sample_batch(batch_size=4, balanced=True)
    assert len(batch) >= 3
    domains = {s.task_domain for s in batch}
    assert len(domains) >= 2

    # Usage counts incremented
    for s in batch:
        refetched = mgr.get_sample(s.id)
        assert refetched.usage_count >= 1

    stats = mgr.get_domain_stats()
    assert stats["coding"] == 2
    assert stats["reasoning"] == 1
    assert stats["tools"] == 1
    assert stats["general"] == 1


def test_ewc_fisher_and_loss_computation(test_db):
    consolidator = EWCConsolidator(test_db)

    weights_a = {
        "layer1.weight": [0.5, -0.2, 0.8, -0.4],
        "layer2.weight": [0.1, 0.3, -0.5, 0.7],
    }

    # Compute Fisher Information
    fisher = consolidator.compute_fisher_information(weights_a)
    assert "layer1.weight" in fisher
    assert len(fisher["layer1.weight"]) == 4
    # All Fisher diagonal elements should be strictly positive
    assert all(f > 0 for f in fisher["layer1.weight"])

    # If weights haven't changed, EWC penalty must be exactly 0.0
    loss_zero = consolidator.compute_ewc_loss(
        current_weights=weights_a,
        optimal_weights=weights_a,
        fisher_diagonal=fisher,
        importance_lambda=1000.0,
    )
    assert loss_zero == 0.0

    # Perturb weights
    perturbed_weights = {
        "layer1.weight": [0.6, -0.2, 0.7, -0.4],  # diff: 0.1 at idx 0, -0.1 at idx 2
        "layer2.weight": [0.1, 0.3, -0.5, 0.7],
    }
    loss_perturbed = consolidator.compute_ewc_loss(
        current_weights=perturbed_weights,
        optimal_weights=weights_a,
        fisher_diagonal=fisher,
        importance_lambda=1000.0,
    )
    assert loss_perturbed > 0.0

    # Save EWC Checkpoint
    record = consolidator.save_checkpoint(
        task_name="task_reasoning_v1",
        optimal_weights=weights_a,
        fisher_diagonal=fisher,
        importance_lambda=1000.0,
        metadata={"val_acc": 0.92},
    )
    assert record.id.startswith("ewc-")
    assert record.parameter_count == 8

    # Fetch and verify
    fetched = consolidator.get_checkpoint(record.id)
    assert fetched is not None
    assert fetched.task_name == "task_reasoning_v1"
    assert fetched.optimal_weights == weights_a

    checkpoints = consolidator.list_checkpoints()
    assert len(checkpoints) == 1
    assert checkpoints[0].id == record.id


def test_prefix_tuning_manager():
    mgr = PrefixTuningManager()

    # Generate random task prefixes
    p_code = mgr.generate_random_prefix(task_name="coding_task", virtual_tokens=8, prefix_dim=16)
    assert p_code.id.startswith("prefix-")
    assert len(p_code.prefix_weights["prefix_key"]) == 8 * 16

    p_math = mgr.generate_random_prefix(task_name="math_task", virtual_tokens=8, prefix_dim=16)

    # Blend prefixes with alpha = 0.5
    blended = mgr.interpolate_prefixes(p_code, p_math, alpha=0.5)
    assert blended.task_name == "coding_task+math_task"
    assert len(blended.prefix_weights["prefix_key"]) == 8 * 16

    # Verify midpoint interpolation
    k_code = p_code.prefix_weights["prefix_key"][0]
    k_math = p_math.prefix_weights["prefix_key"][0]
    expected_mid = round(0.5 * k_code + 0.5 * k_math, 5)
    assert blended.prefix_weights["prefix_key"][0] == expected_mid
