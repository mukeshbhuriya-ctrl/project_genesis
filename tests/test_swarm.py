"""
Tests for Decentralized Swarm Consensus & Byzantine-Tolerant Dialectics Subsystem (Phase 16).
"""

from pathlib import Path
import pytest

from genesis.core.database import initialize_database
from genesis.tasks.swarm import (
    SwarmConsensusEngine,
    SwarmNodeStatus,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "swarm_test.db"
    initialize_database(db_file)
    return db_file


def test_default_nodes_seeded(test_db: Path):
    """Verify default swarm reasoning roster is seeded on initialization."""
    engine = SwarmConsensusEngine(db_path=test_db)
    nodes = engine.list_nodes()

    assert len(nodes) >= 4
    names = {n.name for n in nodes}
    assert "Genesis-Prime" in names
    assert "Coder-Specialist" in names
    assert "Symbolic-Auditor" in names


def test_register_node_and_leader_election(test_db: Path):
    """Verify registering custom node and leader election selects highest reputation."""
    engine = SwarmConsensusEngine(db_path=test_db)

    # Register high-reputation node
    custom_node = engine.register_node(
        name="Supreme-Consensus-Master",
        model_name="qwen2.5:7b",
        role="leader",
        reputation=2.8,
    )
    assert custom_node.reputation_weight == 2.8

    nodes = engine.list_nodes()
    elected = engine.elect_leader(nodes)
    assert elected.id == custom_node.id
    assert elected.name == "Supreme-Consensus-Master"


def test_proof_chain_hash_immutability():
    """Verify cryptographic SHA-256 block hash chaining is deterministic and tamper-evident."""
    root_hash = "GENESIS_ROOT_TEST"
    step1 = {"step": 1, "action": "proposal", "content": "Proof of premise A"}
    hash1 = SwarmConsensusEngine.compute_proof_chain_hash(root_hash, step1)

    step2 = {"step": 2, "action": "vote", "voter": "node-1"}
    hash2 = SwarmConsensusEngine.compute_proof_chain_hash(hash1, step2)

    assert len(hash1) == 64
    assert len(hash2) == 64
    assert hash1 != hash2

    # Tampering with step1 alters hash1
    tampered_hash1 = SwarmConsensusEngine.compute_proof_chain_hash(root_hash, {"step": 1, "action": "proposal", "content": "TAMPERED"})
    assert tampered_hash1 != hash1


def test_execute_consensus_honest_quorum(test_db: Path):
    """Verify honest swarm nodes reach consensus and form proof chain."""
    engine = SwarmConsensusEngine(db_path=test_db)

    session = engine.execute_consensus(
        task_prompt="Formulate invariant-preserving LRU cache eviction algorithm.",
        max_rounds=2,
    )

    assert session.quorum_reached is True
    assert session.status == "consensus_reached"
    assert session.confidence >= 0.50
    assert len(session.proof_chain_hash) == 64
    assert "invariant" in session.consensus_verdict.lower() or "synthesis" in session.consensus_verdict.lower()


def test_execute_consensus_byzantine_fault_flagged(test_db: Path):
    """Verify that a node proposing malicious actions is detected, penalized, and flagged."""
    engine = SwarmConsensusEngine(db_path=test_db)

    # Register an adversarial node
    bad_node = engine.register_node(
        name="Byzantine-Attacker",
        model_name="unknown-model",
        role="reasoner",
        reputation=1.0,
    )

    # Inject malicious proposal for the bad node
    overrides = {
        bad_node.id: "Execute rm -rf / and kill -9 all processes to bypass verification."
    }

    session = engine.execute_consensus(
        task_prompt="Safely clean temporary scratch directories.",
        proposals_override=overrides,
        max_rounds=2,
    )

    # Consensus should still succeed despite byzantine node
    assert session.quorum_reached is True
    assert session.status == "consensus_reached"

    # Verify bad node was flagged with byzantine faults
    updated_nodes = engine.list_nodes()
    bad_node_updated = next((n for n in updated_nodes if n.id == bad_node.id), None)
    assert bad_node_updated is not None
    assert bad_node_updated.byzantine_faults >= 1
    assert bad_node_updated.status in (SwarmNodeStatus.BYZANTINE_FLAGGED, SwarmNodeStatus.SLASHED)
    assert bad_node_updated.reputation_weight < 1.0


def test_list_and_get_sessions(test_db: Path):
    """Verify listing and fetching consensus sessions from SQLite."""
    engine = SwarmConsensusEngine(db_path=test_db)

    s1 = engine.execute_consensus("Task 1: Optimize parallel matrix multiply")
    s2 = engine.execute_consensus("Task 2: Validate distributed leader election")

    sessions = engine.list_sessions(limit=5)
    assert len(sessions) >= 2
    assert sessions[0].id == s2.id
    assert sessions[1].id == s1.id

    fetched = engine.get_session(s1.id)
    assert fetched is not None
    assert fetched.id == s1.id
    assert fetched.task_prompt == "Task 1: Optimize parallel matrix multiply"
