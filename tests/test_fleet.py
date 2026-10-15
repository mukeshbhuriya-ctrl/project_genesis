"""
Unit tests for Distributed Multi-Workstation Fleet Synchronization (Phase 11).
"""

import json
import pytest
from pathlib import Path

from genesis.core.database import initialize_database
from genesis.deployment.fleet import (
    FleetManager,
    FleetNode,
    FleetSyncResult,
)


def test_fleet_self_node_registration(tmp_path):
    """Verify that FleetManager auto-registers the local host node in the database."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    mgr = FleetManager(db_path=test_db)
    nodes = mgr.list_nodes()

    assert len(nodes) >= 1
    self_nodes = [n for n in nodes if n.metadata.get("is_self") is True]
    assert len(self_nodes) == 1
    self_node = self_nodes[0]
    assert self_node.id.startswith("node-self-")
    assert self_node.address == "127.0.0.1"
    assert self_node.status == "online"


def test_fleet_node_crud_and_heartbeat(tmp_path):
    """Verify peer node registration, retrieval, heartbeat update, and removal."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    mgr = FleetManager(db_path=test_db)

    # Register peer
    peer = mgr.register_node(
        hostname="rig-titan-02",
        address="192.168.1.150",
        port=8000,
        gpu_device="RTX 4090",
        vram_gb=24.0,
        metadata={"role": "trainer"},
    )

    assert peer.id.startswith("node-")
    assert peer.hostname == "rig-titan-02"
    assert peer.vram_gb == 24.0

    # Get node
    fetched = mgr.get_node(peer.id)
    assert fetched is not None
    assert fetched.hostname == "rig-titan-02"

    # Update heartbeat
    mgr.update_heartbeat(peer.id, status="syncing")
    updated = mgr.get_node(peer.id)
    assert updated.status == "syncing"

    # Remove node
    deleted = mgr.remove_node(peer.id)
    assert deleted is True
    assert mgr.get_node(peer.id) is None


def test_export_and_import_delta_bundle(tmp_path):
    """Verify delta bundle export and conflict-resolving import between nodes."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    # Insert initial mock knowledge nodes, edge, and paper in DB
    import sqlite3
    conn = sqlite3.connect(test_db)
    conn.execute(
        """
        INSERT INTO knowledge_nodes (id, node_type, label, properties_json, created_at, updated_at, metadata)
        VALUES ('kn-01', 'concept', 'Quantization Kernels', '{"precision":"int8"}', '2026-10-15T10:00:00Z', '2026-10-15T10:00:00Z', '{}'),
               ('kn-02', 'metric', 'Speedup Factor', '{"ratio": 2.5}', '2026-10-15T10:01:00Z', '2026-10-15T10:01:00Z', '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO knowledge_edges (id, source_id, target_id, relation_type, weight, created_at, metadata)
        VALUES ('ke-01', 'kn-01', 'kn-02', 'refines', 0.95, '2026-10-15T10:02:00Z', '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO research_papers (id, title, abstract, content_markdown, campaign_id, cited_nodes_json, status, created_at, updated_at, metadata)
        VALUES ('paper-01', 'Fast Tensor SIMD Kernels', 'High throughput kernel acceleration', '# Fast SIMD', 'camp-01', '["kn-01"]', 'published', '2026-10-15T10:05:00Z', '2026-10-15T10:05:00Z', '{}')
        """
    )
    conn.commit()
    conn.close()

    mgr1 = FleetManager(db_path=test_db)

    # Export delta bundle
    bundle = mgr1.export_delta_bundle()
    assert bundle["version"] == 11
    assert bundle["counts"]["nodes"] == 2
    assert bundle["counts"]["edges"] == 1
    assert bundle["counts"]["papers"] == 1
    assert len(bundle["checksum"]) > 10

    # Import into target node DB
    target_db = tmp_path / "target_genesis.db"
    initialize_database(target_db)
    mgr2 = FleetManager(db_path=target_db)

    result = mgr2.import_delta_bundle(bundle, source_node_id="peer-workstation-1")
    assert isinstance(result, FleetSyncResult)
    assert result.status == "success"
    assert result.items_synced == 4  # 2 nodes + 1 edge + 1 paper

    # Verify audit sync logs recorded
    logs = mgr2.list_sync_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["entity_type"] == "multi_entity_bundle"
    assert logs[0]["items_synced"] == 4
    assert logs[0]["status"] == "success"


def test_import_delta_bundle_conflict_resolution(tmp_path):
    """Verify timestamp-based conflict resolution keeps newest records during delta import."""
    test_db = tmp_path / "test_genesis.db"
    initialize_database(test_db)

    mgr = FleetManager(db_path=test_db)

    # Local has newer record
    import sqlite3
    conn = sqlite3.connect(test_db)
    conn.execute(
        """
        INSERT INTO knowledge_nodes (id, node_type, label, properties_json, created_at, updated_at, metadata)
        VALUES ('kn-shared', 'concept', 'Local Newer Version', '{}', '2026-10-15T12:00:00Z', '2026-10-15T12:00:00Z', '{}')
        """
    )
    conn.commit()
    conn.close()

    # Incoming bundle has older record with same ID
    older_bundle = {
        "version": 11,
        "checksum": "abc123older",
        "entities": {
            "knowledge_nodes": [
                {
                    "id": "kn-shared",
                    "node_type": "concept",
                    "label": "Incoming Older Version",
                    "properties_json": "{}",
                    "created_at": "2026-10-15T10:00:00Z",
                    "updated_at": "2026-10-15T10:00:00Z",
                    "metadata": "{}",
                }
            ],
            "knowledge_edges": [],
            "research_papers": [],
        },
        "counts": {"nodes": 1, "edges": 0, "papers": 0},
    }

    res = mgr.import_delta_bundle(older_bundle, source_node_id="peer-older")
    assert res.status == "success"

    # Query local node; it should still be "Local Newer Version"
    conn = sqlite3.connect(test_db)
    cursor = conn.execute("SELECT label FROM knowledge_nodes WHERE id = 'kn-shared'")
    row = cursor.fetchone()
    conn.close()
    assert row[0] == "Local Newer Version"
