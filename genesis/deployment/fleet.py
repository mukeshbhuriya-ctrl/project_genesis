"""
Distributed Multi-Workstation Fleet Synchronization Engine (Phase 11).

Provides:
- Peer-to-peer fleet node discovery and heartbeat tracking
- Cross-workstation delta synchronization for knowledge nodes, edges, and research papers
- Timestamp-based conflict resolution with audit logging in fleet_sync_logs
- Air-gapped / offline delta bundle export and import
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.device import DeviceOrchestrator
from genesis.core.logging import get_logger

log = get_logger("deployment.fleet")


@dataclass
class FleetNode:
    """A peer workstation node within the Project Genesis distributed fleet."""
    id: str
    hostname: str
    address: str
    port: int = 8000
    gpu_device: str = ""
    vram_gb: float = 0.0
    status: str = "online"  # online, unreachable, syncing, offline
    last_heartbeat: str = ""
    sync_cursor: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hostname": self.hostname,
            "address": self.address,
            "port": self.port,
            "gpu_device": self.gpu_device,
            "vram_gb": round(self.vram_gb, 2),
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "sync_cursor": self.sync_cursor,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class FleetSyncResult:
    """Summary of a delta synchronization operation with a peer node."""
    node_id: str
    direction: str  # push, pull, bidirectional
    entity_type: str
    items_synced: int
    status: str  # success, failed, conflict_resolved
    duration_ms: float = 0.0
    checksum: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "direction": self.direction,
            "entity_type": self.entity_type,
            "items_synced": self.items_synced,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "checksum": self.checksum,
            "message": self.message,
        }


class FleetManager:
    """
    Coordinates fleet membership, heartbeat monitoring, and delta synchronization
    across multi-workstation local clusters.
    """

    def __init__(self, db_path: str | Path | None = None):
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else get_db_path_from_url(settings.database_url)
        self.device_orch = DeviceOrchestrator()
        self._ensure_local_node_registered()

    def _conn(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    def _ensure_local_node_registered(self) -> None:
        """Register the local host machine as self node in fleet_nodes table."""
        import socket
        hostname = socket.gethostname()
        local_id = f"node-self-{hashlib.sha256(hostname.encode()).hexdigest()[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        # Discover hardware capabilities
        try:
            topo = self.device_orch.detect_topology()
            gpu_name = topo.primary_gpu.device_name if topo.primary_gpu else "CPU-only"
            vram_gb = (topo.primary_gpu.total_vram_mb / 1024.0) if topo.primary_gpu else 0.0
        except Exception:
            gpu_name = "CPU-only"
            vram_gb = 0.0

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO fleet_nodes (
                    id, hostname, address, port, gpu_device, vram_gb,
                    status, last_heartbeat, sync_cursor, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    local_id,
                    hostname,
                    "127.0.0.1",
                    get_settings().port,
                    gpu_name,
                    vram_gb,
                    "online",
                    now,
                    now,
                    now,
                    json.dumps({"is_self": True, "os": "Windows"}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Node Registry Operations ──

    def register_node(
        self,
        hostname: str,
        address: str,
        port: int = 8000,
        gpu_device: str = "",
        vram_gb: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> FleetNode:
        """Register a new peer workstation node in the fleet."""
        node_id = f"node-{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        meta = metadata or {}

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO fleet_nodes (
                    id, hostname, address, port, gpu_device, vram_gb,
                    status, last_heartbeat, sync_cursor, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    hostname,
                    address,
                    port,
                    gpu_device,
                    vram_gb,
                    "online",
                    now,
                    "",
                    now,
                    json.dumps(meta),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        log.info(f"Registered fleet node: {node_id} ({hostname} @ {address}:{port})")
        return FleetNode(
            id=node_id,
            hostname=hostname,
            address=address,
            port=port,
            gpu_device=gpu_device,
            vram_gb=vram_gb,
            status="online",
            last_heartbeat=now,
            sync_cursor="",
            created_at=now,
            metadata=meta,
        )

    def list_nodes(self) -> list[FleetNode]:
        """List all known fleet nodes."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, hostname, address, port, gpu_device, vram_gb,
                       status, last_heartbeat, sync_cursor, created_at, metadata
                FROM fleet_nodes ORDER BY created_at ASC
                """
            )
            rows = cursor.fetchall()
            return [
                FleetNode(
                    id=r["id"],
                    hostname=r["hostname"],
                    address=r["address"],
                    port=r["port"],
                    gpu_device=r["gpu_device"],
                    vram_gb=r["vram_gb"],
                    status=r["status"],
                    last_heartbeat=r["last_heartbeat"],
                    sync_cursor=r["sync_cursor"] or "",
                    created_at=r["created_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_node(self, node_id: str) -> FleetNode | None:
        """Get a specific fleet node by ID."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, hostname, address, port, gpu_device, vram_gb,
                       status, last_heartbeat, sync_cursor, created_at, metadata
                FROM fleet_nodes WHERE id = ?
                """,
                (node_id,),
            )
            r = cursor.fetchone()
            if not r:
                return None
            return FleetNode(
                id=r["id"],
                hostname=r["hostname"],
                address=r["address"],
                port=r["port"],
                gpu_device=r["gpu_device"],
                vram_gb=r["vram_gb"],
                status=r["status"],
                last_heartbeat=r["last_heartbeat"],
                sync_cursor=r["sync_cursor"] or "",
                created_at=r["created_at"],
                metadata=json.loads(r["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def remove_node(self, node_id: str) -> bool:
        """Remove a fleet node."""
        conn = self._conn()
        try:
            cursor = conn.execute("DELETE FROM fleet_nodes WHERE id = ?", (node_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def update_heartbeat(self, node_id: str, status: str = "online") -> None:
        """Update a node's heartbeat timestamp and status."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE fleet_nodes SET last_heartbeat = ?, status = ? WHERE id = ?",
                (now, status, node_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Delta Synchronization & Replication ──

    def export_delta_bundle(self, since: str = "") -> dict[str, Any]:
        """
        Export recently created or updated entities (knowledge nodes, edges, research papers)
        for synchronization to another fleet node.
        """
        conn = self._conn()
        try:
            # 1. Knowledge Nodes
            if since:
                k_cursor = conn.execute(
                    "SELECT id, node_type, label, properties_json, created_at, updated_at, metadata FROM knowledge_nodes WHERE updated_at > ? OR created_at > ?",
                    (since, since),
                )
            else:
                k_cursor = conn.execute(
                    "SELECT id, node_type, label, properties_json, created_at, updated_at, metadata FROM knowledge_nodes"
                )
            nodes = [dict(r) for r in k_cursor.fetchall()]

            # 2. Knowledge Edges
            if since:
                e_cursor = conn.execute(
                    "SELECT id, source_id, target_id, relation_type, weight, created_at, metadata FROM knowledge_edges WHERE created_at > ?",
                    (since,),
                )
            else:
                e_cursor = conn.execute(
                    "SELECT id, source_id, target_id, relation_type, weight, created_at, metadata FROM knowledge_edges"
                )
            edges = [dict(r) for r in e_cursor.fetchall()]

            # 3. Research Papers
            if since:
                p_cursor = conn.execute(
                    "SELECT id, title, abstract, content_markdown, campaign_id, cited_nodes_json, status, created_at, updated_at, metadata FROM research_papers WHERE updated_at > ? OR created_at > ?",
                    (since, since),
                )
            else:
                p_cursor = conn.execute(
                    "SELECT id, title, abstract, content_markdown, campaign_id, cited_nodes_json, status, created_at, updated_at, metadata FROM research_papers"
                )
            papers = [dict(r) for r in p_cursor.fetchall()]

            payload_str = json.dumps({"nodes": nodes, "edges": edges, "papers": papers}, sort_keys=True)
            checksum = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

            return {
                "version": 11,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "since": since,
                "checksum": checksum,
                "entities": {
                    "knowledge_nodes": nodes,
                    "knowledge_edges": edges,
                    "research_papers": papers,
                },
                "counts": {
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "papers": len(papers),
                },
            }
        finally:
            conn.close()

    def import_delta_bundle(
        self,
        bundle: dict[str, Any],
        source_node_id: str = "manual",
    ) -> FleetSyncResult:
        """
        Merge an incoming delta bundle into local database using timestamp-based
        conflict resolution.
        """
        start_t = time.perf_counter()
        entities = bundle.get("entities", {})
        nodes = entities.get("knowledge_nodes", [])
        edges = entities.get("knowledge_edges", [])
        papers = entities.get("research_papers", [])

        synced_count = 0
        now = datetime.now(timezone.utc).isoformat()
        checksum = bundle.get("checksum", "")

        conn = self._conn()
        try:
            # 1. Merge Knowledge Nodes
            for n in nodes:
                conn.execute(
                    """
                    INSERT INTO knowledge_nodes (
                        id, node_type, label, properties_json, created_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        label=excluded.label,
                        properties_json=excluded.properties_json,
                        updated_at=excluded.updated_at,
                        metadata=excluded.metadata
                    WHERE excluded.updated_at >= knowledge_nodes.updated_at
                    """,
                    (
                        n["id"],
                        n["node_type"],
                        n["label"],
                        n.get("properties_json", "{}"),
                        n.get("created_at", now),
                        n.get("updated_at", now),
                        n.get("metadata", "{}"),
                    ),
                )
                synced_count += 1

            # 2. Merge Knowledge Edges
            for e in edges:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO knowledge_edges (
                        id, source_id, target_id, relation_type, weight, created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        e["id"],
                        e["source_id"],
                        e["target_id"],
                        e["relation_type"],
                        e.get("weight", 1.0),
                        e.get("created_at", now),
                        e.get("metadata", "{}"),
                    ),
                )
                synced_count += 1

            # 3. Merge Research Papers
            for p in papers:
                camp_id = p.get("campaign_id")
                if camp_id:
                    c_row = conn.execute("SELECT id FROM research_campaigns WHERE id = ?", (camp_id,)).fetchone()
                    if not c_row:
                        camp_id = None

                conn.execute(
                    """
                    INSERT INTO research_papers (
                        id, title, abstract, content_markdown, campaign_id,
                        cited_nodes_json, status, created_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        abstract=excluded.abstract,
                        content_markdown=excluded.content_markdown,
                        status=excluded.status,
                        updated_at=excluded.updated_at,
                        metadata=excluded.metadata
                    WHERE excluded.updated_at >= research_papers.updated_at
                    """,
                    (
                        p["id"],
                        p["title"],
                        p["abstract"],
                        p["content_markdown"],
                        camp_id,
                        p.get("cited_nodes_json", "[]"),
                        p.get("status", "published"),
                        p.get("created_at", now),
                        p.get("updated_at", now),
                        p.get("metadata", "{}"),
                    ),
                )
                synced_count += 1

            # Ensure source node exists in fleet_nodes for foreign key integrity
            cursor = conn.execute("SELECT id FROM fleet_nodes WHERE id = ?", (source_node_id,))
            if not cursor.fetchone():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO fleet_nodes (
                        id, hostname, address, port, gpu_device, vram_gb,
                        status, last_heartbeat, sync_cursor, created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_node_id,
                        f"peer-{source_node_id}",
                        "127.0.0.1",
                        8000,
                        "Unknown",
                        0.0,
                        "online",
                        now,
                        now,
                        now,
                        json.dumps({"auto_created_on_sync": True}),
                    ),
                )

            # Update sync cursor on source node if known
            conn.execute(
                "UPDATE fleet_nodes SET sync_cursor = ?, last_heartbeat = ? WHERE id = ?",
                (now, now, source_node_id),
            )

            # Record audit log
            sync_log_id = f"sync-{uuid.uuid4().hex[:10]}"
            conn.execute(
                """
                INSERT INTO fleet_sync_logs (
                    id, node_id, entity_type, direction, items_synced,
                    status, checksum, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sync_log_id,
                    source_node_id,
                    "multi_entity_bundle",
                    "pull",
                    synced_count,
                    "success",
                    checksum,
                    now,
                    json.dumps({"counts": bundle.get("counts", {})}),
                ),
            )
            conn.commit()

            duration_ms = (time.perf_counter() - start_t) * 1000.0
            return FleetSyncResult(
                node_id=source_node_id,
                direction="pull",
                entity_type="multi_entity_bundle",
                items_synced=synced_count,
                status="success",
                duration_ms=duration_ms,
                checksum=checksum,
                message=f"Synchronized {synced_count} entities cleanly.",
            )

        except Exception as err:
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            log.error(f"Delta bundle import failed from {source_node_id}: {err}")
            return FleetSyncResult(
                node_id=source_node_id,
                direction="pull",
                entity_type="multi_entity_bundle",
                items_synced=0,
                status="failed",
                duration_ms=duration_ms,
                checksum=checksum,
                message=str(err),
            )
        finally:
            conn.close()

    async def sync_with_peer(
        self,
        peer_id: str,
        http_client: Any = None,
    ) -> FleetSyncResult:
        """
        Pull recent delta bundle from a remote fleet peer and import it.
        """
        import httpx

        node = self.get_node(peer_id)
        if not node:
            raise KeyError(f"Fleet node '{peer_id}' not found")

        url = f"http://{node.address}:{node.port}/api/fleet/export-delta?since={node.sync_cursor}"
        start_t = time.perf_counter()

        try:
            client = http_client or httpx.AsyncClient(timeout=10.0)
            res = await client.get(url)
            if res.status_code != 200:
                self.update_heartbeat(peer_id, status="unreachable")
                return FleetSyncResult(
                    node_id=peer_id,
                    direction="pull",
                    entity_type="bundle",
                    items_synced=0,
                    status="failed",
                    duration_ms=(time.perf_counter() - start_t) * 1000.0,
                    message=f"Peer responded with status code {res.status_code}",
                )

            bundle = res.json()
            return self.import_delta_bundle(bundle, source_node_id=peer_id)

        except Exception as e:
            self.update_heartbeat(peer_id, status="unreachable")
            return FleetSyncResult(
                node_id=peer_id,
                direction="pull",
                entity_type="bundle",
                items_synced=0,
                status="failed",
                duration_ms=(time.perf_counter() - start_t) * 1000.0,
                message=f"Connection error to {node.address}:{node.port} — {e}",
            )

    def list_sync_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List historical sync transactions from fleet_sync_logs."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, node_id, entity_type, direction, items_synced,
                       status, checksum, created_at, metadata
                FROM fleet_sync_logs ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()
