"""
Cross-Campaign Meta-Memory Knowledge Graph Subsystem (Phase 10).

Provides:
- KnowledgeNode & KnowledgeEdge dataclasses
- Subgraph & GraphPath representations
- MetaMemoryGraph: Graph query engine, path finding, neighborhood traversal,
  semantic graph search, and campaign/debate synchronization.
"""

from __future__ import annotations

import collections
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.memory.embeddings import cosine_similarity, pack_vector, unpack_vector

log = get_logger("memory.graph")


@dataclass
class KnowledgeNode:
    """A node in the cross-campaign meta-memory knowledge graph."""
    id: str
    node_type: str  # concept, hypothesis, debate, experiment, campaign, metric, finding, paper, component
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """A directed semantic or causal edge between two knowledge nodes."""
    id: str
    source_id: str
    target_id: str
    relation_type: str  # hypothesizes, debates, validates, contradicts, implements, refines, derived_from, measured_by, cites, relates_to
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class Subgraph:
    """A localized sub-network of nodes and connecting edges."""
    nodes: list[KnowledgeNode] = field(default_factory=list)
    edges: list[KnowledgeEdge] = field(default_factory=list)


class MetaMemoryGraph:
    """
    SQLite-backed graph engine maintaining relational and semantic relationships
    across campaigns, debates, code experiments, and long-term memory.
    """

    def __init__(self, db_path: str | Path | None = None):
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else get_db_path_from_url(settings.database_url)

    def _conn(self) -> sqlite3.Connection:
        return get_sync_connection(self.db_path)

    # ── Node Operations ──

    def add_node(
        self,
        id: str,
        node_type: str,
        label: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode:
        """Add or overwrite a knowledge node."""
        now = datetime.now(timezone.utc).isoformat()
        props = properties or {}
        meta = metadata or {}
        blob = pack_vector(embedding) if embedding else None

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO knowledge_nodes (
                    id, node_type, label, properties_json, embedding, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_type = excluded.node_type,
                    label = excluded.label,
                    properties_json = excluded.properties_json,
                    embedding = COALESCE(excluded.embedding, knowledge_nodes.embedding),
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (id, node_type, label, json.dumps(props), blob, now, now, json.dumps(meta)),
            )
            conn.commit()
        finally:
            conn.close()

        return KnowledgeNode(
            id=id,
            node_type=node_type,
            label=label,
            properties=props,
            embedding=embedding,
            created_at=now,
            updated_at=now,
            metadata=meta,
        )

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Retrieve a single knowledge node by ID."""
        conn = self._conn()
        try:
            cursor = conn.execute("SELECT * FROM knowledge_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if not row:
                return None
            emb = unpack_vector(row["embedding"]) if row["embedding"] else None
            return KnowledgeNode(
                id=row["id"],
                node_type=row["node_type"],
                label=row["label"],
                properties=json.loads(row["properties_json"] or "{}"),
                embedding=emb,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        finally:
            conn.close()

    def update_node(
        self,
        node_id: str,
        label: str | None = None,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode | None:
        """Update an existing node's label, properties, or embedding."""
        existing = self.get_node(node_id)
        if not existing:
            return None

        new_label = label if label is not None else existing.label
        new_props = {**existing.properties, **(properties or {})}
        new_emb = embedding if embedding is not None else existing.embedding
        new_meta = {**existing.metadata, **(metadata or {})}
        now = datetime.now(timezone.utc).isoformat()
        blob = pack_vector(new_emb) if new_emb else None

        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE knowledge_nodes
                SET label = ?, properties_json = ?, embedding = ?, updated_at = ?, metadata = ?
                WHERE id = ?
                """,
                (new_label, json.dumps(new_props), blob, now, json.dumps(new_meta), node_id),
            )
            conn.commit()
        finally:
            conn.close()

        return KnowledgeNode(
            id=node_id,
            node_type=existing.node_type,
            label=new_label,
            properties=new_props,
            embedding=new_emb,
            created_at=existing.created_at,
            updated_at=now,
            metadata=new_meta,
        )

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all connected edges via cascading foreign keys."""
        conn = self._conn()
        try:
            cursor = conn.execute("DELETE FROM knowledge_nodes WHERE id = ?", (node_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_nodes(self, node_type: str | None = None, limit: int = 100) -> list[KnowledgeNode]:
        """List knowledge nodes, optionally filtered by node_type."""
        conn = self._conn()
        try:
            if node_type:
                cursor = conn.execute(
                    "SELECT * FROM knowledge_nodes WHERE node_type = ? ORDER BY created_at DESC LIMIT ?",
                    (node_type, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM knowledge_nodes ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            return [
                KnowledgeNode(
                    id=r["id"],
                    node_type=r["node_type"],
                    label=r["label"],
                    properties=json.loads(r["properties_json"] or "{}"),
                    embedding=unpack_vector(r["embedding"]) if r["embedding"] else None,
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ── Edge Operations ──

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
        edge_id: str | None = None,
    ) -> KnowledgeEdge:
        """Add a directed edge connecting source_id to target_id."""
        eid = edge_id or f"edge-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        meta = metadata or {}

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO knowledge_edges (id, source_id, target_id, relation_type, weight, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    relation_type = excluded.relation_type,
                    weight = excluded.weight,
                    metadata = excluded.metadata
                """,
                (eid, source_id, target_id, relation_type, weight, json.dumps(meta), now),
            )
            conn.commit()
        finally:
            conn.close()

        return KnowledgeEdge(
            id=eid,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            metadata=meta,
            created_at=now,
        )

    def get_edges(
        self,
        node_id: str | None = None,
        relation_type: str | None = None,
        limit: int = 150,
    ) -> list[KnowledgeEdge]:
        """Get edges, optionally connected to node_id or matching relation_type."""
        conn = self._conn()
        try:
            query = "SELECT * FROM knowledge_edges WHERE 1=1"
            params: list[Any] = []
            if node_id:
                query += " AND (source_id = ? OR target_id = ?)"
                params.extend([node_id, node_id])
            if relation_type:
                query += " AND relation_type = ?"
                params.append(relation_type)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [
                KnowledgeEdge(
                    id=r["id"],
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relation_type=r["relation_type"],
                    weight=float(r["weight"]),
                    metadata=json.loads(r["metadata"] or "{}"),
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by ID."""
        conn = self._conn()
        try:
            cursor = conn.execute("DELETE FROM knowledge_edges WHERE id = ?", (edge_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[tuple[KnowledgeNode, KnowledgeEdge]]:
        """Retrieve neighboring nodes and their connecting edges."""
        edges = self.get_edges(node_id=node_id, relation_type=relation_type)
        neighbors: list[tuple[KnowledgeNode, KnowledgeEdge]] = []

        for e in edges:
            neighbor_id: str | None = None
            if direction in ("out", "both") and e.source_id == node_id:
                neighbor_id = e.target_id
            elif direction in ("in", "both") and e.target_id == node_id:
                neighbor_id = e.source_id

            if neighbor_id:
                node = self.get_node(neighbor_id)
                if node:
                    neighbors.append((node, e))

        return neighbors

    # ── Graph Traversal & Search ──

    def find_path(self, source_id: str, target_id: str, max_depth: int = 4) -> list[str] | None:
        """Find shortest sequence of node IDs connecting source_id to target_id via BFS."""
        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = collections.deque([[source_id]])

        while queue:
            path = queue.popleft()
            if len(path) > max_depth + 1:
                break

            current_id = path[-1]
            neighbors = self.get_neighbors(current_id, direction="both")

            for neighbor_node, _ in neighbors:
                if neighbor_node.id == target_id:
                    return path + [target_id]
                if neighbor_node.id not in visited:
                    visited.add(neighbor_node.id)
                    queue.append(path + [neighbor_node.id])

        return None

    def get_subgraph(self, center_node_ids: list[str], depth: int = 1) -> Subgraph:
        """Extract a sub-graph of nodes and connecting edges up to depth hops from centers."""
        collected_node_ids = set(center_node_ids)
        frontier = set(center_node_ids)

        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                neighbors = self.get_neighbors(nid, direction="both")
                for node, _ in neighbors:
                    if node.id not in collected_node_ids:
                        next_frontier.add(node.id)
                        collected_node_ids.add(node.id)
            frontier = next_frontier
            if not frontier:
                break

        # Fetch all nodes in set
        nodes: list[KnowledgeNode] = []
        for nid in collected_node_ids:
            node = self.get_node(nid)
            if node:
                nodes.append(node)

        # Fetch all edges connecting any two nodes in collected_node_ids
        conn = self._conn()
        edges: list[KnowledgeEdge] = []
        try:
            all_edges = self.get_edges(limit=500)
            for e in all_edges:
                if e.source_id in collected_node_ids and e.target_id in collected_node_ids:
                    edges.append(e)
        finally:
            conn.close()

        return Subgraph(nodes=nodes, edges=edges)

    def semantic_search_nodes(
        self,
        query_vector: list[float],
        top_k: int = 5,
        node_type: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[KnowledgeNode, float]]:
        """Rank knowledge nodes by dense vector cosine similarity against query_vector."""
        nodes = self.list_nodes(node_type=node_type, limit=500)
        scored: list[tuple[KnowledgeNode, float]] = []

        for n in nodes:
            if n.embedding:
                score = cosine_similarity(query_vector, n.embedding)
                if score >= min_similarity:
                    scored.append((n, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── Campaign Synchronization ──

    def sync_from_campaign(self, campaign_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract conceptual and empirical entities from a research campaign
        and weave them into the knowledge graph with typed relationships.
        """
        cid = campaign_data.get("id") or str(uuid.uuid4())
        title = campaign_data.get("title", "Untitled Campaign")
        goal = campaign_data.get("goal", "")
        artifacts = campaign_data.get("artifacts", {}) or {}

        nodes_created = 0
        edges_created = 0

        # 1. Campaign Node
        camp_nid = f"camp-{cid}"
        self.add_node(
            id=camp_nid,
            node_type="campaign",
            label=title,
            properties={"goal": goal, "status": campaign_data.get("status", "active")},
        )
        nodes_created += 1

        # 2. Hypothesis Node
        hypothesis_text = artifacts.get("hypothesis") or campaign_data.get("initial_hypothesis")
        hypo_nid = f"hypo-{cid}"
        if hypothesis_text:
            self.add_node(
                id=hypo_nid,
                node_type="hypothesis",
                label=f"Hypothesis: {title}",
                properties={"statement": hypothesis_text},
            )
            self.add_edge(camp_nid, hypo_nid, relation_type="hypothesizes")
            nodes_created += 1
            edges_created += 1

        # 3. Debate Consensus Node
        verdict_text = artifacts.get("consensus_verdict")
        deb_nid = f"deb-{cid}"
        if verdict_text:
            self.add_node(
                id=deb_nid,
                node_type="debate",
                label=f"Debate: {title}",
                properties={"verdict": verdict_text, "dissent": artifacts.get("dissenting_views", [])},
            )
            self.add_edge(hypo_nid if hypothesis_text else camp_nid, deb_nid, relation_type="debates")
            nodes_created += 1
            edges_created += 1

        # 4. Code Experiment / Prototype Node
        code_text = artifacts.get("code")
        exp_nid = f"exp-{cid}"
        if code_text:
            self.add_node(
                id=exp_nid,
                node_type="experiment",
                label=f"Prototype: {title}",
                properties={"code_preview": code_text[:200], "tests_passed": artifacts.get("tests_passed", True)},
            )
            self.add_edge(deb_nid if verdict_text else camp_nid, exp_nid, relation_type="implements")
            nodes_created += 1
            edges_created += 1

        # 5. Verified Finding Node
        findings_text = artifacts.get("findings")
        find_nid = f"find-{cid}"
        if findings_text:
            self.add_node(
                id=find_nid,
                node_type="finding",
                label=f"Finding: {title}",
                properties={"summary": findings_text},
            )
            self.add_edge(exp_nid if code_text else camp_nid, find_nid, relation_type="validates")
            nodes_created += 1
            edges_created += 1

        return {
            "campaign_id": cid,
            "nodes_created": nodes_created,
            "edges_created": edges_created,
        }

    def export_graph(self) -> dict[str, Any]:
        """Export full knowledge graph for UI visualization."""
        nodes = self.list_nodes(limit=300)
        edges = self.get_edges(limit=500)
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type,
                    "label": n.label,
                    "properties": n.properties,
                    "created_at": n.created_at,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": e.relation_type,
                    "weight": e.weight,
                }
                for e in edges
            ],
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }
