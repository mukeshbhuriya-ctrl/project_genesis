"""
Tests for Cross-Campaign Meta-Memory Knowledge Graph Engine (Phase 10).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.core.database import initialize_database
from genesis.memory.graph import KnowledgeEdge, KnowledgeNode, MetaMemoryGraph


@pytest.fixture
def graph(tmp_db_path: Path) -> MetaMemoryGraph:
    """Provide an initialized MetaMemoryGraph engine."""
    initialize_database(tmp_db_path)
    return MetaMemoryGraph(db_path=tmp_db_path)


class TestMetaMemoryGraph:
    """Test suite for MetaMemoryGraph."""

    def test_node_crud(self, graph: MetaMemoryGraph):
        """Should add, retrieve, update, and delete nodes."""
        node = graph.add_node(
            id="node-kv-cache",
            node_type="concept",
            label="KV Cache Compression",
            properties={"sparsity": 0.4, "quantization": "int8"},
        )
        assert node.id == "node-kv-cache"
        assert node.label == "KV Cache Compression"

        retrieved = graph.get_node("node-kv-cache")
        assert retrieved is not None
        assert retrieved.label == "KV Cache Compression"
        assert retrieved.properties["sparsity"] == 0.4

        # Update
        updated = graph.update_node(
            node_id="node-kv-cache",
            label="Optimized KV Cache Compression",
            properties={"throughput_gain": "1.35x"},
        )
        assert updated is not None
        assert updated.label == "Optimized KV Cache Compression"
        assert updated.properties["sparsity"] == 0.4
        assert updated.properties["throughput_gain"] == "1.35x"

        # List
        nodes = graph.list_nodes(node_type="concept")
        assert any(n.id == "node-kv-cache" for n in nodes)

        # Delete
        assert graph.delete_node("node-kv-cache") is True
        assert graph.get_node("node-kv-cache") is None

    def test_edge_crud_and_cascade(self, graph: MetaMemoryGraph):
        """Should create edges and cascade-delete when a node is removed."""
        graph.add_node(id="node-a", node_type="concept", label="Attention Engine")
        graph.add_node(id="node-b", node_type="component", label="FlashAttention Kernel")

        edge = graph.add_edge(
            source_id="node-a",
            target_id="node-b",
            relation_type="implements",
            weight=1.5,
        )
        assert edge.source_id == "node-a"
        assert edge.target_id == "node-b"
        assert edge.relation_type == "implements"

        edges = graph.get_edges(node_id="node-a")
        assert len(edges) == 1
        assert edges[0].id == edge.id

        # Cascading deletion
        graph.delete_node("node-a")
        remaining_edges = graph.get_edges(node_id="node-b")
        assert len(remaining_edges) == 0

    def test_neighbors_traversal(self, graph: MetaMemoryGraph):
        """Should traverse inbound, outbound, and bidirectional neighbors."""
        graph.add_node(id="center", node_type="concept", label="Central Optimizer")
        graph.add_node(id="target-1", node_type="component", label="Memory Pool")
        graph.add_node(id="source-1", node_type="hypothesis", label="Initial Guess")

        graph.add_edge("center", "target-1", "implements")
        graph.add_edge("source-1", "center", "hypothesizes")

        out_neighbors = graph.get_neighbors("center", direction="out")
        assert len(out_neighbors) == 1
        assert out_neighbors[0][0].id == "target-1"

        in_neighbors = graph.get_neighbors("center", direction="in")
        assert len(in_neighbors) == 1
        assert in_neighbors[0][0].id == "source-1"

        both = graph.get_neighbors("center", direction="both")
        assert len(both) == 2

    def test_path_finding_bfs(self, graph: MetaMemoryGraph):
        """Should find shortest path between multi-hop nodes."""
        graph.add_node(id="n1", node_type="concept", label="Step 1")
        graph.add_node(id="n2", node_type="concept", label="Step 2")
        graph.add_node(id="n3", node_type="concept", label="Step 3")
        graph.add_node(id="n4", node_type="concept", label="Step 4")

        graph.add_edge("n1", "n2", "relates_to")
        graph.add_edge("n2", "n3", "relates_to")
        graph.add_edge("n3", "n4", "relates_to")

        path = graph.find_path("n1", "n4", max_depth=5)
        assert path == ["n1", "n2", "n3", "n4"]

        # No path case
        graph.add_node(id="isolated", node_type="concept", label="Isolated Island")
        assert graph.find_path("n1", "isolated", max_depth=3) is None

    def test_subgraph_extraction(self, graph: MetaMemoryGraph):
        """Should extract localized subgraphs bounded by depth hops."""
        graph.add_node(id="hub", node_type="concept", label="Hub")
        graph.add_node(id="leaf-1", node_type="concept", label="Leaf 1")
        graph.add_node(id="leaf-2", node_type="concept", label="Leaf 2")
        graph.add_node(id="distant", node_type="concept", label="Distant")

        graph.add_edge("hub", "leaf-1", "relates_to")
        graph.add_edge("hub", "leaf-2", "relates_to")
        graph.add_edge("leaf-2", "distant", "relates_to")

        sub = graph.get_subgraph(center_node_ids=["hub"], depth=1)
        node_ids = {n.id for n in sub.nodes}
        assert "hub" in node_ids
        assert "leaf-1" in node_ids
        assert "leaf-2" in node_ids
        assert "distant" not in node_ids
        assert len(sub.edges) == 2

    def test_semantic_search_nodes(self, graph: MetaMemoryGraph):
        """Should rank nodes by cosine similarity of dense embeddings."""
        v_math = [1.0, 0.0, 0.0] + [0.0] * 765
        v_lang = [0.0, 1.0, 0.0] + [0.0] * 765
        query = [0.95, 0.05, 0.0] + [0.0] * 765

        graph.add_node(id="node-math", node_type="concept", label="Linear Algebra", embedding=v_math)
        graph.add_node(id="node-lang", node_type="concept", label="Syntax Grammar", embedding=v_lang)

        results = graph.semantic_search_nodes(query_vector=query, top_k=2)
        assert len(results) == 2
        assert results[0][0].id == "node-math"
        assert results[0][1] > 0.9

    def test_sync_from_campaign(self, graph: MetaMemoryGraph):
        """Should extract campaign milestones, debate consensus, and findings into typed graph."""
        campaign_payload = {
            "id": "camp-opt-01",
            "title": "Autonomous KV Cache Compression",
            "goal": "Compress attention memory on 8GB VRAM without perplexity regression",
            "status": "completed",
            "artifacts": {
                "hypothesis": "Per-channel symmetric int8 scaling retains >98% accuracy",
                "consensus_verdict": "Consensus: Adopt block-wise dynamic scaling with 32-element chunks",
                "dissenting_views": ["Reviewer noted potential latency spikes during quantization"],
                "code": "def compress_kv(tensor): return tensor.to(torch.int8)",
                "tests_passed": True,
                "findings": "Verified ~38% VRAM reduction with zero accuracy regression.",
            },
        }

        res = graph.sync_from_campaign(campaign_payload)
        assert res["nodes_created"] == 5
        assert res["edges_created"] == 4

        # Verify campaign node
        camp_node = graph.get_node("camp-camp-opt-01")
        assert camp_node is not None
        assert camp_node.label == "Autonomous KV Cache Compression"

        # Verify finding node
        find_node = graph.get_node("find-camp-opt-01")
        assert find_node is not None
        assert "Verified ~38% VRAM reduction" in find_node.properties["summary"]

        # Verify export graph
        export_data = graph.export_graph()
        assert export_data["stats"]["total_nodes"] >= 5
        assert export_data["stats"]["total_edges"] >= 4
