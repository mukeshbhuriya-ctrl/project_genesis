"""
Integration tests for Self-Compilation and Fleet Synchronization API routes (Phase 11).
"""

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core.compiler import C_DOT_PRODUCT_SOURCE


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_api_get_compilers(client):
    """Test GET /api/compiler/compilers returns compiler detection report."""
    res = client.get("/api/compiler/compilers")
    assert res.status_code == 200
    data = res.json()
    assert "compilers" in data
    assert "has_native_compiler" in data
    assert "supported_languages" in data


def test_api_compile_and_list_extensions(client):
    """Test POST /api/compiler/compile and GET /api/compiler/extensions."""
    payload = {
        "name": "vector_dot_product",
        "language": "c",
        "source_code": C_DOT_PRODUCT_SOURCE,
        "metadata": {"test": True},
    }
    compile_res = client.post("/api/compiler/compile", json=payload)
    assert compile_res.status_code == 200
    data = compile_res.json()
    assert data["name"] == "vector_dot_product"
    assert data["status"] in ("compiled", "fallback_python")

    # List extensions
    list_res = client.get("/api/compiler/extensions")
    assert list_res.status_code == 200
    exts = list_res.json()
    assert isinstance(exts, list)
    assert any(e["name"] == "vector_dot_product" for e in exts)


def test_api_benchmark_and_invoke_extension(client):
    """Test POST /api/compiler/benchmark/{name} and POST /api/compiler/invoke/{name}."""
    # Benchmark
    bench_res = client.post(
        "/api/compiler/benchmark/vector_dot_product",
        json={"iterations": 50},
    )
    assert bench_res.status_code == 200
    bench_data = bench_res.json()
    assert bench_data["kernel"] == "vector_dot_product"
    assert "speedup_factor" in bench_data

    # Invoke
    invoke_res = client.post(
        "/api/compiler/invoke/vector_dot_product",
        json={"arguments": [[1.0, 2.0], [3.0, 4.0]]},
    )
    assert invoke_res.status_code == 200
    invoke_data = invoke_res.json()
    assert invoke_data["status"] == "success"
    assert pytest.approx(invoke_data["result"], 0.001) == 11.0  # 1*3 + 2*4 = 11


def test_api_fleet_nodes_crud_and_ping(client):
    """Test /api/fleet/nodes endpoints and ping."""
    # List nodes (at least self node exists)
    list_res = client.get("/api/fleet/nodes")
    assert list_res.status_code == 200
    nodes = list_res.json()
    assert len(nodes) >= 1

    # Register node
    new_node = {
        "hostname": "alpha-station",
        "address": "10.0.0.42",
        "port": 8000,
        "gpu_device": "RTX 3080",
        "vram_gb": 10.0,
        "metadata": {"env": "lab"},
    }
    reg_res = client.post("/api/fleet/nodes", json=new_node)
    assert reg_res.status_code == 200
    node_data = reg_res.json()
    node_id = node_data["id"]
    assert node_data["hostname"] == "alpha-station"

    # Get specific node
    get_res = client.get(f"/api/fleet/nodes/{node_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == node_id

    # Ping node
    ping_res = client.post(f"/api/fleet/nodes/{node_id}/ping")
    assert ping_res.status_code == 200
    assert ping_res.json()["status"] == "online"

    # Delete node
    del_res = client.delete(f"/api/fleet/nodes/{node_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"


def test_api_fleet_delta_bundle_and_logs(client):
    """Test exporting delta bundle, importing it, and viewing sync logs."""
    # Export
    export_res = client.get("/api/fleet/export-delta")
    assert export_res.status_code == 200
    bundle = export_res.json()
    assert "checksum" in bundle
    assert "entities" in bundle

    # Import
    import_res = client.post(
        "/api/fleet/import-delta",
        json={"bundle": bundle, "source_node_id": "test-remote-peer"},
    )
    assert import_res.status_code == 200
    import_data = import_res.json()
    assert import_data["status"] == "success"

    # Check logs
    logs_res = client.get("/api/fleet/logs?limit=10")
    assert logs_res.status_code == 200
    logs = logs_res.json()
    assert isinstance(logs, list)
    assert len(logs) >= 1
