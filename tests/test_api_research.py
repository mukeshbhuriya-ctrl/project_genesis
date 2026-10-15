"""
Integration tests for Research and Capability Expansion REST API endpoints (Phase 6).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database
from genesis.models import GenerationResponse


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    db_file = tmp_path / "api_research_test.db"
    initialize_database(db_file)
    monkeypatch.setenv("GENESIS_DB_URL", f"sqlite:///{db_file}")
    reset_settings()

    app = create_app()
    return TestClient(app)


class TestResearchAPI:
    """Test /api/research routes."""

    def test_get_context_budget(self, client: TestClient):
        res = client.get("/api/research/context-budget?model_id=qwen2.5:7b&context_tokens=4096")
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "qwen2.5:7b"
        assert data["context_length"] == 4096
        assert data["kv_cache_mb"] > 200.0
        assert data["is_safe"] is True
        assert data["headroom_mb"] > 0

    def test_list_profiles_empty_and_retrieval(self, client: TestClient):
        res = client.get("/api/research/profiles")
        assert res.status_code == 200
        data = res.json()
        assert "count" in data
        assert isinstance(data["profiles"], list)

    def test_profile_inference_endpoint(self, client: TestClient):
        # Mock provider streaming generator
        async def mock_stream(request):
            for token in ["Profiling", " response", " tokens."]:
                yield token

        with patch("genesis.models.ollama_backend.OllamaProvider.generate_stream", side_effect=mock_stream):
            res = client.post(
                "/api/research/profile",
                json={
                    "model_id": "qwen2.5:7b",
                    "prompt": "Test profiling speed",
                    "max_tokens": 32,
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert data["model_id"] == "qwen2.5:7b"
            assert data["prompt_tokens"] > 0
            assert data["completion_tokens"] > 0
            assert data["ttft_ms"] >= 0.0

        # Verify listed in profiles
        res_list = client.get("/api/research/profiles")
        assert res_list.status_code == 200
        assert res_list.json()["count"] >= 1

    def test_code_synthesis_and_experiments(self, client: TestClient):
        mock_response = GenerationResponse(
            content=(
                "```python\n"
                "def square(x):\n"
                "    return x * x\n\n"
                "# --- Test Harness ---\n"
                "assert square(4) == 16\n"
                "```"
            ),
            model="qwen2.5-coder:7b",
        )

        with patch("genesis.models.ollama_backend.OllamaProvider.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response

            res = client.post(
                "/api/research/coding/synthesize",
                json={
                    "objective": "Calculate square of a number",
                    "name": "test-square",
                    "max_iterations": 2,
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "passed"
            assert data["name"] == "test-square"
            assert "return x * x" in data["final_code"]
            assert data["iterations"] == 1

        # Check experiments list
        exp_res = client.get("/api/research/coding/experiments")
        assert exp_res.status_code == 200
        exp_data = exp_res.json()
        assert exp_data["count"] >= 1
        assert exp_data["experiments"][0]["name"] == "test-square"

    def test_knowledge_synthesis_endpoint(self, client: TestClient):
        mock_brief = GenerationResponse(
            content=(
                "# Executive Summary\n"
                "Local inference avoids cloud API lock-in [1].\n\n"
                "## Key Synthesized Insights\n"
                "All weights run completely offline [1].\n\n"
                "## Implications & Next Actions\n"
                "Continue incremental development.\n\n"
                "## Reference Sources\n"
                "[1] Local Architecture"
            ),
            model="qwen2.5:7b",
        )

        with patch("genesis.models.ollama_backend.OllamaProvider.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_brief

            res = client.post(
                "/api/research/synthesize",
                json={
                    "topic": "Local Inference Architecture",
                    "max_sources": 3,
                    "save_to_memory": True,
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert data["topic"] == "Local Inference Architecture"
            assert "# Executive Summary" in data["brief_markdown"]

        # Check syntheses list
        syn_res = client.get("/api/research/syntheses")
        assert syn_res.status_code == 200
        syn_data = syn_res.json()
        assert syn_data["count"] >= 1
        assert syn_data["syntheses"][0]["topic"] == "Local Inference Architecture"
