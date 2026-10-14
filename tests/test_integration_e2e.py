"""
End-to-End integration test for Phase 1 Acceptance Criteria.

Acceptance Criterion:
"A real local model can answer a message and the conversation survives an application restart."
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import reset_settings
from genesis.core.database import initialize_database
from genesis.models.ollama_backend import OllamaProvider


@pytest.mark.asyncio
async def test_live_ollama_is_available():
    """Verify that local Ollama instance is running and has qwen2.5:7b."""
    provider = OllamaProvider()
    available = await provider.is_available()
    assert available is True, "Ollama service is not responding at 127.0.0.1:11434"

    models = await provider.list_models()
    model_names = [m.name for m in models]
    assert any("qwen2.5:7b" in name for name in model_names), f"qwen2.5:7b not in installed models: {model_names}"


def test_real_model_chat_and_restart_survival(tmp_path: Path):
    """
    Full Phase 1 integration verification:
    1. Start application with SQLite database
    2. Real model generates a response
    3. Verify conversation and messages are committed to database
    4. Simulate full application shutdown and restart
    5. Re-open database with a fresh application instance
    6. Verify past conversation and message history are intact
    7. Continue the persisted conversation with a followup message
    """
    test_db = tmp_path / "genesis_live_test.db"
    db_url = f"sqlite:///{test_db}"

    # Setup environment
    reset_settings()
    os.environ["GENESIS_DB_URL"] = db_url
    os.environ["GENESIS_DEFAULT_MODEL"] = "qwen2.5:7b"
    initialize_database(test_db)

    # ── SESSION 1: Initial conversation ──
    app1 = create_app()
    client1 = TestClient(app1)

    req_payload = {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Reply in one single word."}
        ],
        "temperature": 0.1,
    }

    response1 = client1.post("/api/chat", json=req_payload)
    assert response1.status_code == 200, f"Chat failed: {response1.text}"
    data1 = response1.json()

    assert "conversation_id" in data1
    conv_id = data1["conversation_id"]
    content1 = data1["message"]["content"].strip()
    assert "Paris" in content1, f"Expected Paris in response, got: {content1}"
    assert data1["usage"]["completion_tokens"] > 0

    # Verify messages in Session 1
    msgs_res1 = client1.get(f"/api/conversations/{conv_id}/messages")
    assert msgs_res1.status_code == 200
    msgs1 = msgs_res1.json()
    assert len(msgs1) == 2
    assert msgs1[0]["role"] == "user"
    assert msgs1[1]["role"] == "assistant"
    assert "Paris" in msgs1[1]["content"]

    # ── SIMULATE APPLICATION RESTART ──
    # Explicitly tear down client1 and app1 references, reset config singletons
    del client1
    del app1
    reset_settings()

    # ── SESSION 2: Post-restart recovery and continuation ──
    app2 = create_app()
    client2 = TestClient(app2)

    # Verify conversation list is preserved
    convs_res2 = client2.get("/api/conversations")
    assert convs_res2.status_code == 200
    convs2 = convs_res2.json()
    assert len(convs2) == 1
    assert convs2[0]["id"] == conv_id

    # Verify full message history survived restart
    msgs_res2 = client2.get(f"/api/conversations/{conv_id}/messages")
    assert msgs_res2.status_code == 200
    msgs2 = msgs_res2.json()
    assert len(msgs2) == 2
    assert msgs2[0]["role"] == "user"
    assert msgs2[1]["role"] == "assistant"
    assert "Paris" in msgs2[1]["content"]

    # Continue the conversation in Session 2
    followup_payload = {
        "conversation_id": conv_id,
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "user", "content": "And what is its primary currency? Reply in one single word."}
        ],
        "temperature": 0.1,
    }

    response2 = client2.post("/api/chat", json=followup_payload)
    assert response2.status_code == 200, f"Followup failed: {response2.text}"
    data2 = response2.json()
    assert data2["conversation_id"] == conv_id
    content2 = data2["message"]["content"].strip()
    assert "Euro" in content2, f"Expected Euro in response, got: {content2}"

    # Verify all 4 messages are now persisted
    final_msgs_res = client2.get(f"/api/conversations/{conv_id}/messages")
    assert final_msgs_res.status_code == 200
    final_msgs = final_msgs_res.json()
    assert len(final_msgs) == 4
    assert final_msgs[2]["content"] == "And what is its primary currency? Reply in one single word."
    assert "Euro" in final_msgs[3]["content"]

    # Cleanup
    del client2
    del app2
    reset_settings()
    os.environ.pop("GENESIS_DB_URL", None)
    os.environ.pop("GENESIS_DEFAULT_MODEL", None)
