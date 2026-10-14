"""Tests for the Chat and Conversation API endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import create_app
from genesis.core import get_settings, reset_settings
from genesis.core.database import initialize_database
from genesis.models import GenerationResponse


@pytest.fixture
def chat_client(tmp_path: Path):
    """Provide a test client with an initialized database."""
    reset_settings()
    test_db = tmp_path / "test_chat.db"
    os.environ["GENESIS_DB_URL"] = f"sqlite:///{test_db}"
    os.environ["GENESIS_DEFAULT_MODEL"] = "test-model"

    initialize_database(str(test_db))
    app = create_app()
    client = TestClient(app)

    yield client

    reset_settings()
    os.environ.pop("GENESIS_DB_URL", None)
    os.environ.pop("GENESIS_DEFAULT_MODEL", None)


class TestChatEndpoints:
    """Test /chat, /conversations, and message retrieval."""

    def test_list_conversations_empty(self, chat_client: TestClient):
        """Conversations list should be empty initially."""
        response = chat_client.get("/api/conversations")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_messages_not_found(self, chat_client: TestClient):
        """Fetching messages for non-existent conversation returns 404."""
        response = chat_client.get("/api/conversations/non-existent-id/messages")
        assert response.status_code == 404

    @patch("genesis.models.ollama_backend.OllamaProvider.is_available", new_callable=AsyncMock)
    @patch("genesis.models.ollama_backend.OllamaProvider.generate", new_callable=AsyncMock)
    def test_chat_flow_and_persistence(
        self,
        mock_generate: AsyncMock,
        mock_available: AsyncMock,
        chat_client: TestClient,
    ):
        """Send a message, verify AI response, and verify conversation persists."""
        mock_available.return_value = True
        mock_generate.return_value = GenerationResponse(
            content="Hello! I am Project Genesis.",
            model="test-model",
            role="assistant",
            token_count_prompt=10,
            token_count_completion=8,
        )

        # 1. Send first message
        payload = {
            "messages": [{"role": "user", "content": "Hello, Genesis!"}],
            "model": "test-model",
        }
        res = chat_client.post("/api/chat", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert "conversation_id" in data
        conv_id = data["conversation_id"]
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello! I am Project Genesis."
        assert data["usage"]["completion_tokens"] == 8

        # 2. Check conversation listing
        conv_res = chat_client.get("/api/conversations")
        assert conv_res.status_code == 200
        convs = conv_res.json()
        assert len(convs) == 1
        assert convs[0]["id"] == conv_id
        assert "Hello, Genesis!" in convs[0]["title"]

        # 3. Check messages persistence
        msg_res = chat_client.get(f"/api/conversations/{conv_id}/messages")
        assert msg_res.status_code == 200
        messages = msg_res.json()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, Genesis!"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello! I am Project Genesis."

        # 4. Continue conversation
        mock_generate.return_value = GenerationResponse(
            content="I can help you build self-improving AI systems.",
            model="test-model",
            role="assistant",
            token_count_prompt=25,
            token_count_completion=12,
        )
        followup_payload = {
            "conversation_id": conv_id,
            "messages": [{"role": "user", "content": "What can you do?"}],
            "model": "test-model",
        }
        res2 = chat_client.post("/api/chat", json=followup_payload)
        assert res2.status_code == 200
        assert res2.json()["conversation_id"] == conv_id

        # 5. Check all 4 messages persisted
        msg_res2 = chat_client.get(f"/api/conversations/{conv_id}/messages")
        messages2 = msg_res2.json()
        assert len(messages2) == 4
        assert messages2[2]["content"] == "What can you do?"
        assert messages2[3]["content"] == "I can help you build self-improving AI systems."
