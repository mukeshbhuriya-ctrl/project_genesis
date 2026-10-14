"""Chat API endpoints."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url
from genesis.core.logging import get_logger
from genesis.models import GenerationRequest
from genesis.models.ollama_backend import OllamaProvider

log = get_logger("api.chat")
router = APIRouter()


class ChatMessageInput(BaseModel):
    """A single message in a chat request."""
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    messages: list[ChatMessageInput]
    model: str = ""
    conversation_id: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    conversation_id: str
    message: dict
    model: str
    usage: dict = Field(default_factory=dict)


def _get_db():
    settings = get_settings()
    db_path = get_db_path_from_url(settings.get_db_url())
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message and get an AI response."""
    settings = get_settings()
    model_name = request.model or settings.default_model

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="No model specified. Set GENESIS_DEFAULT_MODEL or pass 'model' in the request.",
        )

    # Check Ollama availability
    provider = OllamaProvider()
    if not await provider.is_available():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )

    # Create or retrieve conversation
    conn = _get_db()
    try:
        conv_id = request.conversation_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        if not request.conversation_id:
            # Create new conversation
            conn.execute(
                "INSERT INTO conversations (id, title, model_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, request.messages[-1].content[:80], model_name, now, now),
            )

        # Store the user message(s)
        for msg in request.messages:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conv_id, msg.role, msg.content, now),
            )
        conn.commit()

        # Build message history for the model
        history = [{"role": m.role, "content": m.content} for m in request.messages]

        # If continuing a conversation, prepend stored messages
        if request.conversation_id:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conv_id,),
            )
            stored = [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]
            # The last N messages from the request are already in stored
            # Use stored history instead
            history = stored

        # Generate response
        gen_request = GenerationRequest(
            messages=history,
            model=model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        response = await provider.generate(gen_request)

        # Store the assistant response
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, model_id, "
            "token_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()), conv_id, "assistant", response.content,
                response.model, response.token_count_completion, now,
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )
        conn.commit()

        return ChatResponse(
            conversation_id=conv_id,
            message={"role": "assistant", "content": response.content},
            model=response.model,
            usage={
                "prompt_tokens": response.token_count_prompt,
                "completion_tokens": response.token_count_completion,
            },
        )
    finally:
        conn.close()


@router.get("/conversations")
async def list_conversations(limit: int = 20) -> list[dict]:
    """List recent conversations."""
    conn = _get_db()
    try:
        cursor = conn.execute(
            "SELECT id, title, model_id, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str) -> list[dict]:
    """Get all messages in a conversation."""
    conn = _get_db()
    try:
        cursor = conn.execute(
            "SELECT id, role, content, model_id, token_count, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        messages = [dict(row) for row in cursor.fetchall()]
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return messages
    finally:
        conn.close()
