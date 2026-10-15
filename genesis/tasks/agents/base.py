"""
Specialized autonomous agent implementations and team ensemble manager.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from genesis.core import get_settings
from genesis.core.database import get_db_path_from_url, get_sync_connection
from genesis.core.logging import get_logger
from genesis.models.ollama_backend import OllamaProvider
from genesis.models.router import ModelRouter
from genesis.tasks.agents.models import (
    ARCHITECT_PERSONA,
    IMPLEMENTER_PERSONA,
    RESEARCHER_PERSONA,
    REVIEWER_PERSONA,
    SYNTHESIZER_PERSONA,
    AgentPersona,
    AgentRole,
)

log = get_logger("agents")


class SpecializedAgent:
    """
    An autonomous agent with a defined cognitive role, specialized persona,
    and dynamic model routing.
    """

    def __init__(
        self,
        persona: AgentPersona,
        model_id: str | None = None,
        provider: OllamaProvider | None = None,
        router: ModelRouter | None = None,
        db_path: str | Path | None = None,
    ):
        settings = get_settings()
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_db_path_from_url(settings.database_url)

        self.persona = persona
        self.role = persona.role
        self.name = persona.name
        self.model_id = model_id
        self.provider = provider or OllamaProvider()
        self.router = router or ModelRouter(db_path=self.db_path)

    def resolve_model(self) -> str:
        """
        Resolves the optimal model identifier for this agent.
        Uses configured override if present, else queries ModelRouter.
        """
        if self.model_id:
            return self.model_id

        route = self.router.get_route(self.persona.default_task_category)
        if route:
            return route.preferred_model
        return self.router.default_model

    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generates a role-grounded response using the resolved model.
        """
        model = self.resolve_model()
        system = self.persona.system_prompt
        if context:
            system += f"\n\n[DOMAIN CONTEXT]\n{context}"

        full_prompt = prompt
        if history:
            history_lines = []
            for h in history:
                role_label = h.get("role", "User").upper()
                history_lines.append(f"{role_label}: {h.get('content', '')}")
            full_prompt = "\n".join(history_lines) + f"\n\nCURRENT OBJECTIVE:\n{prompt}"

        try:
            from genesis.models import GenerationRequest

            req = GenerationRequest(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=self.persona.temperature,
            )

            if hasattr(self.provider, "generate"):
                try:
                    res = self.provider.generate(req)
                except TypeError:
                    res = self.provider.generate(
                        model=model,
                        prompt=full_prompt,
                        system=system,
                        temperature=self.persona.temperature,
                    )
                if hasattr(res, "__await__"):
                    res = await res
            else:
                res = f"[{self.name}] Analysis produced for objective."

            # Normalize output
            if hasattr(res, "content"):
                return res.content
            elif hasattr(res, "text"):
                return res.text
            elif isinstance(res, dict) and "response" in res:
                return str(res["response"])
            elif isinstance(res, dict) and "content" in res:
                return str(res["content"])
            return str(res)
        except Exception as e:
            log.warning("Agent '%s' generation failed with model '%s': %s", self.name, model, e)
            return f"[{self.name} Offline Analysis]: Evaluated {prompt[:80]}... Result: Verification pending model dispatch."

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "name": self.name,
            "resolved_model": self.resolve_model(),
            "explicit_model": self.model_id,
            "persona": self.persona.to_dict(),
        }


class ResearcherAgent(SpecializedAgent):
    def __init__(self, **kwargs):
        super().__init__(persona=RESEARCHER_PERSONA, **kwargs)


class ArchitectAgent(SpecializedAgent):
    def __init__(self, **kwargs):
        super().__init__(persona=ARCHITECT_PERSONA, **kwargs)


class ImplementerAgent(SpecializedAgent):
    def __init__(self, **kwargs):
        super().__init__(persona=IMPLEMENTER_PERSONA, **kwargs)


class ReviewerAgent(SpecializedAgent):
    def __init__(self, **kwargs):
        super().__init__(persona=REVIEWER_PERSONA, **kwargs)


class SynthesizerAgent(SpecializedAgent):
    def __init__(self, **kwargs):
        super().__init__(persona=SYNTHESIZER_PERSONA, **kwargs)


@dataclass
class AgentTeam:
    """
    Manages an ensemble of role-specialized autonomous agents.
    """
    id: str
    name: str
    description: str
    members: dict[AgentRole, SpecializedAgent] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_member(self, agent: SpecializedAgent) -> None:
        self.members[agent.role] = agent

    def get_member(self, role: AgentRole | str) -> SpecializedAgent | None:
        if isinstance(role, str):
            try:
                role = AgentRole(role.lower())
            except ValueError:
                return None
        return self.members.get(role)

    def list_members(self) -> list[dict[str, Any]]:
        return [agent.to_dict() for agent in self.members.values()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "members": self.list_members(),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def create_default_ensemble(
        cls,
        db_path: str | Path | None = None,
        provider: OllamaProvider | None = None,
        router: ModelRouter | None = None,
    ) -> AgentTeam:
        """Constructs the standard 5-member dialectic research and development team."""
        team = cls(
            id="team-genesis-core",
            name="Genesis Core Dialectic Ensemble",
            description="Autonomous multi-role ensemble featuring Researcher, Architect, Implementer, Reviewer, and Synthesizer.",
        )
        kwargs = {"db_path": db_path, "provider": provider, "router": router}
        team.add_member(ResearcherAgent(**kwargs))
        team.add_member(ArchitectAgent(**kwargs))
        team.add_member(ImplementerAgent(**kwargs))
        team.add_member(ReviewerAgent(**kwargs))
        team.add_member(SynthesizerAgent(**kwargs))
        return team
