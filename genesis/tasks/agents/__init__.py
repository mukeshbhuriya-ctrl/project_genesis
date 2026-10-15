"""
Multi-agent specialization subsystem for Project Genesis (Phase 9).
"""

from genesis.tasks.agents.base import (
    AgentTeam,
    ArchitectAgent,
    ImplementerAgent,
    ResearcherAgent,
    ReviewerAgent,
    SpecializedAgent,
    SynthesizerAgent,
)
from genesis.tasks.agents.models import (
    ARCHITECT_PERSONA,
    IMPLEMENTER_PERSONA,
    RESEARCHER_PERSONA,
    REVIEWER_PERSONA,
    SYNTHESIZER_PERSONA,
    AgentPersona,
    AgentRole,
)

__all__ = [
    "AgentRole",
    "AgentPersona",
    "RESEARCHER_PERSONA",
    "ARCHITECT_PERSONA",
    "IMPLEMENTER_PERSONA",
    "REVIEWER_PERSONA",
    "SYNTHESIZER_PERSONA",
    "SpecializedAgent",
    "ResearcherAgent",
    "ArchitectAgent",
    "ImplementerAgent",
    "ReviewerAgent",
    "SynthesizerAgent",
    "AgentTeam",
]
