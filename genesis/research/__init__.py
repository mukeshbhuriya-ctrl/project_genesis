"""
Research and Capability Expansion Subsystem (Phase 6).

Provides:
- CodeSandbox: Subprocess isolation for automated test execution
- ReflectiveCodeSynthesizer: Autonomous code synthesis with self-repair loops
- AdaptiveKnowledgeSynthesizer: Multi-source research synthesis with citation verification
- ResearchBrief & CodeExperimentRecord: Persistence data structures
"""

from __future__ import annotations

from genesis.research.campaigns import (
    CampaignManager,
    ResearchCampaign,
    ResearchMilestone,
)
from genesis.research.coding import (
    CodeExperimentRecord,
    CodeSandbox,
    ReflectiveCodeSynthesizer,
    SandboxResult,
)
from genesis.research.daemon import (
    AutonomousResearchDaemon,
    ResearchPaper,
)
from genesis.research.meta import (
    CycleStage,
    MetaCycleRecord,
    MetaResearchOrchestrator,
)
from genesis.research.synthesis import (
    AdaptiveKnowledgeSynthesizer,
    Citation,
    ResearchBrief,
)

__all__ = [
    "CodeSandbox",
    "SandboxResult",
    "CodeExperimentRecord",
    "ReflectiveCodeSynthesizer",
    "Citation",
    "ResearchBrief",
    "AdaptiveKnowledgeSynthesizer",
    "CycleStage",
    "MetaCycleRecord",
    "MetaResearchOrchestrator",
    "CampaignManager",
    "ResearchCampaign",
    "ResearchMilestone",
    "AutonomousResearchDaemon",
    "ResearchPaper",
]

