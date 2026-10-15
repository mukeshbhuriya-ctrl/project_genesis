"""
World Model Subsystem (Phase 15).

Provides multimodal environment transition modeling, latent dream rollouts,
cross-modal representation alignment, and speculative Monte Carlo Tree Search (MCTS)
with state-forking and atomic rollback.
"""

from genesis.world_model.alignment import (
    AlignmentResult,
    CrossModalAligner,
    ModalityEmbedding,
)
from genesis.world_model.dream import (
    DreamStep,
    DreamTrajectory,
    LatentDreamEngine,
)
from genesis.world_model.mcts import (
    MCTSNode,
    MCTSRunResult,
    MCTSSearchEngine,
    StateSnapshot,
)
from genesis.world_model.transition import (
    Action,
    EnvironmentState,
    StateTransitionModel,
    TransitionPrediction,
    TransitionRecord,
)

__all__ = [
    "EnvironmentState",
    "Action",
    "TransitionPrediction",
    "TransitionRecord",
    "StateTransitionModel",
    "DreamStep",
    "DreamTrajectory",
    "LatentDreamEngine",
    "ModalityEmbedding",
    "AlignmentResult",
    "CrossModalAligner",
    "StateSnapshot",
    "MCTSNode",
    "MCTSRunResult",
    "MCTSSearchEngine",
]
