# ADR-010: Multi-Agent Specialization, Dialectic Consensus Debates, and Autonomous Long-Horizon Research Campaigns

## Status
Accepted

## Date
2026-10-15

## Context
Project Genesis is a local-first, self-improving AI research and model development platform operating on Windows workstations equipped with local hardware accelerators (NVIDIA RTX 2000 Ada, 8 GB VRAM). Following Phase 8 (autonomous meta-improvement, dynamic multi-model dispatch, and automated benchmark edge-case expansion), Phase 9 advances the cognitive architecture of Project Genesis into **multi-agent specialization, dialectic consensus protocols, and autonomous long-horizon research campaigns**.

Monolithic, single-prompt agent loops suffer from cognitive bias, unexamined assumptions, and difficulty managing multi-step research spanning theoretical formulation, critical review, and implementation. To conduct genuine autonomous scientific inquiry locally, three core architectural subsystems were required:
1. **Role-Specialized Cognitive Agent Ensembles**: Decomposition of the reasoning pipeline into specialized personas with distinct cognitive biases, task boundaries, and model dispatch requirements:
   - **Researcher**: Literature retrieval, empirical evidence gathering, hypothesis formulation.
   - **Architect**: System boundary definition, API contract design, VRAM and computational budgeting.
   - **Implementer**: Executable Python synthesis, algorithmic efficiency, concrete test design.
   - **Reviewer**: Adversarial critique, edge-case vulnerability detection, regression risk auditing.
   - **Synthesizer**: Dialectic arbitration, consensus distillation, dissenting opinion extraction.
2. **Dialectic Consensus Debate Engine**: A formal multi-round protocol where proposals undergo iterative proposal, adversarial critique, implementation defense, and consensus synthesis, preventing hallucination cascades and premature convergence.
3. **Autonomous Long-Horizon Research Campaigns**: Multi-milestone workflow orchestration guiding open-ended research objectives from hypothesis formulation through debate consensus, code prototyping in sandboxed environments, and permanent consolidation into the episodic/semantic memory store.

## Decisions

### 1. Database Schema Version 9 Migrations (`genesis/core/database.py`)
Schema Version 9 introduces four relational tables operating in SQLite WAL mode:
- `agent_teams`: Defines agent ensemble groups (`id`, `name`, `description`, `members`, `created_at`, `updated_at`, `metadata`). Pre-seeded with production default team `team-genesis-core` comprising all 5 cognitive roles.
- `debate_sessions`: Persists dialectic debate sessions (`id`, `topic`, `team_id`, `rounds`, `context`, `status`, `consensus_verdict`, `dissenting_views`, `created_at`, `completed_at`, `metadata`). Indexes: `idx_debate_sessions_status`, `idx_debate_sessions_team`.
- `debate_turns`: Relational log of all agent utterances within a debate (`id`, `session_id`, `speaker_role`, `agent_name`, `round_num`, `content`, `metadata`, `created_at`). Indexes: `idx_debate_turns_session`, `idx_debate_turns_role`.
- `research_campaigns`: Manages long-horizon research projects (`id`, `title`, `goal`, `team_id`, `status`, `current_milestone_index`, `milestones`, `artifacts`, `created_at`, `completed_at`, `metadata`). Indexes: `idx_research_campaigns_status`, `idx_research_campaigns_team`.

### 2. Role-Specialized Autonomous Agent Ensembles (`genesis/tasks/agents/`)
- Encapsulated cognitive personas in `genesis/tasks/agents/models.py` (`AgentRole`, `AgentPersona`) with customized system instructions.
- Implemented `SpecializedAgent` base class and concrete role subclasses (`ResearcherAgent`, `ArchitectAgent`, `ImplementerAgent`, `ReviewerAgent`, `SynthesizerAgent`) in `genesis/tasks/agents/base.py`.
- Agents automatically resolve their optimal local model via `ModelRouter` (e.g. `Implementer` defaults to `coding` route -> `qwen2.5-coder:7b`; `Researcher` and `Synthesizer` default to `reasoning` route -> `qwen2.5:7b`).
- `AgentTeam` coordinates ensemble membership, role dispatch, and SQLite persistence.

### 3. Dialectic Consensus Engine (`genesis/tasks/consensus.py`)
- Coordinates multi-round dialectic debates across ensemble agents.
- **Round Sequence**:
  - Initial Round: Researcher formulates empirical proposal &rarr; Architect defines architectural boundaries &rarr; Implementer generates concrete prototype approach.
  - Review & Defense: Reviewer performs adversarial critique and edge-case challenge &rarr; Implementer/Architect issue rebuttal and refined proposal.
  - Synthesis & Verdict: Synthesizer reviews full transcript, extracts points of universal agreement, defines the actionable consensus verdict, and logs unresolved dissenting views.
- Records all turns and consensus output in SQLite `debate_sessions` and `debate_turns`.

### 4. Autonomous Long-Horizon Research Campaigns (`genesis/research/campaigns.py`)
- `CampaignManager` orchestrates multi-milestone research pipelines spanning 4 canonical stages:
  1. `hypothesis`: Formulate concrete scientific or architectural hypothesis from goal.
  2. `consensus`: Convene multi-agent dialectic debate to validate feasibility and challenge assumptions.
  3. `experiment`: Synthesize prototype code and execute inside sandboxed environment (`SafeCodeSandbox`).
  4. `consolidation`: Synthesize final findings and persist verified knowledge into Genesis memory store (`MemoryStore.create`).
- Supports step-wise progression (`advance_campaign`), artifact preservation, and full JSON audit serialization.

### 5. Multi-Agent & Campaign REST API (`genesis/api/routes/agents.py`)
- `GET /api/agents/roles`: Enumerates specialized roles and cognitive personas.
- `GET /api/agents/teams` & `POST /api/agents/teams`: Lists and creates agent ensemble teams.
- `POST /api/agents/debate/run`: Executes multi-agent dialectic debate with configurable rounds.
- `GET /api/agents/debates` & `GET /api/agents/debates/{id}`: Retrieves debate history and turn transcripts.
- `POST /api/agents/campaigns` & `GET /api/agents/campaigns`: Creates and lists long-horizon research campaigns.
- `GET /api/agents/campaigns/{id}` & `POST /api/agents/campaigns/{id}/advance`: Inspects and advances campaign milestones.

### 6. Web Dashboard Multi-Agent & Campaigns UI (`genesis/web/`)
- Added dedicated **"Multi-Agent & Campaigns"** navigation tab (`#pane-agents`) to `genesis/web/index.html`.
- Implemented active team roster grid showing all 5 specialized roles, responsibilities, and prompt summaries.
- Built interactive Dialectic Consensus Arena with debate history, verdict banners, turn cards color-coded by cognitive role, and launch modal.
- Built Long-Horizon Research Campaigns Board featuring visual milestone stepper, artifacts inspection, one-click milestone progression, and creation modal.
- Added comprehensive frontend controllers and event listeners in `genesis/web/app.js`.

## Consequences
- Complex architectural choices and novel algorithmic implementations are now subject to rigorous peer critique and debate before code synthesis.
- Autonomous research goals can execute over extended horizons with persistent checkpointing across SQLite and structured memory.
- All operations remain strictly local-first with zero external cloud or internet dependencies, running within the 8 GB VRAM budget of the local RTX 2000 Ada GPU.
