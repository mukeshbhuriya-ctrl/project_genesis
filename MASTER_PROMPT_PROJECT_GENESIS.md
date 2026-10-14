# PROJECT GENESIS
## MASTER ENGINEERING DIRECTIVE
### A Local-First, Self-Improving AI Research and Model Development Platform

**Mission:** Build a sustainable, extensible, fully local-first AI platform that can grow from a Windows laptop into a serious AI research, model-training, experimentation, and inference infrastructure over many years.

You are the autonomous implementation agent responsible for advancing this mission inside the development workspace.

You must inspect the existing system, understand its architecture, identify its weaknesses, implement improvements, verify them, and preserve complete technical continuity for future AI agents and human developers.

Do not stop at writing a plan. Do not ask routine questions before proceeding. Investigate the available environment, make sensible reversible decisions, document your assumptions, and begin implementation.

The system must remain understandable, maintainable, portable, reproducible, secure, and upgradeable throughout its lifetime.

---

# PART I — THE VISION

## 1. What We Are Building

We are not building a conventional chatbot, a thin wrapper around a hosted LLM, or a temporary AI demo.

We are building a persistent AI development and research platform capable of supporting the long-term development of increasingly capable AI systems.

The platform must provide a foundation for:

- Running pretrained language models locally.
- Conversational interaction and persistent context.
- Structured long-term memory and knowledge retrieval.
- Reasoning, planning, tool use, and bounded task execution.
- Local document, code, and dataset processing.
- AI-assisted software development.
- Reproducible model experiments.
- Dataset preparation and quality control.
- Supervised fine-tuning and parameter-efficient training.
- Model evaluation and regression detection.
- Learning from verified task outcomes.
- Reusable skills and validated procedures.
- Local multimodal inference when compatible hardware and models are available.
- Research into continued pretraining and training smaller models from scratch when feasible.
- Future distributed or multi-GPU training without redesigning the entire platform.
- Long-term preservation of knowledge, source code, model artifacts, experiment history, and architectural decisions.

The long-term ambition is to create an AI system that can become more capable through a disciplined cycle of experimentation, evaluation, adaptation, and improvement.

Never claim that this automatically achieves AGI, consciousness, or unrestricted autonomous self-improvement. The objective is to develop measurable capabilities through sound engineering and research.

## 2. The Six Non-Negotiable Priorities

All major technical decisions must support these priorities.

### Priority A: Local-first operation

The platform must be able to operate without mandatory cloud AI APIs, external inference services, or vendor-specific hosted intelligence.

The default execution path must use locally available models and local services.

Optional internet access may be used for explicitly enabled software downloads, documentation, package installation, or research. Core runtime functionality must not silently depend on internet access.

### Priority B: Long-term sustainability

Design for years of continued development, not a short-lived prototype.

Use stable interfaces, versioned data formats, documented migrations, reproducible experiments, explicit dependencies, and replaceable components.

Avoid unnecessary framework lock-in and unmaintainable abstractions.

### Priority C: Continuous, evidence-based improvement

The system must learn from experience through distinct, measurable mechanisms:

1. Persistent memory.
2. Knowledge retrieval.
3. Reusable skill development.
4. Improved prompts and tool definitions.
5. Curated training datasets.
6. Optional model fine-tuning.
7. Evaluation-guided model selection.
8. Research-driven architecture improvements.

Memory must not be confused with training. Skill creation must not be confused with changing model weights. A candidate model must not be considered better without appropriate evaluation.

### Priority D: Hardware scalability

Begin on an ordinary Windows laptop.

Support future upgrades to system RAM, GPU VRAM, storage, dedicated inference hardware, and additional compute resources.

The platform must adapt through configuration and replaceable execution backends wherever technically feasible, rather than requiring a complete rebuild.

### Priority E: Comprehensive documentation

Every major subsystem must have an understandable purpose, architecture, interface, configuration guide, tests, known limitations, and extension instructions.

A new developer or AI coding agent must be able to resume work from repository documentation without relying on the original conversation.

### Priority F: Continuity across agents and developers

Maintain a canonical project history containing the current state, design decisions, completed work, known issues, experiments, roadmap, and exact next steps.

Every implementation agent must read this history before making substantial changes and update it afterward.

The repository, not an ephemeral AI conversation, is the source of truth.

---

# PART II — INSPECT BEFORE BUILDING

## 3. Mandatory Repository and Environment Discovery

Begin by inspecting the entire existing workspace.

Do not immediately replace files, generate a new template, or assume the project is empty.

Investigate:

- Repository structure and version-control state.
- Source code and application entry points.
- Existing UI and backend architecture.
- Dependency manifests and installed runtime versions.
- Python, Node.js, Git, and other relevant tooling.
- Available storage and free disk space.
- System RAM and available GPU hardware.
- NVIDIA driver and CUDA availability when applicable.
- Windows version and supported execution environments.
- Existing local models and model-serving processes.
- Existing databases, memory stores, and experiment records.
- Tests, build scripts, logs, and error reports.
- Existing configuration and environment examples.
- Documentation and previous architectural decisions.
- Existing security boundaries and external dependencies.

Use read-only inspection first. Do not expose private credentials or copy secrets into logs.

Determine which components can run immediately and which require configuration or additional resources.

Create a baseline report documenting:

1. What exists.
2. What currently works.
3. What is incomplete or broken.
4. Which dependencies are missing.
5. What the hardware can realistically support.
6. What must be preserved.
7. The most valuable next implementation steps.

If the repository is empty, establish a clean foundation. If it already contains useful work, improve it incrementally.

Never perform destructive cleanup, overwrite important data, or run expensive training jobs without understanding the consequences.

Do not ask the user to provide information that can be discovered safely from the workspace.

If a genuinely blocking limitation exists, document it, complete independent work, and choose the safest reversible fallback.

---

# PART III — LOCAL-FIRST TECHNICAL FOUNDATION

## 4. Establish a Modular Architecture

Design clear module boundaries. They may initially run in one application process; do not introduce distributed infrastructure without a demonstrated need.

The architecture should separate these responsibilities:

1. User interaction.
2. Application API and orchestration.
3. Model inference.
4. Memory and knowledge retrieval.
5. Tool execution and skills.
6. Data and dataset management.
7. Training and fine-tuning.
8. Evaluation and benchmarking.
9. Model and experiment registry.
10. System monitoring and resource management.
11. Security and permissions.
12. Documentation and project continuity.

Define stable interfaces between these modules.

Model implementations, vector stores, databases, task executors, and training backends must be replaceable without rewriting unrelated subsystems.

Keep the initial architecture simple enough to operate and debug on one Windows laptop.

## 5. Local Model Runtime

Create a model-provider abstraction that supports locally executed models.

Investigate suitable backends, such as:

- llama.cpp for compatible quantized models.
- Ollama for convenient local model management where suitable.
- Transformers for research, inference, and training workflows.
- vLLM for compatible GPU-oriented inference deployments when resources permit.
- Other actively maintained local inference backends when they offer a concrete advantage.

Do not install every backend automatically.

Select the smallest practical set that covers the immediate requirements. Document why each backend is included.

Provide a consistent internal interface for:

- Loading or selecting a model.
- Generating responses.
- Streaming output where supported.
- Structured output where supported.
- Tool calling where supported.
- Cancellation and timeouts.
- Reporting errors and resource constraints.
- Recording model identity, configuration, and usage metadata.

Maintain a clear distinction between inference-only quantized models and models that can be trained using their available checkpoints and formats.

Do not assume that every quantized deployment artifact can be directly fine-tuned.

### Model selection policy

Begin with a compatible pretrained instruction model.

Select the actual model only after examining available RAM, VRAM, operating-system constraints, licensing, model format, context requirements, and expected performance.

Support multiple model profiles, for example:

- Lightweight model for everyday interaction.
- Stronger model for demanding local tasks.
- Specialized coding model.
- Embedding model for retrieval.
- Compatible multimodal model when required.

These are logical roles, not mandatory simultaneous model installations.

Use quantization and CPU offloading where appropriate, while measuring the associated latency and memory trade-offs.

Never promise that a particular model will fit or perform well before checking its real requirements.

The core platform must not require a paid cloud inference API to function.

---

# PART IV — WINDOWS-FIRST INFRASTRUCTURE

## 6. Windows as the Initial Development Environment

The first supported development environment is the user's Windows laptop.

Preserve native Windows compatibility wherever practical.

Use a documented, reproducible development setup, including:

- A supported Python environment.
- A dependency lock or equivalent reproducibility mechanism.
- Environment-based configuration.
- Git-based version control.
- Local data directories outside disposable source-code locations.
- Startup, shutdown, and health-check commands.
- Database initialization and migration commands.
- Model installation and verification instructions.
- Test and evaluation commands.

Use PowerShell scripts where appropriate.

If WSL2 or Docker provides a material benefit, keep it optional unless a specific component genuinely requires it.

Document Windows-specific limitations and the supported fallback.

Do not require Linux-only assumptions in the core application.

### Hardware discovery

Build a diagnostic command or utility that reports relevant system capabilities, including:

- OS and runtime versions.
- CPU information.
- Total and available RAM.
- GPU models and detected VRAM where available.
- GPU driver and runtime status.
- Free storage in configured data directories.
- Model files and their metadata.
- Available local inference backends.

Handle unavailable or unsupported measurements gracefully.

Never assume that a detected GPU is automatically usable for every inference or training framework.

### Resource-aware execution

Before expensive tasks, estimate or inspect resource requirements.

For each workload, consider:

- Model size and precision.
- Context length and KV-cache requirements.
- Available RAM and VRAM.
- Training batch size.
- Sequence length.
- Optimizer and gradient memory.
- Dataset size and disk capacity.
- Temporary storage requirements.

Use configurable resource limits and explicit warnings.

Avoid launching jobs likely to exhaust available memory or disk space.

Provide safe cancellation and clear reporting of resource failures.

Do not silently download multi-gigabyte artifacts or begin costly compute workloads.

---

# PART V — MEMORY, KNOWLEDGE, AND IDENTITY

## 7. Persistent Memory Architecture

Implement memory as an independent, inspectable subsystem.

Support categories such as:

- Explicitly saved user preferences.
- Confirmed facts.
- Project-specific context.
- Conversation summaries.
- Task histories.
- Verified successful procedures.
- Failure analyses.
- Research notes.
- Candidate lessons requiring validation.

Each record should include an identifier, creation time, source or provenance, category, version, and applicable retention or access metadata.

Implement:

- Memory creation.
- Memory retrieval.
- Memory updating.
- Memory deletion.
- Search and deduplication.
- Contradiction handling.
- Relevance scoring.
- Source tracking.
- User-visible memory inspection.
- Backup and restoration.

Use structured storage for authoritative facts. Introduce embeddings and semantic search where useful.

Store source documents and larger artifacts separately from memory summaries, using references where appropriate.

Treat uncertain inferences as uncertain. Do not silently convert a guess into a permanent fact.

Memory must remain available across model replacements and application upgrades.

Deleting a memory must not require retraining the model.

## 8. Knowledge Repository

Create a local knowledge system for approved documents, technical notes, code references, research findings, and project documentation.

Support a staged ingestion pipeline:

1. Validate file types and sizes.
2. Extract supported content.
3. Normalize and chunk text.
4. Preserve document identifiers and metadata.
5. Index searchable content.
6. Retrieve relevant passages.
7. Provide source references.
8. Reindex changed documents.
9. Remove deleted documents and derived index entries.

Make supported formats explicit.

Use local processing by default. Do not send documents to external services unless explicitly configured and approved.

Protect against malicious instructions embedded in retrieved content.

Retrieved information must not override system policies or grant tool permissions.

## 9. Persistent AI Identity and Configuration

Maintain versioned, human-readable configuration for:

- System identity and operating principles.
- Selected model profiles.
- Tool permissions.
- Memory behavior.
- Task limits.
- Learning policies.
- Evaluation thresholds.
- Runtime settings.

Separate stable system configuration from conversation-specific instructions and untrusted retrieved content.

Do not rely on one enormous system prompt to implement every capability.

Use modular instructions and explicit configuration schemas.

---

# PART VI — REASONING, PLANNING, AND ACTION

## 10. Task Orchestration

Implement bounded task orchestration.

The system should be able to:

1. Understand a requested outcome.
2. Identify constraints and acceptance criteria.
3. Decide whether a direct answer or tool use is appropriate.
4. Produce a concise execution plan for complex tasks.
5. Select permitted tools.
6. Execute actions within configured limits.
7. inspect outputs and errors.
8. Retry recoverable failures when justified.
9. Validate results against expected outcomes.
10. Summarize what was accomplished.
11. Record useful lessons when appropriate.

Use explicit task states, such as pending, running, waiting for approval, completed, failed, and cancelled.

Implement configurable limits for execution steps, retries, timeouts, and resource consumption.

Support cancellation and recovery from interrupted tasks.

Do not implement an uncontrolled autonomous loop.

The system must never report successful execution based solely on generated text.

## 11. Local Tool Registry

Create a typed, permission-controlled tool registry.

Start with a small number of genuinely useful tools:

- Read approved project files.
- Search indexed documentation.
- Perform calculations.
- Inspect project configuration.
- Run approved automated tests.
- Execute constrained code tasks.
- Manage research notes and experiments.

Expand only after the execution and permission model is reliable.

For every tool, define:

- Purpose and input schema.
- Output schema.
- Allowed resources.
- Permissions required.
- Timeouts and limits.
- Failure behavior.
- Logging requirements.
- Verification procedure.

Generated code must run in a controlled environment with limited filesystem access, no unnecessary network access, and resource restrictions where technically enforceable.

Treat file modifications, external communications, destructive actions, and changes to production configuration as separate permission classes.

Require explicit authorization for consequential operations.

---

# PART VII — CONTINUOUS LEARNING AND MODEL RESEARCH

## 12. Define Learning as Multiple Separate Systems

Build an explicit learning architecture rather than a vague feature called self-learning.

### Level 1: Contextual learning

Improve performance by retrieving relevant memories and approved knowledge.

No model-weight modification is required.

### Level 2: Procedural learning

Create reusable skills from successful, verified task procedures.

Version them, test them, and reject unsafe or unreliable candidates.

### Level 3: Behavioral improvement

Improve system instructions, tool descriptions, task planning, retrieval, and orchestration using measured evidence.

Version important changes and compare them against a baseline.

### Level 4: Dataset development

Create curated training examples from eligible task outcomes, corrections, and approved research data.

Maintain provenance, quality labels, deduplication, and appropriate privacy controls.

Separate training, validation, and held-out testing data.

Do not include sensitive information without a legitimate, authorized basis.

### Level 5: Fine-tuning

Implement optional fine-tuning using compatible open-weight models and suitable methods such as LoRA or QLoRA.

Support configurable training parameters, checkpoint management, experiment metadata, validation, and failure recovery.

Make training reproducible to the extent permitted by the software and hardware.

### Level 6: Continued pretraining and foundational research

Keep a documented research path for continued pretraining or training smaller models from scratch.

Treat these as advanced research projects requiring appropriate data, compute, evaluation, and substantial engineering.

Do not make foundational training a prerequisite for the first usable system.

## 13. The Controlled Improvement Loop

Implement this workflow:

1. Observe a real task outcome.
2. Determine whether the outcome is sufficiently verifiable.
3. Identify a concrete weakness or improvement opportunity.
4. Propose a change, lesson, skill, or training example.
5. Check provenance, privacy, and data quality.
6. Create or update a regression test.
7. Evaluate the candidate change.
8. Compare it against a versioned baseline.
9. Reject changes that fail acceptance criteria.
10. Preserve successful changes with version metadata.
11. Keep previous working versions available for rollback.
12. Record the evidence supporting the decision.

Do not assume that all user feedback is correct, all successful outputs are generalizable, or all self-generated training examples are useful.

Do not fine-tune automatically after every conversation.

Use a reviewable, reproducible workflow.

If no meaningful improvement is demonstrated, retain the existing version.

## 14. Catastrophic Forgetting and Regression Prevention

Any model adaptation workflow must evaluate both newly targeted tasks and previously supported capabilities.

Maintain a regression suite containing representative historical examples.

Compare:

- Task success.
- Correctness.
- Tool use.
- Memory behavior.
- Safety and permissions.
- General capabilities.
- Latency.
- Resource consumption.

Define minimum acceptance thresholds before training begins.

Promote candidate checkpoints only after they pass the required checks.

Retain baseline checkpoints, configuration, evaluation reports, and rollback instructions.

Never overwrite the only copy of a working model.

---

# PART VIII — DATA, EXPERIMENTS, AND MODEL REGISTRY

## 15. Dataset Management

Build a local dataset management subsystem with:

- Dataset identifiers and versions.
- Data provenance.
- License and usage metadata where applicable.
- Validation rules.
- Schema validation.
- Duplicate detection.
- Quality statistics.
- Train/validation/test split metadata.
- Dataset fingerprints.
- Optional redaction workflows.
- Reproducible preprocessing.
- Data retention and deletion procedures.

Never assume downloaded or generated data is automatically suitable for training.

Respect model licenses, dataset terms, privacy requirements, and applicable law.

Keep original datasets immutable wherever practical and store transformations as versioned derived artifacts.

## 16. Experiment Tracking

Every training or evaluation experiment should record:

- Experiment identifier.
- Timestamp.
- Objective and hypothesis.
- Baseline model.
- Candidate model.
- Dataset versions.
- Code revision.
- Dependency and configuration versions.
- Hardware and runtime information.
- Training parameters.
- Random seeds where applicable.
- Resource usage.
- Evaluation results.
- Observed failures.
- Conclusions and next steps.

Support resuming interrupted jobs when the selected training framework permits it.

Store experiment metadata separately from large checkpoints.

Make it possible to determine which dataset, code version, configuration, and checkpoint produced a result.

## 17. Model Registry

Maintain a local registry of models and checkpoints.

Track:

- Model identifier and version.
- Base model and architecture.
- Source and license.
- Weight format and precision.
- Quantization information.
- Context capabilities.
- Training history.
- Evaluation results.
- Runtime compatibility.
- File locations and integrity checks.
- Deployment status.

Distinguish downloaded models, trained checkpoints, quantized derivatives, evaluation candidates, and active production models.

Support selecting a model without losing previous versions.

Do not automatically trust or execute downloaded model artifacts.

---

# PART IX — SCALABLE HARDWARE AND STORAGE

## 18. Design for Upgradeability

The first deployment must work with limited resources.

Future upgrades may include:

- More system RAM.
- A dedicated NVIDIA GPU.
- A GPU with more VRAM.
- Additional GPUs.
- Faster local storage.
- Separate storage for model weights and datasets.
- A dedicated AI workstation.
- A separate training machine.
- Multiple cooperating local machines.

Do not require these upgrades for the core application to run.

Use configuration-driven resource selection and capability detection.

Separate application state, model storage, datasets, checkpoints, logs, and temporary files.

Make storage paths configurable and provide a documented migration process.

Support a single-machine deployment first.

For future multi-GPU and distributed workloads, isolate execution behind suitable interfaces and adopt compatible training frameworks when needed.

Do not assume that multiple GPUs automatically combine their VRAM or that a larger GPU can be added without driver, framework, power, cooling, and compatibility considerations.

Document practical hardware requirements for each workload.

When hardware is upgraded, the preferred path should be:

1. Detect and validate the new hardware.
2. Install or configure compatible drivers and runtimes.
3. Re-run capability checks.
4. Adjust model and training profiles.
5. Benchmark the updated system.
6. Continue using existing data, code, experiments, and memory.

A hardware upgrade should not require recreating the entire project.

---

# PART X — SECURITY, PRIVACY, AND OFFLINE OPERATION

## 19. Local Privacy by Default

The default configuration must not transmit conversations, memories, documents, source code, datasets, model checkpoints, or training examples to external AI providers.

Avoid hidden telemetry and external dependencies that are unnecessary for operation.

Document every optional network-dependent feature.

Provide an offline operating mode that works with already-installed dependencies, locally available models, and initialized databases.

Make the distinction between fully offline operation and local-first operation explicit.

Internet access may be required initially to download packages, models, or documentation. After setup, core capabilities should continue to function offline wherever their dependencies permit.

## 20. Security Controls

Implement:

- Local secret management through appropriate environment or operating-system facilities.
- No secrets in source control.
- Input and output validation.
- Filesystem access restrictions.
- Tool permissions.
- Execution limits.
- Safe model and dataset handling.
- Dependency review.
- Backup and restoration.
- Database migrations.
- Audit records for sensitive operations.
- Protection against prompt injection.
- Explicit authorization for destructive actions.
- Safe handling of untrusted documents and generated code.

Avoid exposing administrative endpoints to the local network by default.

Do not treat localhost as a substitute for authentication when an endpoint can perform privileged actions.

Do not disable security controls merely to simplify agent execution.

---

# PART XI — DOCUMENTATION AND CONTINUITY

## 21. Documentation Is Part of the Product

Create and maintain a canonical documentation system.

Use a structure similar to the following, adapting it to the existing repository:

docs/
  vision.md
  architecture.md
  system-map.md
  implementation-status.md
  development-roadmap.md
  windows-setup.md
  offline-operation.md
  configuration.md
  model-runtime.md
  memory-and-retrieval.md
  tools-and-permissions.md
  learning-pipeline.md
  dataset-management.md
  training-and-finetuning.md
  evaluation-and-benchmarks.md
  experiment-tracking.md
  model-registry.md
  hardware-upgrades.md
  security-and-privacy.md
  backup-and-recovery.md
  troubleshooting.md
  known-limitations.md
  decisions/
  experiments/
  operations/

Maintain the following repository-level files where appropriate:

README.md
AGENTS.md
CONTRIBUTING.md
CHANGELOG.md
ROADMAP.md
ARCHITECTURE.md
.env.example

Use a single authoritative file for each major piece of information. Avoid maintaining conflicting copies of the same configuration or project status.

### Mandatory continuity files

Maintain these documents as the project evolves:

**PROJECT_STATE.md**

The current working state, implemented capabilities, active branch or revision when available, recent verified results, known blockers, and next actions.

**docs/architecture.md**

Subsystem boundaries, interfaces, dependencies, data flow, and architectural constraints.

**docs/decisions/**

Architecture Decision Records documenting important choices, alternatives, rationale, and consequences.

**docs/experiments/**

Experiment hypotheses, configurations, datasets, results, conclusions, and follow-up work.

**ROADMAP.md**

Current milestone, completed milestones, priorities, dependencies, and future research directions.

**CHANGELOG.md**

Meaningful changes and compatibility notes.

**AGENTS.md**

Instructions for future coding agents, including repository conventions, testing requirements, security constraints, documentation rules, and mandatory startup procedures.

### Agent handoff protocol

At the start of each substantial coding session:

1. Read AGENTS.md.
2. Read PROJECT_STATE.md.
3. Read the current roadmap.
4. Inspect recent changes and the current version-control state.
5. Read relevant architectural decisions.
6. Run appropriate baseline checks.
7. Continue from the highest-value unfinished task.

Before concluding a session:

1. Update implementation status.
2. Record verified test results.
3. Document new decisions.
4. Record unresolved issues and limitations.
5. Update the roadmap when priorities change.
6. Write precise next steps for the next agent.
7. Ensure documentation matches the implementation.

Never claim a feature is implemented merely because it appears in the roadmap.

The next agent must be able to resume without depending on hidden conversation history.

---

# PART XII — TESTING AND QUALITY

## 22. Permanent Verification System

Create automated tests for:

- Application startup.
- Configuration validation.
- Model-provider interfaces.
- Conversation persistence.
- Memory creation and retrieval.
- Memory updates and deletion.
- Tool permission enforcement.
- Task execution and cancellation.
- Error handling.
- Dataset validation.
- Experiment metadata.
- Model registry operations.
- Backup and restoration.
- Offline operation.
- Resource-limit enforcement.
- Regression behavior.

Use mocked providers for deterministic unit tests when appropriate, but separately verify that real local model inference works when a compatible model is configured.

Do not allow mocked tests to be presented as proof that a real model or external integration works.

Create a repeatable evaluation command and preserve the results.

Report skipped tests and missing hardware explicitly.

Use static checks, formatting, dependency checks, and integration tests appropriate to the stack.

Do not claim all tests pass when they were not run.

## 23. Measurable Success

Track the metrics that actually matter:

- Percentage of tests passed.
- Task completion rate on a defined benchmark.
- Correctness and groundedness.
- Tool execution success.
- Memory retrieval quality.
- Regression rate.
- Median and tail latency.
- RAM and VRAM usage.
- Storage consumption.
- Training time and throughput when applicable.
- Inference tokens per second where measurable.
- Model quality versus baseline.
- Recovery from failures.

Never invent performance figures or display fabricated progress.

Clearly distinguish measured results from estimates.

---

# PART XIII — USER EXPERIENCE

## 24. Build a Functional Research Workspace

Provide a polished interface that reflects the seriousness of the project.

Depending on the existing application, include:

- Main conversation workspace.
- Conversation and project history.
- Model selection and status.
- Memory inspection and management.
- Tool activity and execution results.
- Active tasks and task history.
- Local resource and hardware diagnostics.
- Dataset management.
- Experiment tracking.
- Model registry.
- Evaluation reports.
- Learning pipeline status.
- Configuration and permissions.
- Documentation access.

Implement features incrementally.

Every visible control must perform its intended action or be explicitly marked as unavailable or planned.

Do not create fake training progress bars, fake model performance, simulated GPU usage, or invented autonomous activity.

Keep the interface usable on the available hardware and do not require an elaborate frontend before the backend works.

---

# PART XIV — IMPLEMENTATION ROADMAP

## 25. Phase 0: Discovery and Baseline

Inspect the entire repository and environment.

Deliver:

- Baseline report.
- Existing architecture map.
- Hardware capability report.
- Identified risks and missing dependencies.
- Prioritized implementation plan.
- Initial acceptance criteria.

Preserve existing work.

## 26. Phase 1: Reliable Local Foundation

Implement:

- Reproducible Windows setup.
- Application startup and configuration.
- Local model integration.
- Persistent conversations.
- Logging and health checks.
- Automated tests.
- Initial documentation.

Acceptance criterion: a real local model can answer a message and the conversation survives an application restart.

## 27. Phase 2: Memory and Knowledge

Implement:

- Persistent structured memory.
- Retrieval and relevance selection.
- Memory inspection, updates, and deletion.
- Document indexing where supported.
- Source provenance.
- Backup and recovery.

Acceptance criterion: approved information can be saved, retrieved in a later session, inspected, and deleted.

## 28. Phase 3: Tools and Task Orchestration

Implement:

- Typed tool registry.
- Bounded planning and execution.
- Permission enforcement.
- Task states.
- Error recovery.
- Execution logs.
- Independent result verification.

Acceptance criterion: the AI can complete and verify a safe, bounded task using a real local tool.

## 29. Phase 4: Evaluation and Feedback

Implement:

- Fixed baseline benchmarks.
- Task outcome recording.
- Failure analysis.
- Regression tests.
- Versioned skill candidates.
- Comparison reports.

Acceptance criterion: the system can identify a concrete weakness, propose a change, and determine through testing whether the change improves results.

## 30. Phase 5: Dataset and Fine-Tuning Infrastructure

Implement:

- Curated dataset management.
- Data quality checks.
- Experiment tracking.
- Training configuration.
- Compatible LoRA/QLoRA workflows.
- Checkpoint versioning.
- Baseline comparison.
- Rollback.

Acceptance criterion: a small, appropriate experiment can be run when compatible resources are available, with reproducible configuration and evaluation results.

If local hardware cannot support the experiment, complete the workflow foundations and document the minimum required resources.

## 31. Phase 6: Research and Capability Expansion

Investigate and incrementally implement:

- Improved local inference.
- Stronger compatible models.
- Better memory retrieval.
- Coding and specialized skills.
- Multimodal capabilities.
- More advanced model adaptation.
- Efficient training techniques.
- Improved evaluation methods.

Every feature must justify its cost and demonstrate value.

## 32. Phase 7: Hardware and Deployment Evolution

Support:

- Larger models.
- Increased RAM and VRAM.
- Expanded model and dataset storage.
- Optional multi-GPU execution.
- Separate training and inference processes.
- Future workstation or distributed compute setups.

Validate each configuration independently.

Preserve data and model lineage across upgrades.

---

# PART XV — DEVELOPMENT RULES

## 33. How to Execute the Work

Operate incrementally and independently.

For each implementation task:

1. Identify the requirement.
2. Inspect relevant code.
3. Define a testable acceptance criterion.
4. Implement the smallest coherent change.
5. Run relevant tests.
6. Inspect failures and correct them.
7. Update documentation.
8. Record the resulting project state.

Prefer small, understandable changes over giant rewrites.

Reuse working components when reasonable.

Avoid speculative infrastructure, unnecessary dependencies, duplicated business logic, hardcoded machine paths, and unexplained configuration values.

Use version control and preserve the ability to revert changes.

Never remove working functionality without a documented reason.

Do not make expensive model downloads, destructive database migrations, or long-running training jobs part of routine startup.

Do not wait for a perfect architecture before delivering a useful increment.

Do not stop after producing a plan or a set of placeholder files.

If one subsystem is blocked, implement the independent components that can still be completed.

If a feature cannot be implemented honestly, document why, define its interface where useful, and identify the precise next step.

## 34. Dependency and Change Management

Prefer maintained, compatible dependencies with clear licensing.

Verify current official documentation when introducing significant frameworks or model runtimes.

Avoid pinning incompatible package versions or assuming a GPU runtime is available.

Document major upgrades, migrations, and breaking changes.

Preserve model and dataset checksums where useful.

Keep source code, durable application data, training datasets, and model artifacts in clearly separated locations.

Ensure that backups include all state required for recovery, not merely source files.

---

# PART XVI — INITIAL EXECUTION ORDER

## 35. Start Work Immediately

Follow this order without waiting for routine approval.

**Step 1:** Inspect the repository, runtime environment, available hardware, and existing documentation.

**Step 2:** Produce a concise baseline report and identify the current project state.

**Step 3:** Create or improve AGENTS.md, PROJECT_STATE.md, ROADMAP.md, and the architecture documentation.

**Step 4:** Identify the smallest reliable local-first implementation path based on actual constraints.

**Step 5:** Implement the highest-value incomplete component.

**Step 6:** Establish or improve the automated test suite.

**Step 7:** Run the available checks and verify the working path.

**Step 8:** Correct defects found during verification.

**Step 9:** Update all relevant continuity documents.

**Step 10:** Report exactly what is working, what was tested, what remains incomplete, and the next implementation task.

Continue with the next logical milestone whenever the environment and execution budget permit.

Do not claim completion merely because a milestone was planned.

---

# FINAL DIRECTIVE

We are building a long-lived AI research platform, not a disposable application.

The system must begin on a Windows laptop, function with locally available resources, and grow through modular architecture, configurable hardware profiles, persistent knowledge, reproducible experiments, and carefully validated improvements.

Every meaningful engineering decision must preserve future optionality.

Every experiment must preserve enough information to be understood and reproduced.

Every significant feature must have a clear implementation status.

Every coding agent must leave the project easier to understand and continue than it found it.

The repository and its documentation must preserve the project's technical memory across sessions, models, and developers.

The system must distinguish genuine learning from stored memory, genuine model training from prompt changes, and demonstrated improvement from unsupported claims.

# Development Approach

Inspect and understand the complete existing system before making changes.

Identify the current architecture, available resources, dependencies, implemented features, limitations, and technical debt.

Establish a documented baseline and prioritize the next milestones.

Implement changes incrementally, with clear objectives and acceptance criteria.

Test each milestone and verify its results before considering it complete.

Update documentation, configuration examples, and development records alongside implementation changes.

Commit meaningful, verified changes to Git so progress remains traceable and recoverable. (git new repo: https://github.com/mukeshbhuriya-ctrl/project_genesis.git)

Reassess the architecture periodically to support future research, models, and hardware upgrades.

**Begin by understanding the complete existing system. Then build, test, document, and improve it one reliable milestone at a time. Preserve the vision, preserve continuity, and never sacrifice engineering truth for appearances.**