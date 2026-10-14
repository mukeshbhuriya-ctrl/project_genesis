# Project Genesis

> A local-first, self-improving AI research and model development platform.

## Vision

Project Genesis is a persistent AI development platform designed to run entirely on local hardware — starting with a Windows laptop — and grow into a serious research and training infrastructure over many years.

**Core capabilities (planned):**
- Local model inference (llama.cpp, Ollama, Transformers)
- Persistent conversations with long-term memory
- Structured knowledge retrieval (RAG)
- Tool use and bounded task orchestration
- Dataset management and quality control
- Supervised fine-tuning (LoRA/QLoRA)
- Experiment tracking and model registry
- Evaluation-driven improvement

## Quick Start

### Prerequisites
- Windows 10/11 (primary), Linux/macOS (future)
- Python 3.12+
- Git 2.40+
- NVIDIA GPU (optional, recommended for inference)
- Ollama (for local model serving)

### Setup

```powershell
# Clone
git clone https://github.com/mukeshbhuriya-ctrl/project_genesis.git
cd project_genesis

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"

# Configure
copy .env.example .env
# Edit .env with your paths and preferences

# Initialize database
python -m genesis.cli init

# Run tests
python -m pytest tests/ -q

# Start the application
python -m genesis.cli serve
```

### Hardware Detection

```powershell
python -m genesis.cli hardware
```

## Project Structure

```
project_genesis/
├── genesis/              # Python application package
│   ├── api/              # REST API (FastAPI)
│   ├── core/             # Configuration, logging, lifecycle
│   ├── models/           # Model provider abstraction
│   ├── memory/           # Persistent memory subsystem
│   ├── knowledge/        # Document ingestion & retrieval
│   ├── tools/            # Tool registry & execution
│   ├── tasks/            # Task orchestration
│   ├── training/         # Fine-tuning & experiment tracking
│   ├── evaluation/       # Benchmarks & regression tests
│   └── cli.py            # CLI entry point
├── tests/                # Automated tests
├── docs/                 # Documentation
├── data/                 # Runtime data (models, datasets, DB)
├── frontend/             # Web UI (future)
└── scripts/              # Utility scripts
```

## Documentation

| Document | Purpose |
|----------|---------|
| [Architecture](docs/architecture.md) | Subsystem boundaries, interfaces, data flow |
| [Roadmap](ROADMAP.md) | Milestones and priorities |
| [Project State](PROJECT_STATE.md) | Current working state and next steps |
| [Windows Setup](docs/windows-setup.md) | Detailed setup guide |
| [Configuration](docs/configuration.md) | All configuration options |
| [Security](docs/security-and-privacy.md) | Security model and privacy controls |

## License

TBD — See `LICENSE` file when added.
